#include <GL/glew.h>
#include "Application.h"
#include "Config.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "imgui.h"
#include "backend/imgui_impl_glfw.h"
#include "backend/imgui_impl_opengl3.h"
#include <iostream>
#include <cstdio>

#define CUDA_CHECK(err) do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// Global sim params (needed by ParticleSystem)
SimulationParams simParams;

Application::Application()
    : m_window(nullptr)
    , m_camera(Config::DEFAULT_WINDOW_WIDTH, Config::DEFAULT_WINDOW_HEIGHT)
    , m_particleSystem(Config::PARTICLE_COUNT, Config::DEFAULT_WINDOW_WIDTH, Config::DEFAULT_WINDOW_HEIGHT)
    , m_physicsEngine(Config::PARTICLE_COUNT, 
                      (int)(Config::DEFAULT_WINDOW_WIDTH / Config::GRID_CELL_SIZE) + 1,
                      (int)(Config::DEFAULT_WINDOW_HEIGHT / Config::GRID_CELL_SIZE) + 1)
    , m_paused(true)
    , m_velocityToHueRange(Config::DEFAULT_VELOCITY_TO_HUE_RANGE)
    , m_windowWidth(Config::DEFAULT_WINDOW_WIDTH)
    , m_windowHeight(Config::DEFAULT_WINDOW_HEIGHT)
{
    // Initialize simulation parameters
    m_simParams.gravity = Config::DEFAULT_GRAVITY;
    m_simParams.dt = Config::DEFAULT_DT;
    m_simParams.dampening = Config::DEFAULT_DAMPENING;
    m_simParams.restitution = Config::DEFAULT_RESTITUTION;
    m_simParams.bounds_width = (float)m_windowWidth;
    m_simParams.bounds_height = (float)m_windowHeight;
    
    // Initialize grid parameters
    updateGridParams();
    
    // Set global simParams
    simParams = m_simParams;
}

Application::~Application() {
    cleanup();
}

bool Application::initialize() {
    try {
        initializeGLFW();
        initializeCUDA();
        
        m_renderer.initialize();
        m_particleSystem.initialize(m_renderer);
        m_physicsEngine.initialize();
        
        // Initialize particles
        m_particleSystem.reset(m_windowWidth, m_windowHeight);
        
        initializeImGui();
        setupCallbacks();
        
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void Application::initializeGLFW() {
    if (!glfwInit()) {
        throw std::runtime_error("Failed to initialize GLFW");
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    
    m_window = glfwCreateWindow(m_windowWidth, m_windowHeight, 
                                 "CUDA Gravity Simulation", nullptr, nullptr);
    if (!m_window) {
        throw std::runtime_error("Failed to create GLFW window");
    }
    
    glfwMakeContextCurrent(m_window);
    glfwSetWindowSizeLimits(m_window, m_windowWidth, m_windowHeight, 
                            GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwSwapInterval(0); // Disable vsync
    
    if (glewInit() != GLEW_OK) {
        throw std::runtime_error("Failed to initialize GLEW");
    }
}

void Application::initializeCUDA() {
    CUDA_CHECK(cudaGLSetGLDevice(0));
}

void Application::initializeImGui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void Application::setupCallbacks() {
    glfwSetWindowUserPointer(m_window, this);
    
    glfwSetWindowSizeCallback(m_window, windowSizeCallback);
    glfwSetScrollCallback(m_window, scrollCallback);
}

void Application::windowSizeCallback(GLFWwindow* window, int width, int height) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    
    glViewport(0, 0, width, height);
    app->m_windowWidth = width;
    app->m_windowHeight = height;
    
    app->m_camera.setWindowSize(width, height);
    app->m_simParams.bounds_width = (float)width;
    app->m_simParams.bounds_height = (float)height;
    app->updateGridParams();
}

void Application::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    Application* app = static_cast<Application*>(glfwGetWindowUserPointer(window));
    app->m_camera.adjustZoom((float)yoffset * 0.1f + 1.0f);
}

void Application::updateGridParams() {
    m_gridParams.grid_width = (int)(m_windowWidth / Config::GRID_CELL_SIZE) + 1;
    m_gridParams.grid_height = (int)(m_windowHeight / Config::GRID_CELL_SIZE) + 1;
    m_gridParams.cell_size = Config::GRID_CELL_SIZE;
}

void Application::run() {
    while (!glfwWindowShouldClose(m_window)) {
        glfwPollEvents();
        handleInput();
        update();
        renderFrame();
        renderUI();
        glfwSwapBuffers(m_window);
    }
}

void Application::handleInput() {
    // Add any keyboard/mouse input handling here
}

void Application::update() {
    if (!m_paused) {
        simParams = m_simParams; // Update global params
        
        m_particleSystem.mapResources();
        m_physicsEngine.simulate(m_particleSystem.getParticles(), m_simParams, m_gridParams);
        m_particleSystem.unmapResources();
    }
}

void Application::renderFrame() {
    m_renderer.beginFrame();
    m_renderer.render(m_particleSystem.getCount(), m_camera, m_velocityToHueRange);
}

void Application::renderUI() {
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Settings");
    ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)", 
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Particles: %d", Config::PARTICLE_COUNT);
    ImGui::Separator();

    ImGui::Checkbox("Pause", &m_paused);
    ImGui::SliderFloat("Gravity", &m_simParams.gravity, -5000.0f, 0.0f);
    ImGui::SliderFloat("Dampening", &m_simParams.dampening, 0.0f, 1.0f);
    ImGui::SliderFloat("Restitution", &m_simParams.restitution, 0.0f, 1.0f);
    ImGui::SliderFloat("Simulation dt", &m_simParams.dt, 0.0001f, 0.3f, 
                       "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Hue range", &m_velocityToHueRange, 10.0f, 300.0f);

    if (ImGui::Button("Reset Particles")) {
        resetParticles();
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::resetParticles() {
    m_particleSystem.reset(m_windowWidth, m_windowHeight);
}

void Application::cleanup() {
    m_particleSystem.cleanup();
    m_physicsEngine.cleanup();
    m_renderer.cleanup();
    
    if (m_window) {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        glfwDestroyWindow(m_window);
        glfwTerminate();
        m_window = nullptr;
    }
}
