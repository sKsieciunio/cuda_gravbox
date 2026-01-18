#include "Application.h"
#include "Config.h"
#include "backend/imgui_impl_glfw.h"
#include "backend/imgui_impl_opengl3.h"
#include "imgui.h"
#include <GL/glew.h>
#include <cstdio>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_CHECK(err)                                                   \
    do                                                                    \
    {                                                                     \
        cudaError_t _err = (err);                                         \
        if (_err != cudaSuccess)                                          \
        {                                                                 \
            fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(_err));                            \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

// Global sim params (needed by ParticleSystem)
SimulationParams simParams;

Application::Application()
    : m_window(nullptr),
      m_camera(Config::DEFAULT_WINDOW_WIDTH, Config::DEFAULT_WINDOW_HEIGHT),
      m_particleSystem(Config::PARTICLE_COUNT, Config::DEFAULT_WINDOW_WIDTH,
                       Config::DEFAULT_WINDOW_HEIGHT),
      m_physicsEngine(
          Config::PARTICLE_COUNT,
          (int)(Config::DEFAULT_WINDOW_WIDTH / Config::GRID_CELL_SIZE) + 1,
          (int)(Config::DEFAULT_WINDOW_HEIGHT / Config::GRID_CELL_SIZE) + 1),
      m_cpuEngine(
          Config::PARTICLE_COUNT,
          (int)(Config::DEFAULT_WINDOW_WIDTH / Config::GRID_CELL_SIZE) + 1,
          (int)(Config::DEFAULT_WINDOW_HEIGHT / Config::GRID_CELL_SIZE) + 1),
      m_paused(true),
      m_velocityToHueRange(Config::DEFAULT_VELOCITY_TO_HUE_RANGE),
      m_particleCount(Config::PARTICLE_COUNT),
      m_particleRadius(Config::PARTICLE_RADIUS),
      m_windowWidth(Config::DEFAULT_WINDOW_WIDTH),
      m_windowHeight(Config::DEFAULT_WINDOW_HEIGHT),
      m_collisionIterations(Config::COLLISION_ITERATIONS),
      m_cudaBlockSize(Config::CUDA_BLOCK_SIZE), m_useCUDA(true),
      m_spawnMode(SpawnMode::DISK_CENTER_EXPLOSION),
      m_coloringMode(ColoringMode::MASS)
{
    m_simParams.gravity = Config::DEFAULT_GRAVITY;
    m_simParams.dt = Config::DEFAULT_DT;
    m_simParams.dampening = Config::DEFAULT_DAMPENING;
    m_simParams.restitution = Config::DEFAULT_RESTITUTION;
    m_simParams.max_speed = Config::DEFAULT_MAX_SPEED;
    m_simParams.bounds_width = (float)m_windowWidth;
    m_simParams.bounds_height = (float)m_windowHeight;
    m_simParams.collision_iterations = m_collisionIterations;
    m_simParams.cuda_block_size = m_cudaBlockSize;
    m_simParams.enable_air_blowers = 0;

    updateGridParams();

    simParams = m_simParams;
}

Application::~Application() { cleanup(); }

bool Application::initialize()
{
    try
    {
        initializeGLFW();
        initializeCUDA();

        m_renderer.initialize();
        m_particleSystem.initialize(m_renderer, m_useCUDA);
        if (m_useCUDA)
        {
            m_physicsEngine.initialize();
        }
        else
        {
            m_cpuEngine.initialize();
        }

        m_particleSystem.reset(m_windowWidth, m_windowHeight, m_particleRadius,
                               m_renderer, m_spawnMode);

        initializeImGui();
        setupCallbacks();

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "Initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void Application::initializeGLFW()
{
    if (!glfwInit())
    {
        throw std::runtime_error("Failed to initialize GLFW");
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);

    m_window = glfwCreateWindow(m_windowWidth, m_windowHeight,
                                "CUDA Gravity Simulation", nullptr, nullptr);
    if (!m_window)
    {
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwGetWindowPos(m_window, &m_windowPosX, &m_windowPosY);

    glfwMakeContextCurrent(m_window);
    glfwSetWindowSizeLimits(m_window, m_windowWidth, m_windowHeight,
                            GLFW_DONT_CARE, GLFW_DONT_CARE);
    glfwSwapInterval(0); // Disable vsync

    if (glewInit() != GLEW_OK)
    {
        throw std::runtime_error("Failed to initialize GLEW");
    }
}

void Application::initializeCUDA() { CUDA_CHECK(cudaGLSetGLDevice(0)); }

void Application::initializeImGui()
{
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui_ImplGlfw_InitForOpenGL(m_window, true);
    ImGui_ImplOpenGL3_Init("#version 330");
}

void Application::setupCallbacks()
{
    glfwSetWindowUserPointer(m_window, this);

    glfwSetScrollCallback(m_window, scrollCallback);
}

void Application::windowSizeCallback(GLFWwindow *window, int width,
                                     int height)
{
    // Deprecated, logic moved to checkForWindowResize
}

void Application::checkForWindowResize()
{
    int width, height;
    glfwGetWindowSize(m_window, &width, &height);

    int x, y;
    glfwGetWindowPos(m_window, &x, &y);

    bool sizeChanged = (width != m_windowWidth || height != m_windowHeight);
    bool posChanged = (x != m_windowPosX || y != m_windowPosY);

    if (sizeChanged)
    {
        float shiftX = (float)(m_windowPosX - x);
        float shiftY = (float)((y + height) - (m_windowPosY + m_windowHeight));

        if (std::abs(shiftX) > 0.0f || std::abs(shiftY) > 0.0f)
        {
            m_particleSystem.shiftParticles(shiftX, shiftY, &m_renderer);
        }

        m_windowWidth = width;
        m_windowHeight = height;
        m_windowPosX = x;
        m_windowPosY = y;

        glViewport(0, 0, width, height);
        m_camera.setWindowSize(width, height);
        m_simParams.bounds_width = (float)width;
        m_simParams.bounds_height = (float)height;
        updateGridParams();

        m_physicsEngine.resize(m_gridParams.grid_width, m_gridParams.grid_height);
        m_cpuEngine.resize(m_gridParams.grid_width, m_gridParams.grid_height);
    }
    else if (posChanged)
    {
        m_windowPosX = x;
        m_windowPosY = y;
    }
}

void Application::scrollCallback(GLFWwindow *window, double xoffset,
                                 double yoffset)
{
    Application *app =
        static_cast<Application *>(glfwGetWindowUserPointer(window));
    app->m_camera.adjustZoom((float)yoffset * 0.1f + 1.0f);
}

void Application::updateGridParams()
{
    m_gridParams.grid_width = (int)(m_windowWidth / Config::GRID_CELL_SIZE) + 1;
    m_gridParams.grid_height = (int)(m_windowHeight / Config::GRID_CELL_SIZE) + 1;
    m_gridParams.cell_size = Config::GRID_CELL_SIZE;
}

void Application::run()
{
    while (!glfwWindowShouldClose(m_window))
    {
        glfwPollEvents();
        handleInput();
        update();
        renderFrame();
        renderUI();
        glfwSwapBuffers(m_window);
    }
}

void Application::handleInput()
{
    // keyboard/mouse input handling here
}

void Application::update()
{
    checkForWindowResize();

    if (!m_paused)
    {
        simParams = m_simParams;
        if (m_useCUDA)
        {
            m_particleSystem.mapResourcesCUDA();
            m_physicsEngine.simulate(m_particleSystem.getParticles(), m_simParams,
                                     m_gridParams);
            m_particleSystem.unmapResourcesCUDA();
        }
        else
        {
            m_particleSystem.mapResourcesCPU(m_renderer);
            m_cpuEngine.simulate(m_particleSystem.getParticles(), m_simParams,
                                 m_gridParams);
            glBindBuffer(GL_ARRAY_BUFFER, m_renderer.getVBO_PosX());
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, m_renderer.getVBO_PosY());
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, m_renderer.getVBO_VelX());
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, m_renderer.getVBO_VelY());
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, m_renderer.getVBO_Radius());
            glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }
}

void Application::renderFrame()
{
    m_renderer.beginFrame();
    m_renderer.render(m_particleSystem.getCount(), m_camera, m_velocityToHueRange,
                      m_coloringMode);
}

void Application::renderUI()
{
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::Begin("Menu");
    ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)",
                1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
    ImGui::Text("Particles: %d", m_particleCount);
    ImGui::Separator();
    ImGui::Text("Backend");
    static int backend = 0; // 0 = CUDA, 1 = CPU
    const char *backends[] = {"GPU (CUDA)", "CPU"};
    if (ImGui::Combo("Physics Backend", &backend, backends,
                     IM_ARRAYSIZE(backends)))
    {
        bool newUseCUDA = (backend == 0);
        if (newUseCUDA != m_useCUDA)
        {
            m_paused = true;
            m_particleSystem.cleanup();
            if (m_useCUDA)
                m_physicsEngine.cleanup();
            else
                m_cpuEngine.cleanup();
            m_useCUDA = newUseCUDA;
            m_particleSystem.initialize(m_renderer, m_useCUDA);
            if (m_useCUDA)
                m_physicsEngine.initialize();
            else
                m_cpuEngine.initialize();
            resetParticles();
        }
    }

    ImGui::Checkbox("Pause", &m_paused);

    ImGui::Separator();
    ImGui::Text("Particle Configuration");

    static int spawnMode = (int)m_spawnMode;
    const char *spawnModes[] = {"Uniform", "Disk Corner", "Disk Center Explosion",
                                "Disk vs Wall"};
    if (ImGui::Combo("Spawn Mode", &spawnMode, spawnModes,
                     IM_ARRAYSIZE(spawnModes)))
    {
        m_spawnMode = (SpawnMode)spawnMode;
        resetParticles();
    }

    static int newParticleCount = m_particleCount;
    static float newParticleRadius = m_particleRadius;

    bool particleCountChanged =
        ImGui::SliderInt("Particle Count", &newParticleCount, 10, 500000, "%d",
                         ImGuiSliderFlags_Logarithmic);
    bool particleRadiusChanged =
        ImGui::SliderFloat("Particle Radius", &newParticleRadius, 0.5f, 50.0f);

    if (particleCountChanged || particleRadiusChanged)
    {
        reinitializeSimulation(newParticleCount, newParticleRadius);
    }

    ImGui::Separator();
    ImGui::Text("Physics Parameters");

    bool airBlowers = m_simParams.enable_air_blowers != 0;
    if (ImGui::Checkbox("Enable Air Blowers", &airBlowers))
    {
        m_simParams.enable_air_blowers = airBlowers ? 1 : 0;
    }

    ImGui::SliderFloat("Gravity", &m_simParams.gravity, -5000.0f, 0.0f);
    ImGui::SliderFloat("Dampening", &m_simParams.dampening, 0.0f, 1.0f);
    ImGui::SliderFloat("Restitution", &m_simParams.restitution, 0.0f, 1.0f);
    ImGui::SliderFloat("Max Speed", &m_simParams.max_speed, 100.0f, 5000.0f);
    ImGui::SliderFloat("Simulation dt", &m_simParams.dt, 0.0001f, 0.3f, "%.4f",
                       ImGuiSliderFlags_Logarithmic);

    if (ImGui::SliderInt("Collision Iterations", &m_collisionIterations, 1, 50))
    {
        m_simParams.collision_iterations = m_collisionIterations;
    }
    if (ImGui::SliderInt("CUDA Block Size", &m_cudaBlockSize, 32, 1024))
    {
        m_simParams.cuda_block_size = m_cudaBlockSize;
    }

    ImGui::Separator();
    ImGui::Text("Rendering");

    static int coloringMode = (int)m_coloringMode;
    const char *coloringModes[] = {"Velocity", "ID", "Mass"};
    if (ImGui::Combo("Coloring Mode", &coloringMode, coloringModes,
                     IM_ARRAYSIZE(coloringModes)))
    {
        m_coloringMode = (ColoringMode)coloringMode;
    }

    ImGui::SliderFloat("Hue range", &m_velocityToHueRange, 10.0f, 300.0f);

    if (ImGui::Button("Reset Particles"))
    {
        resetParticles();
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
}

void Application::resetParticles()
{
    m_particleSystem.reset(m_windowWidth, m_windowHeight, m_particleRadius,
                           m_renderer, m_spawnMode);
}

void Application::reinitializeSimulation(int newParticleCount,
                                         float newParticleRadius)
{
    if (newParticleCount == m_particleCount &&
        newParticleRadius == m_particleRadius)
    {
        return;
    }

    m_paused = true;

    m_particleCount = newParticleCount;
    m_particleRadius = newParticleRadius;

    m_particleSystem.cleanup();
    if (m_useCUDA)
        m_physicsEngine.cleanup();
    else
        m_cpuEngine.cleanup();
    m_renderer.cleanup();

    m_particleSystem =
        ParticleSystem(newParticleCount, m_windowWidth, m_windowHeight);

    float gridCellSize = 2.0f * newParticleRadius;
    m_physicsEngine =
        PhysicsEngine(newParticleCount, (int)(m_windowWidth / gridCellSize) + 1,
                      (int)(m_windowHeight / gridCellSize) + 1);
    m_cpuEngine = CpuPhysicsEngine(newParticleCount,
                                   (int)(m_windowWidth / gridCellSize) + 1,
                                   (int)(m_windowHeight / gridCellSize) + 1);

    m_renderer.initialize(newParticleCount);
    m_particleSystem.initialize(m_renderer, m_useCUDA);
    if (m_useCUDA)
        m_physicsEngine.initialize();
    else
        m_cpuEngine.initialize();

    m_particleSystem.reset(m_windowWidth, m_windowHeight, newParticleRadius,
                           m_renderer, m_spawnMode);

    m_gridParams.grid_width = (int)(m_windowWidth / gridCellSize) + 1;
    m_gridParams.grid_height = (int)(m_windowHeight / gridCellSize) + 1;
    m_gridParams.cell_size = gridCellSize;
}

void Application::cleanup()
{
    m_particleSystem.cleanup();
    if (m_useCUDA)
        m_physicsEngine.cleanup();
    else
        m_cpuEngine.cleanup();
    m_renderer.cleanup();

    if (m_window)
    {
        ImGui_ImplOpenGL3_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        glfwDestroyWindow(m_window);
        glfwTerminate();
        m_window = nullptr;
    }
}
