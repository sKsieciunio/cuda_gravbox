#pragma once

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "Camera.h"
#include "Renderer.h"
#include "ParticleSystem.h"
#include "PhysicsEngine.h"
#include "CpuPhysicsEngine.h"
#include "particle.h"

class Application
{
public:
    Application();
    ~Application();

    bool initialize();
    void run();
    void cleanup();

private:
    GLFWwindow *m_window;
    Camera m_camera;
    Renderer m_renderer;
    ParticleSystem m_particleSystem;
    PhysicsEngine m_physicsEngine; // CUDA engine
    CpuPhysicsEngine m_cpuEngine;  // CPU engine

    SimulationParams m_simParams;
    GridParams m_gridParams;
    bool m_paused;
    bool m_useCUDA;
    float m_velocityToHueRange;

    int m_particleCount;
    float m_particleRadius;
    SpawnMode m_spawnMode;
    ColoringMode m_coloringMode;

    int m_collisionIterations;
    int m_cudaBlockSize;

    int m_windowWidth;
    int m_windowHeight;
    int m_windowPosX;
    int m_windowPosY;

    void initializeGLFW();
    void initializeCUDA();
    void initializeImGui();
    void setupCallbacks();

    void handleInput();
    void update();
    void checkForWindowResize();
    void renderFrame();
    void renderUI();

    void resetParticles();
    void updateGridParams();
    void reinitializeSimulation(int newParticleCount, float newParticleRadius);

    static void windowSizeCallback(GLFWwindow *window, int width, int height);
    static void scrollCallback(GLFWwindow *window, double xoffset, double yoffset);
};
