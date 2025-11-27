#pragma once

namespace Config
{
    // Particle configuration
    constexpr float PARTICLE_RADIUS = 20.0f;
    constexpr int PARTICLE_COUNT = 100;

    // Alternative configurations for different scenarios:
    // Medium: PARTICLE_RADIUS = 2.0f, PARTICLE_COUNT = 10000
    // Large:  PARTICLE_RADIUS = 1.0f, PARTICLE_COUNT = 100000

    // Window configuration
    constexpr int DEFAULT_WINDOW_WIDTH = 1000;
    constexpr int DEFAULT_WINDOW_HEIGHT = 800;

    // Simulation defaults
    constexpr float DEFAULT_GRAVITY = -500.0f;  // Pixels/s^2 (negative = downward)
    constexpr float DEFAULT_DT = 0.0006f;       // Time step
    constexpr float DEFAULT_DAMPENING = 0.6f;   // Energy loss on collision
    constexpr float DEFAULT_RESTITUTION = 0.6f; // Coefficient of restitution

    // Grid configuration
    constexpr float GRID_CELL_SIZE = 2.0f * PARTICLE_RADIUS;

    // Rendering defaults
    constexpr float DEFAULT_ZOOM = 1.0f;
    constexpr float MIN_ZOOM = 1.0f;
    constexpr float MAX_ZOOM = 10.0f;
    constexpr float DEFAULT_VELOCITY_TO_HUE_RANGE = 300.0f;

    // Physics iterations
    constexpr int COLLISION_ITERATIONS = 10;

    // CUDA configuration
    constexpr int CUDA_BLOCK_SIZE = 256;
}
