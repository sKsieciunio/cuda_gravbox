#pragma once

#include <cuda_runtime.h>

struct Particle {
	float2 position;   // x, y coordinates (in pixels or normalized coordinates)
	float2 velocity;   // vx, vy (pixels/second or units/second)
	float radius; // particle radius
	float3 color;      // RGB color (0.0 - 1.0)
};

// Simulation constants
struct SimulationParams {
	float gravity;     // Gravitational acceleration (pixels/s^2)
	float dt;   // Time step (seconds)
	float dampening;   // Velocity dampening on collision (0.0 - 1.0)
	float bounds_width;  // Simulation bounds width
	float bounds_height; // Simulation bounds height
};

