#pragma once

#include <cuda_runtime.h>

struct ParticlesSoA {
	float* position_x; 
	float* position_y;
	float* prev_position_x;
	float* prev_position_y;
	float* radius;
	int count;

	// for coloring
	float* velocity_x;
	float* velocity_y;
};

struct Particle {
	float2 position;   // x, y coordinates (in pixels or normalized coordinates)
	float2 previousPosition;
	float2 velocity;   // vx, vy (pixels/second or units/second)
	float2 acceleration; // ax, ay (pixels/second^2 or units/second^2)
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
	float restitution;  // Coefficient of restitution for collisions
	float max_speed;    // Maximum speed of particles
	int collision_iterations; // Number of collision constraint iterations
	int cuda_block_size;      // CUDA block size for kernel launches
};

struct GridParams {
	int grid_width;    // Number of cells in x direction
	int grid_height;   // Number of cells in y direction
	float cell_size;   // Size of each grid cell (pixels)
};

