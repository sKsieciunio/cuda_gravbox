#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "particle.h"
#include <stdio.h>
#include <curand_kernel.h>

// CUDA kernel for updating particle physics
__global__ void updateParticlesKernel(Particle* particles, int numParticles, SimulationParams params)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numParticles) return;

	Particle& p = particles[idx];

	// Apply gravity
	p.velocity.y += params.gravity * params.dt;

	// Update position
	p.position.x += p.velocity.x * params.dt;
	p.position.y += p.velocity.y * params.dt;

	// Boundary collision detection and response
	// Left wall
	if (p.position.x - p.radius < 0.0f) {
		p.position.x = p.radius;
		p.velocity.x = -p.velocity.x * params.dampening;
	}
	// Right wall
	if (p.position.x + p.radius > params.bounds_width) {
		p.position.x = params.bounds_width - p.radius;
		p.velocity.x = -p.velocity.x * params.dampening;
	}
	// Bottom wall
	if (p.position.y - p.radius < 0.0f) {
		p.position.y = p.radius;
		p.velocity.y = -p.velocity.y * params.dampening;
	}
	// Top wall
	if (p.position.y + p.radius > params.bounds_height) {
		p.position.y = params.bounds_height - p.radius;
		p.velocity.y = -p.velocity.y * params.dampening;
	}
}

// Host function to launch the kernel
void updateParticles(Particle* d_particles, int numParticles, const SimulationParams& params)
{
	int blockSize = 256;
	int numBlocks = (numParticles + blockSize - 1) / blockSize;

	updateParticlesKernel << <numBlocks, blockSize >> > (d_particles, numParticles, params);

	// Check for kernel launch errors
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
	}
}

