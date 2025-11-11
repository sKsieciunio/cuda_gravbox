#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"
#include "particle.h"
#include <stdio.h>
#include <curand_kernel.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>
#include <algorithm>

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

__global__ void assignParticlesToGridKernel(
	Particle* particles,
	int* particleGridIndex,
	int numParticles,
	GridParams gridParams
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	const Particle& p = particles[idx];

	int cellX = (int)floorf(p.position.x / gridParams.cell_size);
	int cellY = (int)floorf(p.position.y / gridParams.cell_size);

	cellX = max(0, min(cellX, gridParams.grid_width - 1));
	cellY = max(0, min(cellY, gridParams.grid_height - 1));

	int cellIndex = cellY * gridParams.grid_width + cellX;

	particleGridIndex[idx] = cellIndex;
}

__global__ void findCellBoundsKernel(
	const int* particleGridIndex,
	int* gridCellStart,
	int* gridCellEnd,
	int numParticles
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	int cellIndex = particleGridIndex[idx];

	if (idx == 0 || cellIndex != particleGridIndex[idx - 1]) {
		gridCellStart[cellIndex] = idx;
	}

	if (idx == numParticles - 1 || cellIndex != particleGridIndex[idx + 1]) {
		gridCellEnd[cellIndex] = idx + 1;
	}
}

__global__ void handleCollisionsKernel(
	Particle* particles,
	float2* positionCorrections,
	const int* particleGridIndex,
	const int* particleIndices,
	const int* gridCellStart,
	const int* gridCellEnd,
	int numParticles,
	GridParams gridParams,
	SimulationParams simParams
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	int particleIdx = particleIndices[idx];
	Particle& p1 = particles[particleIdx];

	int cellIndex = particleGridIndex[idx];
	int cellX = cellIndex % gridParams.grid_width;
	int cellY = cellIndex / gridParams.grid_width;

	float2 correction = make_float2(0.0f, 0.0f);

	for (int dy = -1; dy <= 1; dy++)
	{
		for (int dx = -1; dx <= 1; dx++)
		{
			int neighborX = cellX + dx;
			int neighborY = cellY + dy;

			if (neighborX < 0 || neighborX >= gridParams.grid_width ||
				neighborY < 0 || neighborY >= gridParams.grid_height)
				continue;

			int neighborCellIndex = neighborY * gridParams.grid_width + neighborX;
			int start = gridCellStart[neighborCellIndex];
			int end = gridCellEnd[neighborCellIndex];

			if (start == -1) continue;

			for (int i = start; i < end; i++)
			{
				int otherParticleIdx = particleIndices[i];

				// this is vurnerable to race conditions in case of multiple collisions
				// but i cant think of solution for now
				if (particleIdx >= otherParticleIdx) continue; // Avoid double checking

				Particle& p2 = particles[otherParticleIdx];

				// distance between particles
				float dx_dist = p2.position.x - p1.position.x;
				float dy_dist = p2.position.y - p1.position.y;
				float distSq = dx_dist * dx_dist + dy_dist * dy_dist;
				float minDist = p1.radius + p2.radius;
				float minDistSq = minDist * minDist;

				// colision check
				if (distSq < minDistSq && distSq > 0.0001f)
				{
					float dist = sqrtf(distSq);

					// Normalize collision normal
					float nx = dx_dist / dist;
					float ny = dy_dist / dist;

					float overlap = minDist - dist;
					//float separationX = nx * overlap * 0.5f;
					//float separationY = ny * overlap * 0.5f;

					//p1.position.x -= separationX;
					//p1.position.y -= separationY;
					//p2.position.x += separationX;
					//p2.position.y += separationY;

					correction.x -= nx * overlap * 0.5f;
					correction.y -= ny * overlap * 0.5f;

					// Relative velocity
					float dvx = p2.velocity.x - p1.velocity.x;
					float dvy = p2.velocity.y - p1.velocity.y;

					// Velocity along the normal
					float dvn = dvx * nx + dvy * ny;

					// Only resolve if particles are moving towards each other
					if (dvn < 0.0f)
					{
						float impulse = -(1.0f + simParams.restitution) * dvn * 0.5f;
						//p1.velocity.x -= impulse * nx;
						//p1.velocity.y -= impulse * ny;
						//p2.velocity.x += impulse * nx;
						//p2.velocity.y += impulse * ny;
						atomicAdd(&particles[particleIdx].velocity.x, -impulse * nx);
						atomicAdd(&particles[particleIdx].velocity.y, -impulse * ny);
						atomicAdd(&particles[otherParticleIdx].velocity.x, impulse * nx);
						atomicAdd(&particles[otherParticleIdx].velocity.y, impulse * ny);
					}
				}
			}
		}
	}

	positionCorrections[particleIdx] = correction;
}

__global__ void applyPositionCorrectionsKernal(
	Particle* particles,
	const float2* positionCorrections,
	int numParticles
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	particles[idx].position.x += positionCorrections[idx].x;
	particles[idx].position.y += positionCorrections[idx].y;
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

void handleCollisions(
	Particle* d_particles,
	int* d_particleGridIndex,
	int* d_particleIndices,
	int* d_gridCellStart,
	int* d_gridCellEnd,
	int numParticles,
	const GridParams& gridParams,
	const SimulationParams& simParams
)
{
	int blockSize = 256;
	int numBlocks = (numParticles + blockSize - 1) / blockSize;
	int numCells = gridParams.grid_width * gridParams.grid_height;

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	static float2* d_positionCorrections = nullptr;
	static int allocatedSize = 0;

	if (allocatedSize < numParticles) {
		if (d_positionCorrections) { cudaFree(d_positionCorrections); }
		cudaMalloc(&d_positionCorrections, numParticles * sizeof(float2));
		allocatedSize = numParticles;
	}

	const int interations = 3;
	for (int iter = 0; iter < interations; iter++) {
		cudaMemset(d_gridCellStart, -1, numCells * sizeof(int)); // -1 means empty
		cudaMemset(d_gridCellEnd, 0, numCells * sizeof(int));
		cudaMemset(d_positionCorrections, 0, numParticles * sizeof(float2));

		// Assign particles to grid cells
		assignParticlesToGridKernel << <numBlocks, blockSize, 0, stream >> > (
			d_particles, d_particleGridIndex, numParticles, gridParams
		);

		cudaDeviceSynchronize();

		thrust::sequence(
			thrust::cuda::par.on(stream),
			thrust::device_pointer_cast(d_particleIndices),
			thrust::device_pointer_cast(d_particleIndices + numParticles)
		);

		thrust::sort_by_key(
			thrust::cuda::par.on(stream),
			thrust::device_pointer_cast(d_particleGridIndex),
			thrust::device_pointer_cast(d_particleGridIndex + numParticles),
			thrust::device_pointer_cast(d_particleIndices)
		);

		// Identify start and end indices for each grid cell
		findCellBoundsKernel << <numBlocks, blockSize, 0, stream>> > (
			d_particleGridIndex, d_gridCellStart, d_gridCellEnd, numParticles
		);

		// Handle collisions within each grid cell
		handleCollisionsKernel << <numBlocks, blockSize, 0, stream>> > (
			d_particles,
			d_positionCorrections,
			d_particleGridIndex,
			d_particleIndices,
			d_gridCellStart,
			d_gridCellEnd,
			numParticles,
			gridParams,
			simParams
		);

		applyPositionCorrectionsKernal << <numBlocks, blockSize, 0, stream>> > (
			d_particles,
			d_positionCorrections,
			numParticles
			);
	}

	cudaStreamDestroy(stream);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Collision handling failed: %s\n", cudaGetErrorString(err));
	}
}
