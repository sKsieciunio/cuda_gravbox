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
__global__ void updateParticlesKernel(ParticlesSoA particles, int numParticles, SimulationParams params)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= numParticles) return;

	float pos_x = particles.position_x[idx];
	float pos_y = particles.position_y[idx];
	float prev_pos_x = particles.prev_position_x[idx];
	float prev_pos_y = particles.prev_position_y[idx];
	float radius = particles.radius[idx];
	
	float new_pos_x = 2 * pos_x - prev_pos_x;
	float new_pos_y = 2 * pos_y - prev_pos_y + params.gravity * params.dt * params.dt;

	particles.velocity_x[idx] = (new_pos_x - prev_pos_x) / (2 * params.dt);
	particles.velocity_y[idx] = (new_pos_y - prev_pos_y) / (2 * params.dt);

	particles.prev_position_x[idx] = pos_x;
	particles.prev_position_y[idx] = pos_y;
	particles.position_x[idx] = new_pos_x;
	particles.position_y[idx] = new_pos_y;

	// Boundary collision detection and response
	// Left wall
	if (new_pos_x - radius < 0.0f) {
		float vx = new_pos_x - prev_pos_x;
		particles.position_x[idx] = radius;
		particles.prev_position_x[idx] = particles.position_x[idx] + vx * params.dampening;
	}

	// Right wall
	if (new_pos_x + radius > params.bounds_width) {
		float vx = new_pos_x - prev_pos_x;
		particles.position_x[idx] = params.bounds_width - radius;
		particles.prev_position_x[idx] = particles.position_x[idx] + vx * params.dampening;
	}

	// Bottom wall
	if (new_pos_y - radius < 0.0f) {
		float vy = new_pos_y - prev_pos_y;
		particles.position_y[idx] = radius;
		particles.prev_position_y[idx] = particles.position_y[idx] + vy * params.dampening;
	}

	// Top wall
	if (new_pos_y + radius > params.bounds_height) {
		float vy = new_pos_y - prev_pos_y;
		particles.position_y[idx] = params.bounds_height - radius;
		particles.prev_position_y[idx] = particles.position_y[idx] + vy * params.dampening;
	}
}

__global__ void assignParticlesToGridKernel(
	ParticlesSoA particles,
	int* particleGridIndex,
	int numParticles,
	GridParams gridParams
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	float pos_x = particles.position_x[idx];
	float pos_y = particles.position_y[idx];

	int cellX = (int)floorf(pos_x / gridParams.cell_size);
	int cellY = (int)floorf(pos_y / gridParams.cell_size);

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
	ParticlesSoA particles,
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

	float p1_pos_x = particles.position_x[particleIdx];
	float p1_pos_y = particles.position_y[particleIdx];
	float p1_radius = particles.radius[particleIdx];

	int cellIndex = particleGridIndex[idx];
	int cellX = cellIndex % gridParams.grid_width;
	int cellY = cellIndex / gridParams.grid_width;

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
				if (particleIdx >= otherParticleIdx) continue; // Avoid double checking

				float p2_pos_x = particles.position_x[otherParticleIdx];
				float p2_pos_y = particles.position_y[otherParticleIdx];
				float p2_radius = particles.radius[otherParticleIdx];

				// distance between particles
				float dx_dist = p2_pos_x - p1_pos_x;
				float dy_dist = p2_pos_y - p1_pos_y;
				float distSq = dx_dist * dx_dist + dy_dist * dy_dist;
				float minDist = p1_radius + p2_radius;
				float minDistSq = minDist * minDist;

				// colision check
				if (distSq < minDistSq && distSq > 0.0001f)
				{
					float dist = sqrtf(distSq);

					// Normalize collision normal
					float nx = dx_dist / dist;
					float ny = dy_dist / dist;

					float overlap = minDist - dist;
					float separationX = nx * overlap * 0.5f;
					float separationY = ny * overlap * 0.5f;

					atomicAdd(&particles.position_x[particleIdx], -separationX);
					atomicAdd(&particles.position_y[particleIdx], -separationY);
					atomicAdd(&particles.position_x[otherParticleIdx], separationX);
					atomicAdd(&particles.position_y[otherParticleIdx], separationY);
				}
			}
		}
	}
}

//// Host function to launch the kernel
//void updateParticles(Particle* d_particles, int numParticles, const SimulationParams& params)
//{
//	int blockSize = 256;
//	int numBlocks = (numParticles + blockSize - 1) / blockSize;
//
//	updateParticlesKernel << <numBlocks, blockSize >> > (d_particles, numParticles, params);
//
//	// Check for kernel launch errors
//	cudaError_t err = cudaGetLastError();
//	if (err != cudaSuccess) {
//		fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
//	}
//}

void handleCollisions(
	ParticlesSoA d_particles,
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

	const int interations = 3;
	for (int iter = 0; iter < interations; iter++) {
		cudaMemset(d_gridCellStart, -1, numCells * sizeof(int)); // -1 means empty
		cudaMemset(d_gridCellEnd, 0, numCells * sizeof(int));

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
			d_particleGridIndex,
			d_particleIndices,
			d_gridCellStart,
			d_gridCellEnd,
			numParticles,
			gridParams,
			simParams
		);

		updateParticlesKernel << <numBlocks, blockSize, 0, stream >> > (d_particles, numParticles, simParams);
	}

	cudaStreamDestroy(stream);

	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "Collision handling failed: %s\n", cudaGetErrorString(err));
	}
}
