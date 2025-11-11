#include <cuda_runtime.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

#include "math_functions.h"
#include "particle.h"

__global__ void predictPositionsKernel(Particle* particles, int numParticles, SimulationParams simParams);
__global__ void assignParticlesToGridKernel(Particle* particles, int* particleGridIndex, int numParticles, GridParams gridParams);
__global__ void findCellBoundsKernel(const int* particleGridIndex, int* gridCellStart, int* gridCellEnd, int numParticles);
__global__ void solveConstraintsKernel(Particle* particles, const int* particlesGridIndex, const int* particlesIndices, const int* gridCellStart, const int* gridCellEnd, int numParticles, GridParams gridParams);
__global__ void updateVelocityKernel(Particle* particles, int numParticles, SimulationParams simParams);

void runPhysicsIterationGPU(
	Particle* d_particles,
	int* d_particlesGridIndex,
	int* d_particlesIndices,
	int* d_gridCellStart,
	int* d_gridCellEnd,
	int numParticles,
	const GridParams& gridParams,
	const SimulationParams& simParams,
	const int solverIterations = 5
)
{
	const int blockSize = 256;
	const int numBlocks = (numParticles + blockSize - 1) / blockSize;

	predictPositionsKernel << <numBlocks, blockSize >> > (
		d_particles,
		numParticles,
		simParams
		);
	cudaDeviceSynchronize();

	for (int iter = 0; iter < solverIterations; ++iter)
	{
		int numCells = gridParams.grid_width * gridParams.grid_height;
		cudaMemset(d_gridCellStart, -1, numCells * sizeof(int));
		cudaMemset(d_gridCellEnd, 0, numCells * sizeof(int));

		assignParticlesToGridKernel << <numBlocks, blockSize >> > (
			d_particles,
			d_particlesGridIndex,
			numParticles,
			gridParams
			);
		cudaDeviceSynchronize();

		thrust::sequence(
			thrust::device,
			d_particlesIndices,
			d_particlesIndices + numParticles
		);
		thrust::sort_by_key(
			thrust::device,
			d_particlesGridIndex,
			d_particlesGridIndex + numParticles,
			d_particlesIndices
		);

		findCellBoundsKernel << <numBlocks, blockSize >> > (
			d_particlesGridIndex,
			d_gridCellStart,
			d_gridCellEnd,
			numParticles
			);
		cudaDeviceSynchronize();

		solveConstraintsKernel << <numBlocks, blockSize >> > (
			d_particles,
			d_particlesGridIndex,
			d_particlesIndices,
			d_gridCellStart,
			d_gridCellEnd,
			numParticles,
			gridParams
			);
		cudaDeviceSynchronize();
	}

	updateVelocityKernel << <numBlocks, blockSize >> > (
		d_particles,
		numParticles,
		simParams
		);
	cudaDeviceSynchronize();
}

__global__ void predictPositionsKernel(
	Particle* particles,
	int numParticles,
	SimulationParams simParams
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	Particle& p = particles[idx];

	p.previousPosition = p.position;

	p.predictedPosition.x = p.position.x + p.velocity.x * simParams.dt;
	p.predictedPosition.y = p.position.y + p.velocity.y * simParams.dt + 0.5f * simParams.gravity * simParams.dt * simParams.dt;

	if (p.predictedPosition.x - p.radius < 0.0f)
		p.predictedPosition.x = p.radius;
	if (p.predictedPosition.x + p.radius > simParams.bounds_width)
		p.predictedPosition.x = simParams.bounds_width - p.radius;
	if (p.predictedPosition.y - p.radius < 0.0f)
		p.predictedPosition.y = p.radius;
	if (p.predictedPosition.y + p.radius > simParams.bounds_height)
		p.predictedPosition.y = simParams.bounds_height - p.radius;
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

__global__ void solveConstraintsKernel(
	Particle* particles,
	const int* particlesGridIndex,
	const int* particlesIndices,
	const int* gridCellStart,
	const int* gridCellEnd,
	int numParticles,
	GridParams gridParams
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	int particleIdx = particlesIndices[idx];
	Particle& p1 = particles[particleIdx];

	int cellIndex = particlesGridIndex[idx];
	int cellX = cellIndex % gridParams.grid_width;
	int cellY = cellIndex / gridParams.grid_width;

	float2 correction = make_float2(0.0f, 0.0f);

	for (int dy = -1; dy <= 1; ++dy)
	{
		for (int dx = -1; dx <= 1; ++dx)
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

			for (int i = start; i < end; ++i)
			{
				int otherIdx = particlesIndices[i];

				// this prevents double handling or self-self interaction
				if (particleIdx >= otherIdx) continue;

				Particle& p2 = particles[otherIdx];

				float dx = p2.predictedPosition.x - p1.predictedPosition.x;
				float dy = p2.predictedPosition.y - p1.predictedPosition.y;
				float distSq = dx * dx + dy * dy;
				float minDist = p1.radius + p2.radius;
				float minDistSq = minDist * minDist;

				if (distSq < minDistSq && distSq > 1e-6f)
				{
					float dist = sqrtf(distSq);
					float overlap = minDist - dist;

					float nx = dx / dist;
					float ny = dy / dist;

					float stiffness = 1.0f;
					float2 delta = make_float2(
						nx * overlap * 0.5f * stiffness, 
						ny * overlap * 0.5f * stiffness
					);

					p1.predictedPosition.x -= delta.x;
					p1.predictedPosition.y -= delta.y;
					
					p2.predictedPosition.x += delta.x;
					p2.predictedPosition.y += delta.y;
				}
			}
		}
	}
}

__global__ void updateVelocityKernel(
	Particle* particles,
	int numParticles,
	SimulationParams simParams
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= numParticles) return;

	Particle& p = particles[idx];

	p.velocity.x = (p.predictedPosition.x - p.previousPosition.x) / simParams.dt;
	p.velocity.y = (p.predictedPosition.y - p.previousPosition.y) / simParams.dt;

	p.position = p.predictedPosition;
}
