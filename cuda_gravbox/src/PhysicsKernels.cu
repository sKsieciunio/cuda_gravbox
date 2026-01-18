#include "cuda_runtime.h"
#include "math_functions.h"
#include "device_launch_parameters.h"
#include "particle.h"
#include "Config.h"
#include <stdio.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/sequence.h>

__global__ void shiftParticlesKernel(ParticlesSoA particles, int numParticles, float shiftX, float shiftY)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    particles.position_x[idx] += shiftX;
    particles.position_y[idx] += shiftY;
    particles.prev_position_x[idx] += shiftX;
    particles.prev_position_y[idx] += shiftY;
}

void runShiftParticles(ParticlesSoA d_particles, int numParticles, float shiftX, float shiftY, int blockSize)
{
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    shiftParticlesKernel<<<numBlocks, blockSize>>>(d_particles, numParticles, shiftX, shiftY);
    cudaDeviceSynchronize();
}

__global__ void updateParticlesKernel(ParticlesSoA particles, int numParticles, SimulationParams params)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    float pos_x = particles.position_x[idx];
    float pos_y = particles.position_y[idx];
    float prev_pos_x = particles.prev_position_x[idx];
    float prev_pos_y = particles.prev_position_y[idx];
    float radius = particles.radius[idx];

    float vel_x = (pos_x - prev_pos_x) / params.dt;
    float vel_y = (pos_y - prev_pos_y) / params.dt;

    vel_y += params.gravity * params.dt;

    // Air blowers
    if (params.enable_air_blowers)
    {
        float margin = 150.0f;
        float force = 2000.0f;

        if (pos_x < margin)
        {
            vel_x += force * 0.5f * params.dt;
            vel_y += force * 1.5f * params.dt;
        }
        else if (pos_x > params.bounds_width - margin)
        {
            vel_x -= force * 0.5f * params.dt;
            vel_y += force * 1.5f * params.dt;
        }
    }

    // Clamp velocity
    float speed_sq = vel_x * vel_x + vel_y * vel_y;
    if (speed_sq > params.max_speed * params.max_speed)
    {
        float scale = params.max_speed / sqrtf(speed_sq);
        vel_x *= scale;
        vel_y *= scale;
    }

    particles.prev_position_x[idx] = pos_x;
    particles.prev_position_y[idx] = pos_y;

    float new_pos_x = pos_x + vel_x * params.dt;
    float new_pos_y = pos_y + vel_y * params.dt;

    particles.velocity_x[idx] = vel_x * params.dt; // normalization for shaders
    particles.velocity_y[idx] = vel_y * params.dt;

    particles.position_x[idx] = new_pos_x;
    particles.position_y[idx] = new_pos_y;

    // Left wall
    if (new_pos_x - radius < 0.0f)
    {
        float vx = new_pos_x - pos_x;
        particles.position_x[idx] = radius;
        particles.prev_position_x[idx] = particles.position_x[idx] + vx * params.dampening;
    }

    // Right wall
    if (new_pos_x + radius > params.bounds_width)
    {
        float vx = new_pos_x - pos_x;
        particles.position_x[idx] = params.bounds_width - radius;
        particles.prev_position_x[idx] = particles.position_x[idx] + vx * params.dampening;
    }

    // Bottom wall
    if (new_pos_y - radius < 0.0f)
    {
        float vy = new_pos_y - pos_y;
        particles.position_y[idx] = radius;
        particles.prev_position_y[idx] = particles.position_y[idx] + vy * params.dampening;
    }

    // Top wall
    if (new_pos_y + radius > params.bounds_height)
    {
        float vy = new_pos_y - pos_y;
        particles.position_y[idx] = params.bounds_height - radius;
        particles.prev_position_y[idx] = particles.position_y[idx] + vy * params.dampening;
    }
}

__global__ void assignParticlesToGridKernel(
    ParticlesSoA particles,
    int *particleGridIndex,
    int numParticles,
    GridParams gridParams)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

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
    const int *particleGridIndex,
    int *gridCellStart,
    int *gridCellEnd,
    int numParticles)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    int cellIndex = particleGridIndex[idx];

    if (idx == 0 || cellIndex != particleGridIndex[idx - 1])
    {
        gridCellStart[cellIndex] = idx;
    }

    if (idx == numParticles - 1 || cellIndex != particleGridIndex[idx + 1])
    {
        gridCellEnd[cellIndex] = idx + 1;
    }
}

__global__ void handleCollisionsKernel(
    ParticlesSoA particles,
    const int *particleGridIndex,
    const int *particleIndices,
    const int *gridCellStart,
    const int *gridCellEnd,
    int numParticles,
    GridParams gridParams,
    SimulationParams simParams)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numParticles)
        return;

    int particleIdx = particleIndices[idx];

    float p1_pos_x = particles.position_x[particleIdx];
    float p1_pos_y = particles.position_y[particleIdx];
    float p1_radius = particles.radius[particleIdx];
    float p1_mass = particles.mass[particleIdx];

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

            if (start == -1)
                continue;

            for (int i = start; i < end; i++)
            {
                int otherParticleIdx = particleIndices[i];
                if (particleIdx >= otherParticleIdx)
                    continue; // Avoid double checking

                float p2_pos_x = particles.position_x[otherParticleIdx];
                float p2_pos_y = particles.position_y[otherParticleIdx];
                float p2_radius = particles.radius[otherParticleIdx];
                float p2_mass = particles.mass[otherParticleIdx];

                float dx_dist = p2_pos_x - p1_pos_x;
                float dy_dist = p2_pos_y - p1_pos_y;
                float distSq = dx_dist * dx_dist + dy_dist * dy_dist;
                float minDist = p1_radius + p2_radius;
                float minDistSq = minDist * minDist;

                if (distSq < minDistSq && distSq > 0.0001f)
                {
                    float dist = sqrtf(distSq);

                    float nx = dx_dist / dist;
                    float ny = dy_dist / dist;

                    float overlap = minDist - dist;

                    float totalMass = p1_mass + p2_mass;
                    float p1_factor = p2_mass / totalMass;
                    float p2_factor = p1_mass / totalMass;

                    float separationX = nx * overlap;
                    float separationY = ny * overlap;

                    atomicAdd(&particles.position_x[particleIdx], -separationX * p1_factor);
                    atomicAdd(&particles.position_y[particleIdx], -separationY * p1_factor);
                    atomicAdd(&particles.position_x[otherParticleIdx], separationX * p2_factor);
                    atomicAdd(&particles.position_y[otherParticleIdx], separationY * p2_factor);

                    // Velocity correction (Restitution)
                    float p1_prev_x = particles.prev_position_x[particleIdx];
                    float p1_prev_y = particles.prev_position_y[particleIdx];
                    float p2_prev_x = particles.prev_position_x[otherParticleIdx];
                    float p2_prev_y = particles.prev_position_y[otherParticleIdx];

                    float v1x = (p1_pos_x - p1_prev_x) / simParams.dt;
                    float v1y = (p1_pos_y - p1_prev_y) / simParams.dt;
                    float v2x = (p2_pos_x - p2_prev_x) / simParams.dt;
                    float v2y = (p2_pos_y - p2_prev_y) / simParams.dt;

                    float relVx = v1x - v2x;
                    float relVy = v1y - v2y;
                    float vNormal = relVx * nx + relVy * ny;

                    if (vNormal > 0.0f) // Closing in
                    {
                        float j = -(simParams.restitution) * vNormal;
                        float impulseX = j * nx;
                        float impulseY = j * ny;

                        atomicAdd(&particles.prev_position_x[particleIdx], -impulseX * p1_factor * simParams.dt);
                        atomicAdd(&particles.prev_position_y[particleIdx], -impulseY * p1_factor * simParams.dt);
                        atomicAdd(&particles.prev_position_x[otherParticleIdx], impulseX * p2_factor * simParams.dt);
                        atomicAdd(&particles.prev_position_y[otherParticleIdx], impulseY * p2_factor * simParams.dt);
                    }
                }
            }
        }
    }
}

void runPhysicsSimulation(
    ParticlesSoA d_particles,
    int *d_particleGridIndex,
    int *d_particleIndices,
    int *d_gridCellStart,
    int *d_gridCellEnd,
    int numParticles,
    const GridParams &gridParams,
    const SimulationParams &simParams)
{
    int blockSize = simParams.cuda_block_size;
    int numBlocks = (numParticles + blockSize - 1) / blockSize;
    int numCells = gridParams.grid_width * gridParams.grid_height;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    updateParticlesKernel<<<numBlocks, blockSize, 0, stream>>>(
        d_particles, numParticles, simParams);

    for (int iter = 0; iter < simParams.collision_iterations; iter++)
    {
        cudaMemset(d_gridCellStart, -1, numCells * sizeof(int));
        cudaMemset(d_gridCellEnd, 0, numCells * sizeof(int));

        assignParticlesToGridKernel<<<numBlocks, blockSize, 0, stream>>>(
            d_particles, d_particleGridIndex, numParticles, gridParams);

        cudaDeviceSynchronize();

        thrust::sequence(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(d_particleIndices),
            thrust::device_pointer_cast(d_particleIndices + numParticles));

        thrust::sort_by_key(
            thrust::cuda::par.on(stream),
            thrust::device_pointer_cast(d_particleGridIndex),
            thrust::device_pointer_cast(d_particleGridIndex + numParticles),
            thrust::device_pointer_cast(d_particleIndices));

        findCellBoundsKernel<<<numBlocks, blockSize, 0, stream>>>(
            d_particleGridIndex, d_gridCellStart, d_gridCellEnd, numParticles);

        handleCollisionsKernel<<<numBlocks, blockSize, 0, stream>>>(
            d_particles,
            d_particleGridIndex,
            d_particleIndices,
            d_gridCellStart,
            d_gridCellEnd,
            numParticles,
            gridParams,
            simParams);
    }

    cudaStreamDestroy(stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Physics simulation failed: %s\n", cudaGetErrorString(err));
    }
}
