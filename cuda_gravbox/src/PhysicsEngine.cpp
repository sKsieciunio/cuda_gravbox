#include "PhysicsEngine.h"
#include "Config.h"
#include <cuda_runtime.h>
#include <vector>
#include <cstdio>

#define CUDA_CHECK(err) do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

// External CUDA kernel interface
extern void runPhysicsSimulation(
    ParticlesSoA d_particles,
    int* d_particleGridIndex,
    int* d_particleIndices,
    int* d_gridCellStart,
    int* d_gridCellEnd,
    int numParticles,
    const GridParams& gridParams,
    const SimulationParams& simParams
);

PhysicsEngine::PhysicsEngine(int particleCount, int gridWidth, int gridHeight)
    : m_particleCount(particleCount)
    , m_numCells(gridWidth * gridHeight)
    , d_particleGridIndex(nullptr)
    , d_particleIndices(nullptr)
    , d_gridCellStart(nullptr)
    , d_gridCellEnd(nullptr)
{
}

PhysicsEngine::~PhysicsEngine() {
    cleanup();
}

void PhysicsEngine::initialize() {
    cleanup(); // Ensure clean state

    CUDA_CHECK(cudaMalloc(&d_particleGridIndex, m_particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particleIndices, m_particleCount * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gridCellStart, m_numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gridCellEnd, m_numCells * sizeof(int)));
    
    // Initialize particle indices
    std::vector<int> h_indices(m_particleCount);
    for (int i = 0; i < m_particleCount; i++) {
        h_indices[i] = i;
    }
    CUDA_CHECK(cudaMemcpy(d_particleIndices, h_indices.data(), 
                          m_particleCount * sizeof(int), cudaMemcpyHostToDevice));
}

void PhysicsEngine::cleanup() {
    if (d_particleGridIndex) CUDA_CHECK(cudaFree(d_particleGridIndex));
    if (d_particleIndices) CUDA_CHECK(cudaFree(d_particleIndices));
    if (d_gridCellStart) CUDA_CHECK(cudaFree(d_gridCellStart));
    if (d_gridCellEnd) CUDA_CHECK(cudaFree(d_gridCellEnd));
    
    d_particleGridIndex = nullptr;
    d_particleIndices = nullptr;
    d_gridCellStart = nullptr;
    d_gridCellEnd = nullptr;
}

void PhysicsEngine::resize(int gridWidth, int gridHeight) {
    int newNumCells = gridWidth * gridHeight;
    if (newNumCells == m_numCells) return;

    if (d_gridCellStart) CUDA_CHECK(cudaFree(d_gridCellStart));
    if (d_gridCellEnd) CUDA_CHECK(cudaFree(d_gridCellEnd));

    m_numCells = newNumCells;

    CUDA_CHECK(cudaMalloc(&d_gridCellStart, m_numCells * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_gridCellEnd, m_numCells * sizeof(int)));
}

void PhysicsEngine::simulate(ParticlesSoA& particles, 
                              const SimulationParams& simParams, 
                              const GridParams& gridParams) {
    runPhysicsSimulation(
        particles,
        d_particleGridIndex,
        d_particleIndices,
        d_gridCellStart,
        d_gridCellEnd,
        m_particleCount,
        gridParams,
        simParams
    );
}
