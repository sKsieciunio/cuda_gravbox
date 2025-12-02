#pragma once

#include "particle.h"

class CpuPhysicsEngine
{
public:
    CpuPhysicsEngine(int particleCount, int gridWidth, int gridHeight);
    ~CpuPhysicsEngine();

    void initialize();
    void cleanup();
    void resize(int gridWidth, int gridHeight);

    void simulate(ParticlesSoA &particles, const SimulationParams &simParams, const GridParams &gridParams);

private:
    int m_particleCount;
    int m_numCells;

    // CPU-grid structures
    int *m_particleGridIndex; // size: particleCount
    int *m_particleIndices;   // size: particleCount
    int *m_gridCellStart;     // size: numCells
    int *m_gridCellEnd;       // size: numCells

    void updateParticles(ParticlesSoA &p, const SimulationParams &params);
    void assignToGrid(const ParticlesSoA &p, const GridParams &grid);
    void findCellBounds();
    void handleCollisions(ParticlesSoA &p, const GridParams &grid, const SimulationParams &params);
};
