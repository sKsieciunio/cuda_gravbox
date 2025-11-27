#pragma once

#include "particle.h"

class PhysicsEngine {
public:
    PhysicsEngine(int particleCount, int gridWidth, int gridHeight);
    ~PhysicsEngine();
    
    void initialize();
    void cleanup();
    
    void simulate(ParticlesSoA& particles, const SimulationParams& simParams, const GridParams& gridParams);
    
private:
    int m_particleCount;
    int m_numCells;
    
    // Grid-based collision detection data
    int* d_particleGridIndex;
    int* d_particleIndices;
    int* d_gridCellStart;
    int* d_gridCellEnd;
    
    void handleCollisions(
        ParticlesSoA& particles,
        const GridParams& gridParams,
        const SimulationParams& simParams
    );
};
