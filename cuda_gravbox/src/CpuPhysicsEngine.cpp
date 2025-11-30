#include "CpuPhysicsEngine.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>

CpuPhysicsEngine::CpuPhysicsEngine(int particleCount, int gridWidth, int gridHeight)
    : m_particleCount(particleCount)
    , m_numCells(gridWidth * gridHeight)
    , m_particleGridIndex(nullptr)
    , m_particleIndices(nullptr)
    , m_gridCellStart(nullptr)
    , m_gridCellEnd(nullptr) {}

CpuPhysicsEngine::~CpuPhysicsEngine() { cleanup(); }

void CpuPhysicsEngine::initialize() {
    m_particleGridIndex = new int[m_particleCount];
    m_particleIndices = new int[m_particleCount];
    m_gridCellStart = new int[m_numCells];
    m_gridCellEnd = new int[m_numCells];
    for (int i = 0; i < m_particleCount; ++i) m_particleIndices[i] = i;
}

void CpuPhysicsEngine::cleanup() {
    delete[] m_particleGridIndex; m_particleGridIndex = nullptr;
    delete[] m_particleIndices; m_particleIndices = nullptr;
    delete[] m_gridCellStart; m_gridCellStart = nullptr;
    delete[] m_gridCellEnd; m_gridCellEnd = nullptr;
}

void CpuPhysicsEngine::updateParticles(ParticlesSoA& particles, const SimulationParams& params) {
    for (int idx = 0; idx < m_particleCount; ++idx) {
        float pos_x = particles.position_x[idx];
        float pos_y = particles.position_y[idx];
        float prev_pos_x = particles.prev_position_x[idx];
        float prev_pos_y = particles.prev_position_y[idx];
        float radius = particles.radius[idx];

        float vel_x = (pos_x - prev_pos_x) / params.dt;
        float vel_y = (pos_y - prev_pos_y) / params.dt;
        vel_y += params.gravity * params.dt;

        particles.prev_position_x[idx] = pos_x;
        particles.prev_position_y[idx] = pos_y;

        float new_pos_x = pos_x + vel_x * params.dt;
        float new_pos_y = pos_y + vel_y * params.dt;

        particles.velocity_x[idx] = vel_x;
        particles.velocity_y[idx] = vel_y;

        particles.position_x[idx] = new_pos_x;
        particles.position_y[idx] = new_pos_y;

        // Boundaries - use velocity for proper Verlet dampening
        if (new_pos_x - radius < 0.0f) {
            particles.position_x[idx] = radius;
            particles.prev_position_x[idx] = particles.position_x[idx] + vel_x * params.dt * params.dampening;
        }
        if (new_pos_x + radius > params.bounds_width) {
            particles.position_x[idx] = params.bounds_width - radius;
            particles.prev_position_x[idx] = particles.position_x[idx] + vel_x * params.dt * params.dampening;
        }
        if (new_pos_y - radius < 0.0f) {
            particles.position_y[idx] = radius;
            particles.prev_position_y[idx] = particles.position_y[idx] + vel_y * params.dt * params.dampening;
        }
        if (new_pos_y + radius > params.bounds_height) {
            particles.position_y[idx] = params.bounds_height - radius;
            particles.prev_position_y[idx] = particles.position_y[idx] + vel_y * params.dt * params.dampening;
        }
    }
}

void CpuPhysicsEngine::assignToGrid(const ParticlesSoA& particles, const GridParams& gridParams) {
    for (int idx = 0; idx < m_particleCount; ++idx) {
        float pos_x = particles.position_x[idx];
        float pos_y = particles.position_y[idx];
        int cellX = (int)std::floor(pos_x / gridParams.cell_size);
        int cellY = (int)std::floor(pos_y / gridParams.cell_size);
        cellX = std::max(0, std::min(cellX, gridParams.grid_width - 1));
        cellY = std::max(0, std::min(cellY, gridParams.grid_height - 1));
        int cellIndex = cellY * gridParams.grid_width + cellX;
        m_particleGridIndex[idx] = cellIndex;
    }
    // sort by key
    std::vector<int> order(m_particleCount);
    for (int i = 0; i < m_particleCount; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){ return m_particleGridIndex[a] < m_particleGridIndex[b]; });
    // apply sort to indices and keys
    std::vector<int> sortedKeys(m_particleCount);
    std::vector<int> sortedIdx(m_particleCount);
    for (int i = 0; i < m_particleCount; ++i) { sortedKeys[i] = m_particleGridIndex[order[i]]; sortedIdx[i] = order[i]; }
    std::memcpy(m_particleGridIndex, sortedKeys.data(), m_particleCount * sizeof(int));
    std::memcpy(m_particleIndices, sortedIdx.data(), m_particleCount * sizeof(int));
}

void CpuPhysicsEngine::findCellBounds() {
    std::fill(m_gridCellStart, m_gridCellStart + m_numCells, -1);
    std::fill(m_gridCellEnd, m_gridCellEnd + m_numCells, 0);
    for (int idx = 0; idx < m_particleCount; ++idx) {
        int cellIndex = m_particleGridIndex[idx];
        if (idx == 0 || cellIndex != m_particleGridIndex[idx - 1]) {
            m_gridCellStart[cellIndex] = idx;
        }
        if (idx == m_particleCount - 1 || cellIndex != m_particleGridIndex[idx + 1]) {
            m_gridCellEnd[cellIndex] = idx + 1;
        }
    }
}

void CpuPhysicsEngine::handleCollisions(ParticlesSoA& particles, const GridParams& gridParams, const SimulationParams& simParams) {
    for (int idx = 0; idx < m_particleCount; ++idx) {
        int particleIdx = m_particleIndices[idx];
        float p1_pos_x = particles.position_x[particleIdx];
        float p1_pos_y = particles.position_y[particleIdx];
        float p1_radius = particles.radius[particleIdx];
        int cellIndex = m_particleGridIndex[idx];
        int cellX = cellIndex % gridParams.grid_width;
        int cellY = cellIndex / gridParams.grid_width;
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = cellX + dx, ny = cellY + dy;
                if (nx < 0 || nx >= gridParams.grid_width || ny < 0 || ny >= gridParams.grid_height) continue;
                int nCell = ny * gridParams.grid_width + nx;
                int start = m_gridCellStart[nCell];
                int end = m_gridCellEnd[nCell];
                if (start == -1) continue;
                for (int i = start; i < end; ++i) {
                    int otherIdx = m_particleIndices[i];
                    if (particleIdx >= otherIdx) continue;
                    float p2_pos_x = particles.position_x[otherIdx];
                    float p2_pos_y = particles.position_y[otherIdx];
                    float p2_radius = particles.radius[otherIdx];
                    float dx_d = p2_pos_x - p1_pos_x;
                    float dy_d = p2_pos_y - p1_pos_y;
                    float distSq = dx_d * dx_d + dy_d * dy_d;
                    float minDist = p1_radius + p2_radius;
                    float minDistSq = minDist * minDist;
                    if (distSq < minDistSq && distSq > 1e-4f) {
                        float dist = std::sqrt(distSq);
                        float nxn = dx_d / dist;
                        float nyn = dy_d / dist;
                        float overlap = minDist - dist;
                        float sepX = nxn * overlap * 0.5f;
                        float sepY = nyn * overlap * 0.5f;
                        
                        particles.position_x[particleIdx] -= sepX;
                        particles.position_y[particleIdx] -= sepY;
                        particles.position_x[otherIdx] += sepX;
                        particles.position_y[otherIdx] += sepY;

                        // Update local position for subsequent checks
                        p1_pos_x -= sepX;
                        p1_pos_y -= sepY;
                    }
                }
            }
        }
    }
}

void CpuPhysicsEngine::simulate(ParticlesSoA& particles, const SimulationParams& simParams, const GridParams& gridParams) {
    // single physics update
    updateParticles(particles, simParams);
    // constraints iterations
    for (int iter = 0; iter < simParams.collision_iterations; ++iter) {
        assignToGrid(particles, gridParams);
        findCellBounds();
        handleCollisions(particles, gridParams, simParams);
    }
}
