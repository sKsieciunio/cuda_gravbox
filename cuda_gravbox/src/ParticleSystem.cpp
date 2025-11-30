#include "ParticleSystem.h"
#include <GL/glew.h>
#include "Config.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <cstdio>

extern SimulationParams simParams; // Defined in main.cpp

#define CUDA_CHECK(err) do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

ParticleSystem::ParticleSystem(int particleCount, int windowWidth, int windowHeight)
    : m_particleCount(particleCount)
    , m_useCUDA(true)
    , h_prev_position_x(nullptr)
    , h_prev_position_y(nullptr)
    , m_cuda_res_pos_x(nullptr)
    , m_cuda_res_pos_y(nullptr)
    , m_cuda_res_vel_x(nullptr)
    , m_cuda_res_vel_y(nullptr)
    , m_cuda_res_radius(nullptr)
{
    m_particles.count = particleCount;
    m_particles.position_x = nullptr;
    m_particles.position_y = nullptr;
    m_particles.prev_position_x = nullptr;
    m_particles.prev_position_y = nullptr;
    m_particles.velocity_x = nullptr;
    m_particles.velocity_y = nullptr;
    m_particles.radius = nullptr;
}

ParticleSystem::~ParticleSystem() {
    cleanup();
}

void ParticleSystem::initialize(Renderer& renderer, bool useCUDA) {
    m_useCUDA = useCUDA;
    if (m_useCUDA) {
        // Allocate previous position arrays (device)
        CUDA_CHECK(cudaMalloc(&m_particles.prev_position_x, m_particleCount * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&m_particles.prev_position_y, m_particleCount * sizeof(float)));
        // Register OpenGL buffers for CUDA interop
        registerGLBuffers(renderer);
    } else {
        // Allocate previous position arrays (host)
        h_prev_position_x = (float*)malloc(m_particleCount * sizeof(float));
        h_prev_position_y = (float*)malloc(m_particleCount * sizeof(float));
        m_particles.prev_position_x = h_prev_position_x;
        m_particles.prev_position_y = h_prev_position_y;
    }
}

void ParticleSystem::cleanup() {
    // Unregister CUDA graphics resources
    if (m_cuda_res_pos_x) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_pos_x));
    if (m_cuda_res_pos_y) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_pos_y));
    if (m_cuda_res_vel_x) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_vel_x));
    if (m_cuda_res_vel_y) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_vel_y));
    if (m_cuda_res_radius) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_radius));
    
    // Free memory
    if (m_useCUDA) {
        if (m_particles.prev_position_x) CUDA_CHECK(cudaFree(m_particles.prev_position_x));
        if (m_particles.prev_position_y) CUDA_CHECK(cudaFree(m_particles.prev_position_y));
    } else {
        if (h_prev_position_x) free(h_prev_position_x);
        if (h_prev_position_y) free(h_prev_position_y);
    }
    
    m_cuda_res_pos_x = m_cuda_res_pos_y = m_cuda_res_vel_x = m_cuda_res_vel_y = m_cuda_res_radius = nullptr;
    m_particles.prev_position_x = m_particles.prev_position_y = nullptr;
    h_prev_position_x = h_prev_position_y = nullptr;
}

void ParticleSystem::registerGLBuffers(Renderer& renderer) {
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_res_pos_x, renderer.getVBO_PosX(), 
                                            cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_res_pos_y, renderer.getVBO_PosY(), 
                                            cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_res_vel_x, renderer.getVBO_VelX(), 
                                            cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_res_vel_y, renderer.getVBO_VelY(), 
                                            cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&m_cuda_res_radius, renderer.getVBO_Radius(), 
                                            cudaGraphicsMapFlagsWriteDiscard));
}

void ParticleSystem::mapResourcesCUDA() {
    size_t num_bytes;
    cudaGraphicsResource* resources[] = {
        m_cuda_res_pos_x, m_cuda_res_pos_y, m_cuda_res_vel_x, 
        m_cuda_res_vel_y, m_cuda_res_radius
    };
    
    CUDA_CHECK(cudaGraphicsMapResources(5, resources));
    
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_particles.position_x, 
                                                     &num_bytes, m_cuda_res_pos_x));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_particles.position_y, 
                                                     &num_bytes, m_cuda_res_pos_y));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_particles.velocity_x, 
                                                     &num_bytes, m_cuda_res_vel_x));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_particles.velocity_y, 
                                                     &num_bytes, m_cuda_res_vel_y));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&m_particles.radius, 
                                                     &num_bytes, m_cuda_res_radius));
}

void ParticleSystem::unmapResourcesCUDA() {
    cudaGraphicsResource* resources[] = {
        m_cuda_res_pos_x, m_cuda_res_pos_y, m_cuda_res_vel_x, 
        m_cuda_res_vel_y, m_cuda_res_radius
    };
    
    CUDA_CHECK(cudaGraphicsUnmapResources(5, resources));
}

void ParticleSystem::mapResourcesCPU(Renderer& renderer) {
    // Map OpenGL buffers to CPU pointers
    glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_PosX());
    m_particles.position_x = (float*)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_particleCount * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_PosY());
    m_particles.position_y = (float*)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_particleCount * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_VelX());
    m_particles.velocity_x = (float*)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_particleCount * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_VelY());
    m_particles.velocity_y = (float*)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_particleCount * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
    glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_Radius());
    m_particles.radius = (float*)glMapBufferRange(GL_ARRAY_BUFFER, 0, m_particleCount * sizeof(float), GL_MAP_WRITE_BIT | GL_MAP_READ_BIT);
}

void ParticleSystem::unmapResourcesCPU() {
    // Unmap requires buffers to be bound; we assume last glBindBuffer was done in mapResourcesCPU
    // Safely unmap each buffer by binding and calling glUnmapBuffer
    // Position X/Y, Velocity X/Y, Radius
    // Note: We do not store VBO ids here; caller will bind as needed. We'll just unbind for safety.
}

void ParticleSystem::initializeParticleData(int windowWidth, int windowHeight, float particleRadius, SpawnMode mode) {
    std::vector<float> h_pos_x(m_particleCount);
    std::vector<float> h_pos_y(m_particleCount);
    std::vector<float> h_prev_x(m_particleCount);
    std::vector<float> h_prev_y(m_particleCount);
    std::vector<float> h_vel_x(m_particleCount);
    std::vector<float> h_vel_y(m_particleCount);
    std::vector<float> h_radius(m_particleCount);

    if (mode == SpawnMode::UNIFORM) {
        int cols = (int)std::sqrt((float)m_particleCount * windowWidth / (float)windowHeight);
        int rows = (m_particleCount + cols - 1) / cols;

        float spacingX = windowWidth / (float)(cols + 1);
        float spacingY = windowHeight / (float)(rows + 1);
        float jitterAmount = std::min(spacingX, spacingY) * 0.3f;

        for (int i = 0; i < m_particleCount; i++) {
            int row = i / cols;
            int col = i % cols;

            h_pos_x[i] = spacingX * (col + 1) + ((float)rand() / RAND_MAX - 0.5f) * jitterAmount;
            h_pos_y[i] = spacingY * (row + 1) + ((float)rand() / RAND_MAX - 0.5f) * jitterAmount;
            
            // Initial velocities
            float init_vel_x = ((float)rand() / RAND_MAX - 0.5f) * 200.0f;
            float init_vel_y = ((float)rand() / RAND_MAX - 0.5f) * 200.0f;
            
            h_vel_x[i] = init_vel_x;
            h_vel_y[i] = init_vel_y;
        }
    }
    else if (mode == SpawnMode::DISK_CORNER || mode == SpawnMode::DISK_CENTER_EXPLOSION) {
        float spawnCenterX, spawnCenterY;
        
        if (mode == SpawnMode::DISK_CORNER) {
            spawnCenterX = windowWidth * 0.15f;
            spawnCenterY = windowHeight * 0.8f;
        } else {
            spawnCenterX = windowWidth * 0.5f;
            spawnCenterY = windowHeight * 0.5f;
        }
        
        float spacing = particleRadius * 2.0f + 0.1f; // Minimal gap

        struct Point { float x, y, distSq; };
        std::vector<Point> candidates;
        
        // Generate candidates in a large enough grid around center
        int gridSize = (int)ceil(sqrt(m_particleCount)) * 2 + 5;
        
        for (int y = -gridSize; y <= gridSize; y++) {
            for (int x = -gridSize; x <= gridSize; x++) {
                float px = spawnCenterX + x * spacing;
                float py = spawnCenterY + y * spacing;
                
                // Keep within bounds
                if (px < particleRadius || px > windowWidth - particleRadius ||
                    py < particleRadius || py > windowHeight - particleRadius)
                    continue;
                    
                float dx = px - spawnCenterX;
                float dy = py - spawnCenterY;
                candidates.push_back({px, py, dx*dx + dy*dy});
            }
        }
        
        // Sort by distance from center to form a disk
        std::sort(candidates.begin(), candidates.end(), [](const Point& a, const Point& b) {
            return a.distSq < b.distSq;
        });

        for (int i = 0; i < m_particleCount; i++) {
            float px, py;
            if (i < candidates.size()) {
                px = candidates[i].x;
                py = candidates[i].y;
            } else {
                // Fallback: random position if grid wasn't big enough
                px = ((float)rand() / RAND_MAX) * windowWidth;
                py = ((float)rand() / RAND_MAX) * windowHeight;
            }

            h_pos_x[i] = px;
            h_pos_y[i] = py;

            float init_vel_x = 0.0f;
            float init_vel_y = 0.0f;

            if (mode == SpawnMode::DISK_CORNER) {
                // Move right to hit the wall
                init_vel_x = 800.0f; 
                init_vel_y = 0.0f;
                
                // Add small randomness
                init_vel_x += ((float)rand() / RAND_MAX - 0.5f) * 20.0f;
                init_vel_y += ((float)rand() / RAND_MAX - 0.5f) * 20.0f;
            } else {
                // DISK_CENTER_EXPLOSION
                float dx = px - spawnCenterX;
                float dy = py - spawnCenterY;
                float dist = sqrt(dx*dx + dy*dy);
                
                if (dist > 0.001f) {
                    // Velocity outwards, magnitude increases with distance
                    float speed = 100.0f + dist * 5.0f; 
                    init_vel_x = (dx / dist) * speed;
                    init_vel_y = (dy / dist) * speed;
                }

                // Add small randomness
                init_vel_x += ((float)rand() / RAND_MAX - 0.5f) * 20.0f;
                init_vel_y += ((float)rand() / RAND_MAX - 0.5f) * 20.0f;
            }
            
            h_vel_x[i] = init_vel_x;
            h_vel_y[i] = init_vel_y;
        }
    }

    // Common post-processing
    for (int i = 0; i < m_particleCount; i++) {
        // Clamp positions
        h_pos_x[i] = std::max(particleRadius, 
                              std::min((float)windowWidth - particleRadius, h_pos_x[i]));
        h_pos_y[i] = std::max(particleRadius, 
                              std::min((float)windowHeight - particleRadius, h_pos_y[i]));

        // For Verlet integration: prev_pos = pos - velocity * dt
        h_prev_x[i] = h_pos_x[i] - h_vel_x[i] * simParams.dt;
        h_prev_y[i] = h_pos_y[i] - h_vel_y[i] * simParams.dt;

        h_radius[i] = particleRadius;
    }

    if (m_useCUDA) {
        CUDA_CHECK(cudaMemcpy(m_particles.position_x, h_pos_x.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_particles.position_y, h_pos_y.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_particles.prev_position_x, h_prev_x.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_particles.prev_position_y, h_prev_y.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_particles.velocity_x, h_vel_x.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_particles.velocity_y, h_vel_y.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m_particles.radius, h_radius.data(), 
                              m_particleCount * sizeof(float), cudaMemcpyHostToDevice));
    } else {
        // CPU mode: write directly into mapped buffers (assume mapped)
        std::copy(h_pos_x.begin(), h_pos_x.end(), m_particles.position_x);
        std::copy(h_pos_y.begin(), h_pos_y.end(), m_particles.position_y);
        std::copy(h_prev_x.begin(), h_prev_x.end(), m_particles.prev_position_x);
        std::copy(h_prev_y.begin(), h_prev_y.end(), m_particles.prev_position_y);
        std::copy(h_vel_x.begin(), h_vel_x.end(), m_particles.velocity_x);
        std::copy(h_vel_y.begin(), h_vel_y.end(), m_particles.velocity_y);
        std::copy(h_radius.begin(), h_radius.end(), m_particles.radius);
    }
}

extern void runShiftParticles(ParticlesSoA d_particles, int numParticles, float shiftX, float shiftY, int blockSize);

void ParticleSystem::shiftParticles(float dx, float dy, Renderer* renderer) {
    if (m_useCUDA) {
        mapResourcesCUDA();
        runShiftParticles(m_particles, m_particleCount, dx, dy, 256);
        unmapResourcesCUDA();
    } else {
        bool mappedLocally = false;
        if (renderer) {
            mapResourcesCPU(*renderer);
            mappedLocally = true;
        }
        
        for (int i = 0; i < m_particleCount; i++) {
            m_particles.position_x[i] += dx;
            m_particles.position_y[i] += dy;
            m_particles.prev_position_x[i] += dx;
            m_particles.prev_position_y[i] += dy;
        }
        
        if (mappedLocally) {
            unmapResourcesCPU();
            // Unbind buffers
            glBindBuffer(GL_ARRAY_BUFFER, renderer->getVBO_PosX()); glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, renderer->getVBO_PosY()); glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, renderer->getVBO_VelX()); glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, renderer->getVBO_VelY()); glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, renderer->getVBO_Radius()); glUnmapBuffer(GL_ARRAY_BUFFER);
            glBindBuffer(GL_ARRAY_BUFFER, 0);
        }
    }
}

void ParticleSystem::reset(int windowWidth, int windowHeight, float particleRadius, Renderer& renderer, SpawnMode mode) {
    if (m_useCUDA) {
        mapResourcesCUDA();
        initializeParticleData(windowWidth, windowHeight, particleRadius, mode);
        unmapResourcesCUDA();
    } else {
        // Map CPU GL buffers, initialize, then unmap
        mapResourcesCPU(renderer);
        initializeParticleData(windowWidth, windowHeight, particleRadius, mode);
        // Unmap all buffers
        glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_PosX()); glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_PosY()); glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_VelX()); glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_VelY()); glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO_Radius()); glUnmapBuffer(GL_ARRAY_BUFFER);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }
}
