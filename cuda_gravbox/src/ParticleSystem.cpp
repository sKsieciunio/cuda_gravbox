#include "ParticleSystem.h"
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

void ParticleSystem::initialize(Renderer& renderer) {
    // Allocate previous position arrays (not shared with OpenGL)
    CUDA_CHECK(cudaMalloc(&m_particles.prev_position_x, m_particleCount * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&m_particles.prev_position_y, m_particleCount * sizeof(float)));
    
    // Register OpenGL buffers for CUDA interop
    registerGLBuffers(renderer);
}

void ParticleSystem::cleanup() {
    // Unregister CUDA graphics resources
    if (m_cuda_res_pos_x) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_pos_x));
    if (m_cuda_res_pos_y) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_pos_y));
    if (m_cuda_res_vel_x) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_vel_x));
    if (m_cuda_res_vel_y) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_vel_y));
    if (m_cuda_res_radius) CUDA_CHECK(cudaGraphicsUnregisterResource(m_cuda_res_radius));
    
    // Free CUDA memory
    if (m_particles.prev_position_x) CUDA_CHECK(cudaFree(m_particles.prev_position_x));
    if (m_particles.prev_position_y) CUDA_CHECK(cudaFree(m_particles.prev_position_y));
    
    m_cuda_res_pos_x = m_cuda_res_pos_y = m_cuda_res_vel_x = m_cuda_res_vel_y = m_cuda_res_radius = nullptr;
    m_particles.prev_position_x = m_particles.prev_position_y = nullptr;
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

void ParticleSystem::mapResources() {
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

void ParticleSystem::unmapResources() {
    cudaGraphicsResource* resources[] = {
        m_cuda_res_pos_x, m_cuda_res_pos_y, m_cuda_res_vel_x, 
        m_cuda_res_vel_y, m_cuda_res_radius
    };
    
    CUDA_CHECK(cudaGraphicsUnmapResources(5, resources));
}

void ParticleSystem::initializeParticleData(int windowWidth, int windowHeight) {
    int cols = (int)std::sqrt((float)m_particleCount * windowWidth / (float)windowHeight);
    int rows = (m_particleCount + cols - 1) / cols;

    float spacingX = windowWidth / (float)(cols + 1);
    float spacingY = windowHeight / (float)(rows + 1);
    float jitterAmount = std::min(spacingX, spacingY) * 0.3f;

    std::vector<float> h_pos_x(m_particleCount);
    std::vector<float> h_pos_y(m_particleCount);
    std::vector<float> h_prev_x(m_particleCount);
    std::vector<float> h_prev_y(m_particleCount);
    std::vector<float> h_vel_x(m_particleCount);
    std::vector<float> h_vel_y(m_particleCount);
    std::vector<float> h_radius(m_particleCount);

    for (int i = 0; i < m_particleCount; i++) {
        int row = i / cols;
        int col = i % cols;

        h_pos_x[i] = spacingX * (col + 1) + ((float)rand() / RAND_MAX - 0.5f) * jitterAmount;
        h_pos_y[i] = spacingY * (row + 1) + ((float)rand() / RAND_MAX - 0.5f) * jitterAmount;
        h_pos_x[i] = std::max(Config::PARTICLE_RADIUS, 
                              std::min((float)windowWidth - Config::PARTICLE_RADIUS, h_pos_x[i]));
        h_pos_y[i] = std::max(Config::PARTICLE_RADIUS, 
                              std::min((float)windowHeight - Config::PARTICLE_RADIUS, h_pos_y[i]));

        h_vel_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 200.0f * simParams.dt;
        h_vel_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 200.0f * simParams.dt;

        h_prev_x[i] = h_pos_x[i] - h_vel_x[i];
        h_prev_y[i] = h_pos_y[i] - h_vel_y[i];

        h_radius[i] = Config::PARTICLE_RADIUS;
    }

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
}

void ParticleSystem::reset(int windowWidth, int windowHeight) {
    mapResources();
    initializeParticleData(windowWidth, windowHeight);
    unmapResources();
}
