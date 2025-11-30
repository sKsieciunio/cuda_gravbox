#pragma once

#include "particle.h"
#include "Renderer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class ParticleSystem {
public:
    ParticleSystem(int particleCount, int windowWidth, int windowHeight);
    ~ParticleSystem();
    
    void initialize(Renderer& renderer, bool useCUDA = true);
    void cleanup();
    
    void reset(int windowWidth, int windowHeight, float particleRadius, Renderer& renderer);
    
    void mapResourcesCUDA();
    void unmapResourcesCUDA();
    void mapResourcesCPU(Renderer& renderer);
    void unmapResourcesCPU();
    
    ParticlesSoA& getParticles() { return m_particles; }
    const ParticlesSoA& getParticles() const { return m_particles; }
    
    int getCount() const { return m_particleCount; }
    bool isCUDA() const { return m_useCUDA; }

private:
    int m_particleCount;
    ParticlesSoA m_particles;
    bool m_useCUDA;
    // Host-side arrays when CPU mode
    float* h_prev_position_x;
    float* h_prev_position_y;
    
    
    // CUDA-OpenGL interop resources
    cudaGraphicsResource* m_cuda_res_pos_x;
    cudaGraphicsResource* m_cuda_res_pos_y;
    cudaGraphicsResource* m_cuda_res_vel_x;
    cudaGraphicsResource* m_cuda_res_vel_y;
    cudaGraphicsResource* m_cuda_res_radius;
    
    void registerGLBuffers(Renderer& renderer);
    void initializeParticleData(int windowWidth, int windowHeight, float particleRadius);
};
