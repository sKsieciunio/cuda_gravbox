#pragma once

#include "particle.h"
#include "Renderer.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

class ParticleSystem {
public:
    ParticleSystem(int particleCount, int windowWidth, int windowHeight);
    ~ParticleSystem();
    
    void initialize(Renderer& renderer);
    void cleanup();
    
    void reset(int windowWidth, int windowHeight, float particleRadius);
    
    void mapResources();
    void unmapResources();
    
    ParticlesSoA& getParticles() { return m_particles; }
    const ParticlesSoA& getParticles() const { return m_particles; }
    
    int getCount() const { return m_particleCount; }

private:
    int m_particleCount;
    ParticlesSoA m_particles;
    
    // CUDA-OpenGL interop resources
    cudaGraphicsResource* m_cuda_res_pos_x;
    cudaGraphicsResource* m_cuda_res_pos_y;
    cudaGraphicsResource* m_cuda_res_vel_x;
    cudaGraphicsResource* m_cuda_res_vel_y;
    cudaGraphicsResource* m_cuda_res_radius;
    
    void registerGLBuffers(Renderer& renderer);
    void initializeParticleData(int windowWidth, int windowHeight, float particleRadius);
};
