#pragma once

#include <GL/glew.h>
#include "shader.h"
#include "Camera.h"

enum class ColoringMode {
    VELOCITY = 0,
    ID = 1
};

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    void initialize();
    void initialize(int particleCount);
    void cleanup();
    
    void beginFrame();
    void render(int particleCount, const Camera& camera, float velocityHueRange, ColoringMode coloringMode = ColoringMode::VELOCITY);
    
    GLuint getVBO_PosX() const { return m_vbo_pos_x; }
    GLuint getVBO_PosY() const { return m_vbo_pos_y; }
    GLuint getVBO_VelX() const { return m_vbo_vel_x; }
    GLuint getVBO_VelY() const { return m_vbo_vel_y; }
    GLuint getVBO_Radius() const { return m_vbo_radius; }

private:
    Shader m_particleShader;
    GLuint m_vao;
    GLuint m_vbo_pos_x;
    GLuint m_vbo_pos_y;
    GLuint m_vbo_vel_x;
    GLuint m_vbo_vel_y;
    GLuint m_vbo_radius;
    int m_particleCount;
    
    void setupBuffers(int particleCount);
    void setupVertexAttributes();
};
