#include "Renderer.h"
#include "shadersSourceCode.h"
#include "Config.h"

Renderer::Renderer()
    : m_particleShader(), m_vao(0), m_vbo_pos_x(0), m_vbo_pos_y(0), m_vbo_vel_x(0), m_vbo_vel_y(0), m_vbo_radius(0), m_particleCount(0)
{
}

Renderer::~Renderer()
{
    cleanup();
}

void Renderer::initialize()
{
    initialize(Config::PARTICLE_COUNT);
}

void Renderer::initialize(int particleCount)
{
    m_particleCount = particleCount;

    m_particleShader = Shader::FromSource(
        Shaders::VERTEX_SHADER,
        Shaders::FRAGMENT_SHADER);

    setupBuffers(particleCount);
    setupVertexAttributes();

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void Renderer::cleanup()
{
    if (m_vao)
    {
        glDeleteVertexArrays(1, &m_vao);
        m_vao = 0;
    }

    if (m_vbo_pos_x)
        glDeleteBuffers(1, &m_vbo_pos_x);
    if (m_vbo_pos_y)
        glDeleteBuffers(1, &m_vbo_pos_y);
    if (m_vbo_vel_x)
        glDeleteBuffers(1, &m_vbo_vel_x);
    if (m_vbo_vel_y)
        glDeleteBuffers(1, &m_vbo_vel_y);
    if (m_vbo_radius)
        glDeleteBuffers(1, &m_vbo_radius);

    m_vbo_pos_x = m_vbo_pos_y = m_vbo_vel_x = m_vbo_vel_y = m_vbo_radius = 0;
}

void Renderer::setupBuffers(int particleCount)
{
    glGenBuffers(1, &m_vbo_pos_x);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_pos_x);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &m_vbo_pos_y);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_pos_y);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &m_vbo_vel_x);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_vel_x);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &m_vbo_vel_y);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_vel_y);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

    glGenBuffers(1, &m_vbo_radius);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_radius);
    glBufferData(GL_ARRAY_BUFFER, particleCount * sizeof(float), nullptr, GL_DYNAMIC_DRAW);
}

void Renderer::setupVertexAttributes()
{
    glGenVertexArrays(1, &m_vao);
    glBindVertexArray(m_vao);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_pos_x);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_pos_y);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_vel_x);
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_vel_y);
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);

    glBindBuffer(GL_ARRAY_BUFFER, m_vbo_radius);
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);
}

void Renderer::beginFrame()
{
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void Renderer::render(int particleCount, const Camera &camera, float velocityHueRange, ColoringMode coloringMode)
{
    m_particleShader.use();
    m_particleShader.setMat4("projection", camera.getProjectionMatrix());
    m_particleShader.setFloat("max_speed", velocityHueRange);
    m_particleShader.setFloat("radius_multiplier", camera.getZoom());
    m_particleShader.setInt("coloring_mode", (int)coloringMode);
    m_particleShader.setInt("particle_count", particleCount);

    glBindVertexArray(m_vao);
    glDrawArrays(GL_POINTS, 0, particleCount);
}
