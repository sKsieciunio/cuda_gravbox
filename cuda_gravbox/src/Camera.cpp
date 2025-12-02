#include "Camera.h"

Camera::Camera(int windowWidth, int windowHeight)
    : m_windowWidth(windowWidth), m_windowHeight(windowHeight), m_zoom(Config::DEFAULT_ZOOM)
{
    updateProjectionMatrix();
}

void Camera::setWindowSize(int width, int height)
{
    m_windowWidth = width;
    m_windowHeight = height;
    updateProjectionMatrix();
}

void Camera::setZoom(float zoom)
{
    m_zoom = std::clamp(zoom, Config::MIN_ZOOM, Config::MAX_ZOOM);
    updateProjectionMatrix();
}

void Camera::adjustZoom(float delta)
{
    m_zoom *= delta;
    m_zoom = std::clamp(m_zoom, Config::MIN_ZOOM, Config::MAX_ZOOM);
    updateProjectionMatrix();
}

void Camera::updateProjectionMatrix()
{
    float centerX = m_windowWidth / 2.0f;
    float centerY = m_windowHeight / 2.0f;

    float halfWidth = (m_windowWidth / 2.0f) / m_zoom;
    float halfHeight = (m_windowHeight / 2.0f) / m_zoom;

    float left = centerX - halfWidth;
    float right = centerX + halfWidth;
    float bottom = centerY - halfHeight;
    float top = centerY + halfHeight;

    createOrthographicMatrix(left, right, bottom, top);
}

void Camera::createOrthographicMatrix(float left, float right, float bottom, float top)
{
    for (int i = 0; i < 16; i++)
        m_projectionMatrix[i] = 0.0f;

    m_projectionMatrix[0] = 2.0f / (right - left);
    m_projectionMatrix[5] = 2.0f / (top - bottom);
    m_projectionMatrix[10] = -1.0f;
    m_projectionMatrix[12] = -(right + left) / (right - left);
    m_projectionMatrix[13] = -(top + bottom) / (top - bottom);
    m_projectionMatrix[15] = 1.0f;
}
