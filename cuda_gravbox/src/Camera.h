#pragma once

#include "Config.h"
#include <algorithm>

class Camera
{
public:
    Camera(int windowWidth, int windowHeight);

    void setWindowSize(int width, int height);
    void setZoom(float zoom);
    void adjustZoom(float delta);

    float getZoom() const { return m_zoom; }
    const float *getProjectionMatrix() const { return m_projectionMatrix; }

    void updateProjectionMatrix();

private:
    int m_windowWidth;
    int m_windowHeight;
    float m_zoom;
    float m_projectionMatrix[16];

    void createOrthographicMatrix(float left, float right, float bottom, float top);
};
