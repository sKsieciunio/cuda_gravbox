#pragma once

namespace Shaders {

    constexpr const char* VERTEX_SHADER = R"(
#version 330 core

layout(location = 0) in float pos_x;
layout(location = 1) in float pos_y;
layout(location = 2) in float vel_x;
layout(location = 3) in float vel_y;
layout(location = 4) in float radius;

//layout(location = 0) in vec2 aPos;
//layout(location = 1) in vec2 aVelocity;
//layout(location = 2) in float aRadius;
//layout(location = 3) in vec3 aColor;

out vec3 particleColor;
out float particleRadius;

uniform mat4 projection;
uniform float max_speed;
uniform float radius_multiplier;
uniform int coloring_mode; // 0: Velocity, 1: ID
uniform int particle_count;

vec3 hsv2rgb(vec3 c)
{
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main()
{
    gl_Position = projection * vec4(pos_x, pos_y, 0.0, 1.0);
    gl_PointSize = radius * 2.0 * radius_multiplier;

    vec3 color;
    if (coloring_mode == 1) {
        // ID based coloring
        float hue = (float(gl_VertexID) / float(particle_count)) * 0.6;
        color = hsv2rgb(vec3(hue, 1.0, 1.0));
    } else {
        // Velocity based coloring
        float speed = length(vec2(vel_x, vel_y));
        float speed_norm = clamp(speed / max_speed, 0.0, 1.0);
        float hue = (1.0 - speed_norm) * 0.6; // From blue (0.6) to red (0.0)
        color = hsv2rgb(vec3(hue, 1.0, 1.0));
    }

    particleColor = color;
    particleRadius = radius;
}
)";

    constexpr const char* FRAGMENT_SHADER = R"(
#version 330 core

in vec3 particleColor;
in float particleRadius;

out vec4 FragColor;

void main()
{
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    float intensity = 1.0 - (dist * 2.0);
    vec3 shadedColor = particleColor * (0.6 + 0.4 * intensity);
    
    FragColor = vec4(shadedColor, 1.0);
}
)";

}
