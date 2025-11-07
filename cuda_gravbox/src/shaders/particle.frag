#version 330 core

in vec3 particleColor;
in float particleRadius;

out vec4 FragColor;

void main()
{
    // Calculate distance from center of point sprite
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    // Discard fragments outside the circle
    if (dist > 0.5) {
        discard;
    }
    
    // Simple shading: darker towards edges
    float intensity = 1.0 - (dist * 2.0);
    vec3 shadedColor = particleColor * (0.6 + 0.4 * intensity);
    
    FragColor = vec4(shadedColor, 1.0);
}
