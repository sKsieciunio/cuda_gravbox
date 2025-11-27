# CUDA Gravbox - Refactored Architecture

## Overview

This is a CUDA-accelerated particle physics simulation with OpenGL rendering. The project has been refactored into a clean, modular architecture with proper separation of concerns.

## Project Structure

### Core Classes

#### `Application` (Application.h/cpp)

Main application class that orchestrates all components and manages the main loop.

- Initializes GLFW, CUDA, OpenGL, and ImGui
- Manages the main render loop
- Handles window callbacks
- Coordinates updates between all subsystems

#### `Renderer` (Renderer.h/cpp)

Manages all OpenGL rendering operations.

- Initializes and manages OpenGL buffers (VBOs, VAO)
- Handles shader programs
- Renders particles each frame
- Manages OpenGL state

#### `Camera` (Camera.h/cpp)

Handles view projection and zoom functionality.

- Creates orthographic projection matrices
- Manages zoom level (1x-10x)
- Updates projection on window resize

#### `ParticleSystem` (ParticleSystem.h/cpp)

Manages particle data and CUDA-OpenGL interop.

- Allocates and manages particle memory
- Handles CUDA-OpenGL buffer registration
- Initializes particle positions and velocities
- Provides reset functionality

#### `PhysicsEngine` (PhysicsEngine.h/cpp)

Encapsulates physics simulation logic.

- Manages spatial grid data structures
- Coordinates CUDA kernel execution
- Handles collision detection infrastructure

### CUDA Kernels

#### `PhysicsKernels.cu`

Contains all CUDA kernel implementations:

- `updateParticlesKernel`: Verlet integration and boundary collision
- `assignParticlesToGridKernel`: Spatial hashing for broad-phase collision
- `findCellBoundsKernel`: Grid cell boundary identification
- `handleCollisionsKernel`: Narrow-phase particle-particle collision resolution

### Configuration

#### `Config.h`

Central configuration file with all simulation constants:

- Particle count and radius
- Window dimensions
- Physics parameters (gravity, dampening, restitution)
- Grid cell size
- CUDA block size

### Data Structures

#### `particle.h`

Defines core data structures:

- `ParticlesSoA`: Structure of Arrays for efficient GPU access
- `SimulationParams`: Physics simulation parameters
- `GridParams`: Spatial grid configuration

### Utilities

#### `shader.h`

Shader management class for loading and using GLSL shaders.

#### `shadersSourceCode.h`

Contains vertex and fragment shader source code for particle rendering.

## Key Features

### Physics Simulation

- **Verlet Integration**: Stable numerical integration for particle motion
- **Spatial Grid**: Efficient O(n) collision detection using uniform grid
- **Iterative Constraint Solver**: Multiple iterations for better collision resolution

### Rendering

- **CUDA-OpenGL Interop**: Zero-copy data sharing between CUDA and OpenGL
- **Velocity-based Coloring**: HSV color mapping based on particle speed
- **Point Sprites**: Efficient particle rendering with programmable point size

### Performance

- **GPU Acceleration**: All physics computations run on CUDA
- **Spatial Partitioning**: Grid-based collision reduces complexity from O(n²) to O(n)
- **Structure of Arrays**: Memory layout optimized for GPU coalescing

## Build Instructions

### Prerequisites

- CUDA Toolkit (11.0+)
- Visual Studio 2022 (for Windows)
- CMake 3.20+

### Building

```bash
# Configure
cmake --preset vs2022-release

# Build
cmake --build build/vs/vs2022-release --config Release

# Run
./build/vs/vs2022-release/Release/cuda_gravbox.exe
```

## Configuration Presets

The `Config.h` file includes three particle count presets:

1. **Small** (Default): 100 particles, 20px radius - Good for testing and visualization
2. **Medium**: 10,000 particles, 2px radius - Balanced performance
3. **Large**: 100,000 particles, 1px radius - Stress test

To change presets, modify the `PARTICLE_COUNT` and `PARTICLE_RADIUS` constants in `Config.h`.

## Controls

- **Pause/Resume**: Checkbox in UI
- **Zoom**: Mouse scroll wheel
- **Gravity**: Slider to adjust gravitational force
- **Dampening**: Energy loss on collision
- **Restitution**: Bounciness of particles
- **Reset**: Button to reinitialize particle positions

## Old Files

The original monolithic implementation has been moved to `cuda_gravbox/src/old/` for reference:

- `main.cpp.old`: Original main file with all logic
- `kernel.cu.old`: Original CUDA kernels

## License

See LICENSE file for details.
