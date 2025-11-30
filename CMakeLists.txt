cmake_minimum_required(VERSION 3.20)

project(cuda_gravbox LANGUAGES CXX CUDA)

# Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Dependencies
find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Paths
set(PROJ_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/cuda_gravbox)
set(SRC_DIR   ${PROJ_ROOT}/src)
set(EXT_DIR   ${PROJ_ROOT}/external)

# ImGui sources (vendor)
set(IMGUI_DIR ${EXT_DIR}/imgui)
set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backend/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backend/imgui_impl_opengl3.cpp
)

# App sources
set(APP_SOURCES
    ${SRC_DIR}/main.cpp
    ${SRC_DIR}/Application.cpp
    ${SRC_DIR}/Camera.cpp
    ${SRC_DIR}/Renderer.cpp
    ${SRC_DIR}/ParticleSystem.cpp
    ${SRC_DIR}/PhysicsEngine.cpp
    ${SRC_DIR}/CpuPhysicsEngine.cpp
    ${SRC_DIR}/PhysicsKernels.cu
)

add_executable(${PROJECT_NAME}
    ${APP_SOURCES}
    ${IMGUI_SOURCES}
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE
    ${SRC_DIR}
    ${IMGUI_DIR}
    ${EXT_DIR}/glfw/include
    ${EXT_DIR}/glew/include
    ${CUDAToolkit_INCLUDE_DIRS}
)

# Use GLEW (static) and GLFW (static)
set(GLEW_LIB_DIR ${EXT_DIR}/glew/lib/Release/x64)
set(GLFW_LIB_DIR ${EXT_DIR}/glfw/lib-vc2022)

target_link_directories(${PROJECT_NAME} PRIVATE
    ${GLEW_LIB_DIR}
    ${GLFW_LIB_DIR}
)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    GLEW_STATIC
    IMGUI_IMPL_OPENGL_LOADER_GLEW
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        OpenGL::GL
        glew32s
        glfw3_mt
        CUDA::cudart
        user32
        gdi32
        shell32
        advapi32
        ole32
        uuid
        winmm
        ws2_32
        bcrypt
)

# CUDA compilation options (keep defaults conservative)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Windows: copy GLFW runtime DLL next to the executable after build
# No DLL copy needed: static GLFW linking

# Group sources neatly in IDEs
source_group(TREE ${SRC_DIR} FILES ${APP_SOURCES})
source_group(TREE ${IMGUI_DIR} FILES ${IMGUI_SOURCES})
