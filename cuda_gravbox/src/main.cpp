#include <iostream>
#include <vector>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "imgui.h"
#include "backend/imgui_impl_glfw.h"
#include "backend/imgui_impl_opengl3.h"

#include "shader.h"
#include "shadersSourceCode.h"
#include "particle.h"

//constexpr float PARTICLE_RADIUS = 50.0f;
//constexpr int n = 10;
constexpr float PARTICLE_RADIUS = 2.0f;
constexpr int n = 10000;

int window_width = 1000;
int window_height = 800;
float projection[16];
float velocityToHueRange = 300.0f; // Max speed for color mapping

SimulationParams simParams {
	.gravity = -250.0f,  // Pixels/s^2 (negative = downward)
	.dt = 0.0006f,  
	.dampening = 0.8f,    // Energy loss on collision
	.bounds_width = (float)window_width,
	.bounds_height = (float)window_height,
	.restitution = 0.8f // Coefficient of restitution
};

constexpr float GRID_CELL_SIZE = 2.0f * PARTICLE_RADIUS;
//constexpr float GRID_CELL_SIZE = 1200.0f;
GridParams gridParams {
	.grid_width = (int)(window_width / GRID_CELL_SIZE) + 1,
	.grid_height = (int)(window_height / GRID_CELL_SIZE) + 1,
	.cell_size = GRID_CELL_SIZE,
};

extern void updateParticles(Particle* d_particles, int numParticles, const SimulationParams& params);
extern void handleCollisions(
	Particle* d_particles,
	int* d_particleGridIndex,
	int* d_particleIndices,
	int* d_gridCellStart,
	int* d_gridCellEnd, 
	int numParticles,
	const GridParams& gridParams,
	const SimulationParams& simParams
);

void initializeParticles(Particle* h_particles, int numParticles, float width, float height);
void createOrthographicMatrix(float* matrix, float left, float right, float bottom, float top);

int main()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	GLFWwindow* window = glfwCreateWindow(window_width, window_height, "CUDA Gravity Simulation", nullptr, nullptr);
	glfwMakeContextCurrent(window);

	glfwSetWindowSizeLimits(window, window_width, window_height, GLFW_DONT_CARE, GLFW_DONT_CARE);
	glfwSwapInterval(0); // Disable vsync

	glewInit();

	//cudaSetDevice(0); // this shit causes issues dont touch
	cudaGLSetGLDevice(0);

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	Shader particleShader = Shader::FromSource(
		Shaders::VERTEX_SHADER,
		Shaders::FRAGMENT_SHADER
	);

	createOrthographicMatrix(projection, 0.0f, (float)window_width, 0.0f, (float)window_height);

	std::vector<Particle> h_particles(n);
	initializeParticles(h_particles.data(), n, (float)window_width, (float)window_height);

	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, n * sizeof(Particle), h_particles.data(), GL_DYNAMIC_DRAW);

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, position));
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, velocity));
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, radius));
	glEnableVertexAttribArray(2);
	//glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void*)offsetof(Particle, color));
	//glEnableVertexAttribArray(3);

	cudaGraphicsResource* cuda_vbo_resource;
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

	int* d_particleGridIndex;
	int* d_particleIndices;
	int* d_gridCellStart;
	int* d_gridCellEnd;

	cudaMalloc(&d_particleGridIndex, n * sizeof(int));
	cudaMalloc(&d_particleIndices, n * sizeof(int));
	cudaMalloc(&d_gridCellStart, gridParams.grid_width * gridParams.grid_height * sizeof(int));
	cudaMalloc(&d_gridCellEnd, gridParams.grid_width * gridParams.grid_height * sizeof(int));

	std::vector<int> h_indices(n);
	for (int i = 0; i < n; i++) h_indices[i] = i;
	cudaMemcpy(d_particleIndices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	bool paused = false;
	//float gravityControl = simParams.gravity;
	//float dampeningControl = simParams.dampening;

	glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) {
		glViewport(0, 0, width, height);
		window_width = width;
		window_height = height;
		createOrthographicMatrix(projection, 0.0f, (float)window_width, 0.0f, (float)window_height);
		simParams.bounds_width = (float)window_width;
		simParams.bounds_height = (float)window_height;
	});

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		if (!paused)
		{
			Particle* d_particles;
			size_t num_bytes;
			cudaError_t mapErr = cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
			if (mapErr != cudaSuccess) {
				fprintf(stderr, "Failed to map resource: %s\n", cudaGetErrorString(mapErr));
			}

			cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);

			updateParticles(d_particles, n, simParams);

			cudaDeviceSynchronize();

			handleCollisions(
				d_particles,
				d_particleGridIndex,
				d_particleIndices,
				d_gridCellStart,
				d_gridCellEnd,
				n,
				gridParams,
				simParams
			);

			cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
		}

		particleShader.use();
		particleShader.setMat4("projection", projection);
		particleShader.setFloat("max_speed", velocityToHueRange);

		glBindVertexArray(vao);
		glDrawArrays(GL_POINTS, 0, n);

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		ImGui::Begin("Settings");
		ImGui::Text("Frame Time: %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
		ImGui::Text("Ballz: %d", n);
		ImGui::Separator();

		ImGui::Checkbox("Pause", &paused);
		ImGui::SliderFloat("Gravity", &simParams.gravity, -500.0f, 0.0f);
		ImGui::SliderFloat("Dampening", &simParams.dampening, 0.0f, 1.0f);
		ImGui::SliderFloat("Restitution", &simParams.restitution, 0.0f, 1.0f);
		ImGui::SliderFloat("simulation dt", &simParams.dt, 0.0001f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderFloat("Hue range", &velocityToHueRange, 10.0f, 300.0f);

		if (ImGui::Button("Reset Particles"))
		{
			initializeParticles(h_particles.data(), n, (float)window_width, (float)window_height);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferSubData(GL_ARRAY_BUFFER, 0, n * sizeof(Particle), h_particles.data());
		}

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	cudaGraphicsUnregisterResource(cuda_vbo_resource);
	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	cudaFree(d_particleGridIndex);
	cudaFree(d_particleIndices);
	cudaFree(d_gridCellStart);
	cudaFree(d_gridCellEnd);

	glfwTerminate();
	return 0;
}

void createOrthographicMatrix(float* matrix, float left, float right, float bottom, float top)
{
	for (int i = 0; i < 16; i++) matrix[i] = 0.0f;
	matrix[0] = 2.0f / (right - left);
	matrix[5] = 2.0f / (top - bottom);
	matrix[10] = -1.0f;
	matrix[12] = -(right + left) / (right - left);
	matrix[13] = -(top + bottom) / (top - bottom);
	matrix[15] = 1.0f;
}

void initializeParticles(Particle* h_particles, int numParticles, float width, float height)
{
	for (int i = 0; i < numParticles; i++) {
		// Random position
		h_particles[i].position.x = (float)(rand() % (int)width);
		h_particles[i].position.y = (float)(rand() % (int)height);

		// Random velocity
		h_particles[i].velocity.x = ((float)rand() / RAND_MAX - 0.5f) * 200.0f; // -100 to 100 pixels/s
		h_particles[i].velocity.y = ((float)rand() / RAND_MAX - 0.5f) * 200.0f;

		h_particles[i].radius = PARTICLE_RADIUS;

		// Random color
		h_particles[i].color.x = (float)rand() / RAND_MAX; // R
		h_particles[i].color.y = (float)rand() / RAND_MAX; // G
		h_particles[i].color.z = (float)rand() / RAND_MAX; // B
	}
}
