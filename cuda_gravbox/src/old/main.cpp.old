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

#define CUDA_CHECK(err) do { \
    cudaError_t _err = (err); \
    if (_err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(_err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

constexpr float PARTICLE_RADIUS = 20.0f;
constexpr int n = 100;
//constexpr float PARTICLE_RADIUS = 2.0f;
//constexpr int n = 10000;
//constexpr float PARTICLE_RADIUS = 1.0f;
//constexpr int n = 100000;

int window_width = 1000;
int window_height = 800;
float projection[16];
float zoomLevel = 1.0f;
float velocityToHueRange = 300.0f; // Max speed for color mapping

SimulationParams simParams {
	.gravity = -500.0f,  // Pixels/s^2 (negative = downward)
	.dt = 0.0006f,  
	.dampening = 0.6f,    // Energy loss on collision
	.bounds_width = (float)window_width,
	.bounds_height = (float)window_height,
	.restitution = 0.6f // Coefficient of restitution
};

constexpr float GRID_CELL_SIZE = 2.0f * PARTICLE_RADIUS;
//constexpr float GRID_CELL_SIZE = 1200.0f;
GridParams gridParams {
	.grid_width = (int)(window_width / GRID_CELL_SIZE) + 1,
	.grid_height = (int)(window_height / GRID_CELL_SIZE) + 1,
	.cell_size = GRID_CELL_SIZE,
};

//extern void updateParticles(Particle* d_particles, int numParticles, const SimulationParams& params);
extern void handleCollisions(
	ParticlesSoA d_particles,
	int* d_particleGridIndex,
	int* d_particleIndices,
	int* d_gridCellStart,
	int* d_gridCellEnd, 
	int numParticles,
	const GridParams& gridParams,
	const SimulationParams& simParams
);

void initializeParticles(ParticlesSoA& h_particles, int numParticles, float width, float height);
void updateProjectionMatrix();
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
	CUDA_CHECK(cudaGLSetGLDevice(0));

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	Shader particleShader = Shader::FromSource(
		Shaders::VERTEX_SHADER,
		Shaders::FRAGMENT_SHADER
	);

	updateProjectionMatrix();
	//createOrthographicMatrix(projection, 0.0f, (float)window_width, 0.0f, (float)window_height);

	//ParticlesSoA d_particles;
	//d_particles.count = n;
	//CUDA_CHECK(cudaMalloc(&d_particles.position_x, n * sizeof(float)));
	//CUDA_CHECK(cudaMalloc(&d_particles.position_y, n * sizeof(float)));
	//CUDA_CHECK(cudaMalloc(&d_particles.prev_position_x, n * sizeof(float)));
	//CUDA_CHECK(cudaMalloc(&d_particles.prev_position_y, n * sizeof(float)));
	//CUDA_CHECK(cudaMalloc(&d_particles.velocity_x, n * sizeof(float)));
	//CUDA_CHECK(cudaMalloc(&d_particles.velocity_y, n * sizeof(float)));
	//CUDA_CHECK(cudaMalloc(&d_particles.radius, n * sizeof(float)));

	//initializeParticles(d_particles, n, (float)window_width, (float)window_height);

	GLuint vbo_pos_x, vbo_pos_y, vbo_vel_x, vbo_vel_y, vbo_radius;

	glGenBuffers(1, &vbo_pos_x);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_x);
	glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_pos_y);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_y);
	glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_vel_x);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vel_x);
	glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_vel_y);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_vel_y);
	glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	glGenBuffers(1, &vbo_radius);
	glBindBuffer(GL_ARRAY_BUFFER, vbo_radius);
	glBufferData(GL_ARRAY_BUFFER, n * sizeof(float), nullptr, GL_DYNAMIC_DRAW);

	cudaGraphicsResource *cuda_res_pos_x, *cuda_res_pos_y, *cuda_res_vel_x, *cuda_res_vel_y, *cuda_res_radius;
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_pos_x, vbo_pos_x, cudaGraphicsMapFlagsWriteDiscard));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_pos_y, vbo_pos_y, cudaGraphicsMapFlagsWriteDiscard));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_vel_x, vbo_vel_x, cudaGraphicsMapFlagsWriteDiscard));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_vel_y, vbo_vel_y, cudaGraphicsMapFlagsWriteDiscard));
	CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cuda_res_radius, vbo_radius, cudaGraphicsMapFlagsWriteDiscard));

	ParticlesSoA d_particles;
	d_particles.count = n;
	CUDA_CHECK(cudaMalloc(&d_particles.prev_position_x, n * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&d_particles.prev_position_y, n * sizeof(float)));

	d_particles.position_x = nullptr;
	d_particles.position_y = nullptr;
	d_particles.velocity_x = nullptr;
	d_particles.velocity_y = nullptr;
	d_particles.radius = nullptr;

	{
		size_t num_bytes;

		cudaGraphicsResource* resources[] = {
			cuda_res_pos_x,
			cuda_res_pos_y,
			cuda_res_vel_x,
			cuda_res_vel_y,
			cuda_res_radius
		};
		CUDA_CHECK(cudaGraphicsMapResources(5, resources));

		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.position_x, &num_bytes, cuda_res_pos_x));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.position_y, &num_bytes, cuda_res_pos_y));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.velocity_x, &num_bytes, cuda_res_vel_x));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.velocity_y, &num_bytes, cuda_res_vel_y));
		CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.radius, &num_bytes, cuda_res_radius));

		initializeParticles(d_particles, n, (float)window_width, (float)window_height);

		CUDA_CHECK(cudaGraphicsUnmapResources(5, resources));
	}

	GLuint vao;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_x);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_y);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_vel_x);
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_vel_y);
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

	glBindBuffer(GL_ARRAY_BUFFER, vbo_radius);
	glEnableVertexAttribArray(4);
	glVertexAttribPointer(4, 1, GL_FLOAT, GL_FALSE, 0, (void*)0);

	int* d_particleGridIndex;
	int* d_particleIndices;
	int* d_gridCellStart;
	int* d_gridCellEnd;

	CUDA_CHECK(cudaMalloc(&d_particleGridIndex, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_particleIndices, n * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_gridCellStart, gridParams.grid_width * gridParams.grid_height * sizeof(int)));
	CUDA_CHECK(cudaMalloc(&d_gridCellEnd, gridParams.grid_width * gridParams.grid_height * sizeof(int)));

	std::vector<int> h_indices(n);
	for (int i = 0; i < n; i++) h_indices[i] = i;
	CUDA_CHECK(cudaMemcpy(d_particleIndices, h_indices.data(), n * sizeof(int), cudaMemcpyHostToDevice));

	glEnable(GL_PROGRAM_POINT_SIZE);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	bool paused = true;
	//float gravityControl = simParams.gravity;
	//float dampeningControl = simParams.dampening;

	glfwSetWindowSizeCallback(window, [](GLFWwindow* window, int width, int height) {
		glViewport(0, 0, width, height);
		window_width = width;
		window_height = height;
		updateProjectionMatrix();
		//createOrthographicMatrix(projection, 0.0f, (float)window_width, 0.0f, (float)window_height);
		simParams.bounds_width = (float)window_width;
		simParams.bounds_height = (float)window_height;
	});

	glfwSetScrollCallback(window, [](GLFWwindow* window, double xoffset, double yoffset) {
		zoomLevel *= (float)yoffset * 0.1f + 1.0f;
		zoomLevel = std::max(1.0f, std::min(zoomLevel, 10.0f));
		updateProjectionMatrix();
	});

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		if (!paused)
		{
			size_t num_bytes;

			cudaGraphicsResource* resources[] = {
				cuda_res_pos_x,
				cuda_res_pos_y,
				cuda_res_vel_x,
				cuda_res_vel_y,
				cuda_res_radius
			};
			CUDA_CHECK(cudaGraphicsMapResources(5, resources));

			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.position_x, &num_bytes, cuda_res_pos_x));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.position_y, &num_bytes, cuda_res_pos_y));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.velocity_x, &num_bytes, cuda_res_vel_x));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.velocity_y, &num_bytes, cuda_res_vel_y));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.radius, &num_bytes, cuda_res_radius));

			//cudaError_t mapErr = cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
			//if (mapErr != cudaSuccess) {
			//	fprintf(stderr, "Failed to map resource: %s\n", cudaGetErrorString(mapErr));
			//}
			//cudaGraphicsResourceGetMappedPointer((void**)&d_particles, &num_bytes, cuda_vbo_resource);

			//updateParticles(d_particles, n, simParams);

			//cudaDeviceSynchronize();

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

			CUDA_CHECK(cudaGraphicsUnmapResources(5, resources));
		}

		particleShader.use();
		particleShader.setMat4("projection", projection);
		particleShader.setFloat("max_speed", velocityToHueRange);
		particleShader.setFloat("radius_multiplier", zoomLevel);

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
		ImGui::SliderFloat("Gravity", &simParams.gravity, -5000.0f, 0.0f);
		ImGui::SliderFloat("Dampening", &simParams.dampening, 0.0f, 1.0f);
		ImGui::SliderFloat("Restitution", &simParams.restitution, 0.0f, 1.0f);
		ImGui::SliderFloat("simulation dt", &simParams.dt, 0.0001f, 0.3f, "%.4f", ImGuiSliderFlags_Logarithmic);
		ImGui::SliderFloat("Hue range", &velocityToHueRange, 10.0f, 300.0f);

		if (ImGui::Button("Reset Particles"))
		{
			size_t num_bytes;
			cudaGraphicsResource* resources[] = { cuda_res_pos_x, cuda_res_pos_y, cuda_res_vel_x, cuda_res_vel_y, cuda_res_radius };

			CUDA_CHECK(cudaGraphicsMapResources(5, resources, 0));

			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.position_x, &num_bytes, cuda_res_pos_x));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.position_y, &num_bytes, cuda_res_pos_y));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.velocity_x, &num_bytes, cuda_res_vel_x));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.velocity_y, &num_bytes, cuda_res_vel_y));
			CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_particles.radius, &num_bytes, cuda_res_radius));

			initializeParticles(d_particles, n, (float)window_width, (float)window_height);

			CUDA_CHECK(cudaGraphicsUnmapResources(5, resources, 0));
		}

		ImGui::End();

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		glfwSwapBuffers(window);
	}

	CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_pos_x));
	CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_pos_y));
	CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_vel_x));
	CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_vel_y));
	CUDA_CHECK(cudaGraphicsUnregisterResource(cuda_res_radius));

	glDeleteVertexArrays(1, &vao);
	glDeleteBuffers(1, &vbo_pos_x);
	glDeleteBuffers(1, &vbo_pos_y);
	glDeleteBuffers(1, &vbo_vel_x);
	glDeleteBuffers(1, &vbo_vel_y);
	glDeleteBuffers(1, &vbo_radius);

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();

	CUDA_CHECK(cudaFree(d_particles.prev_position_x));
	CUDA_CHECK(cudaFree(d_particles.prev_position_y));
	CUDA_CHECK(cudaFree(d_particleGridIndex));
	CUDA_CHECK(cudaFree(d_particleIndices));
	CUDA_CHECK(cudaFree(d_gridCellStart));
	CUDA_CHECK(cudaFree(d_gridCellEnd));

	glfwTerminate();
	return 0;
}

void updateProjectionMatrix()
{
	float centerX = window_width / 2.0f;
	float centerY = window_height / 2.0f;

	float halfWidth = (window_width / 2.0f) / zoomLevel;
	float halfHeight = (window_height / 2.0f) / zoomLevel;

	float left = centerX - halfWidth;
	float right = centerX + halfWidth;
	float bottom = centerY - halfHeight;
	float top = centerY + halfHeight;

	createOrthographicMatrix(projection, left, right, bottom, top);
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

void initializeParticles(ParticlesSoA& h_particles, int numParticles, float width, float height)
{
	int cols = (int)std::sqrt((float)numParticles * width / height);
	int rows = (numParticles + cols - 1) / cols;

	float spacingX = width / (float)(cols + 1);
	float spacingY = height / (float)(rows + 1);

	float jitterAmount = std::min(spacingX, spacingY) * 0.3f;

	std::vector<float> h_pos_x(numParticles);
	std::vector<float> h_pos_y(numParticles);
	std::vector<float> h_prev_x(numParticles);
	std::vector<float> h_prev_y(numParticles);
	std::vector<float> h_vel_x(numParticles);
	std::vector<float> h_vel_y(numParticles);
	std::vector<float> h_radius(numParticles);

	for (int i = 0; i < numParticles; i++) {
		int row = i / cols;
		int col = i % cols;

		h_pos_x[i] = spacingX * (col + 1) + ((float)rand() / RAND_MAX - 0.5f) * jitterAmount;
		h_pos_y[i] = spacingY * (row + 1) + ((float)rand() / RAND_MAX - 0.5f) * jitterAmount;
		h_pos_x[i] = std::max(PARTICLE_RADIUS, std::min(width - PARTICLE_RADIUS, h_pos_x[i]));
		h_pos_y[i] = std::max(PARTICLE_RADIUS, std::min(height - PARTICLE_RADIUS, h_pos_y[i]));

		h_vel_x[i] = ((float)rand() / RAND_MAX - 0.5f) * 200.0f * simParams.dt; // -100 to 100 pixels/s
		h_vel_y[i] = ((float)rand() / RAND_MAX - 0.5f) * 200.0f * simParams.dt;

		h_prev_x[i] = h_pos_x[i] - h_vel_x[i];
		h_prev_y[i] = h_pos_y[i] - h_vel_y[i];

		h_radius[i] = PARTICLE_RADIUS;
	}

	CUDA_CHECK(cudaMemcpy(h_particles.position_x, h_pos_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(h_particles.position_y, h_pos_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(h_particles.prev_position_x, h_prev_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(h_particles.prev_position_y, h_prev_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(h_particles.velocity_x, h_vel_x.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(h_particles.velocity_y, h_vel_y.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CHECK(cudaMemcpy(h_particles.radius, h_radius.data(), numParticles * sizeof(float), cudaMemcpyHostToDevice));
}
