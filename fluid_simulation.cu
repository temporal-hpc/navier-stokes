#include <iostream>
#include <vector>
#include <mimir/mimir.hpp>
#include <imgui.h>

#include "kernels.cuh"

using namespace mimir;


// Reduce all block maxima on the CPU to find the global maximum velocity
float reduceMax(const float* d_blockMax, int numBlocks) {
    float* h_blockMax = new float[numBlocks];
    (cudaMemcpy(h_blockMax, d_blockMax, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    float maxVal = 0.0f;
    for (int i = 0; i < numBlocks; ++i) {
        if (h_blockMax[i] > maxVal)
            maxVal = h_blockMax[i];
    }

    delete[] h_blockMax;
    return maxVal;
}


// ----------------------------------------------------------------------------
// Obstacle Parameters (GUI-controlled)
// ----------------------------------------------------------------------------
static int obstacleShape = 0;          // 0: Circle, 1: Square, 2: Diagonal, 3: Combined
static float obstacleCenterX = 0.5f;   // Relative X position (0.0 to 1.0)
static float obstacleCenterY = 0.5f;   // Relative Y position (0.0 to 1.0)
static float obstacleSize = 0.1f;      // Relative size (0.01 to 0.5)
static bool followCursor = true; // Whether the obstacle follows the mouse cursor

// ----------------------------------------------------------------------------
// Emitter Barrier Parameters (GUI-controlled)
// ----------------------------------------------------------------------------
static float barrierCenterX = 0.3f;
static float barrierCenterY = 0.5f;
static float barrierWidth   = 0.05f;
static float barrierHeight  = 0.1f;
static float barrierVelocityU = 2.0f;
static float barrierVelocityV = 0.0f;
static bool showBarrier = false;

void guiCallback_3() {
    ImGui::Begin("Fluid Simulation Controls");

    // Shortcuts
    ImGui::SeparatorText("Emitter Barrier Shortcuts");
    ImGui::BulletText("'V' - Toggle barrier visibility");
    ImGui::BulletText("'F' - Toggle obstacle following cursor");
    if (ImGui::IsKeyPressed(ImGuiKey_V)) {
        showBarrier = !showBarrier;
    }
    if (ImGui::IsKeyPressed(ImGuiKey_F)) {
        followCursor = !followCursor;
    }

    if (showBarrier) {
        ImGui::SeparatorText("Emitter Barrier (manual)");
        ImGui::SliderFloat("Position X", &barrierCenterX, 0.0f, 1.0f);
        ImGui::SliderFloat("Position Y", &barrierCenterY, 0.0f, 1.0f);
        ImGui::SliderFloat("Width", &barrierWidth, 0.01f, 0.2f);
        ImGui::SliderFloat("Height", &barrierHeight, 0.01f, 0.5f);
        ImGui::Text("Barrier Position: (%.2f, %.2f)", barrierCenterX, barrierCenterY);
    }

    // Obstacle follows the mouse
    if (followCursor) {
        ImVec2 mousePos = ImGui::GetMousePos();
        obstacleCenterX = std::clamp(mousePos.x / WIDTH, 0.0f, 1.0f);
        obstacleCenterY = std::clamp(1.0f - mousePos.y / HEIGHT, 0.0f, 1.0f);
    }

    ImGui::SeparatorText("Obstacle (follows cursor)");
    ImGui::Text("Select Shape:");
    ImGui::RadioButton("None",              &obstacleShape, 0);
    ImGui::RadioButton("Square",            &obstacleShape, 1);
    ImGui::RadioButton("Circle",            &obstacleShape, 2);
    ImGui::RadioButton("Triangle",          &obstacleShape, 3);
    ImGui::RadioButton("Vertical Barrier",  &obstacleShape, 4);
    ImGui::RadioButton("Diagonal Barrier",  &obstacleShape, 5);

    ImGui::SliderFloat("Obstacle Size", &obstacleSize, 0.01f, 0.3f);
    ImGui::Text("Obstacle Position: (%.2f, %.2f)", obstacleCenterX, obstacleCenterY);

    ImGui::End();
}

int main() {

    // ----------------------------------------------------------------------------
    // Simulation Parameters
    // ----------------------------------------------------------------------------
    float dt = 0.0002f;                   // Initial time step
    const float viscosity = 0.002f;      // Fluid viscosity coefficient

    // ----------------------------------------------------------------------------
    // Simulation Fields (GPU arrays)
    // ----------------------------------------------------------------------------
    float* velocityU        = nullptr;   // Horizontal velocity component (U)
    float* velocityV        = nullptr;   // Vertical velocity component (V)
    float* velocityU_New    = nullptr;   // Updated horizontal velocity
    float* velocityV_New    = nullptr;   // Updated vertical velocity
    float* pressure         = nullptr;   // Pressure field
    float* divergenceSource = nullptr;   // Divergence source term for pressure solve
    float* vorticity        = nullptr;   // Vorticity field

    // Allocate GPU memory for simulation fields
    cudaMalloc((void**)&velocityU,        sizeof(float) * NX * NY);
    cudaMalloc((void**)&velocityV,        sizeof(float) * NX * NY);
    cudaMalloc((void**)&velocityU_New,    sizeof(float) * NX * NY);
    cudaMalloc((void**)&velocityV_New,    sizeof(float) * NX * NY);
    cudaMalloc((void**)&pressure,         sizeof(float) * NX * NY);
    cudaMalloc((void**)&divergenceSource, sizeof(float) * NX * NY);
    cudaMalloc((void**)&vorticity,        sizeof(float) * NX * NY);

    // ----------------------------------------------------------------------------
    // Buffers for velocity min/max reduction on GPU
    // ----------------------------------------------------------------------------
    float *d_minVelocity, *d_maxVelocity;
    cudaMalloc(&d_minVelocity, sizeof(float));
    cudaMalloc(&d_maxVelocity, sizeof(float));

    // ----------------------------------------------------------------------------
    // Color mapping constants for visualization
    // ----------------------------------------------------------------------------
    const size_t matrix_size = NX * NY * sizeof(float4); // Color buffer size (RGBA)

    constexpr float PRESSURE_MIN = 0.0f;    // Minimum pressure color scale
    constexpr float PRESSURE_MAX = 1.0f;    // Maximum pressure color scale

    constexpr float VORTICITY_MIN = -50.0f; // Vorticity color scale range
    constexpr float VORTICITY_MAX = 50.0f;

    // Smoothed min/max velocity for color normalization (exponential smoothing)
    float smoothedMinVelocity = 0.0f;
    float smoothedMaxVelocity = 1.0f;
    const float alpha = 0.1f;  // Smoothing factor (0.1 = 10% new value)

    // ----------------------------------------------------------------------------
    // Mimir viewer configuration
    // ----------------------------------------------------------------------------
    mimir::ViewerOptions opts;
    opts.window.title = "Mimir: Fluid Simulation";
    opts.window.size = {WIDTH, HEIGHT};
    opts.present.enable_fps_limit = false;
    opts.present.mode = mimir::PresentMode::Immediate;
    opts.present.enable_sync = true;
    opts.present.target_fps = 60; // Ignored if fps limit disabled
    opts.show_panel = true;
    opts.show_metrics = false;
    opts.show_demo_window = false;
    opts.background_color = {0.5f, 0.5f, 0.5f, 1.0f};

    // Create Mimir instance
    InstanceHandle instance = nullptr;
    createInstance(opts, &instance);
    if (!instance) {
        std::cerr << "Failed to create Mimir instance\n";
        return -1;
    }

    // ----------------------------------------------------------------------------
    // Allocate device buffers for visualization colors
    // ----------------------------------------------------------------------------
    float4* d_velocityColors = nullptr;
    float4* d_pressureColors = nullptr;
    float4* d_vorticityColors = nullptr;
    AllocHandle allocVelocityColors, allocPressureColors, allocVorticityColors;

    allocLinear(instance, (void**)&d_velocityColors, matrix_size, &allocVelocityColors);
    allocLinear(instance, (void**)&d_pressureColors, matrix_size, &allocPressureColors);
    allocLinear(instance, (void**)&d_vorticityColors, matrix_size, &allocVorticityColors);

    // ----------------------------------------------------------------------------
    // Define view description for Mimir visualization (structured 2D grid)
    // ----------------------------------------------------------------------------
    ViewDescription viewDesc = {/* common properties */};
    viewDesc.layout = Layout::make(NX, NY);
    viewDesc.type = ViewType::Voxels;
    viewDesc.domain = DomainType::Domain2D;
    viewDesc.attributes[AttributeType::Position] = makeStructuredGrid(instance, viewDesc.layout);
    const float scale = 2.0f / std::max(NX, NY);
    viewDesc.scale = {scale, scale, scale};
    viewDesc.default_size = NX / 2;

    // Create velocity view
    viewDesc.attributes[AttributeType::Color] = {
        .source = allocVelocityColors,
        .size = NX * NY,
        .format = FormatDescription::make<float4>()
    };
    float3 position = {-0.75f, -0.75f, 0.f};
    viewDesc.position = position;
    ViewHandle viewVel; createView(instance, &viewDesc, &viewVel);

    // Create pressure view (shares layout, changes color source)
    viewDesc.attributes[AttributeType::Color].source = allocPressureColors;
    viewDesc.position = position;
    ViewHandle viewPres; createView(instance, &viewDesc, &viewPres);

    // Create vorticity view
    viewDesc.attributes[AttributeType::Color].source = allocVorticityColors;
    viewDesc.position = position;
    ViewHandle viewVort; createView(instance, &viewDesc, &viewVort);

    // ----------------------------------------------------------------------------
    // Set initial camera position
    // ----------------------------------------------------------------------------
    setCameraPosition(instance, {0.f, 0.f, -2.f});

    // ----------------------------------------------------------------------------
    // CUDA kernel execution configuration
    // ----------------------------------------------------------------------------
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks(
        (NX + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (NY + BLOCK_SIZE - 1) / BLOCK_SIZE
    );

    // Allocate buffer for max velocity per block (for CFL condition)
    int totalBlocks = numBlocks.x * numBlocks.y;
    float* d_blockMax = nullptr;
    cudaMalloc(&d_blockMax, totalBlocks * sizeof(float));

    float fixedDt = dt; // Fixed fallback timestep

    // ----------------------------------------------------------------------------
    // Main simulation update lambda
    // ----------------------------------------------------------------------------
    auto update_func = [&]() {
        // Compute max velocity per block (CFL condition)
        size_t sharedMemSize = BLOCK_SIZE * BLOCK_SIZE * sizeof(float);
        computeMaxVelocityBlock<<<numBlocks, threadsPerBlock, sharedMemSize>>>(velocityU, velocityV, d_blockMax);
        cudaDeviceSynchronize();

        // Reduce max velocity globally
        float maxVelocity = reduceMax(d_blockMax, totalBlocks);

        // Dynamically adjust dt based on max velocity (CFL)
        if (maxVelocity > 1e-6f)
            dt = 0.5f * DX / maxVelocity;
        else
            dt = fixedDt;

        // Update velocities using Navier-Stokes kernel
        updateVelocities<<<numBlocks, threadsPerBlock>>>(velocityU, velocityV, velocityU_New, velocityV_New, dt, viscosity);
        cudaDeviceSynchronize();

        // Build divergence source for pressure solve
        buildUpBDensityKernel<<<numBlocks, threadsPerBlock>>>(velocityU_New, velocityV_New, divergenceSource, dt, 1.0f);
        cudaDeviceSynchronize();

        // Solve pressure Poisson equation iteratively
        for(int i = 0; i < 50; ++i) {
            solvePressurePoisson<<<numBlocks, threadsPerBlock>>>(pressure, divergenceSource);
            cudaDeviceSynchronize();
        }

        // Correct velocities based on pressure gradient
        correctVelocities<<<numBlocks, threadsPerBlock>>>(velocityU_New, velocityV_New, pressure, dt);
        cudaDeviceSynchronize();

        // Swap velocity buffers for next iteration
        std::swap(velocityU, velocityU_New);
        std::swap(velocityV, velocityV_New);

        // Compute vorticity field
        computeVorticity<<<numBlocks, threadsPerBlock>>>(velocityU, velocityV, vorticity);
        cudaDeviceSynchronize();

        // Apply emitter barrier if enabled
        if (showBarrier) {
            applyBarrier<<<numBlocks, threadsPerBlock>>>(
                velocityU, velocityV, pressure,
                barrierCenterX * NX, barrierCenterY * NY,
                barrierWidth * NX, barrierHeight * NY,
                barrierVelocityU, barrierVelocityV
            );
            cudaDeviceSynchronize();
        }

        // Apply obstacle boundary conditions
        applyBoundaryConditions<<<numBlocks, threadsPerBlock>>>(
            velocityU, velocityV, pressure, divergenceSource, vorticity,
            obstacleShape, obstacleCenterX, obstacleCenterY, obstacleSize
        );
        cudaDeviceSynchronize();

        // ----------------------------------------------------------------------------
        // Visualization update (color mapping)
        // ----------------------------------------------------------------------------
        int threads = BLOCK_SIZE;
        int blocks = (NX * NY + threads - 1) / threads;

        // Velocity color update with smoothed min/max values
        cudaMemset(d_minVelocity, 0x7F, sizeof(float)); // Initialize to FLT_MAX
        cudaMemset(d_maxVelocity, 0x80, sizeof(float)); // Initialize to -FLT_MAX
        reduceMinMax<<<blocks, threads>>>(velocityU, d_minVelocity, d_maxVelocity, NX * NY);
        cudaDeviceSynchronize();

        float minVel, maxVel;
        cudaMemcpy(&minVel, d_minVelocity, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(&maxVel, d_maxVelocity, sizeof(float), cudaMemcpyDeviceToHost);

        // Exponential smoothing of min/max velocity for stable coloring
        smoothedMinVelocity = alpha * minVel + (1.0f - alpha) * smoothedMinVelocity;
        smoothedMaxVelocity = alpha * maxVel + (1.0f - alpha) * smoothedMaxVelocity;

        // Map velocity scalar field to colors
        fieldToColorKernel<<<blocks, threads>>>(velocityU, d_velocityColors, NX, NY, smoothedMinVelocity, smoothedMaxVelocity);

        // Map pressure to fixed color range
        fieldToColorKernel<<<blocks, threads>>>(pressure, d_pressureColors, NX, NY, PRESSURE_MIN, PRESSURE_MAX);

        // Map vorticity to fixed color range
        fieldToColorKernel<<<blocks, threads>>>(vorticity, d_vorticityColors, NX, NY, VORTICITY_MIN, VORTICITY_MAX);

        // Update views in Mimir
        updateViews(instance);
    };

    // Register the GUI callback
    setGuiCallback(instance, guiCallback_3);

    // Start the rendering and simulation loop
    display(instance, update_func,  0xFFFFFFFF);

    // Cleanup resources on exit
    destroyInstance(instance);
    cudaFree(velocityU);
    cudaFree(velocityV);
    cudaFree(velocityU_New);
    cudaFree(velocityV_New);
    cudaFree(pressure);
    cudaFree(divergenceSource);
    cudaFree(vorticity);
    cudaFree(d_blockMax);
    cudaFree(d_minVelocity);
    cudaFree(d_maxVelocity);
    cudaFree(d_velocityColors);
    cudaFree(d_pressureColors);
    cudaFree(d_vorticityColors);

    return 0;
}
