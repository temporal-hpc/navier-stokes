#include "kernels.cuh"
#include <float.h>



__device__ bool isInCircle(int i, int j, float cx, float cy, float size) {
    float dx = i - cx;
    float dy = j - cy;
    return (dx*dx + dy*dy) <= size*size;
}

__device__ bool isInSquare(int i, int j, float cx, float cy, float size) {
    return (i > cx - size && i < cx + size && j > cy - size && j < cy + size);
}

__device__ bool isInTriangle(int i, int j, float cx, float cy, float size) {
    // size here controls base and height proportionally
    float base = 2*size;
    float halfHeight = size;
    if (i < cx || i > cx + base) return false;
    float widthAtI = halfHeight * (1.0f - (i - cx) / base);
    return (j >= cy - widthAtI) && (j <= cy + widthAtI);
}

__device__ bool isInVerticalBarrier(int i, int j, float cx, float cy, float length) {
    float halfLength = length / 2.0f;
    float barrierWidth = 5.0f; // ancho fijo de 2 celdas
    return (i > cx - barrierWidth && i < cx + barrierWidth) &&
           (j > cy - halfLength && j < cy + halfLength);
}

__device__ bool isInDiagonalBarrier(int i, int j, float cx, float cy, float length) {
    // Barra diagonal 45°, ancho fijo 2
    float halfLength = length / 2.0f;
    float barrierWidth = 5.0f;

    // Para aproximar la barra diagonal, se comprueba que (i,j) esté cerca
    // de la línea i - j = cx - cy (la diagonal desplazada)
    float distToDiagonal = fabsf((i - j) - (cx - cy));
    bool nearDiagonal = distToDiagonal < barrierWidth;

    // También que esté dentro del rango del largo de la barra
    bool withinLength = (i > cx - halfLength) && (i < cx + halfLength) &&
                        (j > cy - halfLength) && (j < cy + halfLength);

    return nearDiagonal && withinLength;
}

__device__ void applyObstacle(float* u, float* v, float* p, float* w, int shapeType, int i, int j, float cx_rel, float cy_rel, float size) {
    int idx = i + j * NX;
    float cx = cx_rel * NX;
    float cy = cy_rel * NY;

    bool inObstacle = false;
    size *= NX; // Convertimos el tamaño a unidades de la malla

    switch(shapeType) {
        case 1:  // Square
            inObstacle = isInSquare(i, j, cx, cy, size);
            break;
        case 2:  // Circle
            inObstacle = isInCircle(i, j, cx, cy, size);
            break;
        case 3:  // Triangle
            // En la función isInTriangle solo pasamos `size`, que usaremos para base/altura
            inObstacle = isInTriangle(i, j, cx, cy, size);
            break;
        case 4:
            inObstacle = isInVerticalBarrier(i, j, cx, cy, size);
            break;
        case 5:
            inObstacle = isInDiagonalBarrier(i, j, cx, cy, size);
            break;
    }

    if (inObstacle) {
        u[idx] = 0.0f;
        v[idx] = 0.0f;
        p[idx] = 0.0f;
        w[idx] = 0.0f;
    }
}


__global__ void applyBarrier(
    float* u, float* v, float* p,
    float cx, float cy,
    float width, float height,
    float velocityX, float velocityY
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    int idx = i + j * NX;

    // Define the bounding box of the barrier
    float left   = cx - width  / 2.0f;
    float right  = cx + width  / 2.0f;
    float top    = cy - height / 2.0f;
    float bottom = cy + height / 2.0f;

    // Check if the current cell lies within the barrier
    if (i >= left && i <= right && j >= top && j <= bottom) {
        u[idx] = velocityX;
        v[idx] = velocityY;
        p[idx] = 0.0f; // Pressure inside the barrier
    }
}

__device__ void applyBoundaryConditionsWithInflow(float* u, float* v, float* p, int i, int j) {
    int idx = i + j * NX;

    // Inflow en la frontera izquierda: velocidad fija hacia la derecha
    if (i == 0) {
        u[idx] = 3.0f;   // Velocidad X constante (puedes ajustar magnitud)
        v[idx] = 0.0f;   // Sin velocidad en Y
        p[idx] = p[idx]; // Presión puede quedar igual o ajustarse si quieres
        return;
    }

    // Outflow o condiciones periódicas en la frontera derecha
    if (i == NX - 1) {
        int idx_from_left = (NX - 2) + j * NX;
        u[idx] = u[idx_from_left];
        v[idx] = v[idx_from_left];
        p[idx] = p[idx_from_left];
    }

    // Periodic en Y (arriba <-> abajo)
    if (j == 0) {
        int idx_from_top = (NY - 2) * NX + i;
        u[idx] = u[idx_from_top];
        v[idx] = v[idx_from_top];
        p[idx] = p[idx_from_top];
    }

    if (j == NY - 1) {
        int idx_from_bottom = NX + i;
        u[idx] = u[idx_from_bottom];
        v[idx] = v[idx_from_bottom];
        p[idx] = p[idx_from_bottom];
    }
}
__device__ void applyBoundaryConditionsEdges(float* u, float* v, float* p, int i, int j) {
    int idx = i + j * NX;

    // Periodic boundary condition in X (left <-> right)
    if (i == 0) {
        int idx_from_right = (NX - 2) + j * NX;
        u[idx] = u[idx_from_right];
        v[idx] = v[idx_from_right];
        p[idx] = p[idx_from_right];
    }

    if (i == NX - 1) {
        int idx_from_left = 1 + j * NX;
        u[idx] = u[idx_from_left];
        v[idx] = v[idx_from_left];
        p[idx] = p[idx_from_left];
    }

    // Periodic boundary condition in Y (top <-> bottom)
    if (j == 0) {
        int idx_from_top = (NY - 2) * NX + i;
        u[idx] = u[idx_from_top];
        v[idx] = v[idx_from_top];
        p[idx] = p[idx_from_top];
    }

    if (j == NY - 1) {
        int idx_from_bottom = NX + i;
        u[idx] = u[idx_from_bottom];
        v[idx] = v[idx_from_bottom];
        p[idx] = p[idx_from_bottom];
    }
}


__global__ void applyBoundaryConditions(
    float* u, float* v, float* pressure, float* divergence, float* vorticity,
    int shapeType, float centerXRel, float centerYRel, float size
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= NX || j >= NY) return;

    applyBoundaryConditionsEdges(u, v, pressure, i, j);
    //applyBoundaryConditionsWithInflow(u, v, pressure, i, j);
    if(shapeType != 0) applyObstacle(u, v, pressure, vorticity, shapeType, i, j, centerXRel, centerYRel, size);
}


__global__ void computeVorticity(const float* u, const float* v, float* vorticity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip boundary cells
    if (x == 0 || x == NX - 1 || y == 0 || y == NY - 1) return;

    int idx = y * NX + x;

    // Compute partial derivative of v with respect to x
    float dv_dx = (v[y * NX + (x + 1)] - v[y * NX + (x - 1)]) / (2.0f * DX);

    // Compute partial derivative of u with respect to y
    float du_dy = (u[(y + 1) * NX + x] - u[(y - 1) * NX + x]) / (2.0f * DY);

    // Vorticity: curl of the velocity field in 2D (dv/dx - du/dy)
    vorticity[idx] = dv_dx - du_dy;
}

__global__ void buildUpBDensityKernel(const float* u, const float* v, float* b, float dt, float rho) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip boundaries
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) return;

    int idx = i + j * NX;

    // First-order derivatives
    float du_dx = (u[idx + 1] - u[idx - 1]) / (2.0f * DX);
    float dv_dy = (v[idx + NX] - v[idx - NX]) / (2.0f * DY);
    float du_dy = (u[(j + 1) * NX + i] - u[(j - 1) * NX + i]) / (2.0f * DY);
    float dv_dx = (v[j * NX + i + 1] - v[j * NX + i - 1]) / (2.0f * DX);

    // Nonlinear (squared) terms
    float du_dx_sq = du_dx * du_dx;
    float dv_dy_sq = dv_dy * dv_dy;
    float cross_term = 2.0f * du_dy * dv_dx;

    // Build-up of the right-hand side of the pressure Poisson equation
    b[idx] = rho * (1.0f / dt) * (du_dx + dv_dy) - (du_dx_sq + cross_term + dv_dy_sq);
    
}



__global__ void solvePressurePoisson(float* pressure, const float* sourceTerm) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip boundaries
    if (i == 0 || i == NX - 1 || j == 0 || j == NY - 1) return;


    int idx = j * NX + i;

    float p_right = pressure[idx + 1];
    float p_left  = pressure[idx - 1];
    float p_up    = pressure[idx + NX];
    float p_down  = pressure[idx - NX];

    float dx2 = DX * DX;
    float dy2 = DY * DY;

    pressure[idx] = ((p_right + p_left) * dy2 + (p_up + p_down) * dx2) / (2.0f * (dx2 + dy2))
                    - (dx2 * dy2) / (2.0f * (dx2 + dy2)) * sourceTerm[idx];
    
}


__global__ void correctVelocities(
    float* u, 
    float* v, 
    const float* pressure, 
    float timeStep
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Skip boundaries
    if (i <= 0 || i >= NX - 1 || j <= 0 || j >= NY - 1) return;
    

    int idx = i + j * NX;

    // Pressure gradients (central difference)
    float dp_dx = (pressure[idx + 1] - pressure[idx - 1]) / (2.0f * DX);
    float dp_dy = (pressure[idx + NX] - pressure[idx - NX]) / (2.0f * DY);

    // Velocity correction
    u[idx] -= timeStep * dp_dx;
    v[idx] -= timeStep * dp_dy;

    // Optional: calculate velocity magnitude 
    //float velocity_mag = sqrtf(u[idx] * u[idx] + v[idx] * v[idx]);
}


// Kernel to update velocity fields (u, v) using Navier-Stokes equations
__global__ void updateVelocities(
    const float* u, 
    const float* v, 
    float* u_next, 
    float* v_next, 
    float timeStep, 
    float viscosity
) {
    // Compute 2D thread indices
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check (ignore edges)
    if (i <= 0 || i >= NX - 1 || j <= 0 || j >= NY - 1) return;
    

    int idx = i + j * NX;
    int idx_right = (i + 1) + j * NX;
    int idx_left = (i - 1) + j * NX;
    int idx_up = i + (j + 1) * NX;
    int idx_down = i + (j - 1) * NX;

    // Velocity gradients (central differences)
    float du_dx = (u[idx_right] - u[idx_left]) / (2.0f * DX);
    float du_dy = (u[idx_up] - u[idx_down]) / (2.0f * DY);
    float dv_dx = (v[idx_right] - v[idx_left]) / (2.0f * DX);
    float dv_dy = (v[idx_up] - v[idx_down]) / (2.0f * DY);

    // Advection terms (u · ∇u and u · ∇v)
    float adv_u = u[idx] * du_dx + v[idx] * du_dy;
    float adv_v = u[idx] * dv_dx + v[idx] * dv_dy;

    // Diffusion terms (Laplacian)
    float laplacian_u = (u[idx_right] - 2.0f * u[idx] + u[idx_left]) / (DX * DX)
                      + (u[idx_up] - 2.0f * u[idx] + u[idx_down]) / (DY * DY);
    float laplacian_v = (v[idx_right] - 2.0f * v[idx] + v[idx_left]) / (DX * DX)
                      + (v[idx_up] - 2.0f * v[idx] + v[idx_down]) / (DY * DY);

    // Update velocities with explicit Euler time stepping
    u_next[idx] = u[idx] + timeStep * (-adv_u + viscosity * laplacian_u);
    v_next[idx] = v[idx] + timeStep * (-adv_v + viscosity * laplacian_v);
}


// CUDA kernel to compute the maximum velocity magnitude in each block
__global__ void computeMaxVelocityBlock(const float* u, const float* v, float* blockMax) {
    extern __shared__ float sdata[];

    int localIdx = threadIdx.x + threadIdx.y * blockDim.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int globalIdx = x + y * NX;

    float localMax = 0.0f;

    if (x < NX && y < NY) {
        float velocity = sqrtf(u[globalIdx] * u[globalIdx] + v[globalIdx] * v[globalIdx]);
        localMax = velocity;
    }

    sdata[localIdx] = localMax;
    __syncthreads();

    // Parallel reduction within the block
    for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1) {
        if (localIdx < s) {
            sdata[localIdx] = fmaxf(sdata[localIdx], sdata[localIdx + s]);
        }
        __syncthreads();
    }

    // Store result from the first thread of the block
    if (localIdx == 0) {
        blockMax[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
    }
}

__device__ float4 colormap_jet(float t) {
    // Clamp t entre 0 y 1
    t = fminf(fmaxf(t, 0.f), 1.f);

    float r = 0.f, g = 0.f, b = 0.f;

    if (t < 0.125f) {
        r = 0.f;
        g = 0.f;
        b = 0.5f + 4.f * t;
    } else if (t < 0.375f) {
        r = 0.f;
        g = 4.f * (t - 0.125f);
        b = 1.f;
    } else if (t < 0.625f) {
        r = 4.f * (t - 0.375f);
        g = 1.f;
        b = 1.f - 4.f * (t - 0.375f);
    } else if (t < 0.875f) {
        r = 1.f;
        g = 1.f - 4.f * (t - 0.625f);
        b = 0.f;
    } else {
        r = 1.f - 4.f * (t - 0.875f);
        g = 0.f;
        b = 0.f;
    }

    return make_float4(r, g, b, 1.f);
}

__global__ void reduceMinMax(const float* input, float* min_out, float* max_out, int size) {
    __shared__ float s_min[BLOCK_SIZE];
    __shared__ float s_max[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;

    if (i < size) {
        local_min = input[i];
        local_max = input[i];
    }

    s_min[tid] = local_min;
    s_max[tid] = local_max;

    __syncthreads();

    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_min[tid] = fminf(s_min[tid], s_min[tid + stride]);
            s_max[tid] = fmaxf(s_max[tid], s_max[tid + stride]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMin((int*)min_out, __float_as_int(s_min[0]));
        atomicMax((int*)max_out, __float_as_int(s_max[0]));
    }
}

__global__ void fieldToColorKernel(const float* input, float4* colors, int nx, int ny, float minVal, float maxVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny) return;

    float normalized = (input[idx] - minVal) / (maxVal - minVal + 1e-8f);
    colors[idx] = colormap_jet(normalized);
}
