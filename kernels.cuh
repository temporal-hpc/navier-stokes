#pragma once
#include "config.h"

__global__ void applyBarrier(float *u, float *v, float *p, float cx, float cy, float ancho, float alto, float velocidadX, float velocidadY);
__device__ void applyBoundaryConditionsEdges(float *u, float *v, float *p, int i, int j);
__device__ void applyBoundaryConditionsWithInflow(float* u, float* v, float* p, int i, int j);
__device__ void applyObstacle(float* u, float* v, float* p, float* w, int shapeType, int i, int j, float cx_rel, float cy_rel, float size);
__global__ void applyBoundaryConditions(float* u, float* v, float* pressure, float* divergence, float* vorticity,
    int shapeType, float centerXRel, float centerYRel, float size);
__global__ void computeVorticity(const float* u, const float* v, float* vorticity);
__global__ void buildUpBDensityKernel(const float *u, const float *v, float *b, float dt, float rho);
__global__ void solvePressurePoisson(float* pressure, const float* sourceTerm);
__global__ void correctVelocities(float* u, float* v, const float* pressure, float timeStep);
__global__ void updateVelocities(const float* u, const float* v, float* u_next, float* v_next, float timeStep, float viscosity); 
__global__ void computeMaxVelocityBlock(const float* u, const float* v, float* blockMax);
__device__ float4 colormap_jet(float t);
__global__ void reduceMinMax(const float* input, float* min_out, float* max_out, int size);
__global__ void fieldToColorKernel(const float* input, float4* colors, int nx, int ny, float minVal, float maxVal);
