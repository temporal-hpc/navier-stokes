# GPU Fluid Simulation with Mimir

## Overview

This project implements an incompressible fluid simulation solving the Navier-Stokes equations on the GPU using CUDA. It serves as an example and testbed for **Mimir**, a library for real-time visualization of GPU data. The simulation supports a variety of obstacle shapes and leverages ImGui for interactive parameter adjustments.

---

## Features

- 2D incompressible Navier-Stokes solver with pressure projection
- Dynamic time step adjustment based on CFL condition
- Multiple obstacle shapes: none, square, circle, triangle, vertical and diagonal barriers
- Visualization via **Mimir** with buffer views updated directly from GPU memory
- Interactive GUI controls using ImGui for obstacle selection and simulation tuning

---

## Requirements

- **CUDA Toolkit 12.6** (or compatible)
- **GCC 12.3** (or newer, supporting C++20)
- **CMake 3.17** or newer
- **C++20** C++20 standard enabled for host-side code
- **Mimir** library
- **Slang**  shader language
---

## Build Instructions

1. Create a build directory and navigate into it:
   ```bash
   mkdir build && cd build
   ```

2. Configure the project specifying:
- `CMAKE_PREFIX_PATH`: the root directory where you installed the Mimir library (contiene sus headers y libs)
- `MIMIR_SHADER_DIR`: the directory containing Mimir shaders 
   ```bash
   cmake .. \
   -DCMAKE_PREFIX_PATH=/path/to/mimir_install \
   -DMIMIR_SHADER_DIR=/path/to/mimir/shaders
   ```

3. Build the project using the following command:
   ```bash
   make -j$(nproc)
   ```

4.  Once the build completes successfully, run the simulation from   the executables directory:
      ```cmake
      ./fluid_simulation
      ```

---

## Usage

- Use the GUI to:
  - Press the key `V` to toggle the visibility of the emitter barrier
  - Press the key `F` to toggle whether the obstacle follows the mouse cursor
  - Select obstacle shape (none, square, circle, triangle, vertical barrier, diagonal barrier)

## Code Structure

- `fluid_simulation.cu`: Main simulation logic and visualization setup  
- `kernels.cu` / `kernels.cuh`: CUDA kernels implementing the fluid dynamics
