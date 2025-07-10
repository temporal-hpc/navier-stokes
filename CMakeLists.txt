cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(fluid_simulation LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

include(FetchContent)

# Parámetros externos en cmake -D
set(MIMIR_SHADER_DIR "" CACHE PATH "Path to Mimir shaders directory")

# Requiere que Mimir haya sido instalado previamente con `cmake --install`
find_package(spdlog REQUIRED)
find_package(mimir REQUIRED PATHS ${CMAKE_PREFIX_PATH})

find_library(SLANG_LIB slang REQUIRED)
if(NOT SLANG_LIB)
  message(FATAL_ERROR "Slang library not found. Instálala antes de continuar.")
endif()

FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.89.8
)
FetchContent_MakeAvailable(imgui)


set(SOURCES
    fluid_simulation.cu
    kernels.cu
)

set(HEADERS
    config.h
    kernels.cuh
)

add_executable(fluid_simulation ${SOURCES} ${HEADERS})

target_include_directories(fluid_simulation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
)

target_link_libraries(fluid_simulation PRIVATE
    mimir
    spdlog::spdlog
    imgui
)

target_compile_definitions(fluid_simulation PRIVATE
    MIMIR_SHADER_DIR="${MIMIR_SHADER_DIR}"
)


if(MIMIR_SHADER_DIR)
    add_custom_command(TARGET fluid_simulation POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${MIMIR_SHADER_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
      COMMENT "Creating symbolic link to shaders directory"
    )
endif()
