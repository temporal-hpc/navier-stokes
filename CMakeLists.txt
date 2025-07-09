cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(fluid_simulation LANGUAGES CXX CUDA)

# C++17 y CUDA moderno
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA necesita estas flags para trabajar con std::variant y std::clamp
add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

include(FetchContent)

# Parámetros externos que pasas en cmake -D
set(MIMIR_SHADER_DIR "" CACHE PATH "Path to Mimir shaders directory")

# Requiere que Mimir haya sido instalado previamente con `cmake --install`
find_package(spdlog REQUIRED)
find_package(mimir REQUIRED PATHS ${CMAKE_PREFIX_PATH})

find_library(SLANG_LIB slang REQUIRED)
if(NOT SLANG_LIB)
  message(FATAL_ERROR "Slang library not found. Instálala antes de continuar.")
endif()

# Descargar ImGui automáticamente si no está presente
FetchContent_Declare(
  imgui
  GIT_REPOSITORY https://github.com/ocornut/imgui.git
  GIT_TAG v1.89.8
)
FetchContent_MakeAvailable(imgui)

# Archivos fuente
set(SOURCES
    fluid_simulation.cu
    kernels.cu
)

# Cabeceras del proyecto
set(HEADERS
    config.h
    kernels.cuh
)

add_executable(fluid_simulation ${SOURCES} ${HEADERS})

# Incluir cabeceras locales y las de ImGui
target_include_directories(fluid_simulation PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${imgui_SOURCE_DIR}
)

# Enlace con Mimir y sus dependencias transitivas (imgui, etc.)
target_link_libraries(fluid_simulation PRIVATE
    mimir
    spdlog::spdlog
    imgui
)

# Pasar la ruta de shaders a una definición para el preprocesador (útil si la usas en código)
target_compile_definitions(fluid_simulation PRIVATE
    MIMIR_SHADER_DIR="${MIMIR_SHADER_DIR}"
)

# Agregar la carpeta shaders al output (para copiar o linkear en build)
if(MIMIR_SHADER_DIR)
    # Por ejemplo, crear un enlace simbólico en build para shaders
    add_custom_command(TARGET fluid_simulation POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
        "${MIMIR_SHADER_DIR}"
        "${CMAKE_CURRENT_BINARY_DIR}/shaders"
      COMMENT "Creating symbolic link to shaders directory"
    )
endif()
