

cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0104 NEW)

# Define a variable for the project name
set(PROJECT_NAME sim)

# Use the variable for the project and executable names
project(${PROJECT_NAME} VERSION 0.1.0 LANGUAGES C CXX CUDA)

# Set the CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Add source files
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/imgui)
set(IMGUI_BACKENDS ${IMGUI_DIR}/backends)

set(IMGUI_SOURCES
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_BACKENDS}/imgui_impl_glut.cpp
    ${IMGUI_BACKENDS}/imgui_impl_opengl2.cpp
)

include_directories(
    ${IMGUI_DIR}
    ${IMGUI_BACKENDS}
)

# Find OpenGL and GLUT packages
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

# Find the CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Find SDL2 package
find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})


# Collect all source files in the src directory
file(GLOB_RECURSE SOURCES "src/*.cpp" "src/*.cu")
file(GLOB_RECURSE HEADERS "src/*.h"  )

# Add the main executable and specify all source files
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS} ${IMGUI_SOURCES})

#Link SDL2 library
target_link_libraries(${PROJECT_NAME} ${SDL2_LIBRARIES})


# Link OpenGL, GLUT, and CUDA libraries
target_link_libraries(${PROJECT_NAME} OpenGL::GL OpenGL::GLU GLUT::GLUT dl ${CUDAToolkit_LIBRARIES})

# Specify properties for CUDA compilation
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/src
    # ${CMAKE_SOURCE_DIR}/src/physics
    ${CUDAToolkit_INCLUDE_DIRS}     # Modern CUDA include directory
    ${CMAKE_SOURCE_DIR}/stb
    # ${CMAKE_SOURCE_DIR}/libraries/tinyobject
)

# Suppress specific CUDA warning and enable relocatable device code
target_compile_options(${PROJECT_NAME} PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=20012>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=550>
    # $<$<COMPILE_LANGUAGE:CUDA>:-maxrregcount=60>
 
)
