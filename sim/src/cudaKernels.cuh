#pragma once

// #include "CUDAHandler.h"
// #include <cuda_runtime.h>
// #include <cstdio>
// #include <iostream>
// #include <chrono>
#include <cuda_gl_interop.h>
// #include <GL/glut.h>




__device__ void drawPixel(cudaSurfaceObject_t surface, int x, int y, uchar4 color, int width, int height);
__device__ void drawLine(cudaSurfaceObject_t surface, int x0, int y0, int x1, int y1,  uchar4 color,  int width, int height);
__device__ void drawRing(cudaSurfaceObject_t surface, float cx, float cy, float radius, float thickness, uchar4 color, int width, int height);
__device__ void drawCircleOutline(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height);
__device__ void drawFilledCircle(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height);
__device__ uchar4 blend(uchar4 dest, uchar4 src);
__device__ void drawBlendedFilledCircle(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height);


// Kernels
__global__ void clearSurface_kernel(cudaSurfaceObject_t surface, int width, int height, uchar4 color);
__global__ void drawCircle_kernel(cudaSurfaceObject_t surface, int width, int height, int cx, int cy, int radius, uchar4 color, bool outline, int thickness, float zoom, float panX, float panY );
__global__ void drawGlowingCircle(cudaSurfaceObject_t surface, int width, int height, int cx, int cy, int radius, uchar4 color, float glowExtent, float zoom, float panX, float panY);







