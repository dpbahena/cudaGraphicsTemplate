#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>


#define checkCuda(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *expr, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,
                "CUDA Error: %s (error code %d)\n"
                "Expression: %s\n"
                "File: %s\n"
                "Line: %d\n",
                cudaGetErrorString(code), code, expr, file, line);
        if (abort) exit(code);
    }
}

const uchar4 BLUE_PLANET = make_uchar4(93, 176, 199,255);
const uchar4 GRAY_ROCKY = make_uchar4(183, 184, 185,255);
const uchar4 SUN_YELLOW = make_uchar4(253, 184, 19, 255);
// const uchar4 JUPITER  = uchar4(188, 175, 178);
// const uchar4 SPACE_NIGHT  = uchar4(3, 0, 53);
// const uchar4 FULL_MOON  = uchar4(245, 238, 188);
// const uchar4 RED_MERCURY = uchar4(120, 6, 6);
// const uchar4 VENUS_TAN = uchar4(248, 226, 176);
// const uchar4 MARS_RED = uchar4(193, 68, 14);
// const uchar4 SATURN_ROSE = uchar4(206, 184, 184);
// const uchar4 NEPTUNE_PURPLE = uchar4(91, 93, 223);
// const uchar4 URANUS_BLUE = uchar4(46, 132, 206);
// const uchar4 PLUTO_TAN = uchar4(255, 241, 213);
// const uchar4 LITE_GREY = uchar4(211, 211, 211);
// const uchar4 CHARCOAL_GREY = uchar4(54, 69, 79);