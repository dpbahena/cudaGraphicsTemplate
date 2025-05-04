#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <random>

const uchar4 BLUE_PLANET    = make_uchar4(93, 176, 199,255);
const uchar4 GRAY_ROCKY     = make_uchar4(183, 184, 185,255);
const uchar4 SUN_YELLOW     = make_uchar4(253, 184, 19, 255);
const uchar4 JUPITER        = make_uchar4(188, 175, 178, 255);
const uchar4 SPACE_NIGH     = make_uchar4(3, 0, 53, 255);
const uchar4 FULL_MOON      = make_uchar4(245, 238, 188, 255);
const uchar4 RED_MERCURY    = make_uchar4(120, 6, 6, 255);
const uchar4 VENUS_TAN      = make_uchar4(248, 226, 176, 255);
const uchar4 MARS_RED       = make_uchar4(193, 68, 14, 255);
const uchar4 SATURN_ROSE    = make_uchar4(206, 184, 184, 255);
const uchar4 NEPTUNE_PURPLE = make_uchar4(91, 93, 223, 255);
const uchar4 URANUS_BLUE    = make_uchar4(46, 132, 206, 255);
const uchar4 PLUTO_TAN      = make_uchar4(255, 241, 213, 255);
const uchar4 LITE_GREY      = make_uchar4(211, 211, 211, 255);
const uchar4 CHARCOAL_GREY  = make_uchar4(54, 69, 79, 255);
const uchar4 DARK  = make_uchar4(0, 0, 0, 255);
const uchar4 GREEN  = make_uchar4(127, 255, 0, 255);
const uchar4 GOLD   = make_uchar4(239, 191, 4, 255);
const uchar4 WHITE  = make_uchar4(255, 255, 255, 255);
const uchar4 PINK   = make_uchar4(255, 192, 203, 255);
const uchar4 ORANGE   = make_uchar4(255, 165, 0, 255);
const uchar4 TAN   = make_uchar4(210, 180, 140, 255);


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



// __device__ __host__
inline bool randomBool() {
    std::random_device rd;
    std::mt19937 gen(rd()); // Mersenne Twister RNG
    std::bernoulli_distribution dist(0.5); // 50% chance for true or false

    return dist(gen);
}

// Random int in [min, max] (inclusive)
inline int random_int(int min, int max) {
    static thread_local std::mt19937 generator(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> distribution(min, max);
    return distribution(generator);
}

template <typename T>
bool justChanged(const T& currentValue) {
    static T previousValue = currentValue;
    bool changed = (currentValue != previousValue);
    previousValue = currentValue;
    return changed;
}
