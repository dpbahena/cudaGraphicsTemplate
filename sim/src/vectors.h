#pragma once
#include <cmath>
#include <cuda_runtime.h>

struct vec2 {
    float x, y;

    __host__ __device__ vec2() : x(0), y(0) {}
    __host__ __device__ vec2(float n) : x(n), y(n) {}
    __host__ __device__ vec2(float x, float y) : x(x), y(y) {}

    __host__ __device__ float magSq() const { return x * x + y * y; }
    __host__ __device__ float mag() const { return sqrtf(magSq()); }

    __host__ __device__ vec2 normalized() const {
        float m = mag();
        return (m > 0.0f) ? vec2(x / m, y / m) : vec2(0.0f, 0.0f);
    }

    __host__ __device__ float dot(const vec2& other) const {
        return x * other.x + y * other.y;
    }

    __host__ __device__ float angleBetween(const vec2& other) const {
        float d = dot(other);
        float m = mag() * other.mag();
        if (m == 0.0f) return 0.0f;
        float cosTheta = fmaxf(fminf(d / m, 1.0f), -1.0f);
        return acosf(cosTheta);
    }

    __host__ __device__ vec2 perpendicular() const {
        return vec2(-y, x);  // 90 degrees CCW
    }

    __host__ __device__ vec2 operator+(const vec2& b) const { return vec2(x + b.x, y + b.y); }
    __host__ __device__ vec2 operator-(const vec2& b) const { return vec2(x - b.x, y - b.y); }
    __host__ __device__ vec2 operator*(float s) const { return vec2(x * s, y * s); }
    __host__ __device__ vec2 operator/(float s) const { return vec2(x / s, y / s); }
    __host__ __device__ vec2 operator+=(const vec2& other) { x += other.x; y += other.x; return *this; }
    __host__ __device__ vec2 operator-=(const vec2& other) { x -= other.x; y -= other.x; return *this; }
    __host__ __device__ vec2 operator*=(const float n) { x *= n; y *= n; return *this; }
    __host__ __device__ vec2 operator/=(const float n) { x /= n; y /= n; return *this; }

};

// ---- vec3 version ----
struct vec3 {
    float x, y, z;

    __host__ __device__ vec3() : x(0), y(0), z(0) {}
    __host__ __device__ vec3(float x, float y, float z) : x(x), y(y), z(z) {}

    __host__ __device__ float magSq() const { return x * x + y * y + z * z; }
    __host__ __device__ float mag() const { return sqrtf(magSq()); }

    __host__ __device__ vec3 normalized() const {
        float m = mag();
        return (m > 0.0f) ? vec3(x / m, y / m, z / m) : vec3(0.0f, 0.0f, 0.0f);
    }

    __host__ __device__ float dot(const vec3& b) const {
        return x * b.x + y * b.y + z * b.z;
    }

    __host__ __device__ vec3 cross(const vec3& b) const {
        return vec3(
            y * b.z - z * b.y,
            z * b.x - x * b.z,
            x * b.y - y * b.x
        );
    }

    __host__ __device__ float angleBetween(const vec3& b) const {
        float d = dot(b);
        float m = mag() * b.mag();
        if (m == 0.0f) return 0.0f;
        float cosTheta = fmaxf(fminf(d / m, 1.0f), -1.0f);
        return acosf(cosTheta);
    }

    __host__ __device__ vec3 operator+(const vec3& b) const { return vec3(x + b.x, y + b.y, z + b.z); }
    __host__ __device__ vec3 operator-(const vec3& b) const { return vec3(x - b.x, y - b.y, z - b.z); }
    __host__ __device__ vec3 operator*(float s) const { return vec3(x * s, y * s, z * s); }
    __host__ __device__ vec3 operator/(float s) const { return vec3(x / s, y / s, z / s); }
};
