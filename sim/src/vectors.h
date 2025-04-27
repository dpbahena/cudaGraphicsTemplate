#pragma once
#include <math.h>
#include <stdio.h>
#include <cuda_runtime.h>




class Vec2 {
    public:
        float x, y;
        __host__ __device__
        Vec2() {}
        __host__ __device__
        Vec2(float v) : x(v), y(v) { };
        __host__ __device__
        Vec2 (float x, float y) : x(x), y(y){}
        
        __host__ __device__
        Vec2& operator=(const Vec2& v)  {
            x = v.x;
            y = v.y;
            return *this;
        }
        __host__ __device__
        bool operator==(const Vec2& v) const {
            return  (x == v.x && y == v.y);
        }

        // bool operator!=(const Vec2& v) const {
        //     return (x != v.x || y != v.y);
        // }
        __host__ __device__
        bool operator!=(const Vec2& v) const {
            return !(*this == v);  // Reuse operator==
        }
        __host__ __device__
        Vec2 operator+(const Vec2& other) const {
            return Vec2(other.x + x, other.y + y);
        }
        __host__ __device__
        Vec2 operator-(const Vec2& other) const {
            return Vec2(x - other.x, y - other.y);
        }
        __host__ __device__
        Vec2 operator*(const float n) const {
            return Vec2(x * n, y * n);
        }
        __host__ __device__
        Vec2 operator/(const float n) {
            return Vec2(x / n, y / n);
        }

        Vec2 operator-() const {   // negation
            return Vec2(-x, -y);
        };
        __host__ __device__
        Vec2& operator+=(const Vec2& other) {
            x += other.x;
            y += other.y;
            return *this;
        }
        __host__ __device__
        Vec2& operator-=(const Vec2& other) {
            x -= other.x;
            y -= other.y;
            return *this;
        }     
        __host__ __device__
        Vec2& operator*=(float n) {
            x*=n;
            y*=n;
            return *this;
        }
        __host__ __device__
        Vec2& operator/=(float n) {
            x/=n;
            y/=n;
            return *this;
        }
        
       
        __host__ __device__
        void scale(float s) {
            x *= s;
            y *= s;
            
        }
        
        // Vec2 rotate(const float a) const {
        //     float angle = toRadians(a);
        //     Vec2 vec{0,0};
        //     vec.x = x * cos(angle) - y * sin(angle);
        //     vec.y = x * sin(angle) + y * cos(angle);
        //     return vec;
        // }
        __device__ __host__
        float mag() const {
            return sqrt(x*x + y*y);
        }

        float magSquared() const {
            return (x*x + y*y);
        }
        __host__ __device__
        Vec2& normalize() {
            float length = this->mag();
            if (length != 0.0f) {
                x /= length;
                y /= length;
            }
            return *this;
        }

        Vec2 unitVector() const {
            float len = this->mag();
            if (len != 0.0f)
                return Vec2(x / len, y / len);
            else
                return Vec2(0.0,0.0);
        }

        Vec2 normal() const {
            return Vec2(y, -x).normalize();
        }
        __host__ __device__
        float dot(const Vec2& v) const {
            return x * v.x + y * v.y;
        }

        float cross(Vec2& v) const {
            return  (x * v.y - v.x * y);  // z-axis component only (determinant) - magnitud of the Z component
        }

        



        static Vec2 add(Vec2 v1, Vec2 v2) {
            return Vec2(v1.x + v2.x, v1.y + v2.y);
        }
        
        // void log(std::string name){
        //     printf("%s: {%f, %f}\n", name.c_str(), x, y);
        // }

        bool hasNaN() const {
            return std::isnan(x) || std::isnan(y);
        }
};
__host__ __device__
inline Vec2 operator*(const float n, const Vec2 &v) {
    return Vec2(v.x * n, v.y * n);
}



class Vec3 {
    public:
        float x, y, z;
        Vec3(){};
        Vec3 (float x, float y, float z) : x(x), y(y), z(z){}
        void add(Vec3 v) {
            x += v.x;
            y += v.y;
            z += v.z;
        }
        void subtract(Vec3 v) {
            x -= v.x;
            y -= v.y;
            z -= v.z;
          
        }

        float mag() {
            return sqrt(x*x + y*y + z*z);
        }

        void scale(float s) {
            x *= s;
            y *= s;
            z *= s;
            
        }

        static Vec3 add(Vec3 v1, Vec3 v2) {
            return Vec3(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
        }

        float dot(Vec3 v){
            return x * v.x + y * v.y, z * v.z;
        }

        Vec3 cross(Vec3& v){
            return Vec3(
                y * v.z - v.y * z,
                x * v.z - v.x * z,
                x * v.y - v.x * y
            );

        }

        Vec3 normalize() {
            float length = mag();
            return Vec3(x/length, y/length, z/length);
        }

};