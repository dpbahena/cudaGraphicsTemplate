#pragma once



#include "nvVector.h"
#include <GL/glut.h>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>


// Shortcuts for nv::vec types
typedef nv::vec2<float> vec2f;
typedef nv::vec3<float> vec3f;
typedef nv::vec4<float> vec4f;

typedef nv::vec2<int> vec2i;
typedef nv::vec3<int> vec3i;
typedef nv::vec4<int> vec4i;


class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();

        // main functions
        void updateDraw(float dt);
        void clearGraphicsDisply(cudaSurfaceObject_t &surface);

        // program variables
        float dt;  // delta time
        int height, width;
        vec2f center;



        // zoom & pan variables
        float zoom = 1.0f;
        float panX = -width / 2.0f, panY = -height / 2.0f;
        int lastMouseX, lastMouseY;
        vec2f lastMousePos = vec2f(0.0f);
        bool isDragging = false;
        bool leftMouseDown = false;
        bool isPanEnabled = false;


        // Draw shapes
        void drawGlowingCircle(cudaSurfaceObject_t &surface, vec2f position, float radius, float glowExtend, uchar4 color);


        


      
    
    private:
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResouse();

};