#pragma once



#include <GL/glut.h>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>


class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();

        // main functions
        void updateDraw(float dt);

        // program variables
        float dt;  // delta time
        int height, width;


      
    
    private:
        // GL resources
        cudaGraphicsResource_t cudaResource;

};