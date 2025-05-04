#pragma once



#include "VecGrapper.h"
#include <GL/glut.h>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>


// Shortcuts for nv::vec types
// typedef nv::vec2<float> vec2f;
// typedef Vec2Wrapper vec2f;
// typedef nv::vec3<float> vec3f;
// typedef nv::vec4<float> vec4f;

// typedef nv::vec2<int> vec2i;
// typedef nv::vec3<int> vec3i;
// typedef nv::vec4<int> vec4i;

struct GameLife{

    vec2 position;
    int radius;
    float glowExtent;
    bool alive;
    bool next;
    int aliveNeighbors;
    uchar4 color;
};

enum ToolMode {
    DISTURBE = 0,
    DRAG = 1,
    TEAR = 2,
    ACTIVE = 3
};

class CUDAHandler {

    public:
    
        static CUDAHandler* instance;

        CUDAHandler(int width, int height, GLuint textureID);
        ~CUDAHandler();

        // main functions
        void updateDraw(float dt);
        void clearGraphicsDisply(cudaSurfaceObject_t &surface, uchar4 color);

        // program variables
        float dt;  // delta time
        int height, width;
        float screenRatio;
        vec2 center;
        int framesCount{};
        int padding = 50;



        // mouse operations
        float zoom = 1.0f;
        float panX = -width / 2.0f, panY = -height / 2.0f;
        int lastMouseX, lastMouseY;
        vec2 lastMousePos = vec2(0.0f, 0.0f);
        bool isDragging = false;
        bool leftMouseDown = false;
        bool isPanEnabled = false;
        float mouseCursorRadius = 50.0f;
        ToolMode toolMode = DISTURBE;


        // Draw shapes
        void drawGroupOfGlowingCircles(cudaSurfaceObject_t &surface, GameLife* &d_gameLife);
        void drawGlowingCircle(cudaSurfaceObject_t &surface, vec2 position, float radius, float glowExtent, uchar4 color);
        void drawRing(cudaSurfaceObject_t &surface, vec2 position, float radius, float thickness, uchar4 color);
        
        // Game of Life
        vec2 topLeft;
        bool startSimulation = false;
        int option = 6;
        float glowExtent = 1.0f;
        
        GameLife* d_gameLife; // for GPU operations
        int gridRows, gridCols;
        std::vector<GameLife> gamelife;
        int numberOfParticles = 1000000;
        float particleRadius = .5f;
        float restLength = 1.0f;
        void activateGameLife();
        void activateGameLife(GameLife* &d_gameLife);
        void initGameLife();
        void setGroupOfParticles(int totalParticles, int2 ratio, bool anchors = 0);
        int2 calculateGrid(int n, int a, int b);
        void drawGameLife(cudaSurfaceObject_t &surface, GameLife* &d_gameLife);
        void changeGlow(cudaSurfaceObject_t &surface, GameLife* &d_gameLife);
        void disturbeGameLife(vec2 mousePosition);

        // options 0 Grid 
        int gridSize = 20;
        // Options 1 - 3  : Vertical, horizontal, checkered
        float widthFactor = 0.1f;
        // Option 7 & 10  : concentric Rings    
        float ringSpacing = 5.0f;
        float thickness = 4.5f;
        // Option 7 : Spiral
        float spacing = 60.0f;
        // Option 13 : diagonal grid
        int blockSize = 10;
        int band = 1;
        // Option 14: full grid
        int border = 3;
        int diagonalBand = 1;


  


      
    
    private:
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResouse();

};