#pragma once



#include "vectors.h"
#include <GL/glut.h>
#include "imgui.h"
#include "imgui_impl_glut.h"
#include "imgui_impl_opengl2.h"
#include "cuda_utils.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <vector>
#include <string>
#include <tuple> // For std::tie


struct GameLife{

    vec2 position;
    int radius;
    bool alive;
    bool next;
    float nextEnergy=0.01f;
    int aliveNeighbors;
    float energy=.01f;
    uchar4 color;
};

enum GameMode {
    gameOfLife = 0,
    sigmoidF = 1,
    hyperbolicTanF = 2,
    reLuF = 3
};

enum ToolMode {
    DISTURBE = 0,
    DRAG = 1,
    TEAR = 2,
    ACTIVE = 3
};

struct Settings {
    // GameMode gameMode = gameOfLife;
    int numberOfParticles = 1000000;
    float particleRadius = .5f;
    float restLength = 1.0f;
    int option = 0;
    float widthFactor = 1.0f;
    int gridSize = 10;
    float thickness = 1.0f;
    float ringSpacing = 1.0f;
    float spacing = 1.0f;
    int band = 0;
    int blockSize = 32;
    int diagonalBand = 0;
    int border = 1;
    uint8_t rule = 30;
      

    bool operator!=(const Settings& other) const {
        return std::tie(/* gameMode, */ numberOfParticles, particleRadius, restLength, option, widthFactor, gridSize, thickness,
                        ringSpacing, spacing, band, blockSize,
                        diagonalBand, border,rule) !=
               std::tie(/* other.gameMode, */ other.numberOfParticles, other.particleRadius, other.restLength, other.option, other.widthFactor, other.gridSize, other.thickness,
                        other.ringSpacing, other.spacing, other.band, other.blockSize,
                        other.diagonalBand, other.border, other.rule);
    }
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
        // int padding = 50;



        // mouse operations
        float zoom = 1.0f;
        float panX = -width / 2.0f, panY = -height / 2.0f;
        int lastMouseX, lastMouseY;
        vec2 lastMousePos = vec2(0.0f, 0.0f);
        bool isDragging = false;
        bool leftMouseDown = false;
        bool isPanEnabled = false;
        float mouseCursorRadius = 50.0f;
        GameMode gameMode = gameOfLife;
        ToolMode toolMode = DISTURBE;


        // Draw shapes
        void drawGlowingCircle(cudaSurfaceObject_t &surface, vec2 position, float radius, float glowExtent, uchar4 color);
        void drawRing(cudaSurfaceObject_t &surface, vec2 position, float radius, float thickness, uchar4 color);
        
        // Game of Life
        vec2 topLeft;
        bool startSimulation = false;
        int option = 15;

        float kernelMatrix[9] = {
            0.5f, 1.0f, 0.5f,
            1.0f, 0.0f, 1.0f,
            0.5f, 1.0f, 0.5f
        };
        
        
        GameLife* d_gameLife; // for GPU operations
        int gridRows, gridCols;
        std::vector<GameLife> gamelife;
        std::vector<uchar4> colorPallete = {BLUE_PLANET, GRAY_ROCKY, SUN_YELLOW, JUPITER, FULL_MOON, VENUS_TAN, RED_MERCURY, GREEN, GOLD, WHITE, PINK, ORANGE, TAN };
        int numberOfParticles = 1000000;
        float particleRadius = .5f;
        float restLength = 1;
        void activateGameLife();
        void activateGameLife(GameLife* &d_gameLife);
        void initGameLife();
        void setGroupOfParticles(int totalParticles, int2 ratio, bool anchors = 0);
        void setGroupOfParticles(int2 ratio);
        int2 calculateGrid(int n, int a, int b);
        int2 calculateGridWithRatio(float ratioX, float ratioY);
        int2 calculateGridClamped(int n, int a, int b);
        void drawGameLife(cudaSurfaceObject_t &surface, GameLife* &d_gameLife);
        void disturbeGameLife(vec2 mousePosition);
        std::vector<float> generateCircularGaussianKernel(int radius, float sigma);

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
        // convolutions
        float sigmoidThreshold = 3.0f;
        float kernelWeightCenter = 0.0f;  // usually ignored
        float kernelWeightEdge = 1.0f;
        float kernelWeightCorner = 0.5f;
        uint8_t rule = 30;
        float sigma= 0.03f;
        float kernelSigma = 0.65f;
        float mu = 0.16f;
        float kernelRadius = 8.0f;


  


      
    
    private:
        // GL resources
        cudaGraphicsResource_t cudaResource;
        cudaSurfaceObject_t MapSurfaceResouse();

};