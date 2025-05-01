#include "CUDAHandler.h"
#include "cudaKernels.cuh"
#include "cuda_utils.h"

__global__ void drawParticles_kernel(cudaSurfaceObject_t surface, GameLife* particles, int numberParticles, int width, int height, float zoom, float panX, float panY){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numberParticles || !particles[i].alive) return;
    GameLife gl = particles[i];
    vec2f pos = gl.position;
    float radius = gl.radius * zoom;
    int x0 = (int)(width / 2.0f + (pos.x + panX) * zoom);
    int y0 = (int)(height / 2.0f + (pos.y + panY) * zoom);
    
    drawFilledCircle(surface, x0, y0, radius, gl.color, width, height);
}

__global__ void commitNextState_kernel(GameLife* gamelife, int totalParticles) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;

    gamelife[i].alive = gamelife[i].next;
    gamelife[i].next = false;
}

__global__ void activate_gameOfLife_kernel(GameLife* gamelife, int totalParticles, int gridRows, int gridCols) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;
  
    int row = i / gridCols;
    int col = i % gridCols;
    int aliveCount = 0;
    
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue; // skip self

            int nr = row + dr;
            int nc = col + dc;
            // * Check if neighbors are within the grid bounds
            if (nr >= 0 && nr < gridRows && nc >= 0 && nc < gridCols) {
                int j = nr * gridCols + nc;
                // * Check if neighbors are alive and count them
                if (gamelife[j].alive) aliveCount++;
            }
        }
    }
    
    // Apply rules
    if (gamelife[i].alive )
        gamelife[i].next = (aliveCount == 2 || aliveCount == 3); // stays alive or not
    else
        gamelife[i].next = (aliveCount == 3);
}




CUDAHandler* CUDAHandler::instance = nullptr;

CUDAHandler::CUDAHandler(int width, int height, GLuint textureID) :  width(width), height(height)
{
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    instance = this; // store global reference (to be used for mouse and imGui User Interface (UI) operations)
    center = vec2f(width / 2.0f, height / 2.0f);
    
}

CUDAHandler::~CUDAHandler()
{
    cudaGraphicsUnregisterResource(cudaResource);
    
}
// _________________________________________________________________________//
void CUDAHandler::updateDraw(float dt)
{
    this->dt = dt;
    framesCount++;

    if (gamelife.empty()) {
        initGameLife();
    } 
    

    GameLife* d_gameLife;
    checkCuda(cudaMalloc(&d_gameLife, gamelife.size() * sizeof(GameLife)));
    checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));
    
    
    activateGameLife(d_gameLife);
    checkCuda(cudaMemcpy(gamelife.data(), d_gameLife, gamelife.size() * sizeof(GameLife), cudaMemcpyDeviceToHost));
    

    cudaSurfaceObject_t surface = MapSurfaceResouse(); 
   
    clearGraphicsDisply(surface, SUN_YELLOW);

    // draw samples to check ZOOM & PAN
    
    // drawCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 200, SUN_YELLOW, 1, 4, zoom, panX, panY);
    // drawGlowingCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 500, RED_MERCURY, 1.5f, zoom, panX, panY);
    // drawRing(surface, center, 500, 4, BLUE_PLANET);

    // drawGlowingCircle(surface, center, 500, 1.5, RED_MERCURY );

    

    drawGameLife(surface, d_gameLife);

    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    cudaFree(d_gameLife);

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);
}

//________________________________________________________________________//

void CUDAHandler::clearGraphicsDisply(cudaSurfaceObject_t &surface, uchar4 color)
{
    int threads = 16; 
    dim3 clearBlock(threads, threads);
    dim3 clearGrid((width + clearBlock.x -1) / clearBlock.x, (height + clearBlock.y - 1) / clearBlock.y);
    clearSurface_kernel<<<clearGrid, clearBlock>>>(surface, width, height, color);
}

void CUDAHandler::drawGlowingCircle(cudaSurfaceObject_t &surface, vec2f position, float radius, float glowExtent, uchar4 color)
{
    // Map world center to screen center
    float screen_cx = (position.x + panX) * zoom + width / 2.0f;
    float screen_cy = (position.y + panY) * zoom + height / 2.0f;

    // Calculate radius in screen pixels
    float screen_radius = radius * zoom;
    float screen_glowRadius = glowExtent * screen_radius;

    int xmin = max(0, (int)(screen_cx - screen_glowRadius));
    int xmax = min(width-1, (int)(screen_cx + screen_glowRadius));
    int ymin = max(0, (int)(screen_cy - screen_glowRadius));
    int ymax = min(height-1, (int)(screen_cy + screen_glowRadius));


    
    // // Calculate bounding box   // if not zoom , nor panx, nor pany involved
    // float glowRadius = glowExtent * radius;
    // int xMin = max(0, (int)(position.x - glowRadius));
    // int xMax = min(width - 1, (int)(position.x + glowRadius));
    // int yMin = max(0, (int)(position.y - glowRadius));
    // int yMax = min(height - 1, (int)(position.y + glowRadius));

    int drawWidth   = xmax - xmin + 1;
    int drawHeight  = ymax - ymin + 1;

    dim3 blockSize(16, 16);
    dim3 gridSize ((drawWidth + blockSize.x - 1) / blockSize.x, (drawHeight + blockSize.y -1) / blockSize.y);
    drawGlowingCircle_kernel<<<gridSize, blockSize>>>(surface, width, height, position.x, position.y, radius,  color, 1.5f, xmin, ymin, zoom, panX, panY);
}

void CUDAHandler::drawRing(cudaSurfaceObject_t &surface, vec2f position, float radius, float thickness, uchar4 color)
{
    // Map world center to screen center
    float screen_cx = (position.x + panX) * zoom + width / 2.0f;
    float screen_cy = (position.y + panY) * zoom + height / 2.0f;

    // Calculate radius in screen pixels
    float screen_radius = radius * zoom;

    int xmin = max(0, (int)(screen_cx - screen_radius - thickness));
    int xmax = min(width - 1, (int)(screen_cx + screen_radius + thickness));
    int ymin = max(0, (int)(screen_cy - screen_radius - thickness));
    int ymax = min(height - 1, (int)(screen_cy + screen_radius + thickness));

    int drawWidth   = xmax - xmin + 1;
    int drawHeight  = ymax - ymin + 1;

    dim3 blockSize(16, 16);
    dim3 gridSize ((drawWidth + blockSize.x - 1) / blockSize.x, (drawHeight + blockSize.y -1) / blockSize.y);
    // Pass world-space center (not screen-space) to the kerne
    // drawRing_kernel<<<gridSize, blockSize>>>(surface, width, height, position.x, position.y, radius,  color, thickness, xmin, ymin, zoom, panX, panY);
    drawRing_sharedMemory_kernel<<<gridSize, blockSize>>>(surface, width, height, position.x, position.y, radius,  color, thickness, xmin, ymin, zoom, panX, panY);



}

void CUDAHandler::activateGameLife()
{
    
    for (auto &gl : gamelife) {
        gl.alive = gl.next;   // next generation is the current generation
        gl.next = false;

    }
    
    int aliveCount;
    // 1. For every particle, calculate its row and column
    for (int i = 0; i < gamelife.size(); ++i) {
        int row = i / gridCols;
        int col = i % gridCols;
        aliveCount = 0;        
         // 2. Loop over all 8 neighbors (including diagonals)
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue; // skip self 
               
                int nr = row + dr;
                int nc = col + dc;
                
                // 3.  Check if neighbor is within the grid bounds
                if (nr >= 0 && nr < gridRows && nc >= 0 && nc < gridCols) {
                    int j = nr * gridCols + nc;
                    // 4. check if the neighbors are alive
                    if (gamelife[j].alive) aliveCount++;
                }
            }
        }
        gamelife[i].aliveNeighbors = aliveCount;  
    }
    


    for (auto &gl : gamelife) {
        if (!gl.alive && gl.aliveNeighbors == 3) gl.next = true;  // revives : reproduction
        if (gl.alive && gl.aliveNeighbors < 2)   gl.next = false; // dies : underpopulation
        if (gl.alive && gl.aliveNeighbors > 3)   gl.next = false;  // dies : overpopulation
        if (gl.alive && (gl.aliveNeighbors == 2 || gl.aliveNeighbors == 3)) gl.next = true;  // stays alive

    }
}

void CUDAHandler::activateGameLife(GameLife* &d_gameLife)
{

    int threads = 128;
    int blocks = (numberOfParticles + threads - 1) / threads;
    commitNextState_kernel<<<blocks, threads>>> (d_gameLife, gamelife.size());
    checkCuda(cudaDeviceSynchronize());
    activate_gameOfLife_kernel<<<blocks, threads>>>(d_gameLife, gamelife.size(), gridRows, gridCols);
}

void CUDAHandler::initGameLife()
{
    
    int top = 1;
    setGroupOfParticles(numberOfParticles, top, {16, 9});

}

int2 CUDAHandler::calculateGrid(int n, int a, int b)
{
    double targetRatio = static_cast<double>(a) / b;
    double bestDiff = std::numeric_limits<double>::max();
    int bestRows = 1, bestCols = n;

    for (int rows = 1; rows <= n; ++rows) {
        int cols = (n + rows - 1) / rows; // ceil(n / rows)
        double currentRatio = static_cast<double>(cols) / rows;
        double diff = std::abs(currentRatio - targetRatio);

        if (diff < bestDiff) {
            bestDiff = diff;
            bestRows = rows;
            bestCols = cols;
        }
    }

    return {bestRows, bestCols};
}

void CUDAHandler::drawGameLife(cudaSurfaceObject_t &surface, GameLife *&d_gameLife)
{
    int threads = 256;
    int blocks = (gamelife.size() + threads -1 ) / threads;
    drawParticles_kernel<<<blocks, threads>>>(surface, d_gameLife, gamelife.size(), width, height, zoom, panX, panY);
}


void CUDAHandler::setGroupOfParticles(int totalParticles, float top, int2 ratio, bool anchors)
{
    
    // ratio refers to the proportion of length vs width
    int2 grid = calculateGrid(totalParticles, ratio.x,ratio.y);
    int rows = grid.x;
    int cols = grid.y;

    gridRows = rows;
    gridCols = cols;    

    int offset = width / 2.0f - (cols - 1) * particleRadius;    
    vec2f topLeft(offset, top);

    // Place particles in a 2D grid at restLength spacing
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float x = topLeft.x + c * restLength;
            float y = topLeft.y + r * restLength;
            GameLife gl;
            gl.position = vec2f(x,y);
            gl.radius = particleRadius;
            gl.alive = gl.next = randomBool();
            // if (c == 2 && ( r == 1 || r == 2 || r == 3)) gl.next = true;
            // else gl.next = false;
            gl.color = RED_MERCURY;
            gamelife.push_back(gl);
        }
    }
}


cudaSurfaceObject_t CUDAHandler::MapSurfaceResouse()
{
    //* Map the resource for CUDA
    cudaArray_t array;
    // glFinish();
    cudaGraphicsMapResources(1, &cudaResource, 0);
    cudaGraphicsSubResourceGetMappedArray(&array, cudaResource, 0, 0);

    //* Create a CUDA surface object
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = array;

    cudaSurfaceObject_t surface = 0;
    cudaCreateSurfaceObject(&surface, &resDesc);
    return surface;
}
