#include "CUDAHandler.h"
#include "cudaKernels.cuh"
#include <curand_kernel.h>
#include "cuda_utils.h"


__global__ void init_random(unsigned int seed, curandState_t* states){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}


// 1D threads
__global__ void disturbeGameLife_kernel(GameLife* gameLife, float mousePosX, float mousePosY, int numberOfCells, float mouseRadius)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= numberOfCells) return;
    
    
    __shared__ float s_mousePosX;
    __shared__ float s_mousePosY;
    __shared__ float s_mouseRadiusSqr;
   
    

    if (threadIdx.x == 0) {
        s_mousePosX = mousePosX;
        s_mousePosY = mousePosY;
        s_mouseRadiusSqr = mouseRadius * mouseRadius;;
        
    }
    __syncthreads();

    // ! Yes! EARLY-EXIT STRATEGY  - Early AABB rejection to skip square root / dotProduct
    // vec2 pos = gameLife[i].position;
    // float dx = pos.x - s_mousePosX;
    // if(fabsf(dx) > mouseRadius) return;

    // float dy = pos.y - s_mousePosY;
    // if(fabsf(dy) > mouseRadius) return;
    
    // float distSq = dx * dx + dy * dy;

    // No EARLY-EXIT STRATEGY
    vec2 pos(s_mousePosX, s_mousePosY);
    float distSq = (gameLife[i].position - pos).magSq();

    if (distSq < s_mouseRadiusSqr) {
        gameLife[i].next ^= true;
        gameLife[i].color = make_uchar4(186, 186, 186, 255);
    }
}

__global__ void disturbeGameLife_kernel_2D(
    GameLife* gameLife,
    int gridRows, int gridCols,
    float cellSpacing,
    float mousePosX, float mousePosY,
    float mouseRadius)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= gridRows || c >= gridCols) return;

    int idx = r * gridCols + c;

    vec2 pos = gameLife[idx].position;

    // AABB rejection
    float dx = pos.x - mousePosX;
    if (fabsf(dx) > mouseRadius) return;

    float dy = pos.y - mousePosY;
    if (fabsf(dy) > mouseRadius) return;

    float distSq = dx * dx + dy * dy;
    if (distSq < mouseRadius * mouseRadius) {
        gameLife[idx].next ^= true;
    }
}

__global__ void disturbGameLife_kernel_windowed(
    GameLife* gameLife,
    int gridRows, int gridCols,
    float cellSpacing,
    float mouseX, float mouseY,
    float radius,
    int rowOffset, int colOffset)
{
    int localRow = blockIdx.y * blockDim.y + threadIdx.y;
    int localCol = blockIdx.x * blockDim.x + threadIdx.x;

    int globalRow = rowOffset + localRow;
    int globalCol = colOffset + localCol;

    if (globalRow >= gridRows || globalCol >= gridCols) return;

    int index = globalRow * gridCols + globalCol;

    vec2 pos = gameLife[index].position;

    float dx = pos.x - mouseX;
    float dy = pos.y - mouseY;

    if (fabsf(dx) > radius || fabsf(dy) > radius) return;

    float distSq = dx * dx + dy * dy;
    if (distSq < radius * radius) {
        gameLife[index].next ^= true;
    }
}

__global__ void disturbGameLife_kernel_windowed_shared(
    GameLife* gameLife,
    int gridRows, int gridCols,
    float cellSpacing,
    float mouseX, float mouseY,
    float radius,
    int rowOffset, int colOffset)
{
    int localRow = blockIdx.y * blockDim.y + threadIdx.y;
    int localCol = blockIdx.x * blockDim.x + threadIdx.x;

    int globalRow = rowOffset + localRow;
    int globalCol = colOffset + localCol;

    if (globalRow >= gridRows || globalCol >= gridCols) return;

    // --- Shared memory for read-only constants
    __shared__ float s_mouseX;
    __shared__ float s_mouseY;
    __shared__ float s_radiusSq;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        s_mouseX = mouseX;
        s_mouseY = mouseY;
        s_radiusSq = radius * radius;
    }

    __syncthreads();  // make sure all threads see the shared values

    int index = globalRow * gridCols + globalCol;

    vec2 pos = gameLife[index].position;

    float dx = pos.x - s_mouseX;
    if (fabsf(dx) > radius) return;

    float dy = pos.y - s_mouseY;
    if (fabsf(dy) > radius) return;

    float distSq = dx * dx + dy * dy;
    if (distSq < s_radiusSq) {
        gameLife[index].next ^= true;
    }
}





__global__ void drawParticles_kernel(cudaSurfaceObject_t surface, GameLife* particles, int numberParticles, int width, int height, float zoom, float panX, float panY){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numberParticles || !particles[i].alive) return;
    GameLife gl = particles[i];
    vec2 pos = gl.position;
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
    else {
        gamelife[i].next = (aliveCount == 3);
    }
}

__device__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ float tanhMapped(float x) {
    return 0.5f * (tanhf(x) + 1.0f);
}

__device__ float reluClamped(float x) {
    return fminf(fmaxf(x, 0.0f), 1.0f);
}
__device__ float reluProb(float x) {
    // scale to a more usable range
    float scaled = 0.25f * x;  // adjust scale as needed
    return fminf(fmaxf(scaled, 0.0f), 1.0f);
}



__global__ void activate_gameOfLife_convolution_kernel(curandState_t* states, GameLife* gamelife, int totalParticles, int gridRows, int gridCols, GameMode gameMode, float threshold, float edgeWeight, float cornerWeight) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= totalParticles) return;

    int row = i / gridCols;
    int col = i % gridCols;

    float neighborSum = 0.0f;
    int aliveCount = 0;

    // Example kernel weights (symmetric, Gaussian-like)
    // const float kernel[3][3] = {
    //     {0.5f, 1.0f, 0.5f},
    //     {1.0f, 0.0f, 1.0f},  // center weight is ignored
    //     {0.5f, 1.0f, 0.5f}
    // };
    // Define your kernel weights dynamically
    const float kernel[3][3] = {
        {cornerWeight, edgeWeight, -cornerWeight},
        {edgeWeight,   0.0f,       -edgeWeight},
        {cornerWeight, -edgeWeight, -cornerWeight}
    };


    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;

            int nr = row + dr;
            int nc = col + dc;
            if (nr >= 0 && nr < gridRows && nc >= 0 && nc < gridCols) {
                int j = nr * gridCols + nc;
                neighborSum += (gamelife[j].alive ? 1.0f : 0.0f) * kernel[dr + 1][dc + 1];
                aliveCount += gamelife[j].alive ? 1 : 0;

            }
        }
    }

    if (gameMode == gameOfLife) {
        if (gamelife[i].alive )
            gamelife[i].next = (aliveCount == 2 || aliveCount == 3); // stays alive or not
        else {
            gamelife[i].next = (aliveCount == 3);
        }
    } else if (gameMode == sigmoidF) {
        // float threshold = 3.0f;  // similar to GoL but flexible
        float probability = sigmoid(neighborSum - threshold);
        // gamelife[i].next = (probability > 0.5f);   // no stochastic
        curandState_t x = states[i];
        gamelife[i].next = (curand_uniform(&x) <= probability); // stochastic : random() < probability
        states[i] = x; // save back

    } else if (gameMode == hyperbolicTanF) {
        float probability = tanhMapped(neighborSum - threshold);
        curandState_t x = states[i];
        gamelife[i].next = (curand_uniform(&x) <= probability);
        states[i] = x; // save back
        // gamelife[i].next = (probability > 0.5f);
    } else if (gameMode == reLuF) {
        float probability = reluProb(neighborSum - threshold);
        curandState_t x = states[i];
        gamelife[i].next = (curand_uniform(&x) <= probability);
        states[i] = x; // save back
    }

    // // float threshold = 3.0f;  // similar to GoL but flexible
    // float probability = sigmoid(neighborSum - threshold);
    // // gamelife[i].next = (probability > 0.5f);  // or experiment with stochastic: random() < probability
    // curandState_t x = states[i];
    // gamelife[i].next = (curand_uniform(&x) <= probability);
    // states[i] = x; // save back

    // float probability = sigmoid(neighborSum - threshold);
    // if (probability > 0.7f) gamelife[i].next = true;
    // else if (probability < 0.1f) gamelife[i].next = false;
    // else gamelife[i].next = gamelife[i].alive;  // keep previous state
}





CUDAHandler* CUDAHandler::instance = nullptr;

CUDAHandler::CUDAHandler(int width, int height, GLuint textureID) :  width(width), height(height)
{
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    instance = this; // store global reference (to be used for mouse and imGui User Interface (UI) operations)
    center = vec2(width / 2.0f, height / 2.0f);
    screenRatio = static_cast<float>(height) / width;
    
}

CUDAHandler::~CUDAHandler()
{
    cudaFree(d_gameLife);

    cudaGraphicsUnregisterResource(cudaResource);
    
}
// _________________________________________________________________________//
void CUDAHandler::updateDraw(float dt)
{
    this->dt = dt;
    framesCount++;


    static Settings previousSettings;
    Settings currentSettings = {
        // .gameMode = gameMode,
        .numberOfParticles = numberOfParticles,
        .particleRadius = particleRadius,
        .restLength = restLength,
        .option = option,
        .widthFactor = widthFactor,
        .gridSize = gridSize,
        .thickness = thickness,
        .ringSpacing = ringSpacing,
        .spacing = spacing,
        .band = band,
        .blockSize = blockSize,
        .diagonalBand = diagonalBand,
        .border = border,
        .rule = rule
    };
    

    if (gamelife.empty() || currentSettings != previousSettings) {
        framesCount = 0;
        initGameLife();
        previousSettings = currentSettings;
    }
    
    
    // GameLife* d_gameLife;
    // checkCuda(cudaMalloc(&d_gameLife, gamelife.size() * sizeof(GameLife)));
    // checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));
    
    
    if(startSimulation) {
        activateGameLife(d_gameLife);
    }
    // checkCuda(cudaMemcpy(gamelife.data(), d_gameLife, gamelife.size() * sizeof(GameLife), cudaMemcpyDeviceToHost));
    

    cudaSurfaceObject_t surface = MapSurfaceResouse(); 
   
    clearGraphicsDisply(surface, DARK);

    // draw samples to check ZOOM & PAN
    
    // drawCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 200, SUN_YELLOW, 1, 4, zoom, panX, panY);
    // drawGlowingCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 500, RED_MERCURY, 1.5f, zoom, panX, panY);
    // drawRing(surface, center, 500, 4, BLUE_PLANET);

    // drawGlowingCircle(surface, center, 500, 1.5, RED_MERCURY );

    

    drawGameLife(surface, d_gameLife);

    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    // cudaFree(d_gameLife);

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

void CUDAHandler::drawGlowingCircle(cudaSurfaceObject_t &surface, vec2 position, float radius, float glowExtent, uchar4 color)
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

void CUDAHandler::drawRing(cudaSurfaceObject_t &surface, vec2 position, float radius, float thickness, uchar4 color)
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
    int threads = 256;
    int blocks = (gamelife.size() + threads - 1) / threads;
    commitNextState_kernel<<<blocks, threads>>> (d_gameLife, gamelife.size());
    checkCuda(cudaDeviceSynchronize());

    //generate random seed to be used in rayTracer kernel
    int num_threads = threads * blocks;
    curandState_t* d_states;
    checkCuda(cudaMalloc(&d_states, num_threads * sizeof(curandState_t)));
    init_random<<<blocks, threads>>>(time(0), d_states);
    checkCuda(cudaDeviceSynchronize() );

    activate_gameOfLife_convolution_kernel<<<blocks, threads>>>(d_states, d_gameLife, gamelife.size(), gridRows, gridCols, gameMode, sigmoidThreshold, kernelWeightEdge, kernelWeightCorner);
    // activate_gameOfLife_kernel<<<blocks, threads>>>(d_gameLife, gamelife.size(), gridRows, gridCols);
    cudaFree(d_states);
}

void CUDAHandler::initGameLife()
{
    
    gamelife.clear();
    startSimulation = false;
    setGroupOfParticles(numberOfParticles, {16, 9});
    // setGroupOfParticles({16, 9});
    checkCuda(cudaMalloc(&d_gameLife, gamelife.size() * sizeof(GameLife)));
    checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));

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

int2 CUDAHandler::calculateGridWithRatio(float ratioX, float ratioY)
{
    float cellSize = 2.0f * particleRadius + restLength;

    int maxCols = static_cast<int>(width  / cellSize);
    int maxRows = static_cast<int>(height / cellSize);

    float aspect = 1.0 ; //ratioX / ratioY;

    // Fit the grid while maintaining the ratio and staying within bounds
    int cols = std::min(maxCols, static_cast<int>(maxRows * aspect));
    int rows = std::min(maxRows, static_cast<int>(cols / aspect));

    return int2{rows, cols};
}

int2 CUDAHandler::calculateGridClamped(int n, int a, int b)
{
    double targetRatio = static_cast<double>(a) / b;
    double bestDiff = std::numeric_limits<double>::max();

    float cellSize = 2.0f * particleRadius + restLength;

    int maxCols = static_cast<int>(width  / cellSize);
    int maxRows = static_cast<int>(height / cellSize);

    int bestRows = 1, bestCols = n;

    for (int rows = 1; rows <= maxRows; ++rows) {
        int cols = (n + rows - 1) / rows; // ceil(n / rows)
        if (cols > maxCols) continue;     // skip: doesn't fit on screen

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

void CUDAHandler::disturbeGameLife(vec2 mousePosition)
{
    // for (int i = 0; i < gamelife.size(); ++i) {

    //     float d2 = (gamelife[i].position - mousePosition).magSq();
    //     if (d2 <  mouseCursorRadius * mouseCursorRadius) {
            
    //         gamelife[i].next ^= true;

    //     }

    // }

    // 1D kernel //
    // checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));
    int threads = 256;
    int blocks = (gamelife.size() + threads - 1) / threads;

    disturbeGameLife_kernel<<<blocks, threads>>>(d_gameLife, mousePosition.x, mousePosition.y, gamelife.size(), mouseCursorRadius);

    // checkCuda(cudaMemcpy(gamelife.data(), d_gameLife, gamelife.size() * sizeof(GameLife), cudaMemcpyDeviceToHost));


    // 2D Kernel
    // // checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));
    // dim3 blockSize(16, 16);
    // dim3 gridSize((gridCols + blockSize.x - 1) / blockSize.x, (gridRows + blockSize.y - 1) / blockSize.y);

    // disturbeGameLife_kernel_2D<<<gridSize, blockSize>>>(d_gameLife, gridRows, gridCols, restLength, mousePosition.x, mousePosition.y, mouseCursorRadius);
    
    // // checkCuda(cudaMemcpy(gamelife.data(), d_gameLife, gamelife.size() * sizeof(GameLife), cudaMemcpyDeviceToHost));

   
    


    // Compute min/max row/col range on host
    // checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));
    // int minCol = max(0, int((mousePosition.x - mouseCursorRadius - topLeft.x) / restLength));
    // int maxCol = min(gridCols, int((mousePosition.x + mouseCursorRadius - topLeft.x) / restLength));
    // int minRow = max(0, int((mousePosition.y - mouseCursorRadius - topLeft.y) / restLength));
    // int maxRow = min(gridRows, int((mousePosition.y + mouseCursorRadius - topLeft.y) / restLength));
    // int drawWidth   = maxCol - minCol + 1;
    // int drawHeight  = maxRow - minRow + 1;
    
    // dim3 blockSize(16, 16);
    // dim3 gridSize((drawWidth + blockSize.x - 1) / blockSize.x, (drawHeight + blockSize.y - 1) / blockSize.y);
    // // disturbGameLife_kernel_windowed<<<gridSize, blockSize>>>(d_gameLife, gridRows, gridCols, restLength, mousePosition.x, mousePosition.y, mouseCursorRadius, minRow, minCol);
    // disturbGameLife_kernel_windowed_shared<<<gridSize, blockSize>>>(d_gameLife, gridRows, gridCols, restLength, mousePosition.x, mousePosition.y, mouseCursorRadius, minRow, minCol);
    // // checkCuda(cudaMemcpy(gamelife.data(), d_gameLife, gamelife.size() * sizeof(GameLife), cudaMemcpyDeviceToHost));




}

void CUDAHandler::setGroupOfParticles(int totalParticles, int2 ratio, bool anchors )
{
    
    // ratio refers to the proportion of length vs width
    int2 grid = calculateGrid(totalParticles, ratio.x,ratio.y);
    // restLength = 2 * particleRadius;
    // int2 grid = calculateGridClamped(totalParticles, ratio.x,ratio.y);
    int rows = grid.x;
    int cols = grid.y;

    // printf("Rows: %d  -  Cols: %d - total: %d\n", rows, cols, rows * cols);

    gridRows = rows;
    gridCols = cols;    

    // int offset = width / 2.0f - (cols - 1) * particleRadius;    
    // float offset = width / 2.0f - (cols - 1) * restLength / 2.0f;
    float offsetX = (width  - (cols - 1) * restLength) / 2.0f;
    float offsetY = (height - (rows - 1) * restLength) / 2.0f;
    topLeft = vec2(offsetX, offsetY);


    // topLeft = vec2(offset, top);
    
    int rowsSize = widthFactor * gridRows;
    int colsSize = widthFactor * gridCols * screenRatio;  // screen ratio for correctness

    // Place particles in a 2D grid at restLength spacing
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float x = topLeft.x + c * restLength;
            float y = topLeft.y + r * restLength;
            GameLife gl;
            gl.position = vec2(x,y);
            gl.radius = particleRadius;
            switch(option){
                case 0:   // grid
                    
                    if (c % gridSize == 0 || r % gridSize == 0) {
                        gl.alive = gl.next = true;
                        gl.color = WHITE;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GREEN;
                    }

                    
                    break;
                case 1: // Vertical
                    if ((c / colsSize) % 2 == 0) {
                        gl.alive = gl.next = true;      // cell is ON
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 2: // horizontal
                    if ((r / rowsSize) % 2 == 0) {
                        gl.alive = gl.next = true;      // cell is ON  
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 3:    // checkered
                    
                    if ((r / rowsSize) % 2 == 0 && c / colsSize % 2 == 0) { 
                        gl.alive = gl.next = true;      // cell is ON
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 4: { // diagonal
                    int band = 0;
                    if (abs(r - c) < band ) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GOLD;
                    }
                    break;
                }
                case 5: {  // x shape
                        int band = 0;
                        int centerOffset = cols - rows;
                        if (abs((r) - (c - centerOffset / 2)) <= band || abs((r) + (c - centerOffset / 2) - (rows - 1)) <= band) {
                        // if (r == c || r + c == rows - 1) {
                        // if (abs(r - c) <= band || abs(r + c - (rows - 1)) <= band) {

                        gl.alive = gl.next = true;
                        gl.color = RED_MERCURY;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GOLD;
                    }
                    break;
                }
                case 6: { // Circle
                    float centerX = cols / 2.0f;
                    float centerY = rows / 2.0f;
                    int gap = 150;
                    int gapSq = gap * gap;
                    float radiusSquared = (rows / 3.0f) * (rows / 3.0f);
                    float radiusSquared2 = (rows / 6.0f) * (rows / 6.0f);
                    float radiusSquared3 = (rows / 9.0f) * (rows / 9.0f);
                    float dx = c - centerX;
                    float dy = r - centerY;
                    float dist = dx * dx + dy * dy;
                    if (dist <= radiusSquared && dist > radiusSquared2 + gapSq) {
                        gl.alive = gl.next = true;
                        gl.color = BLUE_PLANET;
                    } else if (dist > radiusSquared2 && dist < radiusSquared3 + gapSq ){
                        gl.alive = gl.next = true;
                        gl.color = SUN_YELLOW;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = URANUS_BLUE;
                    }
                    break;
                    }
                case 7: {
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                
                    float dx = c - cx;
                    float dy = r - cy;
                    float dist = sqrtf(dx * dx + dy * dy);
                    float angle = atan2f(dy, dx);  // [-π, π]
                    angle = angle < 0 ? angle + 2.0f * M_PI : angle;
                
                    // Parameters
                    // float spacing = 6.0f;        // radial spacing per full rotation (~coil tightness)
                    // float thickness = 2.5f;      // thickness of the spiral band
                
                    // Spiral formula: r = spacing * theta
                    float r_cw = angle * spacing;
                    float diff_cw = fabs(dist - r_cw);
                
                    // Opposite spiral
                    float r_ccw = (2.0f * M_PI - angle) * spacing;
                    float diff_ccw = fabs(dist - r_ccw);
                
                    if (diff_cw < thickness || diff_ccw < thickness) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = PINK;
                    }
                    break;
                }
                case 8: { // border
                    int border = 50;  // thickness of border
                    if (r < border || r >= rows - border || c < border || c >= cols - border) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GOLD;
                    }
                    break;
                }
                case 9: {  // double border
                    int outer = 1;  // outer thickness
                    int inner = 50;  // inner offset
                    bool isOuter = (r < outer || r >= rows - outer || c < outer || c >= cols - outer);
                    bool isInner = (r >= inner && r < rows - inner && c >= inner && c < cols - inner);
                    if (isOuter || isInner) {
                        gl.alive = gl.next = true;
                        gl.color = NEPTUNE_PURPLE;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = SUN_YELLOW;
                    }
                    break;
                }
                case 10: { // concentric Rings
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                    float dx = c - cx;
                    float dy = r - cy;
                    float dist = sqrtf(dx * dx + dy * dy);
                
                    //* ringSpacing :controls distance between rings
                    //* thickness : ring band thickness
                
                    float modVal = fmodf(dist, ringSpacing);
                    if (modVal < thickness) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = NEPTUNE_PURPLE;
                    }
                    break;
                }
                case 11: { // Radial beam
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                    float dx = c - cx;
                    float dy = r - cy;
                
                    float angle = atan2f(dy, dx);  // range: [-π, π]
                    angle = angle < 0 ? angle + 2.0f * M_PI : angle;  // normalize to [0, 2π]
                
                    int numBeams = 16;         // number of sun rays
                    float beamWidth = M_PI * 2.0f / numBeams;  // angle between beams
                
                    int beamIndex = (int)(angle / beamWidth);
                    if (beamIndex % 2 == 0) {
                        gl.alive = gl.next = true;
                        gl.color = SUN_YELLOW;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GREEN;
                    }
                    break;
                }
                case 12: {  // Animated Rotating Sunbeam
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                    float dx = c - cx;
                    float dy = r - cy;
                
                    float angle = atan2f(dy, dx);
                    angle = angle < 0 ? angle + 2.0f * M_PI : angle;
                
                    int numBeams = 16;
                    float beamWidth = 2.0f * M_PI / numBeams;
                
                    float angularOffset = fmodf(framesCount * dt * 0.5f, 2.0f * M_PI);  // rotate over time
                    angle += angularOffset;
                
                    int beamIndex = (int)(angle / beamWidth);
                    if (beamIndex % 2 == 0) {
                        gl.alive = gl.next = true;
                        gl.color = SUN_YELLOW;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = URANUS_BLUE;
                    }
                    break;
                }
            case 13: {
                // int blockSize = 6;      // size of each square block
                // int band = 1;           // diagonal thickness
            
                int blockRow = r / blockSize;
                int blockCol = c / blockSize;
            
                int localR = r % blockSize;
                int localC = c % blockSize;
            
                // Diagonal type: choose one or alternate
                bool useForwardSlash = true;  // true = '/', false = '\'
            
                // Optional: alternate slashes like a checker
                // if ((blockRow + blockCol) % 2 == 0) useForwardSlash = true;
                // else useForwardSlash = false;
            
                bool isDiagonal = false;
            
                if (useForwardSlash) {
                    isDiagonal = abs(localR + localC - (blockSize - 1)) <= band;
                } else {
                    isDiagonal = abs(localR - localC) <= band;
                }
            
                if (isDiagonal) {
                    gl.alive = gl.next = true;
                    gl.color = ORANGE;
                } else {
                    gl.alive = gl.next = false;
                    gl.color = TAN;
                }
                break;
            }
        case 14: {
            // int blockSize = 6;  // size of each square
            // int border = 1;     // thickness of grid lines
            // int diagonalBand = 1;  // diagonal thickness
        
            int blockRow = r / blockSize;
            int blockCol = c / blockSize;
        
            int localR = r % blockSize;
            int localC = c % blockSize;
        
            bool isBorder = (localR < border || localR >= blockSize - border ||
                                localC < border || localC >= blockSize - border);
        
            // Diagonal type: '/' or '\' or alternating
            bool useForwardSlash = ((blockRow + blockCol) % 2 == 0);  // alternate per tile
        
            bool isDiagonal = false;
            if (useForwardSlash) {
                isDiagonal = abs(localR + localC - (blockSize - 1)) <= diagonalBand;
            } else {
                isDiagonal = abs(localR - localC) <= diagonalBand;
            }
        
            if (isBorder || isDiagonal) {
                gl.alive = gl.next = true;
                gl.color = make_uchar4(255, 200, 50, 255);  // gold-orange
            } else {
                gl.alive = gl.next = false;
                gl.color = make_uchar4(20, 20, 20, 255);  // dark background
            }
            break;
        }
        // case 15: { // rule 30
        //     int center = cols / 2;
        //     if (r == 0) {
        //         // First row: set single center cell alive
        //         gl.alive = gl.next = (c == center);
        //     } else {
        //         // Read previous row
        //         int rowAbove = r - 1;
        //         int idxLeft = rowAbove * cols + c - 1;
        //         int idxCenter = rowAbove * cols + c;
        //         int idxRight = rowAbove * cols + c + 1;
        
        //         bool left = (c > 0)              ? gamelife[idxLeft].alive   : false;
        //         bool mid  =                        gamelife[idxCenter].alive;
        //         bool right= (c < cols - 1)       ? gamelife[idxRight].alive  : false;
        
        //         // Apply Rule 30 logic:  new = left XOR (mid OR right)
        //         gl.alive = gl.next = left ^ (mid || right);
        //     }
        
        //     gl.color = gl.alive ? make_uchar4(255, 255, 255, 255) : make_uchar4(0, 0, 0, 255);
        //     break;
        // }
        case 15: {
            int center = cols / 2;
        
            // Expose rule via ImGui slider (external to this loop)
            // extern uint8_t rule;
        
            if (r == 0) {
                // Seed: single dot at center
                gl.alive = gl.next = (c == center);
            } else {
                int rowAbove = r - 1;
                int idxLeft = rowAbove * cols + c - 1;
                int idxMid  = rowAbove * cols + c;
                int idxRight= rowAbove * cols + c + 1;
        
                bool left  = (c > 0)         ? gamelife[idxLeft].alive : false;
                bool mid   = gamelife[idxMid].alive;
                bool right = (c < cols - 1)  ? gamelife[idxRight].alive : false;
        
                int pattern = (left << 2) | (mid << 1) | right;
                gl.alive = gl.next = ((rule >> pattern) & 1);
            }
        
            gl.color = gl.alive ? WHITE : RED_MERCURY;
            break;
        }
                
                default: 
                    break;

            }
            
            gamelife.push_back(gl);
        }
    }
}

void CUDAHandler::setGroupOfParticles(int2 ratio)
{
    // ratio refers to the proportion of length vs width
    restLength = 2 * particleRadius;
    int2 grid = calculateGridWithRatio(ratio.x, ratio.y);
    int rows = grid.x;
    int cols = grid.y;

    // printf("Rows: %d  -  Cols: %d - total: %d\n", rows, cols, rows * cols);

    gridRows = rows;
    gridCols = cols;    

    // int offset = width / 2.0f - (cols - 1) * particleRadius;    
    // float offset = width / 2.0f - (cols - 1) * restLength / 2.0f;
    float offsetX = (width  - (cols - 1) * restLength) / 2.0f;
    float offsetY = (height - (rows - 1) * restLength) / 2.0f;
    topLeft = vec2(offsetX, offsetY);


    // topLeft = vec2(offset, top);
    
    int rowsSize = widthFactor * gridRows;
    int colsSize = widthFactor * gridCols * screenRatio;  // screen ratio for correctness

    // Place particles in a 2D grid at restLength spacing
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float x = topLeft.x + c * restLength;
            float y = topLeft.y + r * restLength;
            GameLife gl;
            gl.position = vec2(x,y);
            gl.radius = particleRadius;
            switch(option){
                case 0:   // grid
                    
                    if (c % gridSize == 0 || r % gridSize == 0) {
                        gl.alive = gl.next = true;
                        gl.color = WHITE;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GREEN;
                    }

                    
                    break;
                case 1: // Vertical
                    if ((c / colsSize) % 2 == 0) {
                        gl.alive = gl.next = true;      // cell is ON
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 2: // horizontal
                    if ((r / rowsSize) % 2 == 0) {
                        gl.alive = gl.next = true;      // cell is ON  
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 3:    // checkered
                    
                    if ((r / rowsSize) % 2 == 0 && c / colsSize % 2 == 0) { 
                        gl.alive = gl.next = true;      // cell is ON
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 4: { // diagonal
                    int band = 0;
                    if (abs(r - c) < band ) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GOLD;
                    }
                    break;
                }
                case 5: {  // x shape
                        int band = 0;
                        int centerOffset = cols - rows;
                        if (abs((r) - (c - centerOffset / 2)) <= band || abs((r) + (c - centerOffset / 2) - (rows - 1)) <= band) {
                        // if (r == c || r + c == rows - 1) {
                        // if (abs(r - c) <= band || abs(r + c - (rows - 1)) <= band) {

                        gl.alive = gl.next = true;
                        gl.color = RED_MERCURY;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GOLD;
                    }
                    break;
                }
                case 6: { // Circle
                    float centerX = cols / 2.0f;
                    float centerY = rows / 2.0f;
                    int gap = 150;
                    int gapSq = gap * gap;
                    float radiusSquared = (rows / 3.0f) * (rows / 3.0f);
                    float radiusSquared2 = (rows / 6.0f) * (rows / 6.0f);
                    float radiusSquared3 = (rows / 9.0f) * (rows / 9.0f);
                    float dx = c - centerX;
                    float dy = r - centerY;
                    float dist = dx * dx + dy * dy;
                    if (dist <= radiusSquared && dist > radiusSquared2 + gapSq) {
                        gl.alive = gl.next = true;
                        gl.color = BLUE_PLANET;
                    } else if (dist > radiusSquared2 && dist < radiusSquared3 + gapSq ){
                        gl.alive = gl.next = true;
                        gl.color = SUN_YELLOW;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = URANUS_BLUE;
                    }
                    break;
                    }
                case 7: {
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                
                    float dx = c - cx;
                    float dy = r - cy;
                    float dist = sqrtf(dx * dx + dy * dy);
                    float angle = atan2f(dy, dx);  // [-π, π]
                    angle = angle < 0 ? angle + 2.0f * M_PI : angle;
                
                    // Parameters
                    // float spacing = 6.0f;        // radial spacing per full rotation (~coil tightness)
                    // float thickness = 2.5f;      // thickness of the spiral band
                
                    // Spiral formula: r = spacing * theta
                    float r_cw = angle * spacing;
                    float diff_cw = fabs(dist - r_cw);
                
                    // Opposite spiral
                    float r_ccw = (2.0f * M_PI - angle) * spacing;
                    float diff_ccw = fabs(dist - r_ccw);
                
                    if (diff_cw < thickness || diff_ccw < thickness) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = PINK;
                    }
                    break;
                }
                case 8: { // border
                    int border = 50;  // thickness of border
                    if (r < border || r >= rows - border || c < border || c >= cols - border) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GOLD;
                    }
                    break;
                }
                case 9: {  // double border
                    int outer = 1;  // outer thickness
                    int inner = 50;  // inner offset
                    bool isOuter = (r < outer || r >= rows - outer || c < outer || c >= cols - outer);
                    bool isInner = (r >= inner && r < rows - inner && c >= inner && c < cols - inner);
                    if (isOuter || isInner) {
                        gl.alive = gl.next = true;
                        gl.color = NEPTUNE_PURPLE;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = SUN_YELLOW;
                    }
                    break;
                }
                case 10: { // concentric Rings
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                    float dx = c - cx;
                    float dy = r - cy;
                    float dist = sqrtf(dx * dx + dy * dy);
                
                    //* ringSpacing :controls distance between rings
                    //* thickness : ring band thickness
                
                    float modVal = fmodf(dist, ringSpacing);
                    if (modVal < thickness) {
                        gl.alive = gl.next = true;
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = NEPTUNE_PURPLE;
                    }
                    break;
                }
                case 11: { // Radial beam
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                    float dx = c - cx;
                    float dy = r - cy;
                
                    float angle = atan2f(dy, dx);  // range: [-π, π]
                    angle = angle < 0 ? angle + 2.0f * M_PI : angle;  // normalize to [0, 2π]
                
                    int numBeams = 16;         // number of sun rays
                    float beamWidth = M_PI * 2.0f / numBeams;  // angle between beams
                
                    int beamIndex = (int)(angle / beamWidth);
                    if (beamIndex % 2 == 0) {
                        gl.alive = gl.next = true;
                        gl.color = SUN_YELLOW;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = GREEN;
                    }
                    break;
                }
                case 12: {  // Animated Rotating Sunbeam
                    float cx = cols / 2.0f;
                    float cy = rows / 2.0f;
                    float dx = c - cx;
                    float dy = r - cy;
                
                    float angle = atan2f(dy, dx);
                    angle = angle < 0 ? angle + 2.0f * M_PI : angle;
                
                    int numBeams = 16;
                    float beamWidth = 2.0f * M_PI / numBeams;
                
                    float angularOffset = fmodf(framesCount * dt * 0.5f, 2.0f * M_PI);  // rotate over time
                    angle += angularOffset;
                
                    int beamIndex = (int)(angle / beamWidth);
                    if (beamIndex % 2 == 0) {
                        gl.alive = gl.next = true;
                        gl.color = SUN_YELLOW;
                    } else {
                        gl.alive = gl.next = false;
                        gl.color = URANUS_BLUE;
                    }
                    break;
                }
            case 13: {
                // int blockSize = 6;      // size of each square block
                // int band = 1;           // diagonal thickness
            
                int blockRow = r / blockSize;
                int blockCol = c / blockSize;
            
                int localR = r % blockSize;
                int localC = c % blockSize;
            
                // Diagonal type: choose one or alternate
                bool useForwardSlash = true;  // true = '/', false = '\'
            
                // Optional: alternate slashes like a checker
                // if ((blockRow + blockCol) % 2 == 0) useForwardSlash = true;
                // else useForwardSlash = false;
            
                bool isDiagonal = false;
            
                if (useForwardSlash) {
                    isDiagonal = abs(localR + localC - (blockSize - 1)) <= band;
                } else {
                    isDiagonal = abs(localR - localC) <= band;
                }
            
                if (isDiagonal) {
                    gl.alive = gl.next = true;
                    gl.color = ORANGE;
                } else {
                    gl.alive = gl.next = false;
                    gl.color = TAN;
                }
                break;
            }
        case 14: {
            // int blockSize = 6;  // size of each square
            // int border = 1;     // thickness of grid lines
            // int diagonalBand = 1;  // diagonal thickness
        
            int blockRow = r / blockSize;
            int blockCol = c / blockSize;
        
            int localR = r % blockSize;
            int localC = c % blockSize;
        
            bool isBorder = (localR < border || localR >= blockSize - border ||
                                localC < border || localC >= blockSize - border);
        
            // Diagonal type: '/' or '\' or alternating
            bool useForwardSlash = ((blockRow + blockCol) % 2 == 0);  // alternate per tile
        
            bool isDiagonal = false;
            if (useForwardSlash) {
                isDiagonal = abs(localR + localC - (blockSize - 1)) <= diagonalBand;
            } else {
                isDiagonal = abs(localR - localC) <= diagonalBand;
            }
        
            if (isBorder || isDiagonal) {
                gl.alive = gl.next = true;
                gl.color = make_uchar4(255, 200, 50, 255);  // gold-orange
            } else {
                gl.alive = gl.next = false;
                gl.color = make_uchar4(20, 20, 20, 255);  // dark background
            }
            break;
        }
        // case 15: { // rule 30
        //     int center = cols / 2;
        //     if (r == 0) {
        //         // First row: set single center cell alive
        //         gl.alive = gl.next = (c == center);
        //     } else {
        //         // Read previous row
        //         int rowAbove = r - 1;
        //         int idxLeft = rowAbove * cols + c - 1;
        //         int idxCenter = rowAbove * cols + c;
        //         int idxRight = rowAbove * cols + c + 1;
        
        //         bool left = (c > 0)              ? gamelife[idxLeft].alive   : false;
        //         bool mid  =                        gamelife[idxCenter].alive;
        //         bool right= (c < cols - 1)       ? gamelife[idxRight].alive  : false;
        
        //         // Apply Rule 30 logic:  new = left XOR (mid OR right)
        //         gl.alive = gl.next = left ^ (mid || right);
        //     }
        
        //     gl.color = gl.alive ? make_uchar4(255, 255, 255, 255) : make_uchar4(0, 0, 0, 255);
        //     break;
        // }
            case 15: {
                int center = cols / 2;
            
                // Expose rule via ImGui slider (external to this loop)
                // extern uint8_t rule;
            
                if (r == 0) {
                    // Seed: single dot at center
                    gl.alive = gl.next = (c == center);
                } else {
                    int rowAbove = r - 1;
                    int idxLeft = rowAbove * cols + c - 1;
                    int idxMid  = rowAbove * cols + c;
                    int idxRight= rowAbove * cols + c + 1;
            
                    bool left  = (c > 0)         ? gamelife[idxLeft].alive : false;
                    bool mid   = gamelife[idxMid].alive;
                    bool right = (c < cols - 1)  ? gamelife[idxRight].alive : false;
            
                    int pattern = (left << 2) | (mid << 1) | right;
                    gl.alive = gl.next = ((rule >> pattern) & 1);
                }
            
                gl.color = gl.alive ? WHITE : RED_MERCURY;
                break;
            }
                default: 
                    break;

            }
            
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
