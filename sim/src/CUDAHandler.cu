#include "CUDAHandler.h"
#include "cudaKernels.cuh"
#include "cuda_utils.h"

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

    static int previousOption = option;
    bool optionJustChaged = (option != previousOption);
    previousOption = option;

    static float previousWidthFactor = widthFactor;
    bool widthFactorJustChaged = (widthFactor != previousWidthFactor);
    previousWidthFactor = widthFactor;

    if (gamelife.empty() || optionJustChaged || (widthFactorJustChaged && option > 0)) {
        framesCount = 0;
        initGameLife();
    } 
    

    // GameLife* d_gameLife;
    // checkCuda(cudaMalloc(&d_gameLife, gamelife.size() * sizeof(GameLife)));
    // checkCuda(cudaMemcpy(d_gameLife, gamelife.data(), gamelife.size() * sizeof(GameLife), cudaMemcpyHostToDevice));
    
    
    if(startSimulation) activateGameLife(d_gameLife);
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

    int threads = 128;
    int blocks = (numberOfParticles + threads - 1) / threads;
    commitNextState_kernel<<<blocks, threads>>> (d_gameLife, gamelife.size());
    checkCuda(cudaDeviceSynchronize());
    activate_gameOfLife_kernel<<<blocks, threads>>>(d_gameLife, gamelife.size(), gridRows, gridCols);
}

void CUDAHandler::initGameLife()
{
    
    gamelife.clear();
    startSimulation = false;
    setGroupOfParticles(numberOfParticles, {16, 9}, 1);
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
    int rows = grid.x;
    int cols = grid.y;

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
                case 0: 
                    gl.alive = gl.next = randomBool();
                    gl.color = RED_MERCURY;
                    break;
                case 1:
                    if ((c / colsSize) % 2 == 0) {
                        gl.alive = gl.next = true;      // cell is ON
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 2:
                    if ((r / rowsSize) % 2 == 0) {
                        gl.alive = gl.next = true;      // cell is ON  // horizontal
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
                case 3:
                    
                    if ((r / rowsSize) % 2 == 0 && c / colsSize % 2 == 0) {  // checkered
                        gl.alive = gl.next = true;      // cell is ON
                        gl.color = GREEN;
                    } else {
                        gl.alive = gl.next = false;     // cell is OFF
                        gl.color = GOLD;
                    }
                    break;
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
