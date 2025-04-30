#include "CUDAHandler.h"
#include "cudaKernels.cuh"
#include "cuda_utils.h"




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

void CUDAHandler::updateDraw(float dt)
{
    this->dt = dt;

    cudaSurfaceObject_t surface = MapSurfaceResouse(); 
   
    clearGraphicsDisply(surface, NEPTUNE_PURPLE);

    // draw samples to check ZOOM & PAN
    
    // drawCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 200, SUN_YELLOW, 1, 4, zoom, panX, panY);
    // drawGlowingCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 500, RED_MERCURY, 1.5f, zoom, panX, panY);
    drawRing(surface, center, 500, 4, BLUE_PLANET);
    
    drawGlowingCircle(surface, center, 500, 1.5, RED_MERCURY );

    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);
}



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
