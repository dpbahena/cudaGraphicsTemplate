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
   
    clearGraphicsDisply(surface);

    // draw samples to check ZOOM & PAN
    
    // drawCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 200, SUN_YELLOW, 1, 4, zoom, panX, panY);
    // drawGlowingCircle_kernel<<<1, 1>>>(surface, width, height, center.x, center.y, 500, RED_MERCURY, 1.5f, zoom, panX, panY);
    
    drawGlowingCircle(surface, center, 500, 1.5, RED_MERCURY );

    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);
}



void CUDAHandler::clearGraphicsDisply(cudaSurfaceObject_t &surface)
{
    int threads = 16; 
    dim3 clearBlock(threads, threads);
    dim3 clearGrid((width + clearBlock.x -1) / clearBlock.x, (height + clearBlock.y - 1) / clearBlock.y);
    clearSurface_kernel<<<clearGrid, clearBlock>>>(surface, width, height, BLUE_PLANET);
}

void CUDAHandler::drawGlowingCircle(cudaSurfaceObject_t &surface, vec2f position, float radius, float glowExtend, uchar4 color)
{
    // Calculate bounding box
    float glowRadius = glowExtend * radius;
    int xMin = max(0, (int)(position.x - glowRadius));
    int xMax = min(width - 1, (int)(position.x + glowRadius));
    int yMin = max(0, (int)(position.y - glowRadius));
    int yMax = min(height - 1, (int)(position.y + glowRadius));

    int drawWidth   = xMax - xMin + 1;
    int drawHeight  = yMax - yMin + 1;

    dim3 blockSize(16, 16);
    dim3 gridSize ((drawWidth + blockSize.x - 1) / blockSize.x, (drawHeight + blockSize.y -1) / blockSize.y);
    drawGlowingCircle_kernel<<<gridSize, blockSize>>>(surface, width, height, position.x, position.y, radius,  RED_MERCURY, 1.5f, xMin, yMin, zoom, panX, panY);
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
