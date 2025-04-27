#include "CUDAHandler.h"
#include "cudaKernels.cuh"
#include "cuda_utils.h"




CUDAHandler* CUDAHandler::instance = nullptr;

CUDAHandler::CUDAHandler(int width, int height, GLuint textureID) :  width(width), height(height)
{
    cudaGraphicsGLRegisterImage(&cudaResource, textureID, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore);
    instance = this; // store global reference (to be used for mouse and imGui User Interface (UI) operations)
}

CUDAHandler::~CUDAHandler()
{
    cudaGraphicsUnregisterResource(cudaResource);
}

void CUDAHandler::updateDraw(float dt)
{
    this->dt = dt;

    

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

    // clear graphics
    int threads = 16; 
    dim3 clearBlock(threads, threads);
    dim3 clearGrid((width + clearBlock.x -1) / clearBlock.x, (height + clearBlock.y - 1) / clearBlock.y);
    clearSurface_kernel<<<clearGrid, clearBlock>>>(surface, width, height, BLUE_PLANET);

    drawCircle_kernel<<<1, 1>>>(surface, width, height, width/2, height/2, 200, SUN_YELLOW, 1, 4, zoom, panX, panY);

    drawGlowingCircle<<<1, 1>>>(surface, width, height, width / 2, height / 2, 50, RED_MERCURY, 1.5f, zoom, panX, panY);
    checkCuda(cudaPeekAtLastError());
    checkCuda(cudaDeviceSynchronize());

    cudaDestroySurfaceObject(surface);
    cudaGraphicsUnmapResources(1, &cudaResource);
}

