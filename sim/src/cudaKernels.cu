#include "cudaKernels.cuh"
#include "helper_math.h"

__device__
void drawPixel(cudaSurfaceObject_t surface, int x, int y, uchar4 color, int width, int height)
{
    if (x >= 0 && x < width && y >= 0 && y < height) {
        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    }
}

__device__ void drawLine(cudaSurfaceObject_t surface, int x0, int y0, int x1, int y1, uchar4 color, int width, int height)
{
    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = -abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = dx + dy, e2;

    while (true) {
        drawPixel(surface, x0, y0, color, width, height);
        if (x0 == x1 && y0 == y1) break;
        e2 = 2 * err;
        if (e2 >= dy) { err += dy; x0 += sx; }
        if (e2 <= dx) { err += dx; y0 += sy; }
    }
}

__device__ void drawCircleOutline(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height) {
    const int segments = 36; // More segments = smoother circle
    for (int i = 0; i < segments; ++i) {
        float theta0 = (2.0f * M_PI * i) / segments;
        float theta1 = (2.0f * M_PI * (i + 1)) / segments;
        
        int x0 = cx + radius * cosf(theta0);
        int y0 = cy + radius * sinf(theta0);
        int x1 = cx + radius * cosf(theta1);
        int y1 = cy + radius * sinf(theta1);

        drawLine(surface, x0, y0, x1, y1, color, width, height);
    }
}

__device__ void drawRing(cudaSurfaceObject_t surface, float cx, float cy, float radius, float thickness, uchar4 color, int width, int height)
{
    int minX = max(0, int(cx - radius - thickness));
    int maxX = min(width - 1, int(cx + radius + thickness));
    int minY = max(0, int(cy - radius - thickness));
    int maxY = min(height - 1, int(cy + radius + thickness));

    for (int y = minY; y <= maxY; ++y) {
        for (int x = minX; x <= maxX; ++x) {
            float dx = x - cx;
            float dy = y - cy;
            float dist = sqrtf(dx * dx + dy * dy);
            if (fabsf(dist - radius) < thickness) {
                surf2Dwrite(color, surface, x * sizeof(uchar4), y);
            }
        }
    }
}


__device__ void drawFilledCircle(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height) {
    int rSquared = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= height) continue;

        for (int dx = -radius; dx <= radius; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= width) continue;

            if (dx * dx + dy * dy <= rSquared) {
                drawPixel(surface, x, y, color, width, height);
            }
        }
    }
}
__device__ uchar4 blend(uchar4 dest, uchar4 src) {
    float alpha = src.w / 255.0f;  // src alpha
    uchar4 result;
    result.x = (unsigned char) ((1.0f - alpha) * dest.x + alpha * src.x);
    result.y = (unsigned char) ((1.0f - alpha) * dest.y + alpha * src.y);
    result.z = (unsigned char) ((1.0f - alpha) * dest.z + alpha * src.z);

    result.w  = 255;  // keep opage or preserve dest.w if needed
    return result;
}
__device__ void drawBlendedFilledCircle(cudaSurfaceObject_t surface, int cx, int cy, int radius, uchar4 color, int width, int height) {
    int rSquared = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= height) continue;

        for (int dx = -radius; dx <= radius; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= width) continue;

            if (dx * dx + dy * dy <= rSquared) {
                uchar4 dst;
                surf2Dread(&dst, surface, x * sizeof(uchar4), y);
                uchar4 blended = blend(dst, color);

                drawPixel(surface, x, y, blended, width, height);
            }
        }
    }
}

__global__ void clearSurface_kernel(cudaSurfaceObject_t surface, int width, int height, uchar4 color) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    if (x >= width || y >= height) return;
    surf2Dwrite(color, surface, x * sizeof(uchar4), y);
}



__global__ void drawCircle_kernel(cudaSurfaceObject_t surface, int width, int height, int cx, int cy, int radius, uchar4 color, bool outline, int thickness, float zoom, float panX, float panY)
{
    cx = (int) (cx + panX) * zoom + width / 2.0f;
    cy = (int) (cy + panY) * zoom + height/ 2.0f;

    radius *= zoom;
    if (outline){ 
        if (thickness > 0) 
            drawRing(surface, cx, cy, radius, thickness, color, width, height);
        else 
            drawCircleOutline(surface, cx, cy, radius, color, width, height);
    } else
        drawBlendedFilledCircle(surface, cx, cy, radius, color, width, height);
}


__global__ void drawGlowingCircle_kernel(cudaSurfaceObject_t surface, int width, int height, int cx, int cy, int radius, uchar4 color, float glowExtent, float zoom, float panX, float panY) {

    cx = (int) (cx + panX) * zoom + width / 2.0f;
    cy = (int) (cy + panY) * zoom + height/ 2.0f;
    
    radius *= zoom;
    drawFilledCircle(surface, cx, cy, radius, color, width, height);

    int rSquared = radius * radius;
    float glowRadius = glowExtent * radius;
    float glowRadiusSquared = glowRadius * glowRadius;

    for (int dy = -glowRadius; dy <= glowRadius; ++dy) {
        int y = cy + dy;
        if (y < 0 || y >= height) continue;

        for (int dx = -glowRadius; dx <= glowRadius; ++dx) {
            int x = cx + dx;
            if (x < 0 || x >= width) continue;

            float distSquared = dx * dx + dy * dy;
            
            if (distSquared > rSquared && distSquared <= glowRadiusSquared) { // Only outside solid
                float intensity = 1.0f - (sqrtf(distSquared) - radius) / (glowRadius - radius);
                intensity = fmaxf(intensity, 0.0f);
                intensity = fminf(intensity, 1.0f);

                uchar4 outColor = make_uchar4(
                    min(255, (int)(color.x * intensity)),
                    min(255, (int)(color.y * intensity)),
                    min(255, (int)(color.z * intensity)),
                    color.w
                );

                // --- Blending with background ---
                uchar4 oldColor;
                surf2Dread(&oldColor, surface, x * sizeof(uchar4), y);

                uchar4 blendedColor = make_uchar4(
                    min(255, oldColor.x + outColor.x),
                    min(255, oldColor.y + outColor.y),
                    min(255, oldColor.z + outColor.z),
                    min(255, oldColor.w + outColor.w)
                );


                drawPixel(surface, x, y, blendedColor, width, height);
            }
        }
    }
}

__global__ void drawGlowingCircle_kernel(cudaSurfaceObject_t surface, int width, int height, int cx, int cy, int radius, uchar4 color, float glowExtent, float xMin, float yMin, float zoom, float panX, float panY)
{
    int local_x = threadIdx.x + blockIdx.x * blockDim.x;
    int local_y = threadIdx.y + blockIdx.y * blockDim.y;

    int x = xMin + local_x;
    int y = yMin + local_y;


    if (x >= width || y >= height) return;

    // Convert screen pixel (x,y) to world space
    float worldX = (x - width * 0.5f) / zoom - panX;
    float worldY = (y - height * 0.5f) / zoom - panY;
    

    float dx = worldX - cx;
    float dy = worldY - cy;

    
    float distSquared = dx * dx + dy * dy;
    // int rSquared = radius * radius;
    // float glowRadius = glowExtent * radius;
    // float glowRadiusSquared = glowRadius * glowRadius;
    float falloffPower = 2.f;

    // Instead of sqrt(distSquared), compare distSquared directly (faster than fsqrt)
    float glowRadiusSquared = (glowExtent * radius) * (glowExtent * radius);

    if (distSquared <= glowRadiusSquared) {
        float normalizedSquared = distSquared / glowRadiusSquared;
        // * Option 1 - Linear distance fade
        // float intensity = 1.0f - normalizedSquared; 

        // * Option 2 - quadratic distance fade (optional softer/harder)

        /** falloffPower = 1.0f → normal fade
            falloffPower = 2.0f → steeper fade  (fast fade)
            falloffPower = 0.5f → softer fade 
        */
        float intensity = powf(1.0f - normalizedSquared, falloffPower);
    
        intensity = fmaxf(intensity, 0.0f);
        intensity = fminf(intensity, 1.0f);
    
        uchar4 newColor = make_uchar4(
            min(255, (int)(color.x * intensity)),
            min(255, (int)(color.y * intensity)),
            min(255, (int)(color.z * intensity)),
            (unsigned char)(color.w * intensity)
        );
    
        uchar4 oldColor;
        surf2Dread(&oldColor, surface, x * sizeof(uchar4), y);
    
        uchar4 blendedColor = make_uchar4(
            min(255, oldColor.x + newColor.x),
            min(255, oldColor.y + newColor.y),
            min(255, oldColor.z + newColor.z),
            min(255, oldColor.w + newColor.w)
        );

        
        // Fade color toward background with slight blending
        // float blendFactor = 0.5f; // between 0 (overwrite) and 1 (fully additive)

        // uchar4 oldColor;
        // surf2Dread(&oldColor, surface, x * sizeof(uchar4), y);

        // uchar4 blendedColor = make_uchar4(
        //     min(255, (unsigned char)(oldColor.x * (1.0f - blendFactor) + newColor.x * blendFactor)),
        //     min(255, (unsigned char)(oldColor.y * (1.0f - blendFactor) + newColor.y * blendFactor)),
        //     min(255, (unsigned char)(oldColor.z * (1.0f - blendFactor) + newColor.z * blendFactor)),
        //     min(255, (unsigned char)(oldColor.w * (1.0f - blendFactor) + newColor.w * blendFactor))
        // );
    
        surf2Dwrite(blendedColor, surface, x * sizeof(uchar4), y);
        // surf2Dwrite(newColor, surface, x * sizeof(uchar4), y);
    }
    
}


__global__ void drawRing_kernel(cudaSurfaceObject_t surface, int width, int height, float centerX_world, float centerY_world, float radius, uchar4 color, float thickness, int xMin, int yMin, float zoom, float panX, float panY) 
{
    int local_x = threadIdx.x + blockIdx.x * blockDim.x;
    int local_y = threadIdx.y + blockIdx.y * blockDim.y;

    int x = xMin + local_x;
    int y = yMin + local_y;


    if (x >= width || y >= height) return;

    // Convert screen pixel (x,y) to world space coordinates
    float worldX = (x - width * 0.5f) / zoom - panX;
    float worldY = (y - height * 0.5f) / zoom - panY;
    // radius *= zoom;

    float dx = worldX - centerX_world;
    float dy = worldY - centerY_world;
    
    float dist = sqrtf(dx * dx + dy * dy);

    // Draw pixel if it falls within the ring (distance ± thickness)
    if (fabsf(dist - radius) < thickness) {
        surf2Dwrite(color, surface, x * sizeof(uchar4), y);
    }


}

/**
 * -- SHARED MEMORY -- in this case can reduce redundant calculations, especially if multiple pixels reuse the same values like zoom, panX, panY, or even values like centerX_world and centerY_world.
    But since shared memory is per block, the biggest gain would come from:
    Avoiding redundant math inside each block
    Optionally using it to cache small constant values shared across threads

    * Use shared memory to cache:

    centerX_world
    centerY_world
    radius
    thickness
    zoom, panX, panY
    These are constant across the whole block.
 */


 __global__ void drawRing_sharedMemory_kernel(cudaSurfaceObject_t surface,
    int width, int height,
    float centerX_world, float centerY_world, float radius,
    uchar4 color, float thickness,
    int xMin, int yMin,
    float zoom, float panX, float panY) {

        // Declare shared memory (one copy per block)
        __shared__ float s_centerX_world;
        __shared__ float s_centerY_world;
        __shared__ float s_radius;
        __shared__ float s_thickness;
        __shared__ float s_zoom;
        __shared__ float s_panX;
        __shared__ float s_panY;

        // Initialized shared memory by the first thread in the block
        if (threadIdx.x == 0 and threadIdx.y == 0) {
            s_centerX_world = centerX_world;
            s_centerY_world = centerY_world;
            s_radius = radius;
            s_thickness = thickness;
            s_zoom = zoom;
            s_panX = panX;
            s_panY = panY;
        }
        // Wait until all threads in the block to finish loading shared memory
        __syncthreads();

        int local_x = threadIdx.x + blockIdx.x * blockDim.x;
        int local_y = threadIdx.y + blockIdx.y * blockDim.y;

        int x = xMin + local_x;
        int y = yMin + local_y;


        if (x >= width || y >= height) return;

        // Convert screen pixel (x,y) to world space coordinates
        float worldX = (x - width * 0.5f) / s_zoom - s_panX;
        float worldY = (y - height * 0.5f) / s_zoom - s_panY;
        
        // Compute distance from current pixel to world center
        float dx = worldX - s_centerX_world;
        float dy = worldY - s_centerY_world;
        float dist = sqrtf(dx * dx + dy * dy);


        // Draw pixel if it falls within the ring (distance ± thickness)
        if (fabsf(dist - s_radius) < s_thickness) {
            surf2Dwrite(color, surface, x * sizeof(uchar4), y);
        }



    }
