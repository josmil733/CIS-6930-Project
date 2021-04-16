#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int TILE_WIDTH = 32;

struct Pixel
{
    int r, g, b, a;

    __host__ __device__ Pixel()
    {
        r, g, b, a = 0;
    }

    Pixel(unsigned char cr, unsigned char cg, unsigned char cb, unsigned char ca)
    {
        r = cr;
        g = cg;
        b = cb;
        a = ca;
    }
};

__global__ void createCocMap(Pixel colorMap[], int depthMap[], float output[], int numPixels)
{
    //Copy maps to shared memory
    __shared__ Pixel ds_ColorMap[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_DepthMap[TILE_WIDTH][TILE_WIDTH];

    int column = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int width = gridDim.x * TILE_WIDTH;
    int index = row * width + column;

    if (index < numPixels)
    {
        ds_ColorMap[threadIdx.y][threadIdx.x] = colorMap[index];
        ds_DepthMap[threadIdx.y][threadIdx.x] = depthMap[index];
    }

    __syncthreads();

    //Calculate blur factor
    float aperture = 0.15;
    float focalLength = 0.0886;
    float focalDistance = 40;
    float farClippingPlane = 150;
    float maxDiameter = 50;

    float realDepth = ((float)ds_DepthMap[threadIdx.y][threadIdx.x] / 255.0f) * farClippingPlane;
    float circleDiameter = aperture * (fabsf(realDepth - focalDistance) / realDepth) * (focalLength / (focalDistance - focalLength));
    float sensorHeight = 0.024f;
    float percentOfSensor = circleDiameter / sensorHeight;
    float blurFactor = percentOfSensor;
    if (blurFactor < 0.0f)
    {
        blurFactor = 0;
    }
    else if (blurFactor > maxDiameter)
    {
        blurFactor = maxDiameter;
    }

    //Assign blur to alpha of color map
    //Pixel blurPixel = ds_ColorMap[threadIdx.y][threadIdx.x];
    //blurPixel.a = blurFactor * 255;
    //ds_ColorMap[threadIdx.y][threadIdx.x] = blurPixel;

    //Output color map
    if (index < numPixels)
    {
        //outputColor[index] = ds_ColorMap[threadIdx.y][threadIdx.x];
        output[index] = blurFactor;
    }
}

__global__ void makeOffsets(float angle)
{

}

__global__ void depthOfField(Pixel colorMap[], int depthMap[], float cocMap[], float offsetData[], int numPixels, int numSamples)
{
    __shared__ Pixel ds_ColorMap[TILE_WIDTH][TILE_WIDTH];
    __shared__ int ds_DepthMap[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_CocMap[TILE_WIDTH][TILE_WIDTH];

    int column = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int width = gridDim.x * TILE_WIDTH;
    int index = row * width + column;

    if (index < numPixels)
    {
        ds_ColorMap[threadIdx.y][threadIdx.x] = colorMap[index];
        ds_DepthMap[threadIdx.y][threadIdx.x] = depthMap[index];
        ds_CocMap[threadIdx.y][threadIdx.x] = cocMap[index];
    }

    __syncthreads();

    float bleedingBias = 0.02f;
    float bleedingMult = 30.0f;

    Pixel centerPixel = ds_ColorMap[threadIdx.y][threadIdx.x];
    int centerDepth = ds_DepthMap[threadIdx.y][threadIdx.x];
    float centerCoc = ds_CocMap[threadIdx.y][threadIdx.x];

    Pixel color;
    float totalWeight;

    for (int i = 0; i < numSamples * 2; i += 2)
    {
        float offsetX = offsetData[i];
        float offsetY = offsetData[i + 1];

        //Pixel samplePixel =
        
    }
}

int main()
{
    int x, y, n;
    unsigned char* colorData = stbi_load("input_color.png", &x, &y, &n, 0);
    unsigned char* depthData = stbi_load("input_depth.png", &x, &y, &n, 0);

    Pixel* h_Color = new Pixel[x * y];
    int* h_Depth = new int[x * y];
    float* h_Coc = new float[x * y];

    size_t colorSize = sizeof(Pixel) * x * y;
    size_t depthSize = sizeof(int) * x * y;
    size_t cocSize = sizeof(float) * x * y;

    if (colorData != NULL)
    {
        printf("Width: %i\nHeight: %i\n", x, y);

        for (int i = 0; i < x * y * 4; i += 4)
        {
            Pixel pixel = Pixel(colorData[i], colorData[i + 1], colorData[i + 2], colorData[i + 3]);
            h_Color[i / 4] = pixel;
        }
        for (int i = 0; i < x * y * 4; i += 4)
        {
            h_Depth[i / 4] = depthData[i];
        }

        Pixel* d_Color;
        int* d_Depth;
        float* d_Coc;

        cudaMalloc((void**)&d_Color, colorSize);
        cudaMalloc((void**)&d_Depth, depthSize);
        cudaMalloc((void**)&d_Coc, cocSize);

        cudaMemcpy(d_Color, h_Color, colorSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Depth, h_Depth, depthSize, cudaMemcpyHostToDevice);

        dim3 dimGrid(ceil((float)x / (float)TILE_WIDTH), ceil((float)y / (float)TILE_WIDTH), 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

        createCocMap<<<dimGrid, dimBlock>>>(d_Color, d_Depth, d_Coc, x * y);

        cudaMemcpy(h_Coc, d_Coc, cocSize, cudaMemcpyDeviceToHost);

        cudaFree(d_Color);
        cudaFree(d_Depth);
        cudaFree(d_Coc);

        stbi_write_png("output.png", x, y, 4, colorData, sizeof(unsigned char) * x * 4);
    }

    return 0;
}