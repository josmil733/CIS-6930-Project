#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

const int TILE_WIDTH = 32;
const int NUM_SAMPLES = 128;

struct Pixel
{
    int r, g, b, a;

    __host__ __device__ Pixel()
    {
        r, g, b, a = 0;
    }

    __host__ __device__ Pixel(unsigned char cr, unsigned char cg, unsigned char cb, unsigned char ca)
    {
        r = cr;
        g = cg;
        b = cb;
        a = ca;
    }

    __host__ __device__ Pixel operator+(Pixel p)
    {
        Pixel pixel;
        pixel.r = this->r + p.r;
        pixel.g = this->g + p.g;
        pixel.b = this->b + p.b;
        pixel.a = this->a + p.a;
        return pixel;
    }

    __host__ __device__ Pixel operator*(float f)
    {
        Pixel pixel;
        pixel.r = this->r * f;
        pixel.g = this->g * f;
        pixel.b = this->b * f;
        pixel.a = this->a * f;
        return pixel;
    }

    __host__ __device__ Pixel operator/(float f)
    {
        Pixel pixel;
        pixel.r = this->r / f;
        pixel.g = this->g / f;
        pixel.b = this->b / f;
        pixel.a = this->a / f;
        return pixel;
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

__host__ __device__ Pixel lerp(Pixel p1, Pixel p2, float t)
{
    Pixel output;
    output.r = p1.r + t * (p2.r - p1.r);
    output.g = p1.g + t * (p2.g - p1.g);
    output.b = p1.b + t * (p2.b - p1.b);
    output.a = p1.a + t * (p2.a - p1.a);
    return output;
}

__host__ __device__ float lerp(float v1, float v2, float t)
{
    return v1 + t * (v2 - v1);
}

float* makeOffsets(float angle, float width, float height)
{
    float* output = new float[NUM_SAMPLES * 2];

    float radius = 5000.0f;

    float radians = angle * 3.14159f / 180.0f;

    float ptX = float(radius * cos(radians));
    float ptY = float(radius * sin(radians));

    for (int i = 0; i < NUM_SAMPLES * 2; i += 2)
    {
        float t = i / ((float)NUM_SAMPLES * 2 - 1.0f);
        output[i] = lerp(-ptX, ptX, t);
        output[i + 1] = lerp(-ptY, ptY, t);
    }

    return output;
}

__global__ void depthOfField(Pixel colorMap[], int depthMap[], float cocMap[], float offsetData[], Pixel outputMap[], int numPixels, int width, int height)
{
    int column = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int index = row * width + column;

    if (index < numPixels)
    {
        float bleedingBias = 0.02f;
        float bleedingMult = 30.0f;

        Pixel centerPixel = colorMap[index];
        int centerDepth = depthMap[index];
        float centerCoc = cocMap[index];

        Pixel color = Pixel(0, 0, 0, 0);
        float totalWeight = 0.0f;

        for (int i = 0; i < NUM_SAMPLES * 2; i += 2)
        {
            //int minIndex = fmax(fmin(floor(row + offsetData[i + 1]) * width + floor(column + offsetData[i]), (float)numPixels - 1), 0.0f);
            //int maxIndex = fmax(fmin(ceil(row + offsetData[i + 1]) * width + ceil(column + offsetData[i]), (float)numPixels - 1), 0.0f);

            //Pixel samplePixel = lerp(colorMap[minIndex], colorMap[maxIndex], 0.5f);
            //int sampleDepth = lerp(depthMap[minIndex], depthMap[maxIndex], 0.5f);
            //float sampleCoc = lerp(cocMap[minIndex], cocMap[maxIndex], 0.5f);

            float y = row + offsetData[i + 1] * centerCoc;
            float x = column + offsetData[i] * centerCoc;
            int sampleIndex = floor(fmax(fmin(y, (float)height), 0.0f)) * width + floor(fmax(fmin(x, (float)width), 0.0f));
            //sampleIndex = fmax(fmin((float)sampleIndex, (float)numPixels), 0.0f);

            Pixel samplePixel = colorMap[sampleIndex];
            int sampleDepth = depthMap[sampleIndex];
            float sampleCoc = cocMap[sampleIndex];

            float weight = sampleDepth < centerDepth ? sampleCoc * bleedingMult : 1.0f;
            weight = (centerCoc > sampleCoc + bleedingBias) ? weight : 1.0f;
            weight = fmax(fmin(weight, 1.0f), 0.0f);

            color = color + samplePixel * weight;
            totalWeight += weight;
        }

        Pixel output = color / totalWeight;
        outputMap[index] = output;
    }    
}

__global__ void finalPass(Pixel colorMapA[], Pixel colorMapB[], Pixel outputMap[], int numPixels, bool max)
{
    __shared__ Pixel ds_ColorA[TILE_WIDTH][TILE_WIDTH];
    __shared__ Pixel ds_ColorB[TILE_WIDTH][TILE_WIDTH];
    __shared__ Pixel ds_Output[TILE_WIDTH][TILE_WIDTH];

    int column = blockIdx.x * TILE_WIDTH + threadIdx.x;
    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int width = gridDim.x * TILE_WIDTH;
    int index = row * width + column;

    if (index < numPixels)
    {
        ds_ColorA[threadIdx.y][threadIdx.x] = colorMapA[index];
        ds_ColorB[threadIdx.y][threadIdx.x] = colorMapB[index];

        Pixel pixelA = ds_ColorA[threadIdx.y][threadIdx.x];
        Pixel pixelB = ds_ColorB[threadIdx.y][threadIdx.x];
        Pixel pixelO = Pixel(0, 0, 0, 0);
        if (max)
        {
            pixelO.r = fmax((float)pixelA.r, (float)pixelB.r);
            pixelO.g = fmax((float)pixelA.g, (float)pixelB.g);
            pixelO.b = fmax((float)pixelA.b, (float)pixelB.b);
            pixelO.a = fmax((float)pixelA.a, (float)pixelB.a);
        }
        else
        {
            pixelO.r = fmin((float)pixelA.r, (float)pixelB.r);
            pixelO.g = fmin((float)pixelA.g, (float)pixelB.g);
            pixelO.b = fmin((float)pixelA.b, (float)pixelB.b);
            pixelO.a = fmin((float)pixelA.a, (float)pixelB.a);
            //pixelO.r = ((float)pixelA.r + (float)pixelB.r) / 2.0f;
            //pixelO.g = ((float)pixelA.g + (float)pixelB.g) / 2.0f;
            //pixelO.b = ((float)pixelA.b + (float)pixelB.b) / 2.0f;
            //pixelO.a = ((float)pixelA.a + (float)pixelB.a) / 2.0f;
        }        
        ds_Output[threadIdx.y][threadIdx.x] = pixelO;

        outputMap[index] = ds_Output[threadIdx.y][threadIdx.x];
    }
}

int main()
{
    int x, y, n;
    unsigned char* colorData = stbi_load("input_color.png", &x, &y, &n, 0);
    unsigned char* depthData = stbi_load("input_depth.png", &x, &y, &n, 0);
    const int numPixels = x * y;

    Pixel* h_Color = new Pixel[numPixels];
    int* h_Depth = new int[numPixels];
    float* h_OffsetA = new float[NUM_SAMPLES * 2];
    float* h_OffsetB = new float[NUM_SAMPLES * 2];
    float* h_OffsetC = new float[NUM_SAMPLES * 2];
    Pixel* h_Output = new Pixel[numPixels];

    size_t colorSize = sizeof(Pixel) * numPixels;
    size_t depthSize = sizeof(int) * numPixels;
    size_t cocSize = sizeof(float) * numPixels;
    size_t offsetSize = sizeof(float) * NUM_SAMPLES * 2;

    if (colorData != NULL)
    {
        printf("Width: %i\nHeight: %i\n", x, y);

        for (int i = 0; i < numPixels * 4; i += 4)
        {
            Pixel pixel = Pixel(colorData[i], colorData[i + 1], colorData[i + 2], colorData[i + 3]);
            h_Color[i / 4] = pixel;
        }
        for (int i = 0; i < numPixels * 4; i += 4)
        {
            h_Depth[i / 4] = depthData[i];
        }

        h_OffsetA = makeOffsets(0, x, y);
        h_OffsetB = makeOffsets(45, x, y);
        h_OffsetC = makeOffsets(-45, x, y);

        Pixel* d_Color;
        int* d_Depth;
        float* d_Coc;
        float* d_OffsetA;
        float* d_OffsetB;
        Pixel* d_ColorA;
        Pixel* d_ColorB;
        Pixel* d_OutputA;
        Pixel* d_OutputB;
        Pixel* d_Final;

        cudaMalloc((void**)&d_Color, colorSize);
        cudaMalloc((void**)&d_Depth, depthSize);
        cudaMalloc((void**)&d_Coc, cocSize);
        cudaMalloc((void**)&d_OffsetA, offsetSize);
        cudaMalloc((void**)&d_OffsetB, offsetSize);
        cudaMalloc((void**)&d_ColorA, colorSize);
        cudaMalloc((void**)&d_ColorB, colorSize);
        cudaMalloc((void**)&d_OutputA, colorSize);
        cudaMalloc((void**)&d_OutputB, colorSize);
        cudaMalloc((void**)&d_Final, colorSize);

        dim3 dimGrid(ceil((float)x / (float)TILE_WIDTH), ceil((float)y / (float)TILE_WIDTH), 1);
        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
        cudaMemcpy(d_Color, h_Color, colorSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_Depth, h_Depth, depthSize, cudaMemcpyHostToDevice);
        createCocMap<<<dimGrid, dimBlock>>>(d_Color, d_Depth, d_Coc, numPixels);


        cudaMemcpy(d_OffsetA, h_OffsetA, offsetSize, cudaMemcpyHostToDevice);
        depthOfField<<<dimGrid, dimBlock>>>(d_Color, d_Depth, d_Coc, d_OffsetA, d_ColorA, numPixels, x, y);
        cudaMemcpy(d_OffsetB, h_OffsetB, offsetSize, cudaMemcpyHostToDevice);
        depthOfField<<<dimGrid, dimBlock>>>(d_Color, d_Depth, d_Coc, d_OffsetB, d_ColorB, numPixels, x, y);
        finalPass<<<dimGrid, dimBlock>>>(d_ColorA, d_ColorB, d_OutputA, numPixels, false);

        cudaMemcpy(d_OffsetA, h_OffsetA, offsetSize, cudaMemcpyHostToDevice);
        depthOfField<<<dimGrid, dimBlock>>>(d_Color, d_Depth, d_Coc, d_OffsetA, d_ColorA, numPixels, x, y);
        cudaMemcpy(d_OffsetB, h_OffsetC, offsetSize, cudaMemcpyHostToDevice);
        depthOfField<<<dimGrid, dimBlock>>>(d_Color, d_Depth, d_Coc, d_OffsetB, d_ColorB, numPixels, x, y);
        finalPass<<<dimGrid, dimBlock>>>(d_ColorA, d_ColorB, d_OutputB, numPixels, false);

        finalPass<<<dimGrid, dimBlock>>>(d_OutputA, d_OutputB, d_Final, numPixels, false);
        cudaMemcpy(h_Output, d_Final, colorSize, cudaMemcpyDeviceToHost);

        cudaFree(d_Color);
        cudaFree(d_Depth);
        cudaFree(d_Coc);
        cudaFree(d_OffsetA);
        cudaFree(d_OffsetB);
        cudaFree(d_ColorA);
        cudaFree(d_ColorB);
        cudaFree(d_OutputA);
        cudaFree(d_OutputA);

        unsigned char* outputData = new unsigned char[numPixels * 4];
        for (int i = 0; i < numPixels; i++)
        {
            outputData[i * 4] = h_Output[i].r;
            outputData[i * 4 + 1] = h_Output[i].g;
            outputData[i * 4 + 2] = h_Output[i].b;
            outputData[i * 4 + 3] = h_Output[i].a;
        }

        stbi_write_png("output.png", x, y, 4, outputData, sizeof(unsigned char) * x * 4);
    }

    return 0;
}