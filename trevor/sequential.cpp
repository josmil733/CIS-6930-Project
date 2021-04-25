#include <stdio.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <chrono>

const int NUM_SAMPLES = 128;

struct Pixel
{
    float r, g, b, a;

    Pixel()
    {
        r, g, b, a = 0;
    }

    Pixel(float fr, float fg, float fb, float fa)
    {
        r = fr;
        g = fg;
        b = fb;
        a = fa;
    }

    Pixel operator+(Pixel p)
    {
        Pixel pixel;
        pixel.r = this->r + p.r;
        pixel.g = this->g + p.g;
        pixel.b = this->b + p.b;
        pixel.a = this->a + p.a;
        return pixel;
    }

    Pixel operator*(float f)
    {
        Pixel pixel;
        pixel.r = this->r * f;
        pixel.g = this->g * f;
        pixel.b = this->b * f;
        pixel.a = this->a * f;
        return pixel;
    }

    Pixel operator/(float f)
    {
        Pixel pixel;
        pixel.r = this->r / f;
        pixel.g = this->g / f;
        pixel.b = this->b / f;
        pixel.a = this->a / f;
        return pixel;
    }
};

Pixel lerp(Pixel p1, Pixel p2, float t)
{
    Pixel output;
    output.r = p1.r + t * (p2.r - p1.r);
    output.g = p1.g + t * (p2.g - p1.g);
    output.b = p1.b + t * (p2.b - p1.b);
    output.a = p1.a + t * (p2.a - p1.a);
    return output;
}

float lerp(float v1, float v2, float t)
{
    return v1 + t * (v2 - v1);
}

float* makeOffsets(float angle, float width, float height)
{
    float* output = new float[NUM_SAMPLES * 2];

    float radius = 960.0f;

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

void createCocMap(Pixel colorMap[], float depthMap[], float output[], int numPixels)
{
    float aperture = 0.5906;
    float focalLength = 0.0886;
    float focalDistance = 40;
    float farClippingPlane = 150;
    float maxDiameter = 50;

    for (int i = 0; i < numPixels; i++)
    {
        float realDepth = depthMap[i];
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

        output[i] = blurFactor;
    }
}

void depthOfField(Pixel colorMap[], float depthMap[], float cocMap[], float offsetData[], Pixel outputMap[], int numPixels, int width, int height)
{
    for (int i = 0; i < numPixels; i++)
    {
        float bleedingBias = 0.02f;
        float bleedingMult = 30.0f;

        Pixel centerPixel = colorMap[i];
        int centerDepth = depthMap[i];
        float centerCoc = cocMap[i];

        Pixel color = Pixel(0, 0, 0, 0);
        float totalWeight = 0.0f;

        int column = i % width;
        int row = i / width;

        for (int j = 0; j < NUM_SAMPLES * 2; j += 2)
        {
            float y = row + offsetData[j + 1] * centerCoc;
            float x = column + offsetData[j] * centerCoc;
            int sampleIndex = floor(fmax(fmin(y, (float)height), 0.0f)) * width + floor(fmax(fmin(x, (float)width), 0.0f));

            Pixel samplePixel;
            int sampleDepth;
            float sampleCoc;
            if (sampleIndex < numPixels)
            {
                samplePixel = colorMap[sampleIndex];
                sampleDepth = depthMap[sampleIndex];
                sampleCoc = cocMap[sampleIndex];
            }
            else
            {
                samplePixel = Pixel(0, 0, 0, 0);
                sampleDepth = 0;
                sampleCoc = 0;
            }

            float weight = sampleDepth < centerDepth ? sampleCoc * bleedingMult : 1.0f;
            weight = (centerCoc > sampleCoc + bleedingBias) ? weight : 1.0f;
            weight = fmax(fmin(weight, 1.0f), 0.0f);

            color = color + samplePixel * weight;
            totalWeight += weight;
        }

        Pixel output = color / totalWeight;
        outputMap[i] = output;
    }
}

void finalPass(Pixel colorMapA[], Pixel colorMapB[], Pixel outputMap[], int numPixels, bool max)
{
    for (int i = 0; i < numPixels; i++)
    {
        Pixel pixelA = colorMapA[i];
        Pixel pixelB = colorMapB[i];
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
        }

        outputMap[i] = pixelO;
    }
}

int main()
{
    int x, y, n;
    float* colorData = stbi_loadf("input_color.hdr", &x, &y, &n, 0);
    float* depthData = stbi_loadf("input_depth.hdr", &x, &y, &n, 0);
    const int numPixels = x * y;

    Pixel* h_Color = new Pixel[numPixels];
    float* h_Depth = new float[numPixels];
    float* h_Coc = new float[numPixels];
    float* h_OffsetA = new float[NUM_SAMPLES * 2];
    float* h_OffsetB = new float[NUM_SAMPLES * 2];
    float* h_OffsetC = new float[NUM_SAMPLES * 2];
    float* h_OffsetD = new float[NUM_SAMPLES * 2];
    Pixel* h_ColorA = new Pixel[numPixels];
    Pixel* h_OutputA = new Pixel[numPixels];
    Pixel* h_OutputB = new Pixel[numPixels];
    Pixel* h_Final = new Pixel[numPixels];

    if (colorData != NULL)
    {
        printf("Width: %i\nHeight: %i\nN: %i\n", x, y, n);

        for (int i = 0; i < numPixels * 3; i += 3)
        {
            Pixel pixel = Pixel(colorData[i], colorData[i + 1], colorData[i + 2], 1.0f);
            h_Color[i / 3] = pixel;
        }
        for (int i = 0; i < numPixels * 3; i += 3)
        {
            h_Depth[i / 3] = depthData[i];
        }

        h_OffsetA = makeOffsets(0, x, y);
        h_OffsetB = makeOffsets(90, x, y);
        h_OffsetC = makeOffsets(45, x, y);
        h_OffsetD = makeOffsets(315, x, y);

        auto t1 = std::chrono::high_resolution_clock::now();

        createCocMap(h_Color, h_Depth, h_Coc, numPixels);

        depthOfField(h_Color, h_Depth, h_Coc, h_OffsetA, h_ColorA, numPixels, x, y);
        depthOfField(h_ColorA, h_Depth, h_Coc, h_OffsetB, h_OutputA, numPixels, x, y);

        depthOfField(h_Color, h_Depth, h_Coc, h_OffsetC, h_ColorA, numPixels, x, y);
        depthOfField(h_ColorA, h_Depth, h_Coc, h_OffsetD, h_OutputB, numPixels, x, y);

        finalPass(h_OutputA, h_OutputB, h_Final, numPixels, false);

        auto t2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;

        printf("%fms\n", ms_double);

        float* outputData = new float[numPixels * 3];
        for (int i = 0; i < numPixels; i++)
        {
            outputData[i * 3] = h_Final[i].r;
            outputData[i * 3 + 1] = h_Final[i].g;
            outputData[i * 3 + 2] = h_Final[i].b;
        }

        stbi_write_hdr("output.hdr", x, y, 3, outputData);
    }

    return 0;
}