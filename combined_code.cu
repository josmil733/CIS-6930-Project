// Circle of Confusion Diameter and Alpha Weight Calculations for Scattering Algorithm

// This is the sequential algorithm

/* TO RUN THIS PROGRAM:
    1) In combined_code.cu, adjust the input image file name and output image file name
	1) nvcc combined_code.cu -o combined_code
	2) ./combined_code

*/


#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WFF 36 // from Liu et. al, Equation 18
#define WSS 60 //no reference frame for this number, this is a random guess

int f_read_Image(FILE * readPtr)
{   //not used in current version. Just useful in order to verify successful file read.
	printf("Depth values from d_input.txt:\n\n");

	int c = getc(readPtr);
	while(c != EOF){
		putchar(c);
		c = getc(readPtr);
	}
	printf("\n");
	rewind(readPtr);
	return 1;
}

float* getDepth(char *input_name, float* depth, int *xPtr, int *yPtr){
	int n;

	unsigned char* depthData = stbi_load(input_name, xPtr, yPtr, &n, 0);

	float meters_cnvt = 0.588235; //8-bit representation to meters conversion 					// = 150/255

	depth = (float *) malloc(sizeof(float)*4*(*xPtr)*(*yPtr));
	if(depthData != NULL){
		for(int i=0; i < 4*(*xPtr)*(*yPtr); i+=4){
			*(depth + i/4) = depthData[i]* meters_cnvt;
		}
		
	}
	return depth;

}

void calculateBlur(float *caPtr, unsigned char* image_Original, int *xPtr1, int *yPtr1, unsigned char* image_Bokeh)
{    
    float dist = 0; //distance between q and p
    int *circle_set = (int *) malloc(sizeof(int)*4*( *xPtr1)*( *yPtr1)); //array of 1 and 0s. Value of 1 means that the CoC of p impacts q; 0 otherwise.
    
    for(int n=0;n<4*( *xPtr1)*( *yPtr1);n++) 
    {
        circle_set[n] = 0;
    }   
   
    image_Bokeh = (unsigned char *) malloc(sizeof(unsigned char)*3*( *xPtr1)*( *yPtr1)); //this will hold the final, blurred pixel rgbs.
    
    for(int pix=0; pix < (4 * (*xPtr1) * (*yPtr1)); pix += 4) //loops over output pixels q. Each q is located at row i, column j in the output. += 3 because each pixel contains an R,G,B.
    { 
        for(int p=0; p < (4 * (*xPtr1) * (*yPtr1)); p += 4) // loops over all other output pixels, which are located at i' and j' in the output.
        {    
            int i = pix/(*yPtr1); 
            int i_prime = p/(*yPtr1); 
            int j = pix % (*yPtr1); 
            int j_prime = p % (*yPtr1); //calculates the indices of each output pixel

            dist = sqrt(pow(i-i_prime,2) + pow(j-j_prime,2)); //this is just the distance formula

            if(caPtr != NULL)
            { 
                if(p==0)
                {
                    if(dist < *(caPtr+p))
                    {
                        *(circle_set+p) = 1;
                    }
                    else
                    {
                        *(circle_set+p) = 0;
                    }
                }
                else
                {
                    if((p/4)%2 == 0)
                    {
                        if(dist < *(caPtr+p/4))
                        {
                            *(circle_set+p) = 1;
                        }
                    }
                    else
                    { 
                        if(dist < *(caPtr+(p/4)))
                        {
                            *(circle_set+p) = 1;
                        }
                    }
                }
            }
        }

        float omega_alpha = 0; //same thing that's defined in equation 23
        for(int p=0; p < (4 * (*xPtr1) * (*yPtr1)); p += 4 )
        {
            if(caPtr != NULL)
            {
            if(p==0)
            {
                if(circle_set[p] == 1)
                {   
                    omega_alpha += *(caPtr+p+1);
                }
            }
            else
            {
                if((p/4)%2 == 0)
                {
                    if(circle_set[p] == 1)
                    {   
                        omega_alpha += *(caPtr+(p/4)+1);
                    }
                }
                else
                {
                    if(circle_set[p] == 1)
                    {
                        omega_alpha += *(caPtr+(p/4)+1);
                    }
                }   
            }
            }
        }

        for(int p=0; p < (3 * (*xPtr1) * (*yPtr1)); p += 3)
        {
            if(p==0)
            {
                if(circle_set[p] == 1)
                {
                    image_Bokeh[pix] += image_Original[pix] * (*(caPtr+p+1));     //new R value for pixel q
                    image_Bokeh[pix+1] += image_Original[pix+1] * (*caPtr+p+1); //new G value for pixel q
                    image_Bokeh[pix+2] += image_Original[pix+2] * (*caPtr+p+1); //new B value for pixel q
                }
            }
            else
            {
                if(circle_set[p+1] == 1)
                {
                    image_Bokeh[pix] += image_Original[pix] * (*caPtr+p-1);     //new R value for pixel q
                    image_Bokeh[pix+1] += image_Original[pix+1] * (*caPtr+p-1); //new G value for pixel q
                    image_Bokeh[pix+2] += image_Original[pix+2] * (*caPtr+p-1); //new B value for pixel q
                }
            }
        }
    }
}

float* coc_alpha(float* caPtr, char *output_name, int *xPtr, int *yPtr, float* depthMap, int ncols, float ap_diam, float f, float d_focus){

	FILE * writePtr = fopen(output_name, "w");
	
	float f_f = f*WFF/WSS;
	float F = f/ap_diam;

	printf("Writing CoCs and alphas to %s:\n\n", output_name);

	for(int i=0; i < *(xPtr)*(*yPtr); i++){
		*(caPtr+2*i) = abs(ncols*pow(WSS*f_f,2)*(d_focus-depthMap[i])/(WFF*F*depthMap[i]*(d_focus*WFF - f_f*WSS)));
		*(caPtr+2*i+1) = 4/pow(*(caPtr+2*i), 2);
		fprintf(writePtr, "%f    %f\n", *(caPtr+2*i), *(caPtr+2*i+1));
        }

	fclose(writePtr);
        
	return caPtr;
}

int main(int argc, char * argv[])
{
	float* depthPtr;
	int *xPtr; int* yPtr;
	xPtr = (int *) malloc(sizeof(int));
	yPtr = (int *) malloc(sizeof(int));

	
    depthPtr = getDepth("plant_image.png", depthPtr, xPtr, yPtr);

    int *xPtr1, *yPtr1, n;
    xPtr1 = (int *) malloc(sizeof(int));
    yPtr1 = (int *) malloc(sizeof(int));

    unsigned char* original_image = stbi_load("plant_image.png", xPtr1, yPtr1, &n, 0);

	float* caPtr = (float *)malloc(sizeof(float)*2*(*xPtr)*(*yPtr));
	caPtr = coc_alpha(caPtr, "output_image.png", xPtr, yPtr, depthPtr, 3, 2.5, 5.5, 4.5);

        unsigned char* output_image;

        calculateBlur(caPtr, original_image, xPtr1, yPtr1, output_image);
        
        stbi_write_png("output_image.png", *xPtr1, *yPtr1, 3, output_image, (*xPtr1)*3);	

        return 1;
}
