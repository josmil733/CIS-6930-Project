// Circle of Confusion Diameter and Alpha Weight Calculations for Scattering Algorithm

// This is the sequential algorithm

/* TO RUN THIS PROGRAM:
	1) nvcc coc_alpha.cu -o coc_alpha
	2) ./coc_alpha <input_file_name.ext> <output_file_name.ext>

*/


#include <stdio.h>
#include <math.h>
#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WFF 36 // from Liu et. al, Equation 18
#define WSS 60 //no reference frame for this number, this is a random guess
//#define RAND_MAX 255

//FILE * read_Ptr;

int f_read_Image(FILE * readPtr){ //not used in current version. Just useful in order to verify successful file read.
	printf("Depth values from d_input.txt:\n\n");
	//while(feof(readPtr) != 1 && ){
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
	//unsigned char* colorData = stbi_load("input_color.png", &x, &y, &n, 0);
	unsigned char* depthData = stbi_load(input_name, xPtr, yPtr, &n, 0);

	float meters_cnvt = 0.588235; //8-bit representation to meters conversion 					// = 150/255

	depth = (float *) malloc(sizeof(float)*4*(*xPtr)*(*yPtr));
	if(depthData != NULL){
		for(int i=0; i < 4*(*xPtr)*(*yPtr); i+=4){
			*(depth + i/4) = depthData[i]* meters_cnvt;
		}
		/*for(int j=0; j < 60; j+=4){
			printf("%.2f  %.2f  %.2f  %.2f\n", *(depth + j), *(depth +j+1), *(depth + j+2),  *(depth+j+3));
		}
		printf("\n\n\n");*/
	}
	return depth;

}

int x, y; // x is number of rows of image array and y is number of columns of image array

void calculateBlur(float *caPtr, float image_Original[x][y][3])
{

    // float *caPtr: circle of confusion diameters and alpha weights for each pixel in the input image
    // float image_Original [x*y*3]: the RGB values from the input image

    float dist = 0; //distance between q and p

    int circle_set[x][y] = {0}; //array of 1 and 0s. Value of 1 means that the CoC of p impacts q; 0 otherwise.

    float omega_alpha = 0; //same thing that's defined in equation 23

    image_Bokeh[x][y][3] = {0}; //this will hold the final, blurred pixel rgbs.

    float CoCs[x][y] = {0}; // Circle of Confusion 2D array that matches the structure of the image

    for(int c=0; c < 2*x*y; c+=2) // organizes the CoC diameters of the output pixels into a 2D array that matches the structure of the image
    {
        CoCs[c / y][c % y] = *(caPtr +j);
    }

    float alphas[x][y] = {0}; // Alpha 2D array that matches the structure of the image

    for(int v=0; v < 2*x*y; v+=2) // organizes the Alpha values into a 2D array that matches the structure of the image
    {
        alphas[v / y][v % y] = *(caPtr +j +1);
    }

    for(int i=0; i < x; i++) // This for loop and the following for loop, loop over output pixels q. Each q is located at row i, column j in the output.
    {
        for(int j=0; j < y; j++)
        {
            for(int k=0; k < x; k++) // This for loop and the following for loop, loop over all other output pixels, which are located at k and l in the output.
            {
                for(int l=0; l < y; l++)
                {
                    dist = sqrt(pow(i-k,2) + pow(j-l,2)); //this is just the distance formula
                    
                    if( dist < CoC[k][l]) // If the distace is less than the diameter of circle of confusion then mark circle_set to 1
                    {
                        circle_set[k][l] = 1;
                    }
                }
            }

            for(int p=0; p < x; p++) // This block of code is responsible for calculating omega alpha utilizing equation 23
            {
                for(int q=0; q < y; q++)
                {
                    if(circle_set[p][q] == 1)
                    {
                        omega_alpha += alpha[p][q];
                    }
                }
            }
            
            for(int m=0; m < x; m++) // This for loop primarily calculates the output image pixel RGB band valures utilizing equation 23
            {
                for(int n=0; n < y; n++)
                {
                    if(circle_set[m][n] == 1)
                    {
                        image_Bokeh[m][n][0] += image_Original[m][n][0]*alpha[m][n];
                        image_Bokeh[m][n][1] += image_Original[m][n][1]*alpha[m][n];
                        image_Bokeh[m][n][2] += image_Original[m][n][2]*alpha[m][n];
                    }

                    image_Bokeh[m][n][0] = (1/omega_alpha)*(image_Bokeh[m][n][0]); //new R value for pixel q
                    image_Bokeh[m][n][1] = (1/omega_alpha)*(image_Bokeh[m][n][1]); //new G value for pixel q
                    image_Bokeh[m][n][2] = (1/omega_alpha)*(image_Bokeh[m][n][2]); //new B value for pixel q
                }
            }

            for(int r=0; r < x; r++) // This block of code is responsible for resetting circle_set to 0
            {
                for(int t=0; t < y; t++)
                {
                    circle_set[r][t] == 0;
                }
            }
        }
        
    }
}

/*int coc_alpha(float * depth, char *output_name, int ncols, float ap_diam, float f, float d_focus){

	//char out_file = *output_name;
	
	float f_f = f*WFF/WSS;
	float F = f/ap_diam;

	float coc_alpha[2] = {0};
	float depth = 1.00;
	printf("Writing CoCs and alphas to %s:\n\n", output_name);
	int i=0;
	while(fscanf(readPtr, "%f", &depth) ==1 ){
		i+=1;
		//fscanf(readPtr, "%f", &depth);
		coc_alpha[0] = abs(ncols*pow(WSS*f_f,2)*(d_focus-depth)/(WFF*F*depth*(d_focus*WFF - f_f*WSS)));
		coc_alpha[1] = 4/pow(coc_alpha[0], 2);
		fprintf(writePtr, "%f    %f\n", coc_alpha[0], coc_alpha[1]);
		printf("%f    %f\n",coc_alpha[0], coc_alpha[1]);
	}

	rewind(writePtr);
	rewind(readPtr);

	return 1;
}*/

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

int main(int argc, char * argv[]){
	
	float* depthPtr;
	int *xPtr; int* yPtr;
	xPtr = (int *) malloc(sizeof(int));
	yPtr = (int *) malloc(sizeof(int));

	depthPtr = getDepth(argv[1], depthPtr, xPtr, yPtr);
	//printf("%.2f  %.2f  %.2f  %d\n\n", *(depthPtr), *(depthPtr+1), *(depthPtr+2), *xPtr);

	float* caPtr = (float *)malloc(sizeof(float)*2*(*xPtr)*(*yPtr));
	caPtr = coc_alpha(caPtr, argv[2], xPtr, yPtr, depthPtr, 3, 2.5, 5.5, 4.5);
	//printf("\n%.2f   %.2f\n", *(caPtr), *(caPtr+1));

	return 1;
}