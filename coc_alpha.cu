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
