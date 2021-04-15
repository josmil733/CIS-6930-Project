// Circle of Confusion Diameter and Alpha Weight Calculations for Scattering Algorithm

// This is the sequential algorithm


#include <stdio.h>
#include <math.h>
#include <iostream>

#define NPIXELS_X 4
#define NPIXELS_Y 4
#define DIM 4

#define WFF 36 // from Liu et. al, Equation 18
#define WSS 60 //no reference frame for this number, this is a random guess
//#define RAND_MAX 255



int read_Image(int* picPtr){ //reads depth map and rgb values
	//int pic[NPIXELS][DIM];
	for(size_t i=0; i < NPIXELS_X*NPIXELS_Y; i++){
		for(size_t j=0; j < DIM-1; j++){
			//pic[i][j] = rand() % 255;
			*(picPtr +i*DIM + j) = rand() % 255;
			printf("%d   ", *(picPtr +i*DIM + j));
		}
	//pic[i][DIM-1] = rand() % 10;
	*(picPtr + i*DIM + DIM-1) = rand() % 10;
	printf("%d   \n", *(picPtr +i*DIM + DIM-1));
	}
	
	/*for(int i=0; i < NPIXELS; i++){
		printf("%d   %d   %d   %d\n", pic[i][0], pic[i][1], pic[i][2], pic[i][3]);
	}
*/
//	return pic[][];
	//return pic;

}

void coc_alpha(float* coc_alphaPtr, int image[NPIXELS_X*NPIXELS_Y], float ap_diam, float f, float d_focus){
	//image is meant to be a 1D array with depth values
	float f_f = f*WFF/WSS;
	float F = f/ap_diam;
	//float phi[NPIXELS_X*NPIXELS_Y] = {0};
	//float alpha[NPIXELS_X*NPIXELS_Y] = {0};
	printf("COC diameters | alpha weights:\n\n");
	for(size_t i=0; i < NPIXELS_X*NPIXELS_Y; i++){
		*(coc_alphaPtr +2*i) = abs(NPIXELS_X*pow(WSS*f_f,2)*(d_focus-image[i])/(WFF*F*image[i]*(d_focus*WFF - f_f*WSS)));
		*(coc_alphaPtr +2*i +1) = 4/pow(*(coc_alphaPtr + 2*i), 2);
		printf("%.2f   %.2f\n", *(coc_alphaPtr + 2*i), *(coc_alphaPtr + 2*i + 1) );
	}
}

int main(){
	int* picPtr = (int*) malloc(NPIXELS_X*NPIXELS_Y*DIM*sizeof(int));
	int pic[NPIXELS_X*NPIXELS_Y][DIM] = {0};
	picPtr = &pic[0][0];
	//printf("%d\n", *pic);
	read_Image(picPtr);
	//printf("\n\n\n%d   %d   %d   %d\n", *(picPtr), *(picPtr+1), *(picPtr+2), *(picPtr+3));
	float* coc_alphaPtr = (float*) malloc(NPIXELS_X*NPIXELS_Y*2*sizeof(double));
	float Coc_alpha[NPIXELS_X*NPIXELS_Y][2] = {0};
	coc_alphaPtr = &Coc_alpha[0][0];
	int depth[NPIXELS_X*NPIXELS_Y] = {0};
	for(int i=0; i < NPIXELS_X*NPIXELS_Y; i++){
		depth[i] = pic[i][3];
}
	printf("%d   %d   %d\n", depth[0], depth[1], depth[2]);
	coc_alpha(coc_alphaPtr, depth, 2.5,5.5,4.5);
}
