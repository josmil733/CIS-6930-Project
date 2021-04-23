// Circle of Confusion Diameter and Alpha Weight Calculations for Scattering Algorithm

// This is the first optimization of the parallel algorithm.

/* TO RUN THIS PROGRAM:
	1) nvcc par1_coc_alpha.cu -o par1_coc_alpha
	2) ./par1_coc_alpha <input_file_name.ext> <output_file_name.ext>
	3) If you want to record the execution time of the program over several repetitions (as specified in the #define REPS directive), simply enter any string as the third command line argument. If you don't give a third command line argument, the program will not record its execution time.

*/


#include <stdio.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include "cudaCheck.cuh"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define WFF 36 // from Liu et. al, Equation 18
#define WSS 60 //no reference frame for this number, this is a random guess
//#define RAND_MAX 255
#define REPS 20

const int TILE_WIDTH = 16;



__global__ void getDepth(char *input_name, float depth[], const unsigned char* depthData, int *xPtr, int *yPtr){

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	float meters_cnvt = 0.588235; //8-bit representation to meters conversion 					// = 150/255
					// put in shared memory for optimization
	depth[ x*(*xPtr)+y ] = depthData[4*(x*(*xPtr)+y)]* meters_cnvt;//not coalesced; would need to switch roles of x and y, or something similar.

}



__global__ void coc_alpha(float* caPtr, char *output_name, int *xPtr, int *yPtr, float* depthMap, int ncols, float ap_diam, float f, float d_focus){

	float f_f = f*WFF/WSS;//can write this one and next line into shared memory as an optimization
	float F = f/ap_diam;

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;

	*(caPtr+2*(x*(*xPtr)+y) ) = abs(ncols*pow(WSS*f_f,2)*(d_focus-*(depthMap + x*(*xPtr)+y))/(WFF*F*(*(depthMap + x*(*xPtr)+y))*(d_focus*WFF - f_f*WSS)));
	*(caPtr+2*(x*(*xPtr)+y)+1) = 4/pow(*(caPtr+2*(x*(*xPtr)+y) ), 2);

}



/*__global__ void time(char *input_file, char *output_file, float* depthPtr, int* xPtr, int* yPtr){
	clock_t start[REPS], end[REPS];
	//clock_t start, end;
	clock_t total = 0; //counts total time
	for(int i=0; i < REPS; i++){
		start[i] = clock();

		depthPtr = getDepth(input_file, depthPtr, xPtr, yPtr);

		float* caPtr = (float *)malloc(sizeof(float)*2*(*xPtr)*(*yPtr));
		caPtr = coc_alpha(caPtr, output_file, xPtr, yPtr, depthPtr, 3, 2.5, 5.5, 4.5);
		end[i] = clock() - start[i];
		total += end[i];
	}

	printf("The program took an average of %.4f seconds per repetition, averaged across %d repetitions.\n", (float)total/(CLOCKS_PER_SEC*REPS), REPS);

}

*/

int main(int argc, char *argv[]){

	int *h_xPtr; int* h_yPtr;
	h_xPtr = (int *) malloc(sizeof(int));
	h_yPtr = (int *) malloc(sizeof(int));
	int n;

	unsigned char* h_depthData = stbi_load(argv[1], h_xPtr, h_yPtr, &n, 0);//must run on host first so that size of d_depthPtr can be determined before allocation
	
	int n_pixels = (*h_xPtr)*(*h_yPtr);
	unsigned char* h_depthPtr = (unsigned char*) malloc(sizeof(unsigned char)*n_pixels);
	
	dim3 dimGrid(ceil((float)(*h_xPtr)/(float)TILE_WIDTH), ceil((float)(*h_yPtr)/(float)TILE_WIDTH), 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	float *d_depthPtr;
	unsigned char* d_depthData;
	int *d_xPtr, *d_yPtr;
	
	cudaCheck( cudaMalloc((void**)&d_depthPtr, sizeof(float)*n_pixels) );
	cudaCheck( cudaMalloc((void**)&d_depthData, 4*sizeof(unsigned char)*n_pixels) );
	cudaCheck( cudaMalloc((void**)&d_xPtr, sizeof(int)) );
	cudaCheck( cudaMalloc((void**)&d_yPtr, sizeof(int)) );

	cudaMemcpy(d_xPtr, h_xPtr, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_yPtr, h_yPtr, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_depthData, h_depthData, 4*sizeof(unsigned char)*n_pixels, cudaMemcpyHostToDevice);

	getDepth<<<dimGrid,dimBlock>>>(argv[1], d_depthPtr, d_depthData, d_xPtr, d_yPtr);
	cudaError_t err2 = cudaGetLastError();
	if(err2 != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err2));
	
	cudaDeviceSynchronize();
	cudaMemcpy(h_depthPtr, d_depthPtr, sizeof(float)*n_pixels, cudaMemcpyDeviceToHost);
	cudaCheck( cudaMemcpy(h_depthPtr, d_depthPtr, sizeof(unsigned char)*n_pixels, cudaMemcpyDeviceToHost) );
	cudaError_t err3 = cudaGetLastError();
	if(err3 != cudaSuccess){
		 printf("Error: %s\n", cudaGetErrorString(err2));
		cudaFree(d_xPtr);
		cudaFree(d_yPtr);
		cudaFree(d_depthPtr);
		cudaFree(d_depthData);
	}

	float* h_caPtr = (float *)malloc(sizeof(float)*2*n_pixels);
	float* d_caPtr;
	cudaMalloc(&d_caPtr, sizeof(float)*2*n_pixels);
	coc_alpha<<<dimGrid,dimBlock>>>(d_caPtr, argv[2], d_xPtr, d_yPtr, d_depthPtr, 3, 2.5, 5.5, 4.5);
	cudaDeviceSynchronize();	
	cudaCheck( cudaMemcpy(h_caPtr, d_caPtr, sizeof(float)*2*n_pixels, cudaMemcpyDeviceToHost) );

	cudaFree(d_xPtr);
	cudaFree(d_yPtr);
	cudaFree(d_depthPtr);
	cudaFree(d_depthData);
	cudaFree(d_caPtr);

	return 1;
}
