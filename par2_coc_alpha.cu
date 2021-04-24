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
#define REPS 1000

const int TILE_WIDTH = 16;

__constant__ int xdim;
__constant__ int ydim;

//__constant__ float meters_cnvt = 0.588235; //8-bit representation to meters conversion 					// = 150/255

//__constant__ float f_f = f*WFF/WSS;
//__constant__ float F = f/ap_diam; //<- declared in device const memory, allocated to kernels in main. Optimization 1.





__global__ void getDepth(char *input_name, float depth[], const unsigned char* depthData){

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;
	float meters_cnvt = 0.588235;

	if(x < xdim && y < ydim){
	depth[ y*xdim+x ] = depthData[4*(y*xdim+x)]* meters_cnvt;//not coalesced; would need to switch roles of x and y, or something similar.
	}

}



__global__ void coc_alpha(float* caPtr, float* depthMap, int ncols, float ap_diam, float f, float d_focus){

	int x = threadIdx.x + blockDim.x*blockIdx.x;
	int y = threadIdx.y + blockDim.y*blockIdx.y;

	/*__shared__ float f_f = f*WFF/WSS;
	__shared__ float F = f/ap_diam;*/
	// compile error: cannot initialize a__shared__ variable.

	float f_f = f*WFF*WSS;
	float F = f/ap_diam;

	if(x < xdim && y < ydim){
		*(caPtr+2*(y*xdim+x) ) = abs(ncols*pow(WSS*f_f,2)*(d_focus-*(depthMap + y*xdim+x))/(WFF*F*(*(depthMap + y*xdim+x))*(d_focus*WFF - f_f*WSS)));
		*(caPtr+2*(y*xdim+x)+1) = 4/pow(*(caPtr+2*(y*xdim+x) ), 2);
	}

}



void time(char *input_file, float* depthPtr, const unsigned char* depthData, int* h_xPtr, int* h_yPtr, float* caPtr){

	int grid_x = ceil((float)(*h_xPtr)/(float)TILE_WIDTH);
	int grid_y = ceil((float)(*h_yPtr)/(float)TILE_WIDTH);
	dim3 dimGrid(grid_x, grid_y, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	
	clock_t start[REPS], end[REPS];
	//clock_t start, end;
	clock_t total = 0; //counts total time
	for(int i=0; i < REPS; i++){
		start[i] = clock();

		getDepth<<<dimGrid,dimBlock>>>(input_file, depthPtr, depthData);

		coc_alpha<<<dimGrid,dimBlock>>>(caPtr, depthPtr, 3, 2.5, 5.5, 4.5);

		end[i] = clock() - start[i];
		total += end[i];
	}

	printf("Total clock counts: %d\n\nExpression denominator: %d\n\nThe program took an average of %.7f seconds per repetition, averaged across %d repetitions.\n", total, CLOCKS_PER_SEC*REPS, (float)total/(float)(CLOCKS_PER_SEC*REPS), REPS);

}



int main(int argc, char *argv[]){

	int *h_xPtr; int* h_yPtr;
	h_xPtr = (int *) malloc(sizeof(int));
	h_yPtr = (int *) malloc(sizeof(int));
	int n;

	unsigned char* h_depthData = stbi_load(argv[1], h_xPtr, h_yPtr, &n, 0);//must run on host first so that size of d_depthPtr can be determined before allocation
	
	int n_pixels = (*h_xPtr)*(*h_yPtr);
	float* h_depthPtr = (float*) malloc(sizeof(float)*n_pixels);

	int grid_x = ceil((float)(*h_xPtr)/(float)TILE_WIDTH);
	int grid_y = ceil((float)(*h_yPtr)/(float)TILE_WIDTH);
	dim3 dimGrid(grid_x, grid_y, 1);
	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
	float *d_depthPtr;
	unsigned char* d_depthData;

	cudaCheck( cudaMalloc(&d_depthPtr, sizeof(float)*n_pixels) );
	cudaCheck( cudaMalloc(&d_depthData, 4*sizeof(unsigned char)*n_pixels) );
	
	cudaMemcpyToSymbol(xdim, h_xPtr, sizeof(int));
	cudaMemcpyToSymbol(ydim, h_yPtr, sizeof(int));

	cudaMemcpy(d_depthData, h_depthData, 4*sizeof(unsigned char)*n_pixels, cudaMemcpyHostToDevice);

	//cudaMemcpytoSymbol(

	float* h_caPtr = (float *)malloc(sizeof(float)*2*n_pixels);
	float* d_caPtr;
	cudaMalloc(&d_caPtr, sizeof(float)*2*n_pixels);




	if(argv[2] != NULL){
		time(argv[1], d_depthPtr, d_depthData, h_xPtr, h_yPtr, d_caPtr);
		cudaDeviceSynchronize();
		exit(0);
	}

	
	getDepth<<<dimGrid,dimBlock>>>(argv[1], d_depthPtr, d_depthData);
	/*cudaError_t err2 = cudaGetLastError();
	if(err2 != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err2));*/
	
	cudaDeviceSynchronize();
	cudaMemcpy(h_depthPtr, d_depthPtr, sizeof(float)*n_pixels, cudaMemcpyDeviceToHost);
	
	coc_alpha<<<dimGrid,dimBlock>>>(d_caPtr, d_depthPtr, 3, 2.5, 5.5, 4.5);
	cudaDeviceSynchronize();	
	cudaCheck( cudaMemcpy(h_caPtr, d_caPtr, sizeof(float)*2*n_pixels, cudaMemcpyDeviceToHost) );
	
	printf("%.2f  %.4f\n%.2f  %.4f\n", *h_caPtr,*(h_caPtr+1),*(h_caPtr+2),*(h_caPtr+3));

	cudaFree(d_depthPtr);
	cudaFree(d_depthData);
	cudaFree(d_caPtr);

	return 1;
}
