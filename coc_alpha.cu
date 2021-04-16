// Circle of Confusion Diameter and Alpha Weight Calculations for Scattering Algorithm

// This is the sequential algorithm


#include <stdio.h>
#include <math.h>
#include <iostream>

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

int coc_alpha(FILE *writePtr, FILE *readPtr, int ncols, float ap_diam, float f, float d_focus){
	float f_f = f*WFF/WSS;
	float F = f/ap_diam;

	float coc_alpha[2] = {0};
	float depth = 1.00;
	printf("Writing CoCs and alphas to output.txt:\n\n");
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
}

int main(){
	
	FILE * read_Ptr = fopen("d_input.txt", "r");
	//f_read_Image(read_Ptr);

	FILE * write_Ptr = fopen("output.txt", "w");
	coc_alpha(write_Ptr, read_Ptr, 3, 2.5, 5.5, 4.5);
	fclose(read_Ptr);
	fclose(write_Ptr);
	return 1;
}
