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