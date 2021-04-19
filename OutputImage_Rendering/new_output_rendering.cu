void calculateBlur(float *caPtr, float image_Original[4*x*y]){

    // float CoC[x*y]: circle of confusion diameters. x = # rows of pixels; y = # columns of pixels
    // float alpha[x*y]: alpha weights for each pixel in the input image
    // float image_Original [4*x*y]: the RGB values from the input image
    
    float dist = 0; //distance between q and p
    int circle_set[4*x*y] = {0}; //array of 1 and 0s. Value of 1 means that the CoC of p impacts q; 0 otherwise.
    
    image_Bokeh[4*x*y] = {0}; //this will hold the final, blurred pixel rgbs.
    
    for(int pix=0; pix < 4*x*y; pix += 4) //loops over output pixels q. Each q is located at row i, column j in the output. += 3 because each pixel contains an R,G,B.
    { 
        for(int p=0; p < 4*x*y; p += 4) // loops over all other output pixels, which are located at i' and j' in the output.
        {    
            int i = pix/y; 
            int i_prime = p/y; 
            int j = pix % y; 
            int j_prime = p % y; //calculates the indices of each output pixel

            dist = sqrt(pow(i-i_prime,2) + pow(j-j_prime,2)); //this is just the distance formula
            
            if(p==0)
            {
                if(dist < *(caPtr+p)
                {
                    circle_set[p] = 1;
                }
            }
            else
            {
                if(dist < *(caPtr+p-2)
                {
                    circle_set[p] = 1;
                }
            }
        }

        float omega_alpha = 0; //same thing that's defined in equation 23
        for(int p=0; p < 4*x*y, p += 4 )
        {
            if(p==0)
            {
                if(circle_set[p] == 1)
                {   
                    omega_alpha += *caPtr(p+1);
                }
            }
            else
            {
                if(circle_set[p] == 1)
                {   
                    omega_alpha += *caPtr(p-1);
                }
            }
        }
    
        for(int p=0; p < 4*x*y; p += 4)
        {
            if(p==0)
            {
                if(circle_set[p] == 1)
                {
                    image_Bokeh[pix] += image_Original[pix] * caPtr(p+1)     //new R value for pixel q
                    image_Bokeh[pix+1] += image_Original[pix+1] * caPtr(p+1) //new G value for pixel q
                    image_Bokeh[pix+2] += image_Original[pix+2] * caPtr(p+1) //new B value for pixel q
                }
            }
            else
            {
                if(circle_set[p] == 1)
                {
                    image_Bokeh[pix] += image_Original[pix] * caPtr(p-1)     //new R value for pixel q
                    image_Bokeh[pix+1] += image_Original[pix+1] * caPtr(p-1) //new G value for pixel q
                    image_Bokeh[pix+2] += image_Original[pix+2] * caPtr(p-1) //new B value for pixel q
                }
            }
        }
    }
}