import os
import numpy as np
import matplotlib.pyplot as plt

class cannyEdgeDetector:
    def __init__(self, img_name,img, sigma=1, kernel_size=3, lowthreshold=50, highthreshold=80):
        self.img_name = img_name
        self.img = img
        self.imgs_final = []
        self.sigma = sigma
        self.kernel_size = kernel_size
        self.Ix_prime = None
        self.Iy_prime = None
        self.lowThreshold = lowthreshold
        self.highThreshold = highthreshold
        self.nms_thinned_output = None
        return 
    
    def gaussian(self):
        '''
         1D gaussian filter 
         '''
        size = int(self.kernel_size) // 2
        x = np.arange(-size,size+1,1)
        normal = 1 / (np.sqrt(2.0*np.pi)*self.sigma)
        g = np.exp(-(x*x)/(2*self.sigma*self.sigma))*normal
        return g

    def derivative_gaussian(self):
        '''
         1D gaussian derivate filter
         '''
        size = int(self.kernel_size) // 2
        x = np.arange(-size,size+1,1)
        normal = -x / ((np.sqrt(2.0*np.pi) * self.sigma * self.sigma * self.sigma))
        g = np.exp(-(x*x)/(2*self.sigma*self.sigma))*normal
        return g

    def convolution(self,image, kernel):
        """
        This function which takes an image and a kernel and returns the convolution of them
        """
        # convolution output
        size = int(np.max(kernel.shape)/2)
        output = np.zeros_like(image)
        image_padded = np.zeros((image.shape[0] + 2*size, image.shape[1] + 2*size))
        padsize = 2*(size)-1
        image_padded[1:-padsize, 1:-padsize] = image
        # Loop over every pixel of the image
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                # element-wise multiplication of the kernel and the image
                output[y, x]=(kernel * image_padded[y: (y+kernel.shape[0]), x:(x+kernel.shape[1]) ]).sum()

        return output

    def min_max_normalize(self,image):
        '''
        Normalizing the Image to the scale of [0 - 255]
        '''
        low = image.min()
        high = image.max()
        image = (image-low) * 1.0 / (high - low)
        return image * 255

    def magnitude_angle(self):
        '''
        Calculating magnitude and angle by combining x and y components
        '''
        self.magnitude = np.sqrt(self.Ix_prime**2+self.Iy_prime**2)
        self.magnitude = np.round(self.magnitude/np.max(self.magnitude)*255)
        self.angle = np.degrees(np.arctan2(self.Iy_prime, self.Ix_prime))
        return self.magnitude,self.angle

    def non_maximum_suppression(self) :
        '''
        Implementation of NMS on image to thin the Edges
        '''
        thinned_output = np.zeros(self.magnitude.shape)
        for i in range(1, int(self.magnitude.shape[0] - 1)) :
            for j in range(1, int(self.magnitude.shape[1] - 1)) :

                if(0 <= self.angle[i,j] < 22.5) or (157.5 <= self.angle[i,j] <= 180):
                    if((self.magnitude[i, j] > self.magnitude[i, j+1]) and (self.magnitude[i, j] > self.magnitude[i, j-1])) :
                        thinned_output[i, j] = self.magnitude[i, j]
                    else :
                        thinned_output[i, j] = 0

                elif(22.5 <= self.angle[i,j] < 67.5):
                    if((self.magnitude[i, j] > self.magnitude[i+1, j+1]) and (self.magnitude[i, j] > self.magnitude[i-1, j-1])) :
                        thinned_output[i, j] = self.magnitude[i, j]
                    else :
                        thinned_output[i, j] = 0.0

                elif(67.5 <= self.angle[i,j] < 112.5):
                    if((self.magnitude[i, j] > self.magnitude[i+1, j]) and (self.magnitude[i, j] > self.magnitude[i-1, j])) :
                        thinned_output[i, j] = self.magnitude[i, j]
                    else :
                        thinned_output[i, j] = 0
                
                elif(112.5 <= self.angle[i,j] < 157.5):
                    if((self.magnitude[i, j] > self.magnitude[i+1, j]) and (self.magnitude[i, j] > self.magnitude[i-1, j])) :
                        thinned_output[i, j] = self.magnitude[i, j]
                    else :
                        thinned_output[i, j] = 0

                else :
                    if((self.magnitude[i, j] > self.magnitude[i+1, j-1]) and (self.magnitude[i, j] > self.magnitude[i-1, j+1])) :
                        thinned_output[i, j] = self.magnitude[i, j]
                    else :
                        thinned_output[i, j] = 0

        return thinned_output

    def hysteresis(self):
        '''
        Implementation of hystersis thresholding on image based on Thresholdings {High and Low}
        '''

        M, N = self.nms_thinned_output.shape  
        for i in range(1, M-1):
            for j in range(1, N-1):
                if (self.nms_thinned_output[i,j] > self.highThreshold):

                    self.nms_thinned_output[i,j]=255
                elif (self.nms_thinned_output[i,j] < self.lowThreshold):

                    self.nms_thinned_output[i,j]=0
                else:

                    if ((self.nms_thinned_output[i+1,j]>self.highThreshold) or (self.nms_thinned_output[i-1,j]>self.highThreshold) or (self.nms_thinned_output[i,j-1]>self.highThreshold) 
                      or (self.nms_thinned_output[i,j+1]>self.highThreshold) or (self.nms_thinned_output[i+1,j-1]>self.highThreshold) 
                      or (self.nms_thinned_output[i-1,j-1]>self.highThreshold) or (self.nms_thinned_output[i+1,j+1]>self.highThreshold) or (self.nms_thinned_output[i-1,j+1]>self.highThreshold)):
                        self.nms_thinned_output[i,j]=255
                    else:
                        self.nms_thinned_output[i,j]=0   
        return self.nms_thinned_output

    def detect(self):
        
        name,ext = self.img_name.split(".")
        self.img = plt.imread(self.img)

        # Step 1: create 1D Gaussian mask (G)
        g = self.gaussian()
        g = g.reshape(1,-1)

        # Step 2: create derivate of gaussian masks Gx and Gy 
        gx = self.derivative_gaussian()
        gx = gx.reshape(1,-1)

        # Step 3: convolve Gaussian (G) with I along x & y to get Ix & Iy 
        Ix = self.convolution(self.img, g)
        Iy = self.convolution(self.img, np.transpose(g))

        # Normalize output
        Iy = self.min_max_normalize(Iy)
        Ix = self.min_max_normalize(Ix)

        # Step 4: obtain Ix' and Iy' the derivative of filted image
        self.Ix_prime = self.convolution(Ix, gx)
        self.Iy_prime = self.convolution(Iy, np.transpose(gx))


        # Step 5: compute the magnitude (M) and angle
        magnitude,_ = self.magnitude_angle()
        self.magnitude = self.min_max_normalize(magnitude)

        # Step 6: non-maximum supression
        self.nms_thinned_output = self.non_maximum_suppression()
        #I = np.round(I/np.max(I)*255) 

        # Step 7: Hystersis Thresholding
        image_hysteresis_output = self.hysteresis()

        path = os.getcwd() + "/tempDir/output/"

        ixplot = plt.imshow(Ix,cmap='gray') 
        plt.axis('off')
        plt.title("Image's X component of the convolution with a Gaussian for std = " + str(self.sigma))
        plt.savefig(path + '1.Ix_result_'+ name + '_' + str(self.sigma) + "." + ext)


        iyplot = plt.imshow(Iy,cmap='gray') 
        plt.axis('off')
        plt.title("Image's Y component of the convolution with a Gaussian for std = " + str(self.sigma))
        plt.savefig(path + '2.Iy_result_' + name + '_' + str(self.sigma)+ "." + ext)


        ixxplot = plt.imshow(self.Ix_prime,cmap='gray') 
        plt.axis('off')
        plt.title("X component of the image with the derivative of a Gaussian for std = " + str(self.sigma))
        plt.savefig(path + '3.Ixx_result_'+ name + '_' + str(self.sigma)+ "." + ext)


        iyyplot = plt.imshow(self.Iy_prime,cmap='gray') 
        plt.axis('off')
        plt.title("Y component of the image with the derivative of a Gaussian for std = " + str(self.sigma))
        plt.savefig(path + '4.Iyy_result_'+ name + '_' + str(self.sigma)+ "." + ext)


        implot = plt.imshow(self.magnitude,cmap='gray') 
        plt.axis('off')
        plt.title("Resulting magnitude image for std  = " + str(self.sigma))
        plt.savefig(path + '5.M_result_'+ name + '_' + str(self.sigma)+ "." + ext)

        inmsplot = plt.imshow(self.nms_thinned_output,cmap='gray') 
        plt.axis('off')
        plt.title("Canny Edge image after Non-maximum suppression for std = " + str(self.sigma))
        plt.savefig(path + '6.Inms_result_'+ name + '_' + str(self.sigma)+ "." + ext)

        ihysplot = plt.imshow(image_hysteresis_output,cmap='gray')
        plt.axis('off')
        plt.title("Image with final output for std =  " + str(self.sigma))
        plt.savefig(path + '7.Ihyst_result_'+ name + '_' + str(self.sigma)+ "." + ext)
