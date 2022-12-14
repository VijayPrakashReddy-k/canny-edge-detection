# Canny Edge Detection

`The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in the images. It was developed by John F. Canny in 1986.
`
<p align = 'center'>
            <img src = canny_edge_algorithm.png/>
</p>

***The Canny Edge Detection Algorithm is composed of 5 steps:***

#### 1. Reduce noise by smoothing the Image with Gaussian Filter

β β To remove noise, the image is smoothed by Gaussian blur with the kernel of size 3Γ3 and π = 1 . Since the sum of the elements in the Gaussian kernel equals 1 , the kernel should be normalized before the convolution.

#### 2. Compute Derivative (Gradient) of Filtered Image

β β When the image πΌ is smoothed, the derivatives πΌπ₯ and πΌπ¦ w.r.t. π₯ and π¦ are calculated. It can be implemented by convolving πΌ with Sobel kernels πΎπ₯ and πΎπ¦ , respectively:
    
<p align = 'center'>
            <img src = sobel_kernel.png/>
</p>

<p align = 'center'>
            <img src = magnitude_angle.png/>
</p>

#### 3. Non-maximum suppression (for Thinning the Edges)

β β For each pixel find two neighbors (in the positive and negative gradient directions, supposing that each neighbor occupies the angle of π/4 , and 0 is the direction straight to the right). If the magnitude of the current pixel is greater than the magnitudes of the neighbors, nothing changes, otherwise, the magnitude of the current pixel is set to zero.

#### 4. Double threshold

β β The gradient magnitudes are compared with two specified threshold values, the first one is less than the second. The gradients that are smaller than the low threshold value are suppressed; the gradients higher than the high threshold value are marked as strong ones and the corresponding pixels are included in the final edge map. All the rest gradients are marked as weak ones and pixels corresponding to these gradients are considered in the next step.

#### 5. Edge tracking by Hysteresis Thresholding

β β Since a weak edge pixel caused from true edges will be connected to a strong edge pixel, pixel π€ with weak gradient is marked as edge and included in the final edge map if and only if it is involved in the same blob (connected component) as some pixel π  with strong gradient. In other words, there should be a chain of neighbor weak pixels connecting π€ and π  (the neighbors are 8 pixels around the considered one).
    
    
#### Conclusion:

- ***`Higher sigma values identify larger scale edges and produce a coarse output, whereas lower sigma values produce more edges and detect finer characteristics.`***
