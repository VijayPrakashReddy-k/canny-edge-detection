import os
import matplotlib.pyplot as plt 
from PIL import Image,ImageChops

# def visualize(img, format=None, gray=False):
#     plt.figure(figsize=(20, 40))
#     plt.imshow(img) 
#     plt.show()
#     #     plt.imshow(img, format)
#     # plt.show()

#     # for i, img in enumerate(imgs):
#     #     if img.shape[0] == 3:
#     #         img = img.transpose(1,2,0)
#     #     plt_idx = i+1
#     #     plt.subplot(2, 2, plt_idx)
#     #     plt.imshow(img, format)
#     # plt.show()

def visualize(imgs,num_rows = 4, num_cols = 2):
    _, axs = plt.subplots(num_rows, num_cols, figsize=(10, 20))
    axs = axs.flatten()
    for img, ax in zip(imgs, axs):
        ax.imshow(img)
    plt.savefig('/tempDir/final_output/full_figure.jpeg')
    
def is_greyscale(image):
    """
    Check if image is monochrome (1 channel or 3 identical channels)
    """
    im = Image.open(image)
    if im.mode not in ("L", "RGB"):
        raise ValueError("Unsuported image mode")

    if im.mode == "RGB":
        rgb = im.split()
        if ImageChops.difference(rgb[0],rgb[1]).getextrema()[1]!=0: 
            return im,False #"Colored Image uploaded"
        if ImageChops.difference(rgb[0],rgb[2]).getextrema()[1]!=0: 
            return im,False #"Colored Image uploaded"
    return im,True #"Grey scale Image uploaded"

def load_data(dir_name):    
    '''
    Load images from the "tempDir" directory If the Image is in RGB then we will convert it to gray scale image
    '''
    for filename in os.listdir(dir_name):
        if os.path.isfile(dir_name + '/' + filename):
            im,check = is_greyscale(dir_name + '/' + filename)
            return im,check