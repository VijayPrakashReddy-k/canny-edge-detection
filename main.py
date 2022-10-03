import os
import glob
import utils
import shutil
import base64
import numpy as np
from PIL import Image
import streamlit as st
from canny import cannyEdgeDetector as ced

# pandas display options
#pd.set_option('display.max_colwidth', None)

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.jpeg')

### Function to save the uploaded files:
def save_uploaded_file(uploadedfile):
	try:
		shutil.rmtree("./tempDir")
	except Exception:
		pass
	try:
		os.makedirs("./tempDir")
		os.makedirs("./tempDir/output")
		os.makedirs("./tempDir/final_output")         
	except Exception:
		pass
	with open(os.path.join("tempDir",uploadedfile.name),"wb") as f:
		f.write(uploadedfile.getbuffer())
		#st.balloons()
	return st.success("Saved file : {} in tempDir folder".format(uploadedfile.name))

# Function to Read and Manupilate Images
def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image

def side_show():
    """Shows the sidebar components for the template and returns user inputs as dict."""
    inputs = {}
    with st.sidebar:
        st.write("#### Standard Deviation for the Gaussian Distribution")
        inputs["sigma"] = st.number_input(
            "Sigma Value",min_value = 1, max_value = 5, value = 1, step = 1,
        )

        st.write("#### Low value for Hysteresis Thresholding")
        inputs["low"] = st.number_input(
            "Low Value",min_value = 10, max_value = 75, value = 20, step = 5,
        )
        st.write("#### High value for Hysteresis Thresholding")
        inputs["high"] = st.number_input(
            "High Value",min_value = 60, max_value = 120, value = 75, step = 5,
        )
    return inputs

def main():
    title = '<p style="font-family:Courier; color:Blue; font-size: 30px;">Computer Vision Class Implementations</p>'
    st.markdown(title, unsafe_allow_html=True)
    menu = ["Home","About"]
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        # st.subheader("Home")
        home = '<p style="font-family:Courier; color:Green; font-size: 25px;"> Home </p>'
        st.markdown(home, unsafe_allow_html=True)

        canny = '<p style="font-family:Courier; color:Red; font-size: 20px;">Canny Edge Detection for uploading Images</p>'
        st.markdown(canny, unsafe_allow_html=True)
        hello = '<p style="font-family:Courier; color:Black; font-size: 20px;"><b>Hello, World! &#x1F981;<b></p>'
        st.markdown(hello, unsafe_allow_html=True)

        download = '<p style="font-family:Courier; color:Black; font-size: 15px;">"check out this link for Images to download <a href="https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300/html/dataset/images.html">link</a>"</p>'
        st.markdown(download, unsafe_allow_html=True)

        params = side_show()
        uploadFile = st.file_uploader("Upload File",type=['png','jpeg','jpg'])
        if st.button("Process"):
            # Checking the Format of the page
            if uploadFile is not None:
                file_details = {"Filename":uploadFile.name,"FileType":uploadFile.type,"FileSize":uploadFile.size}
                st.markdown(file_details, unsafe_allow_html=True)
                img = load_image(uploadFile)
                success = '<p style="font-family:Courier; color:Black; font-size: 20px;">Image Uploaded Successfully</p>'
                st.markdown(success, unsafe_allow_html=True)
                st.balloons()
                st.image(img)
                save_uploaded_file(uploadFile)
                image,image_mode_check = utils.load_data("tempDir")
                if image_mode_check:
                    #utils.visualize(image, 'gray')
                    st.image(image)
                else:
                    image = image.convert('L')
                    #gray_image = utils.rgb2gray(image)
                    st.write("Uploaded RGB into Gray scale")
                    st.image(image)
                    #utils.visualize(gray_image, 'gray')
                    path = os.path.join("tempDir",file_details["Filename"])
                    image.save(path, 'JPEG')
                
                img_path = os.getcwd() + "/tempDir/" + uploadFile.name
                detector = ced(uploadFile.name,img_path, sigma = params['sigma'], kernel_size=3, lowthreshold = params["low"], highthreshold = params["high"])
                imgs_final = detector.detect()

                image_list = []
                folder_dir = os.getcwd() + "/tempDir/output/"
                for filename in glob.iglob(f'{folder_dir}/*'):
                #for filename in glob.glob('/tempDir/output/*'): #assuming jpeg
                    print(filename)
                    im=Image.open(filename)
                    image_list.append(im)
                #st.image(imgs_final)

                st.image(image_list, use_column_width=True, caption=["Canny Edge Detection"] * len(image_list))

                # name,ext = uploadFile.name.split(".")
                # st.subheader("Canny Ende Detection Output")
                # folder_dir = os.getcwd() + "/tempDir/output/"
                # for images in glob.iglob(f'{folder_dir}/*'):
                #     if (images.endswith("."+ext)):
                #         st.image(images)

                #st.image("full_figure.png")
                #utils.visualize(image_list)

            else:
                #st.write("Please Upload the Image and make sure your image is in JPG/PNG Format.")
                failed = '<p style="font-family:Courier; color:Black; font-size: 20px;">"Please Upload the Image and make sure your image is in JPG/PNG Format."</p>'
                st.markdown(failed, unsafe_allow_html=True)

                st.write("##### Please check out above mentioned Link for Images to test the Canny Edge Detection")
        
    else:
        with st.sidebar:
            title = '<p style="font-family:Courier; color:Green; font-size: 18px;"> The Canny Edge Detection Algorithm is based on Gray scale Images </p>'
            st.markdown(title, unsafe_allow_html=True)
            #st.write("### The Canny Edge Detection Algorithm is based on Gray scale Images")

        title = '<p style="font-family:Courier; color:Red; font-size: 30px;"> The Canny Edge Detection </p>'
        st.markdown(title, unsafe_allow_html=True)

        text = '<p style="font-family:Courier; color:Black; font-size: 18px;">The Canny edge detector is an edge detection operator that uses a multi-stage algorithm to detect a wide range of edges in images. It was developed by John F. Canny in 1986.</p>'
        st.markdown(text, unsafe_allow_html=True)

        st.image("canny_edge_algorithm.png")

        t1 = '<p style="font-family:Courier; color:Black; font-size: 16px;">The Canny Edge Detection Algorithm is composed of 5 steps: <br> <br> <b>1. Reduce noise by smoothing the Image with Gaussian Filter </b><br> <br> &emsp; &emsp; To remove noise, the image is smoothed by Gaussian blur with the kernel of size  3×3  and  𝜎 = 1 . Since the sum of the elements in the Gaussian kernel equals  1 , the kernel should be normalized before the convolution. </p>'
        st.markdown(t1, unsafe_allow_html=True)

        t2 = '<p style="font-family:Courier; color:Black; font-size: 16px;"> <b> 2. Compute Derivative (Gradient) of Filtered Image </b><br> <br> &emsp; &emsp; When the image  𝐼  is smoothed, the derivatives  𝐼𝑥  and  𝐼𝑦  w.r.t.  𝑥  and  𝑦  are calculated. It can be implemented by convolving  𝐼  with Sobel kernels  𝐾𝑥  and  𝐾𝑦 , respectively: </p>'
        st.markdown(t2, unsafe_allow_html=True)

        st.latex(r'''
        K_{x}=
        \begin{pmatrix}
        -1 & 0 & 1\\
        -2 & 0 & 2\\
        -1 & 0 & 1
        \end{pmatrix}
        ''')

        st.latex(r'''
        K_{y}=
        \begin{pmatrix}
        1 & 2 & 1\\
        0 & 0 & 0\\
        -1 & -2 & -1
        \end{pmatrix}
        ''')

        ps = '<p style="font-family:Courier; color:Brown; font-size: 12px;"> P.S : used 1-dimensional Gaussian filter for the assignment </p>'
        st.markdown(ps, unsafe_allow_html=True)

        para = '<p style="font-family:Courier; color:Black; font-size: 14px;"> Then, the magnitude  𝐺  and the slope  𝜃  of the gradient are calculated: </p>'
        st.markdown(para, unsafe_allow_html=True)

        st.latex(r'''
        |G|=
        \sqrt{I_{x}^2 + I_{y}^2}
        ''')

        st.latex(r'''
        \theta(x,y) = arctan 
        \begin{pmatrix}
        I_{y}/I_{x}
        \end{pmatrix}
        ''')

        t3 = '<p style="font-family:Courier; color:Black; font-size: 16px;"> <b> 3. Non-maximum suppression (for Thinning the Edges) </b><br> <br> &emsp; &emsp; For each pixel find two neighbors (in the positive and negative gradient directions, supposing that each neighbor occupies the angle of  𝜋/4 , and  0  is the direction straight to the right). If the magnitude of the current pixel is greater than the magnitudes of the neighbors, nothing changes, otherwise, the magnitude of the current pixel is set to zero.</p>'
        st.markdown(t3, unsafe_allow_html=True)

        t4 = '<p style="font-family:Courier; color:Black; font-size: 16px;"> <b> 4. Double threshold </b><br> <br> &emsp; &emsp; The gradient magnitudes are compared with two specified threshold values, the first one is less than the second. The gradients that are smaller than the low threshold value are suppressed; the gradients higher than the high threshold value are marked as strong ones and the corresponding pixels are included in the final edge map. All the rest gradients are marked as weak ones and pixels corresponding to these gradients are considered in the next step.</p>'
        st.markdown(t4, unsafe_allow_html=True)

        t5 = '<p style="font-family:Courier; color:Black; font-size: 16px;"> <b> 5. Edge tracking by Hysteresis Thresholding </b><br> <br> &emsp; &emsp;  Since a weak edge pixel caused from true edges will be connected to a strong edge pixel, pixel  𝑤  with weak gradient is marked as edge and included in the final edge map if and only if it is involved in the same blob (connected component) as some pixel  𝑠  with strong gradient. In other words, there should be a chain of neighbor weak pixels connecting  𝑤  and  𝑠  (the neighbors are 8 pixels around the considered one). </p>'
        st.markdown(t5, unsafe_allow_html=True)

        conclude = '<p style="font-family:Courier; color:red; font-size: 14px;"> <b> Conclusion:</b> Higher sigma values identify larger scale edges and produce a coarse output, whereas lower sigma values produces more edges and detect finer characteristics. </b></p>'
        st.markdown(conclude, unsafe_allow_html=True)

if __name__ == '__main__':
	main()