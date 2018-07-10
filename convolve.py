import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt


def img_to_matrix(image_path): #converts image to rgb valye matrix
    return cv2.imread(image_path)

def get_mag_complex_image(img):
    mag_image = []
    curr_line = []
    for line in img:
        for pixel in line:
            curr_line.append([np.absolute(pixel[0]), np.absolute(pixel[1]), np.absolute(pixel[2])])
        mag_image.append(curr_line)
        curr_line = []
    return mag_image

def pad_kernel(img, kernel):
    diff_in_size = img.shape[0] - kernel.shape[0]
    return np.pad(kernel, ((diff_in_size//2, diff_in_size//2+1), (diff_in_size//2, diff_in_size//2+1)), "constant", constant_values = np.array([0,0,0]))

def normalize(img):
    max_val = np.amax(img)
    norm_img = []
    curr_line = []
    for line in img:
        for pixel in line:
            curr_line.append([np.uint8(46*np.log(pixel[0]+1)), np.uint8(46*np.log(pixel[1]+1)), np.uint8(46*np.log(pixel[2]+1))])
        norm_img.append(curr_line)
        curr_line=[]
    return norm_img

ripple = img_to_matrix("ripple.jpg")
circle = img_to_matrix("test.png")

ripple_circle = np.fft.ifft2(np.multiply(np.fft.fft2(ripple), np.fft.fft2(circle)))
plt.imshow(normalize(get_mag_complex_image(ripple_gauss)))
plt.show()
