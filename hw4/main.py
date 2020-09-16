"""
    @HW4: Binary Morphology
"""

import numpy as np 
from PIL import Image

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

# Utils

def binarize(img, threshold):
    """ Binarize image given threshold """
    return np.where(img < threshold, 0, 255).astype(np.uint8)

def binary_morph(img, func):
    kernel = np.array([
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0]
    ])

    h, w = img.shape
    r = kernel.shape[0] // 2  # radius of kernel
    new_img = np.zeros((h+2*r, w+2*r))
    img = np.pad(img, r, mode='constant')
    for i in range(r, h+r):
        for j in range(r, w+r):
            patch = img[i-r:i+r+1, j-r:j+r+1]
            pixel = func(patch, kernel)
            new_img[i][j] = pixel
    return new_img[r:-r, r:-r].astype(np.uint8) * 255

def dilate_pixel(patch, kernel):
    masked = np.multiply(patch, kernel)
    pixel = np.any(masked.ravel())
    return pixel

def erode_pixel(patch, kernel):
    masked = np.multiply(patch, kernel) + (1 - kernel)
    pixel = np.all(masked.ravel())
    return pixel

# Binary morphology operation functions

def dilation(img):
    return binary_morph(img, dilate_pixel)

def erosion(img):
    return binary_morph(img, erode_pixel)

def opening(img):
    return dilation(erosion(img))

def closing(img):
    return erosion(dilation(img))

def hit_and_miss(img):
    pass

def main():
    img = read_image('../data/lena.bmp')
    img = binarize(img, 128)

    save_image(dilation(img), './results/a.png')
    save_image(erosion(img),  './results/b.png')
    save_image(opening(img),  './results/c.png')
    save_image(closing(img),  './results/d.png')

if __name__ == '__main__':
    main()