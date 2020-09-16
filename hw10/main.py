import numpy as np 
from PIL import Image

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

# utils

def convolve2d(img, kernel):
    h, w = img.shape
    r = kernel.shape[0] // 2  # radius of kernel
    new_img = np.zeros((h+2*r, w+2*r))
    img = np.pad(img, r, mode='constant').astype(np.float64)
    for i in range(r, h+r):
        for j in range(r, w+r):
            patch = img[i-r:i+r+1, j-r:j+r+1]
            pixel = np.sum(np.multiply(patch.ravel(), kernel.ravel()))
            new_img[i][j] = pixel
    new_img = np.clip(new_img, 0, 255)
    return new_img[r:-r, r:-r].astype(np.uint8)

def binarize(img, threshold=128):
    """ Binarize image given threshold """
    return np.where(img < threshold, 255, 0).astype(np.uint8)

def laplacian_mask(img, threshold):
    pass

# 

def laplacian_1(img, threshold):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    L = convolve2d(img, kernel)
    edge = binarize(L, threshold)
    return edge

def laplacian_2(img, threshold):
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]) / 3
    L = convolve2d(img, kernel)
    edge = binarize(L, threshold)
    return edge

def laplacian_3(img, threshold):
    kernel = np.array([
        [2, -1, 2],
        [-1, -4, -1],
        [2, -1, 2]
    ]) / 3
    L = convolve2d(img, kernel)
    edge = binarize(L, threshold)
    return edge

def main():
    img = read_image('../data/lena.bmp')
    img = laplacian_3(img, 20)
    save_image(img, 'test.png')

if __name__ == '__main__':
    main()