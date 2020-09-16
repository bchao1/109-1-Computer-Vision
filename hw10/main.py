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
    return new_img[r:-r, r:-r]

# 

def laplacian_zero_crossing(img, kernel, threshold):
    h, w = img.shape
    L = convolve2d(img, kernel) # compute laplacian
    L = np.where(L <= -threshold, -1, L)
    L = np.where(L >= threshold, 1, L)
    L = np.pad(L, 1, mode='constant').astype(np.float64)
    edge = np.ones((h+2, w+2)) * 255
    for i in range(1, h+1):
        for j in range(1, w+1):
            patch = L[i-1:i+2, j-1:j+2]
            if patch[1][1] == 1 and np.any(patch.ravel() == -1):
                edge[i][j] = 0
    return edge[1:-1, 1:-1].astype(np.uint8)

def laplacian_1(img, threshold):
    kernel = np.array([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ])
    return laplacian_zero_crossing(img, kernel, threshold)

def laplacian_2(img, threshold):
    kernel = np.array([
        [1, 1, 1],
        [1, -8, 1],
        [1, 1, 1]
    ]) / 3
    return laplacian_zero_crossing(img, kernel, threshold)

def laplacian_3(img, threshold):
    kernel = np.array([
        [2, -1, 2],
        [-1, -4, -1],
        [2, -1, 2]
    ]) / 3
    return laplacian_zero_crossing(img, kernel, threshold)

def LoG(img, threshold):
    kernel = np.array([
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-2, -9, -23, -1, 103, 178, 103, -1, -23, -9, -2],
        [-1, -8, -22, -14, 52, 103, 52, -14, -22, -8, -1],
        [-1, -4, -15, -24, -14, -1, -14, -24, -15, -4, -1],
        [0, -2, -7, -15, -22, -23, -22, -15, -7, -2, 0],
        [0, 0, -2, -4, -8, -9, -8, -4, -2, 0, 0],
        [0, 0, 0, -1, -1, -2, -1, -1, 0, 0, 0]
    ])
    return laplacian_zero_crossing(img, kernel, threshold)

def DoG(img, threshold):
    kernel = np.array([
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-4, -8, -12, -16, -17, -17, -17, -16, -12, -8, -4],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-8, -13, -17, 15, 160, 283, 160, 15, -17, -13, -8],
        [-7, -13, -17, 0, 85, 160, 85, 0, -17, -13, -7],
        [-6, -11, -16, -16, 0, 15, 0, -16, -16, -11, -6],
        [-4, -8, -12, -16, 0, 15, 0, -16, -16, -11, -6],
        [-3, -5, -8, -11, -13, -13, -13, -11, -8, -5, -3],
        [-1, -3, -4, -6, -7, -8, -7, -6, -4, -3, -1]
    ])
    return laplacian_zero_crossing(img, kernel, threshold)
    

def main():
    img = read_image('../data/lena.bmp')

    res1 = laplacian_1(img, 15)
    res2 = laplacian_2(img, 15)
    res3 = laplacian_3(img, 20)
    res4 = LoG(img, 3000)
    res5 = DoG(img, 1)
    
    save_image(res1, './results/1.png')
    save_image(res2, './results/2.png')
    save_image(res3, './results/3.png')
    save_image(res4, './results/4.png')
    save_image(res5, './results/5.png')

if __name__ == '__main__':
    main()