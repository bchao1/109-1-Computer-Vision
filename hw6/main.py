"""
    @HW6: Yokoi Connectivity Number
"""

import numpy as np 
from PIL import Image, ImageDraw

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

# Utils

def binarize(img, threshold=128):
    """ Binarize image given threshold """
    return np.where(img < threshold, 0, 255).astype(np.uint8)

def downsample(img, scale=8):
    h, w = img.shape
    assert h % scale == w % scale == 0
    downsampled_img = img[::scale, ::scale]
    return downsampled_img

"""
    Connectivities

| 7| 2| 6|
| 3| 0| 1|
| 8| 4| 5|

"""

def h(b, c, d, e):
    if b == c:
        if (d != b or e != b):
            return 0
        else:
            return 1
    return 2

def f(a1, a2, a3, a4):
    if [a1, a2, a3, a4].count(1) == 4:
        return 5
    return [a1, a2, a3, a4].count(0)

def yokoi(img):
    _h, w = img.shape
    new_img = np.zeros(img.shape)
    img = np.pad(img, 1, mode='constant') # pad with black edge pixels
    for i in range(1, _h+1):
        for j in range(1, w+1):
            if img[i, j] == 255:
                n = img[i-1:i+2, j-1:j+2]
                a1 = h(n[1, 1], n[1, 2], n[0, 2], n[0, 1])
                a2 = h(n[1, 1], n[0, 1], n[0, 0], n[1, 0])
                a3 = h(n[1, 1], n[1, 0], n[2, 0], n[2, 1])
                a4 = h(n[1, 1], n[2, 1], n[2, 2], n[1, 2])
                new_img[i - 1, j - 1] = f(a1, a2, a3, a4)
    return new_img
 
def draw_yokoi(c, outfile):
    h, w = c.shape
    with open(outfile, 'w') as file:
        for i in range(h):
            for j in range(w):
                if c[i, j] != 0:
                    file.write(str(int(c[i, j])) + ' ')
                else:
                    file.write('  ')
            file.write('\n')

def main():
    img = read_image('../data/lena.bmp')
    img = binarize(img, 128)
    img = downsample(img, 8)
    #draw_connectivity(img)
    #np.set_printoptions(threshold=np.inf)
    #print(img)
    draw_yokoi(yokoi(img), 'out.txt')
    save_image(img, 'test.png')

if __name__ == '__main__':
    main()