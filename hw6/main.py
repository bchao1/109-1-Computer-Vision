"""
    @HW6: Yokoi Connectivity Number
"""

import numpy as np 
from PIL import Image

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

def 
def main():
    img = read_image('../data/lena.bmp')
    img = binarize(img, 128)
    img = downsample(img, 8)
    print(img.shape)
    save_image(img, 'test.png')

if __name__ == '__main__':
    main()