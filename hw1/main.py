import numpy as np 
from PIL import Image

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

"""
    Part 1. Required to implement from scratch.
"""

def vertical_flip(img):
    return img[::-1, :]

def horizontal_flip(img):
    return img[:, ::-1]

def diagonal_flip(img):
    return np.transpose(img)

"""
    Part 2. Free to use functions
"""

def rotate(img, deg):
    """ Counter-clockwise rotation """
    return np.asarray(Image.fromarray(img).rotate(deg, expand=True))

def shrink(img, downsample=2):
    """ Shrink image by downsample rate """
    h, w = img.shape
    return np.asarray(Image.fromarray(img).resize((h//2, w//2), Image.ANTIALIAS))

def binarize(img, threshold):
    """ Binarize image given threshold """
    return np.where(img < threshold, 0, 255).astype(np.uint8)


def main():
    img = read_image('../data/lena.bmp')
    save_image(vertical_flip(img),   './results/1-a.png')
    save_image(horizontal_flip(img), './results/1-b.png')
    save_image(diagonal_flip(img),   './results/1-c.png')
    save_image(rotate(img, -45),     './results/2-a.png')
    save_image(shrink(img, 2),       './results/2-b.png')
    save_image(binarize(img, 128),   './results/2-c.png')

if __name__ == '__main__':
    main()