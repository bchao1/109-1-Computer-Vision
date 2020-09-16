import numpy as np 
from PIL import Image

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

def binarize(img, threshold=128):
    """ Binarize image given threshold """
    return np.where(img < threshold, 255, 0).astype(np.uint8)

def convolve2d(img, kernel_maps):
    n = kernel_maps.shape[-1]
    h, w = img.shape
    r = kernel_maps[0].shape[0] // 2  # radius of kernel
    img = np.pad(img, r, mode='constant')
    stacked_img = np.dstack([img] * n)
    new_img = np.zeros((h+2*r, w+2*r, n))
    for i in range(r, h+r):
        for j in range(r, w+r):
            patch = stacked_img[i-r:i+r+1, j-r:j+r+1, :]
            pixel = np.sum(np.multiply(patch.reshape(-1, n), kernel_maps.reshape(-1, n)), axis=0) # convolution
            new_img[i, j, :] = pixel
    return new_img[r:-r, r:-r]

# Edge detection algorithm classes

def xy_grad_filters(img, x_kernel, y_kernel, threshold):
    """
        Edge detection filters in form sqrt(Gx**2+Gy**2)
    """
    grad_maps = convolve2d(img, np.dstack([x_kernel, y_kernel]))
    G = np.sqrt(np.sum(grad_maps**2, axis=-1))
    edge = binarize(G, threshold)
    return edge

def max_grad_filters(img, kernel_maps, threshold):
    """
        Edge detection filters in form max(list of G's)
    """

    grad_maps = convolve2d(img, kernel_maps)
    G = np.max(grad_maps, axis=-1)
    edge = binarize(G, threshold)
    return edge

# Edge detection filters

def robert(img, threshold):
    # pad with 0 just for convenience
    x_kernel = np.array([
        [0, 0, 0],
        [0, -1, 0],
        [0, 0, 1]
    ])
    y_kernel = np.array([
        [0, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ])
    return xy_grad_filters(img, x_kernel, y_kernel, threshold)

def prewitt(img, threshold):
    x_kernel = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    y_kernel = np.array([
        [-1, 0, 1],
        [-1, 0, 1],
        [-1, 0, 1]
    ])
    return xy_grad_filters(img, x_kernel, y_kernel, threshold)

def sobel(img, threshold):
    x_kernel = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    y_kernel = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    return xy_grad_filters(img, x_kernel, y_kernel, threshold)

def frei_chen(img, threshold):
    x_kernel = np.array([
        [-1, -np.sqrt(2), -1],
        [0, 0, 0],
        [1, np.sqrt(2), 1]
    ])
    y_kernel = np.array([
        [-1, 0, 1],
        [-np.sqrt(2), 0, np.sqrt(2)],
        [-1, 0, 1]
    ])
    return xy_grad_filters(img, x_kernel, y_kernel, threshold)

def kirsch_compass(img, threshold):
    kernels = [
        np.array([
            [-3, -3, 5],
            [-3, 0, 5],
            [-3, -3, 5]
        ]),
        np.array([
            [-3, 5, 5],
            [-3, 0, 5],
            [-3, -3, -3]
        ]),
        np.array([
            [5, 5, 5],
            [-3, 0, -3],
            [-3, -3, -3]
        ]),
        np.array([
            [5, 5, -3],
            [5, 0, -3],
            [-3, -3, -3]
        ]),
        np.array([
            [5, -3, -3],
            [5, 0, -3],
            [5, -3, -3]
        ]),
        np.array([
            [-3, -3, -3],
            [5, 0, -3],
            [5, 5, -3]
        ]),
        np.array([
            [-3, -3, -3],
            [-3, 0, -3],
            [5, 5, 5]
        ]),
        np.array([
            [-3, -3, -3],
            [-3, 0, 5],
            [-3, 5, 5]
        ]),
    ]
    kernel_maps = np.dstack(kernels)
    return max_grad_filters(img, kernel_maps, threshold)

def robinson_compass(img, threshold):
    kernels = [
        np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]),
        np.array([
            [0, 1, 2],
            [-1, 0, 1],
            [-2, -1, 0]
        ]),
        np.array([
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]),
        np.array([
            [2, 1, 0],
            [1, 0, -1],
            [0, -1, -2]
        ]),
        np.array([
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1]
        ]),
        np.array([
            [0, -1, -2],
            [1, 0, -1],
            [2, 1, 0]
        ]),
        np.array([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]),
        np.array([
            [-2, -1, 0],
            [-1, 0, 1],
            [0, 1, 2]
        ]),
    ]
    kernel_maps = np.dstack(kernels)
    return max_grad_filters(img, kernel_maps, threshold)

def nevatia_babu(img, threshold):
    kernels = [
        np.array([
            [100, 100, 100, 100, 100],
            [100, 100, 100, 100, 100],
            [0, 0, 0, 0, 0],
            [-100, -100, -100, -100, -100],
            [-100, -100, -100, -100, -100]
        ]),
        np.array([
            [100, 100, 100, 100, 100],
            [100, 100, 100, 78, -32],
            [100, 92, 0, -92, -100],
            [32, -78, -100, -100, -100],
            [-100, -100, -100, -100, -100]
        ]),
        np.array([
            [100, 100, 100, 32, -100],
            [100, 100, 92, -78, -100],
            [100, 100, 0, -100, -100],
            [100, 78, -92, -100, -100],
            [100, -32, -100, -100, -100]
        ]),
        np.array([
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, 0, 100, 100],
        ]),
        np.array([
            [-100, 32, 100, 100, 100],
            [-100, -78, 92, 100, 100],
            [-100, -100, 0, 100, 100],
            [-100, -100, -92, 78, 100],
            [-100, -100, -100, -32, 100]
        ]),
        np.array([
            [100, 100, 100, 100, 100],
            [-32, 78, 100, 100, 100],
            [-100, -92, 0, 92, 100],
            [-100, -100, -100, -78, 32],
            [-100, -100, -100, -100, -100]
        ])
    ]
    kernel_maps = np.dstack(kernels)
    return max_grad_filters(img, kernel_maps, threshold)

def main():
    img = read_image('../data/lena.bmp')
    
    a = robert(img, 12)
    b = prewitt(img, 24)
    c = sobel(img, 38)
    d = frei_chen(img, 30)
    e = kirsch_compass(img, 135)
    f = robinson_compass(img, 43)
    g = nevatia_babu(img, 12500)


    save_image(a, './results/a.png')
    save_image(b, './results/b.png')
    save_image(c, './results/c.png')
    save_image(d, './results/d.png')
    save_image(e, './results/e.png')
    save_image(f, './results/f.png')
    save_image(g, './results/g.png')

if __name__ == '__main__':
    main()