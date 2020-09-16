import numpy as np 
from PIL import Image

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

# Noise functions

def gaussian_noise(img, amplitude):
    noise = np.random.normal(scale=amplitude, size=img.shape)
    return np.clip(img + noise, a_min=0, a_max=255).astype(np.uint8)

def salt_and_pepper_noise(img, prob):
    h, w = img.shape
    total_pixels = h * w
    coords = np.random.choice(total_pixels, size=int(total_pixels * prob), replace=False)
    salt_coords = np.random.choice(coords, size=int(len(coords) * 0.5), replace=False)
    noisy_img = img.copy().ravel()
    noisy_img[coords] = 0
    noisy_img[salt_coords] = 255
    return noisy_img.reshape(h, w).astype(np.uint8)

# Filters

def filter(img, kernel_size, func):
    h, w = img.shape
    r = kernel_size // 2
    new_img = np.zeros((h+2*r, w+2*r))
    img = np.pad(img, r, mode='edge')  # pad with edge value
    for i in range(r, h+r):
        for j in range(r, w+r):
            patch = img[i-r:i+r+1, j-r:j+r+1]
            pixel = func(patch)
            new_img[i][j] = pixel
    return new_img[r:-r, r:-r].astype(np.uint8)

def box(patch):
    return np.mean(patch.ravel())

def median(patch):
    patch = sorted(patch.ravel())
    return patch[len(patch) // 2]

def box_filter(img, kernel_size):
    return filter(img, kernel_size, box)

def median_filter(img, kernel_size):
    return filter(img, kernel_size, median)

# IQA

def SNR(img):
    pixels = img.ravel()
    mean = np.mean(img)
    sigma = np.sqrt(np.var(img))
    return mean / sigma

def main():
    img = read_image('../data/lena.bmp')
    # Generate noisy images
    g_10 = gaussian_noise(img, 10)
    g_30 = gaussian_noise(img, 30)
    sp_01 = salt_and_pepper_noise(img, 0.1)
    sp_005 = salt_and_pepper_noise(img, 0.05)

    # Box filter
    box3_g10 = box_filter(g_10, 3)
    box5_g10 = box_filter(g_10, 5)
    box3_g30 = box_filter(g_30, 3)
    box5_g30 = box_filter(g_30, 5)

    # Median filter
    median3_g10 = median_filter(g_10, 3)
    median5_g10 = median_filter(g_10, 5)
    median3_g30 = median_filter(g_30, 3)
    median5_g30 = median_filter(g_30, 5)

    # Opening then closing

    # Closing then opening
    

if __name__ == '__main__':
    main()