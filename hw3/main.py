import numpy as np 
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

# 

def reduce_intensity(img, factor):
    return (img/factor).astype(np.uint8)

def compute_histogram(img):
    count = Counter(img.ravel())  # all pixel value occurences
    hist = np.zeros(256, dtype=np.uint)  # histogram for pixel values
    for i, cnt in count.items():
        hist[i] = cnt
    return hist

def plot_histogram(hist, title, outfile):
    plt.figure()
    plt.bar(np.arange(256), hist)
    plt.title(title)
    plt.savefig(outfile)
    plt.close()
    
def histogram_eqalize(img, hist):
    cdf = np.cumsum(hist)
    eq_cdf = 255 * (cdf - np.min(cdf)) / (np.sum(hist) - np.min(cdf))
    eq_cdf = eq_cdf.astype(np.uint8)
    eq_img = eq_cdf[img]
    return eq_img

def main():
    img = read_image('../data/lena.bmp')

    hist1 = compute_histogram(img)
    plot_histogram(hist1, '1. Original', './results/1-hist.png')
    save_image(img, './results/1.png')

    img2 = reduce_intensity(img, 3)
    hist2 = compute_histogram(img2)
    plot_histogram(hist2, '2. Reduced by 3', './results/2-hist.png')
    save_image(img2, './results/2.png')

    img3 = histogram_eqalize(img2, hist2)
    hist3 = compute_histogram(img3)
    plot_histogram(hist3, '3. Histogram Equalized', './results/3-hist.png')
    save_image(img3, './results/3.png')

if __name__ == '__main__':
    main()