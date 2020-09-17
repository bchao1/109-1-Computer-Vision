import numpy as np 
from PIL import Image, ImageDraw
from collections import Counter
import matplotlib.pyplot as plt
plt.style.use('ggplot')

def read_image(file):
    img = np.array(Image.open(file))
    return img

def save_image(img, file):
    Image.fromarray(img).save(file)

def binarize(img, threshold):
    """ Binarize image given threshold """
    return np.where(img < threshold, 0, 255).astype(np.uint8)

# Histogram 

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

def flood_fill(img, threshold=500):
    """ Connected components in binarized image. Walk on white pixels (255) """
    visit = img.copy().astype(np.uint8)
    cc = np.zeros(img.shape)
    h, w = img.shape
    regions = []

    def bfs(x, y, c):
        q = []
        dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        q.append((x, y))
        visit[x, y] = 0
        cc[x, y] = c
        area = 1
        while q:
            x, y = q.pop(0)
            for i, j in dirs:
                new_x, new_y = x + i, y + j
                if 0 <= new_x < h and 0 <= new_y < w and visit[new_x, new_y] == 255:
                    visit[new_x, new_y] = 0
                    cc[new_x, new_y] = c
                    q.append((new_x, new_y))
                    area += 1

        if area >= threshold:
            regions.append(c)

    c = 5
    for i in range(h):
        for j in range(w):
            if visit[i, j] == 255:  # unvisited
                bfs(i, j, c)
                c += 1
    return cc, regions

def bounding_box(cc, regions, img, outfile):
    img = Image.fromarray(img).convert('RGB')
    draw = ImageDraw.Draw(img)

    for r in regions:
        y, x = np.where(cc == r)
        center_y, center_x = np.mean(y), np.mean(x)
        x_min, x_max = min(x), max(x)
        y_min, y_max = min(y), max(y)
        draw.rectangle([(x_min, y_min), (x_max, y_max)], fill=None, width=2, outline="blue")
        draw.line([(center_x - 5, center_y), (center_x + 5, center_y)], width=2, fill="red")
        draw.line([(center_x, center_y - 5), (center_x, center_y + 5)], width=2, fill="red")
    
    img.save(outfile)


def main():
    img = read_image('../data/lena.bmp')

    b_img = binarize(img, 128)
    save_image(b_img, './results/a.png')

    hist = compute_histogram(img)
    plot_histogram(hist, 'Color histogram', './results/b.png')

    cc, regions = flood_fill(b_img)
    bounding_box(cc, regions, b_img, './results/c.png')

if __name__ == '__main__':
    main()