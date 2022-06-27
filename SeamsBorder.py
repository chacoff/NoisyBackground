import cv2
import numpy as np
import matplotlib.pyplot as plt


def main():
    pass


if __name__ == '__main__':

    main()

    source = cv2.imread('images\\ZH024_1480_069470_TXnoradar34.bmp', 0)  # read already in 8bit

    try:
        r, g, b = cv2.split(source)[:3]  # splitting in RGB channels
    except ValueError as e:
        print(f'{e}: Using the source image as input channel')
    finally:
        r = source.copy()

    height, width = r.shape

    scale_percent = 100  # percent of original size
    new_width = int(r.shape[1] * scale_percent / 100)
    new_height = int(r.shape[0] * scale_percent / 100)

    r_resized = cv2.resize(r, (new_width, new_height), interpolation=cv2.INTER_AREA)

    kern = 7
    r_resized = cv2.normalize(r_resized, np.zeros((height, width)), 0, 220, cv2.NORM_MINMAX)
    median = cv2.medianBlur(r_resized.copy(), kern)  # Filters to denoise the image
    r_denoise = cv2.GaussianBlur(median, (kern, kern), 0)

    gx, gy = np.gradient(r_denoise)
    gr = abs(np.gradient(gy[80]))
    gr = np.where(gr <= 0.5, 0, gr)
    gr_x = np.arange(0, len(gr)) * 1 + 1

    fig = plt.figure()

    fig.add_subplot(2, 2, 1)
    plt.imshow(r_resized, cmap='gray', aspect="auto")
    plt.axis('off')
    plt.title('Original')

    fig.add_subplot(2, 2, 2)
    plt.imshow(r_denoise, cmap='gray', aspect="auto")
    plt.axis('off')
    plt.title('Denoise')

    fig.add_subplot(2, 2, 3)
    plt.imshow(gy, cmap='gray', aspect="auto")
    plt.axis('off')
    plt.title('gradient on Y')

    fig.add_subplot(2, 2, 4)
    plt.plot(gr_x, gr)
    # plt.axis('off')
    plt.title('plot of gradient on Y')

    plt.show()