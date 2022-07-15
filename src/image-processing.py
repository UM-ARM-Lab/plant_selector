"""
image-processing.py

This python file turns an RGB image to a binary image. Applies Edge detection and skeletonization for weed extraction.

Created by: Miguel Munoz
Date: July 7th, 2022
"""
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


def adaptive_gaussian_thresholding(img):
    img = cv.imread(img, 0)
    img = cv.medianBlur(img, 5)
    ret, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    th2 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                               cv.THRESH_BINARY, 11, 2)
    th3 = cv.adaptiveThreshold(img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv.THRESH_BINARY, 11, 2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',
              'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2, 2, i + 1), plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def otsu_thresholding(img):
    img = cv.imread(img, 0)
    # global thresholding
    ret1, th1 = cv.threshold(img, 127, 255, cv.THRESH_BINARY)
    # Otsu's thresholding
    ret2, th2 = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # Otsu's thresholding after Gaussian filtering
    blur = cv.GaussianBlur(img, (65, 65), 0)
    ret3, th3 = cv.threshold(blur, 200, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    # plot all the images and their histograms
    images = [img, 0, th1,
              img, 0, th2,
              blur, 0, th3]
    titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
              'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
              'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
    for i in range(3):
        plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
        plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
        plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
        plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
        plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
    plt.show()


def canny_edge_detection(img):
    edges = cv.Canny(img, 200, 400)
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(edges, cmap='gray')
    plt.title('Canny Edge Detection', fontsize=30), plt.xticks([]), plt.yticks([])
    plt.show()


def skeletonization(img):
    """================WORKS VERY GOOD================="""
    img_or = cv.imread(img, 1)

    hsv = img_or.copy()
    col_filter = img_or.copy()

    for x in range(img_or.shape[0]):
        for y in range(img_or.shape[1]):
            cur_color = img_or[x][y]
            gr_diff = cur_color[1] - cur_color[2]
            print(gr_diff)
            gb_diff = int(cur_color[1]) - int(cur_color[0])
            if gr_diff > 10 and gb_diff > 10:
               continue
            else:
                hsv[x][y] = [0, 0, 0]
    res = hsv

    img_blurred = cv.medianBlur(res, 15)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    erosion = cv.erode(img_blurred, kernel, iterations=3)
    # opening = cv.morphologyEx(erosion, cv.MORPH_OPEN, kernel)

    erosion_reshaped = erosion.reshape((-1, 3))
    erosion_reshaped = np.float32(erosion_reshaped)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    k = 2
    _, labels, (centers) = cv.kmeans(erosion_reshaped, k, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)

    # flatten the labels array
    labels = labels.flatten()

    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(erosion.shape)

    plt.figure(1)
    plt.imshow(img_or), plt.xticks([]), plt.yticks([])
    plt.title("OG Image")
    plt.figure(2)
    plt.imshow(res), plt.xticks([]), plt.yticks([])
    plt.title("After green mask")
    plt.figure(3)
    plt.subplot(1, 2, 1)
    plt.title("Image blurred")
    plt.imshow(img_blurred), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.title("Erosion")
    plt.imshow(erosion), plt.xticks([]), plt.yticks([])
    plt.figure(4)
    plt.imshow(segmented_image), plt.xticks([]), plt.yticks([])
    plt.title("K-Means")

    cv.waitKey(0)
    plt.show()
    cv.destroyAllWindows()

    # ##################SKELETONIZATION PART########################
    # size = np.size(img_or)
    # skel = np.zeros(img_or.shape, np.uint8)
    #
    # img_blurred = cv.medianBlur(img_or, 15)
    # img = cv.adaptiveThreshold(img_blurred, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv.THRESH_BINARY, 3, 2)
    # element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    # done = False
    #
    # while not done:
    #     eroded = cv.erode(img, element)
    #     temp = cv.dilate(eroded, element)
    #     temp = cv.subtract(img, temp)
    #     skel = cv.bitwise_or(skel, temp)
    #     img = eroded.copy()
    #
    #     zeros = size - cv.countNonZero(img)
    #     if zeros == size:
    #         done = True
    # plt.figure()
    # plt.imshow(skel, cmap='gray')
    # plt.title('Skeletonization'), plt.xticks([]), plt.yticks([])
    # plt.show()
    # cv.waitKey(0)
    # cv.destroyAllWindows()


def contours_canny(img):
    """==========WORKS==========="""
    # Let's load a simple img with 3 black squares
    # Grayscale
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    # Find Canny edges
    edged = cv.Canny(gray, 400, 450)

    # Finding Contours
    # Use a copy of the img e.g. edged.copy()
    # since findContours alters the img
    contours, hierarchy = cv.findContours(edged, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    plt.figure(1)
    plt.imshow(edged, cmap='gray')
    plt.title('Canny Edges After Contouring'), plt.xticks([]), plt.yticks([])

    print("Number of Contours found = " + str(len(contours)))

    # Draw all contours
    # -1 signifies drawing all contours
    cv.drawContours(img, contours, -1, (0, 255, 0), 3)

    plt.figure(6)
    plt.imshow(img, cmap='gray')
    plt.title('Contours'), plt.xticks([]), plt.yticks([])


def christian(img_path):
    img_or = cv.imread(img_path, 1)

    col_filter = img_or.copy()
    lower_green = np.array([5, 20, 5])
    upper_green = np.array([255, 255, 255])
    mask = cv.inRange(img_or, lower_green, upper_green)
    res = cv.bitwise_and(img_or, img_or, mask=mask)

    mask = np.zeros(col_filter.shape)
    for x in range(img_or.shape[0]):
        for y in range(img_or.shape[1]):
            cur_color = res[x][y]
            gr_diff = int(cur_color[1]) - int(cur_color[2])
            gb_diff = int(cur_color[1]) - int(cur_color[0])
            if gr_diff > 7 and gb_diff > 7:
                mask[x][y] = 1
                continue
            else:
                mask[x][y] = 0
                col_filter[x][y] = [0, 0, 0]

    plt.figure(1)
    plt.imshow(img_or), plt.xticks([]), plt.yticks([])
    plt.title("OG Image")
    plt.figure(2)
    plt.imshow(col_filter), plt.xticks([]), plt.yticks([])
    plt.title("After green mask")
    plt.figure(3)
    plt.imshow(mask), plt.xticks([]), plt.yticks([])
    plt.title("Binary Mask")

    img_data = np.asarray(mask[:, :, 0], dtype=np.uint8)
    # gx, gy = np.gradient(img_data)
    # temp_edge = gy * gy + gx * gx
    # temp_edge[temp_edge != 0.0] == 255.0
    # plt.figure(4)
    # plt.imshow(temp_edge), plt.xticks([]), plt.yticks([])
    # plt.title("Binary Mask")

    ##################SKELETONIZATION PART########################
    size = np.size(img_or)
    skel = np.zeros(img_or.shape, np.uint8)
    img = np.zeros(img_or.shape, np.uint8)
    img[mask == 1] = 255.0
    element = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    x = 0
    while True:
        eroded = cv.erode(img, element)
        temp = cv.dilate(eroded, element)
        temp = cv.subtract(img, temp)
        skel = cv.bitwise_or(skel, temp)
        img = eroded.copy()
        x += 1
        if x == 7:
            break
    plt.figure()
    plt.imshow(skel, cmap='gray')
    plt.title('Skeletonization'), plt.xticks([]), plt.yticks([])

    cv.waitKey(0)
    plt.show()
    cv.destroyAllWindows()



def main():
    img = '/home/christianforeman/catkin_ws/src/plant_selector/good.png'
    # adaptive_gaussian_thresholding(img)
    # otsu_thresholding(img)
    # canny_edge_detection(img)
    # skeletonization(img)
    # contours_canny(img)
    # gabor_filter(img)
    christian(img)

if __name__ == '__main__':
    main()
