import cv2
import numpy as np

import matplotlib.pyplot as plt
from comparison_metrics import *
from custom_plots import *

metrics = {
    "ssd": ssd,
    "ncc": ncc,
    "ssim": ssim,
}

def main():
    
    #<<<<========optimal lambda for each metrics across each dataset=========>>>>>
    # fig = plt.figure()
    # x, y = optimal_lambda_plot("ssd")
    # print("Optimal lambda for min SSD for Art Image: ", x[np.argmin(y[0])])
    # print("Optimal lambda for min SSD for Books Image: ", x[np.argmin(y[1])])
    # print("Optimal lambda for min SSD for Dolls Image: ", x[np.argmin(y[2])])
    # print("Optimal lambda for min SSD for Laundry Image: ", x[np.argmin(y[3])])
    # print("Optimal lambda for min SSD for Reindeer Image: ", x[np.argmin(y[4])])
    # print("Optimal lambda for min SSD for Moebius Image: ", x[np.argmin(y[5])])

    # fig = plt.figure()
    # x, y = optimal_lambda_plot("ssim")
    # print("Optimal lambda for max SSIM for Art Image: ", x[np.argmax(y[0])])
    # print("Optimal lambda for max SSIM for Books Image: ", x[np.argmax(y[1])])
    # print("Optimal lambda for max SSIM for Dolls Image: ", x[np.argmax(y[2])])
    # print("Optimal lambda for max SSIM for Laundry Image: ", x[np.argmin(y[3])])
    # print("Optimal lambda for max SSIM for Reindeer Image: ", x[np.argmin(y[4])])
    # print("Optimal lambda for max SSIM for Moebius Image: ", x[np.argmin(y[5])])

    # fig = plt.figure()
    # x, y = optimal_lambda_plot("ncc")
    # print("Optimal lambda for max NCC for Art Image: ", x[np.argmax(y[0])])
    # print("Optimal lambda for max NCC for Books Image: ", x[np.argmax(y[1])])
    # print("Optimal lambda for max NCC for Dolls Image: ", x[np.argmax(y[2])])
    # print("Optimal lambda for max NCC for Laundry Image: ", x[np.argmax(y[3])])
    # print("Optimal lambda for max NCC for Reindeer Image: ", x[np.argmax(y[4])])
    # print("Optimal lambda for max NCC for Moebius Image: ", x[np.argmax(y[5])])

    #<<<<================ Difference Image Computation ================>>>>>
    img1 = cv2.imread('../results/Dolls/Dolls_window1_dynamic.png')
    img2 = cv2.imread('../datasets/Dolls/Dolls_GT.png')
    difference_image(img1, img2)

    # #<<<<================== Metrics vs Window size ====================>>>>>
    # for metric in metrics:
    #     fig, ax = plt.subplots(2, 3, figsize=(15, 15))
    #     fig.suptitle(f"{metric.upper()} vs window_size")
    #     metrics_vs_window_plot(metric, ax)

    # #<<<<=================== Time vs Window size ======================>>>>>
    # fig, ax = plt.subplots(2, 3, figsize=(15, 15))
    # fig.suptitle("Processing Time (s) vs window_size")
    # time_vs_window_plot(ax)
    plt.show()


def difference_image(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (score, diff) = skt_ssim(img1_gray, img2_gray, full=True)

    diff = (diff * 255).astype("uint8")

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    filled_after = img2.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area > 40:
            x,y,w,h = cv2.boundingRect(c)
            cv2.drawContours(filled_after, [c], 0, (0,0,255), -1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 15))
    ax[0].imshow(img1)
    ax[0].set_title("Disparity by Dynamic Programming")
    ax[1].imshow(img2)
    ax[1].set_title("Ground Truth Disparity")
    ax[2].imshow(filled_after)
    ax[2].set_title("Difference Image")


    
if __name__ == "__main__":
    main()

