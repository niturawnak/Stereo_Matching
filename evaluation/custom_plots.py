import cv2
import numpy as np

from skimage.metrics import structural_similarity as skt_ssim
from comparison_metrics import *
import matplotlib.pyplot as plt
import re



metrics = {
    "ssd": ssd,
    "ncc": ncc,
    "ssim": ssim,
}

dataset = ["Art", "Books", "Dolls", "Laundry", "Moebius", "Reindeer"]
lambda_val = [1, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
algorithms = ["opencv", "naive", "dynamic"]
window_size = [1, 3, 5]

data_dir = "../results/"
data_dir_GT= "../datasets/"

def optimal_lambda_plot(metric):
    dp_lambda_dict = {}
    for data in dataset:
        dp_lambda_dict[data] = []
        for val in lambda_val:
            metric_func = metrics[metric]
            image_name = data_dir + f"{data}/Different_Lamdas/" + "{}_window1_lambda{}_dynamic.png".format(data, val)
            gt_img = read_image(data_dir_GT + f"{data}/" + f"{data}_GT.png")
            calc_img = read_image(image_name)
            out_value = float(metric_func(gt_img, calc_img))
            dp_lambda_dict[data].append(out_value)

    x  = lambda_val
    y1 = dp_lambda_dict["Art"]
    y2 = dp_lambda_dict["Books"]
    y3 = dp_lambda_dict["Dolls"]
    y4 = dp_lambda_dict["Laundry"]
    y5 = dp_lambda_dict["Reindeer"]
    y6 = dp_lambda_dict["Moebius"]
    plt.plot(x, y1, label="Art Dataset")
    plt.plot(x, y2, label="Books Dataset")
    plt.plot(x, y3, label="Dolls Dataset")
    plt.plot(x, y4, label="Laundry Dataset")
    plt.plot(x, y5, label="Reindeer Dataset")
    plt.plot(x, y6, label="Moebius Dataset")
    plt.plot()

    plt.xlabel("Lambda")
    plt.ylabel(metric.upper())
    plt.title(f"{metric.upper()} vs Lambda (Dynamic Programming Approach)")
    plt.legend(loc="upper right")

    return x, [y1, y2, y3, y4, y5, y6]
    


def metrics_vs_window_plot(metric, ax):
    output_dict = {}
    for name in dataset:
        output_dict[name] = {}
        for algo in algorithms:
            output_dict[name][algo] = {}
            for size in window_size:
                metric_func = metrics[metric]
                image_name = "{}_window{}_{}.png".format(name, size, algo)
                gt_img = read_image(data_dir_GT + f"{name}/" + f"{name}_GT.png")
                calc_img = read_image(data_dir + f"{name}/" + image_name)
                out_value = float(metric_func(gt_img, calc_img))
                output_dict[name][algo][size] = out_value

    for id, name in enumerate(dataset):
        first_index = int(id/3)

        x1 = [1, 3, 5]
        y1 = [output_dict[name]['opencv'][1], output_dict[name]['opencv'][3], output_dict[name]['opencv'][5]]

        x2 = [1, 3, 5]
        y2 = [output_dict[name]['naive'][1], output_dict[name]['naive'][3], output_dict[name]['naive'][5]]

        x3 = [1, 3, 5]
        y3 = [output_dict[name]['dynamic'][1], output_dict[name]['dynamic'][3], output_dict[name]['dynamic'][5]]

        width = np.min(np.diff(x3))/5

        ax[first_index][id % 3].bar(x1-width, y1, label="OpenCV_StereBM", color='royalblue', width=0.5)
        ax[first_index][id % 3].bar(x2, y2, label="Naive Approach", color='green', width=0.5)
        ax[first_index][id % 3].bar(x3+width, y3, label="Dynamic Approach ", color='orange', width=0.5)
        ax[first_index][id % 3].plot()

        ax[first_index][id % 3].set_xlabel("window size")
        ax[first_index][id % 3].set_ylabel(metric.upper())
        ax[first_index][id % 3].set_title("{} Dataset".format(name))
        ax[first_index][id % 3].legend()

def read_time_file(file):
    """From a list of integers in a file, creates a list of tuples"""
    with open(file, 'r') as f:
        return([float(x) for x in re.findall(r'[\d]*[.][\d]+', f.read())])

def time_vs_window_plot(ax):
    processing_time_dict = {}
    for name in dataset:
        processing_time_dict[name] = {}
        for size in window_size:
            processing_time_dict[name][size] = {}
            file_name = "{}_window{}_processing_time.txt".format(name, size)
            time_arr = read_time_file(data_dir + f"{name}/" + file_name)
            processing_time_dict[name][size] = time_arr

    for id, name in enumerate(dataset):
        first_index = int(id/3)

        x1 = [1, 3, 5]
        y1 = [processing_time_dict[name][1][0], processing_time_dict[name][3][0], processing_time_dict[name][5][0]]

        x2 = [1, 3, 5]
        y2 = [processing_time_dict[name][1][1], processing_time_dict[name][3][1], processing_time_dict[name][5][1]]

        x3 = [1, 3, 5]
        y3 = [processing_time_dict[name][1][2] * 100, processing_time_dict[name][3][2], processing_time_dict[name][5][2] * 100]

        width = np.min(np.diff(x3))/5

        ax[first_index][id % 3].bar(x1-width, y1, label="Naive", color='green', width=0.5)
        ax[first_index][id % 3].bar(x2, y2, label="DP", color='royalblue', width=0.5)
        ax[first_index][id % 3].bar(x3+width, y3, label="OpenCV_StereoBM", color='', width=0.5)
        ax[first_index][id % 3].plot()

        ax[first_index][id % 3].set_xlabel("window size")
        ax[first_index][id % 3].set_ylabel("Time (seconds)")
        ax[first_index][id % 3].set_title("{} Dataset".format(name))
        ax[first_index][id % 3].legend()


    
