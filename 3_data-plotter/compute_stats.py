import os
import numpy as np
import pandas as pd
import math
from matplotlib import pyplot as plt
import argparse

image_name = "plot"
image_format = ".png"
image_counter = 1
surface_thresholds = range(0, 300, 50)
complexity_thresholds = range(5)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--inference", required=True, help="Path to where detection data resides")
ap.add_argument("-gt", "--groundtruth", required=True, help="Path to where groundtruth data resides")
ap.add_argument("-o", "--output_graphs", required=True, help="Output folder fo the graphs")
args = vars(ap.parse_args())

detection_path = args["inference"]
groundtruth_path = args["groundtruth"]
output_folder = args["output_graphs"]

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

detection = pd.read_csv(detection_path, na_values="-")
groundtruth = pd.read_csv(groundtruth_path, na_values="-")

detection_values = detection.values
groundtruth_values = groundtruth.values

headers = detection.columns

def create_ranges(thresholds):
    thresholds_length = len(thresholds)
    interv = [[thresholds[n], thresholds[n+1]] for n in range(thresholds_length) if n < thresholds_length-1]
    interv.append([thresholds[thresholds_length-1], math.inf])
    print(interv)
    return interv


def count_thresholds(ranges, graphtype="2D"):
    col_index = 1 if graphtype == "2D" else (2 if graphtype == "3D" else 3)
    a = detection_values[:, col_index] - groundtruth_values[:, col_index]
    total_values = len(a)
    error_count = np.sum(np.isnan(a))
    a = np.abs(np.where(np.isfinite(a), a, 0))
    result = [(np.count_nonzero((a >= ranges[n][0]) & (a < ranges[n][1]))/total_values)*100 for n in range(len(ranges))]
    result.append(error_count)
    print(result)
    return result


def plot_results(values, labels, tittle="", iscomplexity=False):
    plt.title(tittle)
    if not iscomplexity:
        plt.xlabel('Squared feet error')
    else:
        plt.xlabel('Inches per foot')
    plt.ylabel('Percentage of blueprints')
    plt.bar(range(len(values)), values)
    plt.xticks(range(len(values)), labels)
    plt.ylim([0, 100])
    plt.savefig(output_folder + "/" + image_name + str(image_counter) + image_format)
    plt.show()


def ranges_to_string(ranges):
    return ["[" + str(ranges[n][0]) + ", " + str(ranges[n][1]) + ")" for n in range(len(ranges))]


ranges = create_ranges(surface_thresholds)
labels = ranges_to_string(ranges)
labels.append("Errors")
#Plotting 2D
results = count_thresholds(ranges)
plot_results(results, labels, headers[1])
image_counter += 1
#Plotting 3D
results = count_thresholds(ranges, "3D")
plot_results(results, labels, headers[2])
image_counter += 1
#Plotting Complexity
ranges = create_ranges(complexity_thresholds)
labels = ranges_to_string(ranges)
labels.append("Errors")
results = count_thresholds(ranges, "Complexity")
plot_results(results, labels, headers[3], True)
