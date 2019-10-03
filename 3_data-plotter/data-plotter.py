from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

detection = pd.read_csv('./data/detection.csv', skiprows=1, na_values='-')
groundtruth = pd.read_csv('./data/groundtruth.csv', skiprows=1, na_values='-')


