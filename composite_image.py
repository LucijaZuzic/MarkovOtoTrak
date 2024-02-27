from utilities import load_object, save_object, fix_prob, predict_prob, predict_prob_with_array
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
    
all_subdirs = os.listdir() 
testn = 0
trainn = 0

for subdir_name in all_subdirs: 
    if not os.path.isdir(subdir_name) or "Vehicle" not in subdir_name:
        continue

    all_files = os.listdir(subdir_name + "/cleaned_csv/") 
    bad_rides_filenames = set()
    if os.path.isfile(subdir_name + "/bad_rides_filenames"):
        bad_rides_filenames = load_object(subdir_name + "/bad_rides_filenames")
    gap_rides_filenames = set()
    if os.path.isfile(subdir_name + "/gap_rides_filenames"):
        gap_rides_filenames = load_object(subdir_name + "/gap_rides_filenames")
    test_rides = set()
    if os.path.isfile(subdir_name + "/test_rides"):
        test_rides = load_object(subdir_name + "/test_rides")

    testn1 = 0
    trainn1 = 0
        
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames:
            continue
        if some_file in test_rides:
            testn += 1
            testn1 += 1
        else:
            trainn += 1
            trainn1 += 1
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"]) 
        latitudes = list(file_with_ride["fields_latitude"]) 
    print(subdir_name, testn1, trainn1)

print(testn, trainn)
        