from utilities import load_object, save_object, preprocess_long_lat, scale_long_lat, fix_prob, predict_prob, predict_prob_with_array
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

all_subdirs = os.listdir() 

if not os.path.isdir("num_occurences"):
    os.makedirs("num_occurences")
if not os.path.isdir("probability"):
    os.makedirs("probability")
if not os.path.isdir("predicted"):
    os.makedirs("predicted")
flag_replace = True

if flag_replace or not os.path.isfile("num_occurences/num_occurences_of_longitude_no_abs"):
    num_occurences_of_longitude_no_abs = dict()
    num_occurences_of_longitude_no_abs_in_next_step = dict()
    num_occurences_of_longitude_no_abs_in_next_next_step = dict()

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
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in test_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            longitudes = list(file_with_ride["fields_longitude"]) 
            latitudes = list(file_with_ride["fields_latitude"]) 
            longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
            longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
            longitude_int = [np.round(longitudes[longitude_index + 1] - longitudes[longitude_index], 10) for longitude_index in range(len(longitudes) - 1)]

            for longitude in longitude_int:
                if longitude not in num_occurences_of_longitude_no_abs:
                    num_occurences_of_longitude_no_abs[longitude] = 0
                num_occurences_of_longitude_no_abs[longitude] += 1
 
            for longitude_index in range(len(longitude_int) - 1):
                longitude = longitude_int[longitude_index]
                next_longitude_no_abs = longitude_int[longitude_index + 1]
                if longitude not in num_occurences_of_longitude_no_abs_in_next_step:
                    num_occurences_of_longitude_no_abs_in_next_step[longitude] = dict()
                if next_longitude_no_abs not in num_occurences_of_longitude_no_abs_in_next_step[longitude]:
                    num_occurences_of_longitude_no_abs_in_next_step[longitude][next_longitude_no_abs] = 0
                num_occurences_of_longitude_no_abs_in_next_step[longitude][next_longitude_no_abs] += 1
                if longitude_index < len(longitude_int) - 2:
                    next_next_longitude_no_abs = longitude_int[longitude_index + 2]
                    if longitude not in num_occurences_of_longitude_no_abs_in_next_next_step:
                        num_occurences_of_longitude_no_abs_in_next_next_step[longitude] = dict()
                    if next_longitude_no_abs not in num_occurences_of_longitude_no_abs_in_next_next_step[longitude]:
                        num_occurences_of_longitude_no_abs_in_next_next_step[longitude][next_longitude_no_abs] = dict()
                    if next_next_longitude_no_abs not in num_occurences_of_longitude_no_abs_in_next_next_step[longitude][next_longitude_no_abs]:
                        num_occurences_of_longitude_no_abs_in_next_next_step[longitude][next_longitude_no_abs][next_next_longitude_no_abs] = 0
                    num_occurences_of_longitude_no_abs_in_next_next_step[longitude][next_longitude_no_abs][next_next_longitude_no_abs] += 1
 
    save_object("num_occurences/num_occurences_of_longitude_no_abs", num_occurences_of_longitude_no_abs)
    
    plt.bar(num_occurences_of_longitude_no_abs.keys(), num_occurences_of_longitude_no_abs.values())
    plt.show()
 
    save_object("num_occurences/num_occurences_of_longitude_no_abs_in_next_step", num_occurences_of_longitude_no_abs_in_next_step)
    save_object("num_occurences/num_occurences_of_longitude_no_abs_in_next_next_step", num_occurences_of_longitude_no_abs_in_next_next_step)

    probability_of_longitude_no_abs, probability_of_longitude_no_abs_in_next_step, probability_of_longitude_no_abs_in_next_next_step = fix_prob(num_occurences_of_longitude_no_abs, num_occurences_of_longitude_no_abs_in_next_step, num_occurences_of_longitude_no_abs_in_next_next_step)
    
    save_object("probability/probability_of_longitude_no_abs", probability_of_longitude_no_abs)
    save_object("probability/probability_of_longitude_no_abs_in_next_step", probability_of_longitude_no_abs_in_next_step)
    save_object("probability/probability_of_longitude_no_abs_in_next_next_step", probability_of_longitude_no_abs_in_next_next_step)

probability_of_longitude_no_abs = load_object("probability/probability_of_longitude_no_abs") 
probability_of_longitude_no_abs_in_next_step = load_object("probability/probability_of_longitude_no_abs_in_next_step") 
probability_of_longitude_no_abs_in_next_next_step = load_object("probability/probability_of_longitude_no_abs_in_next_next_step")   

x = predict_prob(probability_of_longitude_no_abs, probability_of_longitude_no_abs_in_next_step, probability_of_longitude_no_abs_in_next_next_step, -1, 1, 10 ** -10)

plt.plot(x)
plt.xlabel('Longitudes',fontsize=20)
plt.ylabel(r'$S_{n}$',fontsize=20)
plt.show()

total_match_score = 0
total_guesses = 0 
total_guesses_no_empty = 0
delta_series_total = [] 
all_x = dict()
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
    train_rides = set()
    if os.path.isfile(subdir_name + "/train_rides"):
        train_rides = load_object(subdir_name + "/train_rides")
        
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
            continue
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes = list(file_with_ride["fields_longitude"]) 
        latitudes = list(file_with_ride["fields_latitude"]) 
        longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
        longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
        longitude_int = [np.round(longitudes[longitude_index + 1] - longitudes[longitude_index], 10) for longitude_index in range(len(longitudes) - 1)]
 
        x, n, match_score, no_empty, delta_series = predict_prob_with_array(probability_of_longitude_no_abs, probability_of_longitude_no_abs_in_next_step, probability_of_longitude_no_abs_in_next_next_step, longitude_int, -1, 1, 10 ** -10)
        total_guesses += n
        total_guesses_no_empty += no_empty
        total_match_score += match_score 
        for value_delta in delta_series:
            delta_series_total.append(value_delta)
        all_x[subdir_name + "/cleaned_csv/" + some_file] = x
save_object("predicted/predicted_longitude_no_abs", all_x)
print(total_match_score / total_guesses, total_match_score / total_guesses_no_empty, min(delta_series_total), np.quantile(delta_series_total, 0.25), np.quantile(delta_series_total, 0.5), np.quantile(delta_series_total, 0.75), max(delta_series_total), np.average(delta_series_total), np.std(delta_series_total), np.var(delta_series_total))
plt.hist(delta_series_total)
plt.show()