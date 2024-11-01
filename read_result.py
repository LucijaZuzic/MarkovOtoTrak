from utilities import load_object, preprocess_long_lat, scale_long_lat, process_time, fill_gap, format_e, translate_var
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

all_subdirs = os.listdir() 
  
def read_heading(title): 
    all_x = load_object("predicted/predicted_direction")
    all_mine = dict()
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
            train_rides= load_object(subdir_name + "/train_rides")
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            directions = list(file_with_ride["fields_direction"]) 
            direction_int = [np.round(direction, 0) for direction in directions]
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = direction_int
    return end_read(title, all_x, all_mine, True)
 
def read_latitude_no_abs(title): 
    all_x = load_object("predicted/predicted_latitude_no_abs")
    all_mine = dict()
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
            train_rides= load_object(subdir_name + "/train_rides")
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            longitudes = list(file_with_ride["fields_longitude"]) 
            latitudes = list(file_with_ride["fields_latitude"]) 
            longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
            longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
            latitude_int = [np.round(latitudes[latitude_index + 1] - latitudes[latitude_index], 10) for latitude_index in range(len(latitudes) - 1)]
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = latitude_int
    return end_read(title, all_x, all_mine)
  
def read_longitude_no_abs(title): 
    all_x = load_object("predicted/predicted_longitude_no_abs")
    all_mine = dict()
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
            train_rides= load_object(subdir_name + "/train_rides")
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            longitudes = list(file_with_ride["fields_longitude"]) 
            latitudes = list(file_with_ride["fields_latitude"]) 
            longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
            longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
            longitude_int = [np.round(longitudes[longitude_index + 1] - longitudes[longitude_index], 10) for longitude_index in range(len(longitudes) - 1)]
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = longitude_int
    return end_read(title, all_x, all_mine)
  
def read_speed(title): 
    all_x = load_object("predicted/predicted_speed")
    all_mine = dict()
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
            train_rides= load_object(subdir_name + "/train_rides")
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            speeds = list(file_with_ride["fields_speed"]) 
            speed_int = [np.round(speed, 0) for speed in speeds] 
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = speed_int
    return end_read(title, all_x, all_mine)

def read_time(title): 
    all_x = load_object("predicted/predicted_time")
    all_mine = dict()
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
            train_rides= load_object(subdir_name + "/train_rides")
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
                continue
        
            file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
            times = list(file_with_ride["time"])
            times_processed = [process_time(time_new) for time_new in times] 
            time_int = [np.round(times_processed[time_index + 1] - times_processed[time_index], 3) for time_index in range(len(times_processed) - 1)] 
            for time_index in range(len(time_int)):
                    if time_int[time_index] == 0: 
                        time_int[time_index] = 10 ** -20 
            all_mine[subdir_name + "/cleaned_csv/" + some_file] = time_int
    return end_read(title, all_x, all_mine)
   
def end_read(title, all_x, all_mine, isangle = False):
    total_match_score = 0
    total_guesses = 0 
    total_guesses_no_empty = 0
    delta_series_total = []  
    data_series_total = []  
    minval = 10000000
    maxval = -10000000 
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
            train_rides= load_object(subdir_name + "/train_rides")
            
        for some_file in all_files:  
            if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
                continue

            x = all_x[subdir_name + "/cleaned_csv/" + some_file]
            other = all_mine[subdir_name + "/cleaned_csv/" + some_file]
            minval = min(minval, min(fill_gap(x)))
            maxval = max(maxval, max(fill_gap(x))) 
            n = len(x)
            match_score = 0 
            no_empty = 0
            delta_series = [] 
            for i in range(n):
                if x[i] == other[i]:
                    match_score += 1
                if x[i] != "undefined":
                    no_empty += 1
                    delta_x = abs(other[i] - x[i])
                    if isangle:
                        if delta_x > 180:
                            delta_x = 360 - delta_x
                    delta_series.append(delta_x)
                    delta_series_total.append(delta_x)
                    data_series_total.append(x[i]) 

            total_guesses += n
            total_guesses_no_empty += no_empty
            total_match_score += match_score  

    no_extension = title.replace(".png", "").replace("markov_hist/", "").capitalize()
    plt.rcParams.update({'font.size': 22})
    
    lines_ret = [
        [translate_var[no_extension], format_e(minval), format_e(maxval), "$" + str(np.round(total_match_score / total_guesses * 100, 2)) + "\\%$"],
        [translate_var[no_extension], format_e(np.average(delta_series_total)), format_e(np.std(delta_series_total)), format_e(np.var(delta_series_total))],
        [translate_var[no_extension], format_e(np.quantile(delta_series_total, 0.25)), format_e(np.quantile(delta_series_total, 0.50)), format_e(np.quantile(delta_series_total, 0.75)), format_e(max(delta_series_total))],
        [translate_var[no_extension], format_e(np.average(delta_series_total)), format_e(np.std(delta_series_total)), format_e(np.var(delta_series_total)), format_e(minval), format_e(np.quantile(delta_series_total, 0.25)), format_e(np.quantile(delta_series_total, 0.50)), format_e(np.quantile(delta_series_total, 0.75)), format_e(np.quantile(delta_series_total, 0.80)),format_e(np.quantile(delta_series_total, 0.85)), format_e(np.quantile(delta_series_total, 0.90)), format_e(np.quantile(delta_series_total, 0.95)), format_e(max(delta_series_total))],
        [translate_var[no_extension], format_e(np.average(delta_series_total)), format_e(np.std(delta_series_total)), format_e(np.var(delta_series_total)), format_e(minval), format_e(np.quantile(delta_series_total, 0.25)), format_e(np.quantile(delta_series_total, 0.50)), format_e(np.quantile(delta_series_total, 0.75)), format_e(max(delta_series_total))]
    ]    
    plt.figure(figsize=(25, 10))
    plt.subplot(1, 2, 1)
    plt.title(translate_var[no_extension] + " - Estimated")   
    plt.hist(data_series_total)
    plt.xlabel("Estimated (" + unit_engl[no_extension] + ")")
    plt.ylabel("Number of occurences")
    plt.subplot(1, 2, 2) 
    plt.title(no_extension + " delta")  
    plt.hist(delta_series_total)
    plt.xlabel("Delta (" + unit_engl[no_extension] + ")")
    plt.ylabel("Number of occurences")
    plt.savefig(title, bbox_inches = "tight")
    plt.close()

    plt.figure(figsize=(30, 10))
    plt.subplot(1, 2, 1)
    plt.title(translate_var[no_extension] + " - Procjena")  
    plt.hist(data_series_total)
    plt.xlabel("Procjena (" + unit[no_extension] + ")")
    plt.ylabel("Broj pojavljivanja")
    plt.subplot(1, 2, 2) 
    plt.title(translate_var[no_extension] + " - Razlika")  
    plt.hist(delta_series_total)
    plt.xlabel("Razlika (" + unit[no_extension] + ")")
    plt.ylabel("Broj pojavljivanja")
    plt.savefig(title.replace(".png", "_hr.png"), bbox_inches = "tight")
    plt.close()

    return lines_ret

if not os.path.isdir("markov_hist"):
    os.makedirs("markov_hist")
 
unit = { 
        "Direction": "stupnjevi",  
        "Latitude no abs": "desetinke stupnja",  
        "Longitude no abs": "desetinke stupnja",   
        "Time": "sekunde",
        "Speed": "km / h", 
    }

unit_engl = { 
        "Direction": "degrees",  
        "Latitude no abs": "0.1 degrees",  
        "Longitude no abs": "0.1 degrees",   
        "Time": "seconds",
        "Speed": "km / h", 
    }

all_lines_ret = [] 
all_lines_ret.append(read_heading("markov_hist/direction.png")) 
all_lines_ret.append(read_latitude_no_abs("markov_hist/latitude no abs.png"))  
all_lines_ret.append(read_longitude_no_abs("markov_hist/longitude no abs.png")) 
all_lines_ret.append(read_time("markov_hist/time.png"))
all_lines_ret.append(read_speed("markov_hist/speed.png")) 

for table_number in range(len(all_lines_ret[0])):
    strpr = ""
    for var_index in range(len(all_lines_ret)):
        for val_index in range(len(all_lines_ret[var_index][table_number])):
            if val_index != len(all_lines_ret[var_index][table_number]) - 1:
                strpr += all_lines_ret[var_index][table_number][val_index] + " & "
            else:
                strpr += all_lines_ret[var_index][table_number][val_index] + " \\\\ \\hline\n"
    print(strpr)