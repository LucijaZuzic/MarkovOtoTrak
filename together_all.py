from utilities import load_object, save_object, compare_traj_and_sample, load_traj_name, preprocess_long_lat, scale_long_lat, process_time, get_sides_from_angle, fill_gap
import os
import numpy as np
import pandas as pd

predicted_time = load_object("predicted/predicted_time")    
predicted_longitude_no_abs = load_object("predicted/predicted_longitude_no_abs")  
predicted_latitude_no_abs = load_object("predicted/predicted_latitude_no_abs")  
predicted_direction = load_object("predicted/predicted_direction")   
predicted_speed = load_object("predicted/predicted_speed")   
  
metric_names = ["simpson x", "trapz x", "simpson y", "trapz y", "euclidean"]  

metric_names_longer = []
for metric_name in metric_names:  
    metric_names_longer.append(metric_name)

distance_predicted = dict()
distance_array = dict()
 
count_best_longit = dict()
count_best_latit = dict()
count_best_longit_latit = dict()

count_best_longit_metric = dict()
count_best_latit_metric = dict()
count_best_longit_latit_metric = dict() 
 
for metric_name in metric_names_longer:
    count_best_longit_metric[metric_name] = dict()
    count_best_latit_metric[metric_name] = dict()
    count_best_longit_latit_metric[metric_name] = dict() 

best_match_for_metric = dict()
best_match_for_metric_long_lat = dict()
best_match_for_metric_long = dict()
best_match_for_metric_lat = dict()
worst_match_for_metric = dict()
worst_match_for_metric_long_lat = dict()
worst_match_for_metric_long = dict()
worst_match_for_metric_lat = dict()
best_match_name_for_metric = dict()
best_match_name_for_metric_long_lat = dict()
best_match_name_for_metric_long = dict()
best_match_name_for_metric_lat = dict()
worst_match_name_for_metric = dict()
worst_match_name_for_metric_long_lat = dict()
worst_match_name_for_metric_long = dict()
worst_match_name_for_metric_lat = dict()
for metric_name in metric_names_longer:  
    best_match_for_metric[metric_name] = 100000
    best_match_for_metric_long_lat[metric_name] = dict()
    best_match_for_metric_long[metric_name] = dict()
    best_match_for_metric_lat[metric_name] = dict()
    worst_match_for_metric[metric_name] = 0
    worst_match_for_metric_long_lat[metric_name] = dict()
    worst_match_for_metric_long[metric_name] = dict()
    worst_match_for_metric_lat[metric_name] = dict()
    best_match_name_for_metric[metric_name] = ""
    best_match_name_for_metric_long_lat[metric_name] = dict()
    best_match_name_for_metric_long[metric_name] = dict()
    best_match_name_for_metric_lat[metric_name] = dict()
    worst_match_name_for_metric[metric_name] = ""
    worst_match_name_for_metric_long_lat[metric_name] = dict()
    worst_match_name_for_metric_long[metric_name] = dict()
    worst_match_name_for_metric_lat[metric_name] = dict()

all_subdirs = os.listdir() 
  
times_cumulative = dict()
 
long_no_abs_cumulative = dict()
lat_no_abs_cumulative = dict()
   
longitude_from_speed_time_heading = dict()
latitude_from_speed_time_heading = dict()
 
longitude_from_speed_time_ones_heading = dict()
latitude_from_speed_time_ones_heading = dict()

longitude_from_speed_time_actual_heading = dict()
latitude_from_speed_time_actual_heading = dict()

size = 8 
long_dict = dict()
lat_dict = dict()
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

    distance_predicted[subdir_name] = dict() 
        
    for some_file in all_files:  
        if subdir_name + "/cleaned_csv/" + some_file in bad_rides_filenames or subdir_name + "/cleaned_csv/" + some_file in gap_rides_filenames or some_file in train_rides: 
            continue
    
        file_with_ride = pd.read_csv(subdir_name + "/cleaned_csv/" + some_file)
        longitudes, latitudes, times = load_traj_name(subdir_name + "/cleaned_csv/" + some_file)
        longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
        longitudes, latitudes = scale_long_lat(longitudes, latitudes, 0.1, 0.1, True)
        times_processed = [process_time(time_new) for time_new in times] 
        time_int = [np.round(times_processed[time_index + 1] - times_processed[time_index], 3) for time_index in range(len(times_processed) - 1)] 
        times = [np.round(times_processed[time_index] - times_processed[0], 3) for time_index in range(len(times_processed))] 
        longer_file_name = subdir_name + "/cleaned_csv/" + some_file
        time_no_gap = fill_gap(predicted_time[longer_file_name]) 
        longitudes_no_abs_no_gap = fill_gap(predicted_longitude_no_abs[longer_file_name])
        latitudes_no_abs_no_gap = fill_gap(predicted_latitude_no_abs[longer_file_name])
        direction_no_gap = fill_gap(predicted_direction[longer_file_name]) 
        speed_no_gap = fill_gap(predicted_speed[longer_file_name]) 
        
        x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
        y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]

        times_cumulative[longer_file_name] = [0]
        for time_index in range(len(time_no_gap)):
            times_cumulative[longer_file_name].append(times_cumulative[longer_file_name][-1] + time_no_gap[time_index])

        long_no_abs_cumulative[longer_file_name] = [longitudes[0]]
        lat_no_abs_cumulative[longer_file_name] = [latitudes[0]]
        for index_long in range(len(longitudes_no_abs_no_gap)):
            long_no_abs_cumulative[longer_file_name].append(long_no_abs_cumulative[longer_file_name][-1] + longitudes_no_abs_no_gap[index_long])
            lat_no_abs_cumulative[longer_file_name].append(lat_no_abs_cumulative[longer_file_name][-1] + latitudes_no_abs_no_gap[index_long])
            
        longitude_from_speed_time_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_heading[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(time_no_gap)): 
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600 * time_no_gap[index_long], new_dir)
            longitude_from_speed_time_heading[longer_file_name].append(longitude_from_speed_time_heading[longer_file_name][-1] + new_long)
            latitude_from_speed_time_heading[longer_file_name].append(latitude_from_speed_time_heading[longer_file_name][-1] + new_lat) 
         
        longitude_from_speed_time_actual_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_actual_heading[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(time_int)): 
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600 * time_int[index_long], new_dir)
            longitude_from_speed_time_actual_heading[longer_file_name].append(longitude_from_speed_time_actual_heading[longer_file_name][-1] + new_long)
            latitude_from_speed_time_actual_heading[longer_file_name].append(latitude_from_speed_time_actual_heading[longer_file_name][-1] + new_lat) 
         
        longitude_from_speed_time_ones_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_ones_heading[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(time_int)): 
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600, new_dir)
            longitude_from_speed_time_ones_heading[longer_file_name].append(longitude_from_speed_time_ones_heading[longer_file_name][-1] + new_long)
            latitude_from_speed_time_ones_heading[longer_file_name].append(latitude_from_speed_time_ones_heading[longer_file_name][-1] + new_lat) 
        
        long_dict[longer_file_name] = {
            "long no abs": long_no_abs_cumulative[longer_file_name], 
            "long speed dir": longitude_from_speed_time_heading[longer_file_name],  
            "long speed ones dir": longitude_from_speed_time_ones_heading[longer_file_name], 
            "long speed actual dir": longitude_from_speed_time_actual_heading[longer_file_name], 
        }
         
        lat_dict[longer_file_name] = {
            "lat no abs": lat_no_abs_cumulative[longer_file_name], 
            "lat speed dir": latitude_from_speed_time_heading[longer_file_name],      
            "lat speed ones dir": latitude_from_speed_time_ones_heading[longer_file_name], 
            "lat speed actual dir": latitude_from_speed_time_actual_heading[longer_file_name], 
        }
         
        long_names = long_dict[longer_file_name].keys() 
        lat_names = lat_dict[longer_file_name].keys() 
         
        distance_predicted[subdir_name][some_file] = dict() 
        for metric_name in metric_names_longer: 
            distance_predicted[subdir_name][some_file][metric_name] = dict()  
            for latit in lat_names:
                for longit in long_names: 
                    distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name)
 
        if "long" not in count_best_longit:
            for latit in lat_names:
                count_best_latit[latit] = 0 
                for metric_name in metric_names_longer:
                    count_best_latit_metric[metric_name][latit] = 0  
                    best_match_for_metric_lat[metric_name][latit] = 100000
                    worst_match_for_metric_lat[metric_name][latit] = 0
                    best_match_name_for_metric_lat[metric_name][latit] = ""
                    worst_match_name_for_metric_lat[metric_name][latit] = ""
                for longit in long_names:
                    count_best_longit[longit] = 0 
                    count_best_longit_latit[longit + "-" + latit] = 0 
                    for metric_name in metric_names_longer:  
                        count_best_longit_metric[metric_name][longit] = 0 
                        best_match_for_metric_long[metric_name][longit] = 100000
                        worst_match_for_metric_long[metric_name][longit] = 0
                        best_match_name_for_metric_long[metric_name][longit] = ""
                        worst_match_name_for_metric_long[metric_name][longit] = ""
                        count_best_longit_latit_metric[metric_name][longit + "-" + latit] = 0 
                        best_match_for_metric_long_lat[metric_name][longit + "-" + latit] = 100000
                        worst_match_for_metric_long_lat[metric_name][longit + "-" + latit] = 0
                        best_match_name_for_metric_long_lat[metric_name][longit + "-" + latit] = ""
                        worst_match_name_for_metric_long_lat[metric_name][longit + "-" + latit] = ""
  
        for metric_name in metric_names_longer:  
            min_for_metric = 100000
            best_name = "" 
            for latit in lat_names: 
                for longit in long_names: 
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] < min_for_metric:
                        min_for_metric = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        best_name = longit + "-" + latit
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] < best_match_for_metric[metric_name]:
                        best_match_for_metric[metric_name] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        best_match_name_for_metric[metric_name] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] > worst_match_for_metric[metric_name]:
                        worst_match_for_metric[metric_name] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        worst_match_name_for_metric[metric_name] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] < best_match_for_metric_long_lat[metric_name][longit + "-" + latit]:
                        best_match_for_metric_long_lat[metric_name][longit + "-" + latit] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        best_match_name_for_metric_long_lat[metric_name][longit + "-" + latit] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] > worst_match_for_metric_long_lat[metric_name][longit + "-" + latit]:
                        worst_match_for_metric_long_lat[metric_name][longit + "-" + latit] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        worst_match_name_for_metric_long_lat[metric_name][longit + "-" + latit] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] < best_match_for_metric_lat[metric_name][latit]:
                        best_match_for_metric_lat[metric_name][latit] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        best_match_name_for_metric_lat[metric_name][latit] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] > worst_match_for_metric_lat[metric_name][latit]:
                        worst_match_for_metric_lat[metric_name][latit] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        worst_match_name_for_metric_lat[metric_name][latit] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] < best_match_for_metric_long[metric_name][longit]:
                        best_match_for_metric_long[metric_name][longit] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        best_match_name_for_metric_long[metric_name][longit] = subdir_name + "/cleaned_csv/" + some_file
                    if distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] > worst_match_for_metric_long[metric_name][longit]:
                        worst_match_for_metric_long[metric_name][longit] = distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit]
                        worst_match_name_for_metric_long[metric_name][longit] = subdir_name + "/cleaned_csv/" + some_file
                         
            if best_name != '':
                    count_best_longit_latit[best_name] += 1
                    count_best_longit[best_name.split("-")[0]] += 1
                    count_best_latit[best_name.split("-")[1]] += 1
                    count_best_longit_latit_metric[metric_name][best_name] += 1
                    count_best_longit_metric[metric_name][best_name.split("-")[0]] += 1
                    count_best_latit_metric[metric_name][best_name.split("-")[1]] += 1  
        print("Done ", some_file)
    print("Done ", subdir_name)

if not os.path.isdir("markov_result"):
    os.makedirs("markov_result")

save_object("markov_result/long_dict", long_dict)
save_object("markov_result/lat_dict", lat_dict)
   
save_object("markov_result/distance_predicted", distance_predicted)

save_object("markov_result/count_best_longit_latit", count_best_longit_latit)
save_object("markov_result/count_best_longit", count_best_longit)
save_object("markov_result/count_best_latit", count_best_latit)
save_object("markov_result/count_best_longit_latit_metric", count_best_longit_latit_metric)
save_object("markov_result/count_best_longit_metric", count_best_longit_metric)
save_object("markov_result/count_best_latit_metric", count_best_latit_metric)

save_object("markov_result/best_match_for_metric", best_match_for_metric)
save_object("markov_result/worst_match_for_metric", worst_match_for_metric)
save_object("markov_result/best_match_for_metric_long_lat", best_match_for_metric_long_lat)
save_object("markov_result/worst_match_for_metric_long_lat", worst_match_for_metric_long_lat)
save_object("markov_result/best_match_for_metric_long", best_match_for_metric_long)
save_object("markov_result/worst_match_for_metric_long", worst_match_for_metric_long)
save_object("markov_result/best_match_for_metric_lat", best_match_for_metric_lat)
save_object("markov_result/worst_match_for_metric_lat", worst_match_for_metric_lat)
   
save_object("markov_result/best_match_name_for_metric", best_match_name_for_metric)
save_object("markov_result/worst_match_name_for_metric", worst_match_name_for_metric)
save_object("markov_result/best_match_name_for_metric_long_lat", best_match_name_for_metric_long_lat)
save_object("markov_result/worst_match_name_for_metric_long_lat", worst_match_name_for_metric_long_lat)
save_object("markov_result/best_match_name_for_metric_long", best_match_name_for_metric_long)
save_object("markov_result/worst_match_name_for_metric_long", worst_match_name_for_metric_long)
save_object("markov_result/best_match_name_for_metric_lat", best_match_name_for_metric_lat)
save_object("markov_result/worst_match_name_for_metric_lat", worst_match_name_for_metric_lat)