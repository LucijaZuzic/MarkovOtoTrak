from utilities import *
     
predicted_time = load_object("predicted/predicted_time")  
predicted_distance = load_object("predicted/predicted_distance")  
predicted_longitude = load_object("predicted/predicted_longitude")  
predicted_latitude = load_object("predicted/predicted_latitude")  
predicted_longitude_sgn = load_object("predicted/predicted_longitude_sgn")  
predicted_latitude_sgn = load_object("predicted/predicted_latitude_sgn")  
predicted_longitude_no_abs = load_object("predicted/predicted_longitude_no_abs")  
predicted_latitude_no_abs = load_object("predicted/predicted_latitude_no_abs")  
predicted_direction = load_object("predicted/predicted_direction")  
predicted_direction_alternative = load_object("predicted/predicted_direction_alternative")
predicted_direction_abs_alternative = load_object("predicted/predicted_direction_abs_alternative")
predicted_speed = load_object("predicted/predicted_speed")  
predicted_speed_alternative = load_object("predicted/predicted_speed_alternative")  
predicted_x_speed_alternative = load_object("predicted/predicted_x_speed_alternative")  
predicted_y_speed_alternative = load_object("predicted/predicted_y_speed_alternative")  
predicted_x_speed_no_abs_alternative = load_object("predicted/predicted_x_speed_no_abs_alternative")  
predicted_y_speed_no_abs_alternative = load_object("predicted/predicted_y_speed_no_abs_alternative")  

metric_names = ["simpson", "trapz", "simpson x", "trapz x", "simpson y", "trapz y", "euclidean", "custom", "rays", "dtw"]
metric_names = ["simpson", "trapz", "simpson x", "trapz x", "simpson y", "trapz y", "euclidean", "custom", "rays"]  
metric_names = ["simpson x", "trapz x", "simpson y", "trapz y", "euclidean", "custom", "rays"]  
metric_names = ["simpson x", "trapz x", "simpson y", "trapz y", "euclidean", "custom"]  

metric_names_longer = []
for metric_name in metric_names: 
    if "simpson " in metric_name:
        metric_names_longer.append(metric_name)
        metric_names_longer.append(metric_name + " no time t1")
        metric_names_longer.append(metric_name + " no time sample")
        metric_names_longer.append(metric_name + " no time t1 no time sample") 
        continue
    if "rays" in metric_name: 
        metric_names_longer.append(metric_name + " scale")
        metric_names_longer.append(metric_name + " no scale")
        metric_names_longer.append(metric_name + " offset") 
        metric_names_longer.append(metric_name + " only offset") 
        continue
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

long_cumulative = dict()
lat_cumulative = dict()

long_no_abs_cumulative = dict()
lat_no_abs_cumulative = dict()

x_speed_cumulative = dict()
y_speed_cumulative = dict()
x_speed_cumulative_actual = dict()
y_speed_cumulative_actual = dict()
x_speed_cumulative_ones = dict()
y_speed_cumulative_ones = dict()

x_speed_no_abs_cumulative = dict()
y_speed_no_abs_cumulative = dict()
x_speed_no_abs_cumulative_actual = dict()
y_speed_no_abs_cumulative_actual = dict()
x_speed_no_abs_cumulative_ones = dict()
y_speed_no_abs_cumulative_ones = dict()

longitude_from_distance_heading_abs_alternative = dict()
latitude_from_distance_heading_abs_alternative = dict()

longitude_from_distance_heading_alternative = dict()
latitude_from_distance_heading_alternative = dict()

longitude_from_distance_heading = dict()
latitude_from_distance_heading = dict()

longitude_from_speed_alternative_time_heading_abs_alternative = dict()
latitude_from_speed_alternative_time_heading_abs_alternative = dict()

longitude_from_speed_alternative_time_heading_alternative = dict()
latitude_from_speed_alternative_time_heading_alternative = dict()

longitude_from_speed_alternative_time_heading = dict()
latitude_from_speed_alternative_time_heading = dict()

longitude_from_speed_time_heading_abs_alternative = dict()
latitude_from_speed_time_heading_abs_alternative = dict()

longitude_from_speed_time_heading_alternative = dict()
latitude_from_speed_time_heading_alternative = dict()

longitude_from_speed_time_heading = dict()
latitude_from_speed_time_heading = dict()

longitude_from_speed_alternative_time_actual_heading_abs_alternative = dict()
latitude_from_speed_alternative_time_actual_heading_abs_alternative = dict()

longitude_from_speed_alternative_time_actual_heading_alternative = dict()
latitude_from_speed_alternative_time_actual_heading_alternative = dict()

longitude_from_speed_alternative_time_actual_heading = dict()
latitude_from_speed_alternative_time_actual_heading = dict()

longitude_from_speed_time_actual_heading_abs_alternative = dict()
latitude_from_speed_time_actual_heading_abs_alternative = dict()

longitude_from_speed_time_actual_heading_alternative = dict()
latitude_from_speed_time_actual_heading_alternative = dict()

longitude_from_speed_time_actual_heading = dict()
latitude_from_speed_time_actual_heading = dict()
 
longitude_from_speed_alternative_time_ones_heading_abs_alternative = dict()
latitude_from_speed_alternative_time_ones_heading_abs_alternative = dict()

longitude_from_speed_alternative_time_ones_heading_alternative = dict()
latitude_from_speed_alternative_time_ones_heading_alternative = dict()

longitude_from_speed_alternative_time_ones_heading = dict()
latitude_from_speed_alternative_time_ones_heading = dict()

longitude_from_speed_time_ones_heading_abs_alternative = dict()
latitude_from_speed_time_ones_heading_abs_alternative = dict()

longitude_from_speed_time_ones_heading_alternative = dict()
latitude_from_speed_time_ones_heading_alternative = dict()

longitude_from_speed_time_ones_heading = dict()
latitude_from_speed_time_ones_heading = dict()

size = 8
dotsx_original, dotsy_original = make_rays(size)
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
        distance_no_gap = fill_gap(predicted_distance[longer_file_name])
        longitudes_no_gap = fill_gap(predicted_longitude[longer_file_name])
        latitudes_no_gap = fill_gap(predicted_latitude[longer_file_name])
        longitudes_no_abs_no_gap = fill_gap(predicted_longitude_no_abs[longer_file_name])
        latitudes_no_abs_no_gap = fill_gap(predicted_latitude_no_abs[longer_file_name])
        direction_no_gap = fill_gap(predicted_direction[longer_file_name])
        direction_alternative_no_gap = fill_gap(predicted_direction_alternative[longer_file_name]) 
        direction_abs_alternative_no_gap = fill_gap(predicted_direction_abs_alternative[longer_file_name])
        speed_no_gap = fill_gap(predicted_speed[longer_file_name])
        speed_alternative_no_gap = fill_gap(predicted_speed_alternative[longer_file_name])
        x_speed_alternative_no_gap = fill_gap(predicted_x_speed_alternative[longer_file_name]) 
        y_speed_alternative_no_gap = fill_gap(predicted_y_speed_alternative[longer_file_name])
        x_speed_no_abs_alternative_no_gap = fill_gap(predicted_x_speed_no_abs_alternative[longer_file_name])
        y_speed_no_abs_alternative_no_gap = fill_gap(predicted_y_speed_no_abs_alternative[longer_file_name])
        
        x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
        y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]

        times_cumulative[longer_file_name] = [0]
        for time_index in range(len(time_no_gap)):
            times_cumulative[longer_file_name].append(times_cumulative[longer_file_name][-1] + time_no_gap[time_index])

        long_cumulative[longer_file_name] = [longitudes[0]]
        lat_cumulative[longer_file_name] = [latitudes[0]]
        for index_long in range(len(longitudes_no_gap)):
            long_cumulative[longer_file_name].append(long_cumulative[longer_file_name][-1] - longitudes_no_gap[index_long] + longitudes_no_gap[index_long] * 2 * predicted_longitude_sgn[longer_file_name][index_long])
            lat_cumulative[longer_file_name].append(lat_cumulative[longer_file_name][-1] - latitudes_no_gap[index_long] + latitudes_no_gap[index_long] * 2 * predicted_latitude_sgn[longer_file_name][index_long])

        long_no_abs_cumulative[longer_file_name] = [longitudes[0]]
        lat_no_abs_cumulative[longer_file_name] = [latitudes[0]]
        for index_long in range(len(longitudes_no_abs_no_gap)):
            long_no_abs_cumulative[longer_file_name].append(long_no_abs_cumulative[longer_file_name][-1] + longitudes_no_abs_no_gap[index_long])
            lat_no_abs_cumulative[longer_file_name].append(lat_no_abs_cumulative[longer_file_name][-1] + latitudes_no_abs_no_gap[index_long])
           
        x_speed_cumulative[longer_file_name] = [longitudes[0]] 
        y_speed_cumulative[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(x_speed_alternative_no_gap)):
            x_speed_cumulative[longer_file_name].append(x_speed_cumulative[longer_file_name][-1] - x_speed_alternative_no_gap[index_long] * time_no_gap[index_long] + x_speed_alternative_no_gap[index_long] * time_no_gap[index_long] * 2 * predicted_longitude_sgn[longer_file_name][index_long])
            y_speed_cumulative[longer_file_name].append(y_speed_cumulative[longer_file_name][-1] - y_speed_alternative_no_gap[index_long] * time_no_gap[index_long] + y_speed_alternative_no_gap[index_long] * time_no_gap[index_long] * 2 * predicted_latitude_sgn[longer_file_name][index_long])
        
        x_speed_no_abs_cumulative[longer_file_name] = [longitudes[0]]    
        y_speed_no_abs_cumulative[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(x_speed_no_abs_alternative_no_gap)):
            x_speed_no_abs_cumulative[longer_file_name].append(x_speed_no_abs_cumulative[longer_file_name][-1] + x_speed_no_abs_alternative_no_gap[index_long] * time_no_gap[index_long])
            y_speed_no_abs_cumulative[longer_file_name].append(y_speed_no_abs_cumulative[longer_file_name][-1] + y_speed_no_abs_alternative_no_gap[index_long] * time_no_gap[index_long])
        
        x_speed_cumulative_actual[longer_file_name] = [longitudes[0]] 
        y_speed_cumulative_actual[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(x_speed_alternative_no_gap)):
            x_speed_cumulative_actual[longer_file_name].append(x_speed_cumulative_actual[longer_file_name][-1] - x_speed_alternative_no_gap[index_long] * time_int[index_long] + x_speed_alternative_no_gap[index_long] * time_int[index_long] * 2 * predicted_longitude_sgn[longer_file_name][index_long])
            y_speed_cumulative_actual[longer_file_name].append(y_speed_cumulative_actual[longer_file_name][-1] - y_speed_alternative_no_gap[index_long] * time_int[index_long] + y_speed_alternative_no_gap[index_long] * time_int[index_long] * 2 * predicted_latitude_sgn[longer_file_name][index_long])
        
        x_speed_no_abs_cumulative_actual[longer_file_name] = [longitudes[0]]    
        y_speed_no_abs_cumulative_actual[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(x_speed_no_abs_alternative_no_gap)):
            x_speed_no_abs_cumulative_actual[longer_file_name].append(x_speed_no_abs_cumulative_actual[longer_file_name][-1] + x_speed_no_abs_alternative_no_gap[index_long] * time_int[index_long])
            y_speed_no_abs_cumulative_actual[longer_file_name].append(y_speed_no_abs_cumulative_actual[longer_file_name][-1] + y_speed_no_abs_alternative_no_gap[index_long] * time_int[index_long])
        
        x_speed_cumulative_ones[longer_file_name] = [longitudes[0]] 
        y_speed_cumulative_ones[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(x_speed_alternative_no_gap)):
            x_speed_cumulative_ones[longer_file_name].append(x_speed_cumulative_ones[longer_file_name][-1] - x_speed_alternative_no_gap[index_long] + x_speed_alternative_no_gap[index_long] * 2 * predicted_longitude_sgn[longer_file_name][index_long])
            y_speed_cumulative_ones[longer_file_name].append(y_speed_cumulative_ones[longer_file_name][-1] - y_speed_alternative_no_gap[index_long] + y_speed_alternative_no_gap[index_long] * 2 * predicted_latitude_sgn[longer_file_name][index_long])
        
        x_speed_no_abs_cumulative_ones[longer_file_name] = [longitudes[0]]    
        y_speed_no_abs_cumulative_ones[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(x_speed_no_abs_alternative_no_gap)):
            x_speed_no_abs_cumulative_ones[longer_file_name].append(x_speed_no_abs_cumulative_ones[longer_file_name][-1] + x_speed_no_abs_alternative_no_gap[index_long])
            y_speed_no_abs_cumulative_ones[longer_file_name].append(y_speed_no_abs_cumulative_ones[longer_file_name][-1] + y_speed_no_abs_alternative_no_gap[index_long])
        
        longitude_from_distance_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_distance_heading_abs_alternative[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir
            new_long, new_lat = get_sides_from_angle(distance_no_gap[index_long], new_dir)
            longitude_from_distance_heading_abs_alternative[longer_file_name].append(longitude_from_distance_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_distance_heading_abs_alternative[longer_file_name].append(latitude_from_distance_heading_abs_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_distance_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_distance_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(distance_no_gap)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(distance_no_gap[index_long], cumulative_heading)
            longitude_from_distance_heading_alternative[longer_file_name].append(longitude_from_distance_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_distance_heading_alternative[longer_file_name].append(latitude_from_distance_heading_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_distance_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_distance_heading[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(distance_no_gap)):
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(distance_no_gap[index_long], new_dir)
            longitude_from_distance_heading[longer_file_name].append(longitude_from_distance_heading[longer_file_name][-1] + new_long)
            latitude_from_distance_heading[longer_file_name].append(latitude_from_distance_heading[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir 
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long] * time_no_gap[index_long], new_dir)
            longitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name].append(longitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name].append(latitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(time_no_gap)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long] * time_no_gap[index_long], cumulative_heading)
            longitude_from_speed_alternative_time_heading_alternative[longer_file_name].append(longitude_from_speed_alternative_time_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_heading_alternative[longer_file_name].append(latitude_from_speed_alternative_time_heading_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_heading[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(time_no_gap)): 
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long] * time_no_gap[index_long], new_dir)
            longitude_from_speed_alternative_time_heading[longer_file_name].append(longitude_from_speed_alternative_time_heading[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_heading[longer_file_name].append(latitude_from_speed_alternative_time_heading[longer_file_name][-1] + new_lat) 
          
        longitude_from_speed_time_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_heading_abs_alternative[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir 
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600 * time_no_gap[index_long], new_dir)
            longitude_from_speed_time_heading_abs_alternative[longer_file_name].append(longitude_from_speed_time_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_time_heading_abs_alternative[longer_file_name].append(latitude_from_speed_time_heading_abs_alternative[longer_file_name][-1] + new_lat) 
        
        longitude_from_speed_time_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(time_no_gap)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600 * time_no_gap[index_long], cumulative_heading)
            longitude_from_speed_time_heading_alternative[longer_file_name].append(longitude_from_speed_time_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_time_heading_alternative[longer_file_name].append(latitude_from_speed_time_heading_alternative[longer_file_name][-1] + new_lat) 
        
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
        
        longitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir 
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long] * time_int[index_long], new_dir)
            longitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name].append(longitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name].append(latitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(time_int)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long] * time_int[index_long], cumulative_heading)
            longitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name].append(longitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name].append(latitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_actual_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_actual_heading[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(time_int)): 
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long] * time_int[index_long], new_dir)
            longitude_from_speed_alternative_time_actual_heading[longer_file_name].append(longitude_from_speed_alternative_time_actual_heading[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_actual_heading[longer_file_name].append(latitude_from_speed_alternative_time_actual_heading[longer_file_name][-1] + new_lat) 
          
        longitude_from_speed_time_actual_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_actual_heading_abs_alternative[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir 
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600 * time_int[index_long], new_dir)
            longitude_from_speed_time_actual_heading_abs_alternative[longer_file_name].append(longitude_from_speed_time_actual_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_time_actual_heading_abs_alternative[longer_file_name].append(latitude_from_speed_time_actual_heading_abs_alternative[longer_file_name][-1] + new_lat) 
        
        longitude_from_speed_time_actual_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_actual_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(time_int)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600 * time_int[index_long], cumulative_heading)
            longitude_from_speed_time_actual_heading_alternative[longer_file_name].append(longitude_from_speed_time_actual_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_time_actual_heading_alternative[longer_file_name].append(latitude_from_speed_time_actual_heading_alternative[longer_file_name][-1] + new_lat) 
        
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
     
        longitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir 
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long], new_dir)
            longitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name].append(longitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name].append(latitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(time_int)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long], cumulative_heading)
            longitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name].append(longitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name].append(latitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name][-1] + new_lat)
        
        longitude_from_speed_alternative_time_ones_heading[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_alternative_time_ones_heading[longer_file_name] = [latitudes[0]]  
        for index_long in range(len(time_int)): 
            new_dir = (90 - direction_no_gap[index_long] + 360) % 360 
            if not x_dir: 
                new_dir = (180 - new_dir + 360) % 360
            if not y_dir: 
                new_dir = 360 - new_dir 
            new_long, new_lat = get_sides_from_angle(speed_alternative_no_gap[index_long], new_dir)
            longitude_from_speed_alternative_time_ones_heading[longer_file_name].append(longitude_from_speed_alternative_time_ones_heading[longer_file_name][-1] + new_long)
            latitude_from_speed_alternative_time_ones_heading[longer_file_name].append(latitude_from_speed_alternative_time_ones_heading[longer_file_name][-1] + new_lat) 
          
        longitude_from_speed_time_ones_heading_abs_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_ones_heading_abs_alternative[longer_file_name] = [latitudes[0]]   
        for index_long in range(len(direction_abs_alternative_no_gap)):
            new_dir = direction_abs_alternative_no_gap[index_long]
            if not predicted_longitude_sgn[longer_file_name][index_long] and predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 - new_dir
            if predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 360 - new_dir
            if not predicted_longitude_sgn[longer_file_name][index_long] and not predicted_latitude_sgn[longer_file_name][index_long]:
                new_dir = 180 + new_dir 
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600, new_dir)
            longitude_from_speed_time_ones_heading_abs_alternative[longer_file_name].append(longitude_from_speed_time_ones_heading_abs_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_time_ones_heading_abs_alternative[longer_file_name].append(latitude_from_speed_time_ones_heading_abs_alternative[longer_file_name][-1] + new_lat) 
        
        longitude_from_speed_time_ones_heading_alternative[longer_file_name] = [longitudes[0]]  
        latitude_from_speed_time_ones_heading_alternative[longer_file_name] = [latitudes[0]]  
        cumulative_heading = 0
        for index_long in range(len(time_int)):
            cumulative_heading = direction_alternative_no_gap[index_long]
            new_long, new_lat = get_sides_from_angle(speed_no_gap[index_long] / 111 / 0.1 / 3600, cumulative_heading)
            longitude_from_speed_time_ones_heading_alternative[longer_file_name].append(longitude_from_speed_time_ones_heading_alternative[longer_file_name][-1] + new_long)
            latitude_from_speed_time_ones_heading_alternative[longer_file_name].append(latitude_from_speed_time_ones_heading_alternative[longer_file_name][-1] + new_lat) 
        
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
            "long": long_cumulative[longer_file_name],
            "long no abs": long_no_abs_cumulative[longer_file_name], 
            "x speed": x_speed_cumulative[longer_file_name], 
            "x speed no abs": x_speed_no_abs_cumulative[longer_file_name],
            "long dist dir abs alt": longitude_from_distance_heading_abs_alternative[longer_file_name],
            "long dist dir alt": longitude_from_distance_heading_alternative[longer_file_name],
            "long dist dir": longitude_from_distance_heading[longer_file_name],
            "long speed alt dir abs alt": longitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name],
            "long speed alt dir alt": longitude_from_speed_alternative_time_heading_alternative[longer_file_name],
            "long speed alt dir": longitude_from_speed_alternative_time_heading[longer_file_name], 
            "long speed dir abs alt": longitude_from_speed_time_heading_abs_alternative[longer_file_name],
            "long speed dir alt": longitude_from_speed_time_heading_alternative[longer_file_name],
            "long speed dir": longitude_from_speed_time_heading[longer_file_name], 
            "x speed actual": x_speed_cumulative_actual[longer_file_name], 
            "x speed actual no abs": x_speed_no_abs_cumulative_actual[longer_file_name],
            "long speed actual alt dir abs alt": longitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name],
            "long speed actual alt dir alt": longitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name],
            "long speed actual alt dir": longitude_from_speed_alternative_time_actual_heading[longer_file_name], 
            "long speed actual dir abs alt": longitude_from_speed_time_actual_heading_abs_alternative[longer_file_name],
            "long speed actual dir alt": longitude_from_speed_time_actual_heading_alternative[longer_file_name],
            "long speed actual dir": longitude_from_speed_time_actual_heading[longer_file_name], 
            "x speed ones": x_speed_cumulative_ones[longer_file_name], 
            "x speed ones no abs": x_speed_no_abs_cumulative_ones[longer_file_name],
            "long speed ones alt dir abs alt": longitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name],
            "long speed ones alt dir alt": longitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name],
            "long speed ones alt dir": longitude_from_speed_alternative_time_ones_heading[longer_file_name], 
            "long speed ones dir abs alt": longitude_from_speed_time_ones_heading_abs_alternative[longer_file_name],
            "long speed ones dir alt": longitude_from_speed_time_ones_heading_alternative[longer_file_name],
            "long speed ones dir": longitude_from_speed_time_ones_heading[longer_file_name], 
        }
         
        lat_dict[longer_file_name] = {
            "lat": lat_cumulative[longer_file_name],
            "lat no abs": lat_no_abs_cumulative[longer_file_name], 
            "y speed": y_speed_cumulative[longer_file_name], 
            "y speed no abs": y_speed_no_abs_cumulative[longer_file_name],
            "lat dist dir abs alt": latitude_from_distance_heading_abs_alternative[longer_file_name],
            "lat dist dir alt": latitude_from_distance_heading_alternative[longer_file_name],
            "lat dist dir": latitude_from_distance_heading[longer_file_name],
            "lat speed alt dir abs alt": latitude_from_speed_alternative_time_heading_abs_alternative[longer_file_name],
            "lat speed alt dir alt": latitude_from_speed_alternative_time_heading_alternative[longer_file_name],
            "lat speed alt dir": latitude_from_speed_alternative_time_heading[longer_file_name], 
            "lat speed dir abs alt": latitude_from_speed_time_heading_abs_alternative[longer_file_name],
            "lat speed dir alt": latitude_from_speed_time_heading_alternative[longer_file_name],
            "lat speed dir": latitude_from_speed_time_heading[longer_file_name],     
            "y speed actual": y_speed_cumulative_actual[longer_file_name], 
            "y speed actual no abs": y_speed_no_abs_cumulative_actual[longer_file_name],
            "lat speed actual alt dir abs alt": latitude_from_speed_alternative_time_actual_heading_abs_alternative[longer_file_name],
            "lat speed actual alt dir alt": latitude_from_speed_alternative_time_actual_heading_alternative[longer_file_name],
            "lat speed actual alt dir": latitude_from_speed_alternative_time_actual_heading[longer_file_name], 
            "lat speed actual dir abs alt": latitude_from_speed_time_actual_heading_abs_alternative[longer_file_name],
            "lat speed actual dir alt": latitude_from_speed_time_actual_heading_alternative[longer_file_name],
            "lat speed actual dir": latitude_from_speed_time_actual_heading[longer_file_name], 
            "y speed ones": y_speed_cumulative_ones[longer_file_name], 
            "y speed ones no abs": y_speed_no_abs_cumulative_ones[longer_file_name],
            "lat speed ones alt dir abs alt": latitude_from_speed_alternative_time_ones_heading_abs_alternative[longer_file_name],
            "lat speed ones alt dir alt": latitude_from_speed_alternative_time_ones_heading_alternative[longer_file_name],
            "lat speed ones alt dir": latitude_from_speed_alternative_time_ones_heading[longer_file_name], 
            "lat speed ones dir abs alt": latitude_from_speed_time_ones_heading_abs_alternative[longer_file_name],
            "lat speed ones dir alt": latitude_from_speed_time_ones_heading_alternative[longer_file_name],
            "lat speed ones dir": latitude_from_speed_time_ones_heading[longer_file_name], 
        }
        
        #plot_long_lat_dict(some_file, long_dict, lat_dict, subdir_name + "/cleaned_csv/" + some_file, long_dict[longer_file_name].keys(), lat_dict[longer_file_name].keys(), 0.1, 0.1)

        long_names = long_dict[longer_file_name].keys() 
        lat_names = lat_dict[longer_file_name].keys() 
         
        distance_predicted[subdir_name][some_file] = dict() 
        for metric_name in metric_names_longer: 
            distance_predicted[subdir_name][some_file][metric_name] = dict()  
            for latit in lat_names:
                for longit in long_names:
                    if " no time t1 no time sample" in metric_name:   
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" no time t1 no time sample", ""), True, True)
                        continue
                    if " no time t1" in metric_name:    
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" no time t1", ""), True, False)
                        continue
                    if " no time sample" in metric_name:  
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" no time sample", ""), False, True)
                        continue
                    if " only offset" in metric_name:    
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" only offset", ""), False, False, False, True, dotsx_original, dotsy_original)
                        continue
                    if " no scale" in metric_name:    
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" no scale", ""), False, False, False, False, dotsx_original, dotsy_original)
                        continue
                    if " scale" in metric_name:
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" scale", ""), False, False, True, False, dotsx_original, dotsy_original)
                        continue
                    if " offset" in metric_name:    
                        distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit] = compare_traj_and_sample(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], times_cumulative[longer_file_name], {"long": longitudes, "lat": latitudes, "time": times}, metric_name.replace(" offset", ""), False, False, True, True, dotsx_original, dotsy_original)
                        continue
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
                        
            #print(metric_name, best_name)
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
'''
print("count_best_longit_latit")
for x in dict(sorted(count_best_longit_latit.items(), key=lambda item: item[1], reverse=True)):
    if count_best_longit_latit[x] > 0:
        print(x, count_best_longit_latit[x]) 
 
print("count_best_latit")
for x in dict(sorted(count_best_latit.items(), key=lambda item: item[1], reverse=True)):
    if count_best_latit[x] > 0:
        print(x, count_best_latit[x]) 

print("count_best_longit")
for x in dict(sorted(count_best_longit.items(), key=lambda item: item[1], reverse=True)):
    if count_best_longit[x] > 0:
        print(x, count_best_longit[x]) 

for metric_name in metric_names_longer:   
    print(metric_name, "count_best_longit_latit")
    for x in dict(sorted(count_best_longit_latit_metric[metric_name].items(), key=lambda item: item[1], reverse=True)):
        if count_best_longit_latit_metric[metric_name][x] > 0:
            print(x, count_best_longit_latit_metric[metric_name][x]) 
    
    print(metric_name, "count_best_latit")
    for x in dict(sorted(count_best_latit_metric[metric_name].items(), key=lambda item: item[1], reverse=True)):
        if count_best_latit_metric[metric_name][x] > 0:
            print(x, count_best_latit_metric[metric_name][x]) 

    print(metric_name, "count_best_longit")
    for x in dict(sorted(count_best_longit_metric[metric_name].items(), key=lambda item: item[1], reverse=True)):
        if count_best_longit[x] > 0:
            print(x, count_best_longit_metric[metric_name][x]) 
 
for metric_name in metric_names_longer:   
    metric_file_name = best_match_name_for_metric[metric_name]    
    plot_long_lat_dict("Best for " + metric_name, long_dict, lat_dict, metric_file_name, long_dict[longer_file_name].keys(), lat_dict[longer_file_name].keys(), 0.1, 0.1)
    for long in long_dict[longer_file_name].keys():
        long_file_name = best_match_name_for_metric_long[metric_name][long]
        plot_long_lat_dict("Best for long " + metric_name, long_dict, lat_dict, long_file_name, [long], lat_dict[longer_file_name].keys(), 0.1, 0.1) 
        for lat in lat_dict[longer_file_name].keys():
            lat_file_name = best_match_name_for_metric_lat[metric_name][lat]
            plot_long_lat_dict("Best for lat " + metric_name, long_dict, lat_dict, lat_file_name, long_dict[longer_file_name].keys(), [lat], 0.1, 0.1) 
            long_lat_file_name = best_match_name_for_metric_long_lat[metric_name][long + "-" + lat]
            plot_long_lat_dict("Best for long lat " + metric_name, long_dict, lat_dict, long_lat_file_name, [long], [lat], 0.1, 0.1) 

for metric_name in metric_names_longer:   
    metric_file_name = worst_match_name_for_metric[metric_name]    
    plot_long_lat_dict("Worst for " + metric_name, long_dict, lat_dict, metric_file_name, long_dict[longer_file_name].keys(), lat_dict[longer_file_name].keys(), 0.1, 0.1)
    for long in long_dict[longer_file_name].keys():
        long_file_name = worst_match_name_for_metric_long[metric_name][long]
        plot_long_lat_dict("Worst for long " + metric_name, long_dict, lat_dict, long_file_name, [long], lat_dict[longer_file_name].keys(), 0.1, 0.1) 
        for lat in lat_dict[longer_file_name].keys():
            lat_file_name = worst_match_name_for_metric_lat[metric_name][lat]
            plot_long_lat_dict("Worst for lat " + metric_name, long_dict, lat_dict, lat_file_name, long_dict[longer_file_name].keys(), [lat], 0.1, 0.1) 
            long_lat_file_name = worst_match_name_for_metric_long_lat[metric_name][long + "-" + lat]
            plot_long_lat_dict("Worst for long lat " + metric_name, long_dict, lat_dict, long_lat_file_name, [long], [lat], 0.1, 0.1)
'''