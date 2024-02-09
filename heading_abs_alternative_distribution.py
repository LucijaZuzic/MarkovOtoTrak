from utilities import *
    
all_subdirs = os.listdir() 

if not os.path.isdir("num_occurences"):
    os.makedirs("num_occurences")
if not os.path.isdir("probability"):
    os.makedirs("probability")
if not os.path.isdir("predicted"):
    os.makedirs("predicted")
flag_replace = True

if flag_replace or not os.path.isfile("num_occurences/num_occurences_of_direction_abs_alternative"):
    num_occurences_of_direction_abs_alternative = dict()
    num_occurences_of_direction_abs_alternative_diff = dict()
    num_occurences_of_direction_abs_alternative_in_next_step = dict()
    num_occurences_of_direction_abs_alternative_in_next_next_step = dict()

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
            xdistance_int = [abs(longitudes[distance_index + 1] - longitudes[distance_index]) for distance_index in range(len(longitudes) - 1)]
            ydistance_int = [abs(latitudes[distance_index + 1] - latitudes[distance_index]) for distance_index in range(len(latitudes) - 1)]
            direction_alternative_int = []
            for heading_alternative_index in range(len(longitudes) - 1): 
                direction_alternative_int.append((360 + np.round(np.arctan2(ydistance_int[heading_alternative_index], xdistance_int[heading_alternative_index]) / np.pi * 180, 0)) % 360)
                   
            for direction_alternative in direction_alternative_int:
                if direction_alternative not in num_occurences_of_direction_abs_alternative:
                    num_occurences_of_direction_abs_alternative[direction_alternative] = 0
                num_occurences_of_direction_abs_alternative[direction_alternative] += 1

            for direction_alternative_index in range(len(direction_alternative_int) - 1):
                direction_alternative = direction_alternative_int[direction_alternative_index]
                next_direction_abs_alternative = direction_alternative_int[direction_alternative_index + 1]
                direction_alternative_diff = abs(next_direction_abs_alternative - direction_alternative)
                if direction_alternative_diff > 180:
                    direction_alternative_diff = 360 - direction_alternative_diff
                if direction_alternative_diff not in num_occurences_of_direction_abs_alternative_diff: 
                    num_occurences_of_direction_abs_alternative_diff[direction_alternative_diff] = 0
                num_occurences_of_direction_abs_alternative_diff[direction_alternative_diff] += 1
                if direction_alternative not in num_occurences_of_direction_abs_alternative_in_next_step:
                    num_occurences_of_direction_abs_alternative_in_next_step[direction_alternative] = dict()
                if next_direction_abs_alternative not in num_occurences_of_direction_abs_alternative_in_next_step[direction_alternative]:
                    num_occurences_of_direction_abs_alternative_in_next_step[direction_alternative][next_direction_abs_alternative] = 0
                num_occurences_of_direction_abs_alternative_in_next_step[direction_alternative][next_direction_abs_alternative] += 1
                if direction_alternative_index < len(direction_alternative_int) - 2:
                    next_next_direction_abs_alternative = direction_alternative_int[direction_alternative_index + 2]
                    if direction_alternative not in num_occurences_of_direction_abs_alternative_in_next_next_step:
                        num_occurences_of_direction_abs_alternative_in_next_next_step[direction_alternative] = dict()
                    if next_direction_abs_alternative not in num_occurences_of_direction_abs_alternative_in_next_next_step[direction_alternative]:
                        num_occurences_of_direction_abs_alternative_in_next_next_step[direction_alternative][next_direction_abs_alternative] = dict()
                    if next_next_direction_abs_alternative not in num_occurences_of_direction_abs_alternative_in_next_next_step[direction_alternative][next_direction_abs_alternative]:
                        num_occurences_of_direction_abs_alternative_in_next_next_step[direction_alternative][next_direction_abs_alternative][next_next_direction_abs_alternative] = 0
                    num_occurences_of_direction_abs_alternative_in_next_next_step[direction_alternative][next_direction_abs_alternative][next_next_direction_abs_alternative] += 1

    #print(num_occurences_of_direction_abs_alternative)
    save_object("num_occurences/num_occurences_of_direction_abs_alternative", num_occurences_of_direction_abs_alternative)
    #print(num_occurences_of_direction_abs_alternative.keys())

    #print(num_occurences_of_direction_abs_alternative_diff)
    plt.bar(num_occurences_of_direction_abs_alternative_diff.keys(), num_occurences_of_direction_abs_alternative_diff.values())
    plt.show()

    #print(num_occurences_of_direction_abs_alternative_in_next_step)
    save_object("num_occurences/num_occurences_of_direction_abs_alternative_in_next_step", num_occurences_of_direction_abs_alternative_in_next_step)
    #print(num_occurences_of_direction_abs_alternative_in_next_next_step)
    save_object("num_occurences/num_occurences_of_direction_abs_alternative_in_next_next_step", num_occurences_of_direction_abs_alternative_in_next_next_step)

    probability_of_direction_abs_alternative, probability_of_direction_abs_alternative_in_next_step, probability_of_direction_abs_alternative_in_next_next_step = fix_prob(num_occurences_of_direction_abs_alternative, num_occurences_of_direction_abs_alternative_in_next_step, num_occurences_of_direction_abs_alternative_in_next_next_step)
    #print(probability_of_direction_abs_alternative)
    save_object("probability/probability_of_direction_abs_alternative", probability_of_direction_abs_alternative)
    #print(probability_of_direction_abs_alternative_in_next_step)
    save_object("probability/probability_of_direction_abs_alternative_in_next_step", probability_of_direction_abs_alternative_in_next_step)
    #print(probability_of_direction_abs_alternative_in_next_next_step)
    save_object("probability/probability_of_direction_abs_alternative_in_next_next_step", probability_of_direction_abs_alternative_in_next_next_step)

probability_of_direction_abs_alternative = load_object("probability/probability_of_direction_abs_alternative") 
probability_of_direction_abs_alternative_in_next_step = load_object("probability/probability_of_direction_abs_alternative_in_next_step") 
probability_of_direction_abs_alternative_in_next_next_step = load_object("probability/probability_of_direction_abs_alternative_in_next_next_step")  

x = predict_prob(probability_of_direction_abs_alternative, probability_of_direction_abs_alternative_in_next_step, probability_of_direction_abs_alternative_in_next_next_step, 0, 90, 1)

plt.plot(x)
plt.xlabel('Directions',fontsize=20)
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
        xdistance_int = [abs(longitudes[distance_index + 1] - longitudes[distance_index]) for distance_index in range(len(longitudes) - 1)]
        ydistance_int = [abs(latitudes[distance_index + 1] - latitudes[distance_index]) for distance_index in range(len(latitudes) - 1)]
        direction_alternative_int = []
        for heading_alternative_index in range(len(longitudes) - 1): 
            direction_alternative_int.append((360 + np.round(np.arctan2(ydistance_int[heading_alternative_index], xdistance_int[heading_alternative_index]) / np.pi * 180, 0)) % 360)
           
        x, n, match_score, no_empty, delta_series = predict_prob_with_array(probability_of_direction_abs_alternative, probability_of_direction_abs_alternative_in_next_step, probability_of_direction_abs_alternative_in_next_next_step, direction_alternative_int, 0, 90, 1, isangle = True)
        total_guesses += n
        total_guesses_no_empty += no_empty
        total_match_score += match_score 
        for value_delta in delta_series:
            delta_series_total.append(value_delta)
        all_x[subdir_name + "/cleaned_csv/" + some_file] = x
save_object("predicted/predicted_direction_abs_alternative", all_x)
print(total_match_score / total_guesses, total_match_score / total_guesses_no_empty, min(delta_series_total), np.quantile(delta_series_total, 0.25), np.quantile(delta_series_total, 0.5), np.quantile(delta_series_total, 0.75), max(delta_series_total), np.average(delta_series_total), np.std(delta_series_total), np.var(delta_series_total))
plt.hist(delta_series_total)
plt.show()