from utilities import *
    
all_subdirs = os.listdir()  

if not os.path.isdir("num_occurences"):
    os.makedirs("num_occurences")
if not os.path.isdir("probability"):
    os.makedirs("probability")
if not os.path.isdir("predicted"):
    os.makedirs("predicted")
flag_replace = True

if flag_replace or not os.path.isfile("num_occurences/num_occurences_of_distance"):
    num_occurences_of_distance = dict()
    num_occurences_of_distance_in_next_step = dict()
    num_occurences_of_distance_in_next_next_step = dict()
    
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
            distance_int = [np.round(np.sqrt((longitudes[distance_index + 1] - longitudes[distance_index]) ** 2 + (latitudes[distance_index + 1] - latitudes[distance_index]) ** 2), 5) for distance_index in range(len(longitudes) - 1)]

            for distance in distance_int:
                if distance not in num_occurences_of_distance:
                    num_occurences_of_distance[distance] = 0
                num_occurences_of_distance[distance] += 1

            for distance_index in range(len(distance_int) - 1):
                distance = distance_int[distance_index]
                next_distance = distance_int[distance_index + 1]
                if distance not in num_occurences_of_distance_in_next_step:
                    num_occurences_of_distance_in_next_step[distance] = dict()
                if next_distance not in num_occurences_of_distance_in_next_step[distance]:
                    num_occurences_of_distance_in_next_step[distance][next_distance] = 0
                num_occurences_of_distance_in_next_step[distance][next_distance] += 1
                if distance_index < len(distance_int) - 2:
                    next_next_distance = distance_int[distance_index + 2]
                    if distance not in num_occurences_of_distance_in_next_next_step:
                        num_occurences_of_distance_in_next_next_step[distance] = dict()
                    if next_distance not in num_occurences_of_distance_in_next_next_step[distance]:
                        num_occurences_of_distance_in_next_next_step[distance][next_distance] = dict()
                    if next_next_distance not in num_occurences_of_distance_in_next_next_step[distance][next_distance]:
                        num_occurences_of_distance_in_next_next_step[distance][next_distance][next_next_distance] = 0
                    num_occurences_of_distance_in_next_next_step[distance][next_distance][next_next_distance] += 1

    #print(num_occurences_of_distance)
    save_object("num_occurences/num_occurences_of_distance", num_occurences_of_distance)
    #print(num_occurences_of_distance.keys())

    plt.bar(num_occurences_of_distance.keys(), num_occurences_of_distance.values())
    plt.show()

    #print(num_occurences_of_distance_in_next_step)
    save_object("num_occurences/num_occurences_of_distance_in_next_step", num_occurences_of_distance_in_next_step)
    #print(num_occurences_of_distance_in_next_next_step)
    save_object("num_occurences/num_occurences_of_distance_in_next_next_step", num_occurences_of_distance_in_next_next_step)

    probability_of_distance, probability_of_distance_in_next_step, probability_of_distance_in_next_next_step = fix_prob(num_occurences_of_distance, num_occurences_of_distance_in_next_step, num_occurences_of_distance_in_next_next_step)
    #print(probability_of_distance)
    save_object("probability/probability_of_distance", probability_of_distance)
    #print(probability_of_distance_in_next_step)
    save_object("probability/probability_of_distance_in_next_step", probability_of_distance_in_next_step)
    #print(probability_of_distance_in_next_next_step)
    save_object("probability/probability_of_distance_in_next_next_step", probability_of_distance_in_next_next_step)

probability_of_distance = load_object("probability/probability_of_distance") 
probability_of_distance_in_next_step = load_object("probability/probability_of_distance_in_next_step") 
probability_of_distance_in_next_next_step = load_object("probability/probability_of_distance_in_next_next_step")  

x = predict_prob(probability_of_distance, probability_of_distance_in_next_step, probability_of_distance_in_next_next_step, 0, 1, 10 ** -5)

plt.plot(x)
plt.xlabel('Distances',fontsize=20)
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
        distance_int = [np.round(np.sqrt((longitudes[distance_index + 1] - longitudes[distance_index]) ** 2 + (latitudes[distance_index + 1] - latitudes[distance_index]) ** 2), 5) for distance_index in range(len(longitudes) - 1)]

        x, n, match_score, no_empty, delta_series = predict_prob_with_array(probability_of_distance, probability_of_distance_in_next_step, probability_of_distance_in_next_next_step, distance_int, 0, 1, 10 ** -5)
        total_guesses += n
        total_guesses_no_empty += no_empty
        total_match_score += match_score 
        for value_delta in delta_series:
            delta_series_total.append(value_delta)
        all_x[subdir_name + "/cleaned_csv/" + some_file] = x
save_object("predicted/predicted_distance", all_x)
print(total_match_score / total_guesses, total_match_score / total_guesses_no_empty, min(delta_series_total), np.quantile(delta_series_total, 0.25), np.quantile(delta_series_total, 0.5), np.quantile(delta_series_total, 0.75), max(delta_series_total), np.average(delta_series_total), np.std(delta_series_total), np.var(delta_series_total))
plt.hist(delta_series_total)
plt.show()