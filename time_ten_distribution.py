from utilities import *
    
all_subdirs = os.listdir() 

if not os.path.isdir("num_occurences"):
    os.makedirs("num_occurences")
if not os.path.isdir("probability"):
    os.makedirs("probability")
if not os.path.isdir("predicted"):
    os.makedirs("predicted")
flag_replace = True

if flag_replace or not os.path.isfile("num_occurences/num_occurences_of_time_ten"):
    num_occurences_of_time_ten = dict()
    num_occurences_of_time_ten_ten_in_next_step = dict()
    num_occurences_of_time_ten_ten_in_next_next_step = dict()

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
            times = list(file_with_ride["time"])
            times_processed = [process_time(time_ten_new) for time_ten_new in times] 
            time_ten_int = [np.round(times_processed[time_ten_index + 1] - times_processed[time_ten_index], 1) for time_ten_index in range(len(times_processed) - 1)] 
            for time_ten_index in range(len(time_ten_int)):
                if time_ten_int[time_ten_index] == 0: 
                    time_ten_int[time_ten_index] = 10 ** -20
                                
            for time in time_ten_int:
                if time not in num_occurences_of_time_ten:
                    num_occurences_of_time_ten[time] = 0
                num_occurences_of_time_ten[time] += 1

            for time_ten_index in range(len(time_ten_int) - 1):
                time = time_ten_int[time_ten_index]
                next_time = time_ten_int[time_ten_index + 1]
                if time not in num_occurences_of_time_ten_ten_in_next_step:
                    num_occurences_of_time_ten_ten_in_next_step[time] = dict()
                if next_time not in num_occurences_of_time_ten_ten_in_next_step[time]:
                    num_occurences_of_time_ten_ten_in_next_step[time][next_time] = 0
                num_occurences_of_time_ten_ten_in_next_step[time][next_time] += 1
                if time_ten_index < len(time_ten_int) - 2:
                    next_next_time = time_ten_int[time_ten_index + 2]
                    if time not in num_occurences_of_time_ten_ten_in_next_next_step:
                        num_occurences_of_time_ten_ten_in_next_next_step[time] = dict()
                    if next_time not in num_occurences_of_time_ten_ten_in_next_next_step[time]:
                        num_occurences_of_time_ten_ten_in_next_next_step[time][next_time] = dict()
                    if next_next_time not in num_occurences_of_time_ten_ten_in_next_next_step[time][next_time]:
                        num_occurences_of_time_ten_ten_in_next_next_step[time][next_time][next_next_time] = 0
                    num_occurences_of_time_ten_ten_in_next_next_step[time][next_time][next_next_time] += 1

    #print(num_occurences_of_time_ten)
    save_object("num_occurences/num_occurences_of_time_ten", num_occurences_of_time_ten)
    #print(num_occurences_of_time_ten.keys())

    plt.bar(num_occurences_of_time_ten.keys(), num_occurences_of_time_ten.values())
    plt.show()

    #print(num_occurences_of_time_ten_ten_in_next_step)
    save_object("num_occurences/num_occurences_of_time_ten_ten_in_next_step", num_occurences_of_time_ten_ten_in_next_step)
    #print(num_occurences_of_time_ten_ten_in_next_next_step)
    save_object("num_occurences/num_occurences_of_time_ten_ten_in_next_next_step", num_occurences_of_time_ten_ten_in_next_next_step)

    probability_of_time_ten, probability_of_time_ten_ten_in_next_step, probability_of_time_ten_ten_in_next_next_step = fix_prob(num_occurences_of_time_ten, num_occurences_of_time_ten_ten_in_next_step, num_occurences_of_time_ten_ten_in_next_next_step)
    #print(probability_of_time_ten)
    save_object("probability/probability_of_time_ten", probability_of_time_ten)
    #print(probability_of_time_ten_ten_in_next_step)
    save_object("probability/probability_of_time_ten_ten_in_next_step", probability_of_time_ten_ten_in_next_step)
    #print(probability_of_time_ten_ten_in_next_next_step)
    save_object("probability/probability_of_time_ten_ten_in_next_next_step", probability_of_time_ten_ten_in_next_next_step)

probability_of_time_ten = load_object("probability/probability_of_time_ten") 
probability_of_time_ten_ten_in_next_step = load_object("probability/probability_of_time_ten_ten_in_next_step") 
probability_of_time_ten_ten_in_next_next_step = load_object("probability/probability_of_time_ten_ten_in_next_next_step")  
 
x = predict_prob(probability_of_time_ten, probability_of_time_ten_ten_in_next_step, probability_of_time_ten_ten_in_next_next_step, 0, 5, 10 ** -3)

plt.plot(x)
plt.xlabel('Times',fontsize=20)
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
        times = list(file_with_ride["time"])
        times_processed = [process_time(time_ten_new) for time_ten_new in times] 
        time_ten_int = [np.round(times_processed[time_ten_index + 1] - times_processed[time_ten_index], 1) for time_ten_index in range(len(times_processed) - 1)] 
        for time_ten_index in range(len(time_ten_int)):
            if time_ten_int[time_ten_index] == 0: 
                time_ten_int[time_ten_index] = 10 ** -20
 
        x, n, match_score, no_empty, delta_series = predict_prob_with_array(probability_of_time_ten, probability_of_time_ten_ten_in_next_step, probability_of_time_ten_ten_in_next_next_step, time_ten_int, 0, 5, 10 ** -3)
        total_guesses += n
        total_guesses_no_empty += no_empty
        total_match_score += match_score 
        for value_delta in delta_series:
            delta_series_total.append(value_delta)
        all_x[subdir_name + "/cleaned_csv/" + some_file] = x
save_object("predicted/predicted_time_ten", all_x)
print(total_match_score / total_guesses, total_match_score / total_guesses_no_empty, min(delta_series_total), np.quantile(delta_series_total, 0.25), np.quantile(delta_series_total, 0.5), np.quantile(delta_series_total, 0.75), max(delta_series_total), np.average(delta_series_total), np.std(delta_series_total), np.var(delta_series_total))
plt.hist(delta_series_total)
plt.show()