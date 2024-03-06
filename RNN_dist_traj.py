import os
import numpy as np
from utilities import load_object, save_object, compare_traj_and_sample

predicted_all = load_object("RNN_result/predicted_all")
y_test_all = load_object("RNN_result/y_test_all")
ws_all = load_object("RNN_result/ws_all")

actual_long = load_object("RNN_result/actual_long")
actual_lat = load_object("RNN_result/actual_lat")
predicted_long = load_object("RNN_result/predicted_long")
predicted_lat = load_object("RNN_result/predicted_lat")

distance_predicted_new = dict()

metric_names = ["trapz x", "trapz y", "euclidean"]  

for metric in metric_names:
    distance_predicted_new[metric] = dict()
    for model_name in predicted_long:
        distance_predicted_new[metric][model_name] = dict()
        for dist_name in predicted_long[model_name]:
            distance_predicted_new[metric][model_name][dist_name] = dict()
            for k in predicted_long[model_name][dist_name]:

                actual_long_one = actual_long[model_name][k]
                actual_lat_one = actual_lat[model_name][k]

                predicted_long_one = predicted_long[model_name][dist_name][k]
                predicted_lat_one = predicted_lat[model_name][dist_name.replace("long", "lat")][k]

                use_len = min(len(actual_long_one), len(predicted_long_one))
                
                actual_long_one = actual_long_one[:use_len]
                actual_lat_one = actual_lat_one[:use_len]

                predicted_long_one = predicted_long_one[:use_len]
                predicted_lat_one = predicted_lat_one[:use_len]
                   
                time_actual = y_test_all["time"][model_name][k]
                time_predicted = predicted_all["time"][model_name][k]

                time_actual_cumulative = [0]
                time_predicted_cumulative = [0]
                
                for ix in range(len(time_actual)):
                    time_actual_cumulative.append(time_actual_cumulative[-1] + time_actual[ix])
                    time_predicted_cumulative.append(time_predicted_cumulative[-1] + time_predicted[ix])
                    
                use_len_time = min(use_len, len(time_actual_cumulative))

                actual_long_one = actual_long_one[:use_len_time]
                actual_lat_one = actual_lat_one[:use_len_time]

                predicted_long_one = predicted_long_one[:use_len_time]
                predicted_lat_one = predicted_lat_one[:use_len_time]
                
                time_actual_cumulative = time_actual_cumulative[:use_len_time]
                time_predicted_cumulative = time_predicted_cumulative[:use_len_time]

                distance_predicted_new[metric][model_name][dist_name][k] = compare_traj_and_sample(actual_long_one, predicted_lat_one, time_actual_cumulative, {"long": predicted_long_one, "lat": predicted_lat_one, "time": time_predicted_cumulative}, metric)

if not os.path.isdir("RNN_result"):
    os.makedirs("RNN_result")

save_object("RNN_result/distance_predicted_new", distance_predicted_new)

for metric in distance_predicted_new:
    for model_name in distance_predicted_new[metric]:
        for dist_name in distance_predicted_new[metric][model_name]:
            vals_list = list(distance_predicted_new[metric][model_name][dist_name].values())
            print(metric, model_name, dist_name, np.quantile(vals_list, 0.5))

choose_best = dict()
for metric in metric_names:
    choose_best[metric] = dict()
    for name in list(distance_predicted_new["euclidean"]["RNN"]["long no abs"].keys()):
        choose_best[metric][name] = ("RNN", "long no abs", distance_predicted_new[metric]["RNN"]["long no abs"][name])
        for model_name in distance_predicted_new[metric]:
            for dist_name in distance_predicted_new[metric][model_name]:
                if distance_predicted_new[metric][model_name][dist_name][name] < choose_best[metric][name][2]:
                    choose_best[metric][name] = (model_name, dist_name, distance_predicted_new[metric][model_name][dist_name][name])

count_best = dict()
for metric in choose_best: 
    count_best[metric] = dict()
    for name in choose_best[metric]:
        if choose_best[metric][name][0] + "_" + choose_best[metric][name][1] not in count_best[metric]:
            count_best[metric][choose_best[metric][name][0] + "_" + choose_best[metric][name][1]] = 0
        count_best[metric][choose_best[metric][name][0] + "_" + choose_best[metric][name][1]] += 1
    print(metric, count_best[metric])

distance_predicted = load_object("markov_result/distance_predicted")
#distance_predicted[subdir_name][some_file][metric_name][longit + "-" + latit

choose_best_new = dict()
for metric in choose_best:
    choose_best_new[metric] = dict()
    for name in choose_best[metric]:
        choose_best_new[metric][name] = choose_best[metric][name]

for subdir_name in distance_predicted:
    for some_file in distance_predicted[subdir_name]:
        name = subdir_name + "/cleaned_csv/" + some_file
        for metric in choose_best_new:
            for dist_name in distance_predicted[subdir_name][some_file][metric]:
                if distance_predicted[subdir_name][some_file][metric][dist_name] < choose_best_new[metric][name][2]:
                    choose_best_new[metric][name] = ("Markov", dist_name, distance_predicted[subdir_name][some_file][metric][dist_name])

count_best_new = dict()
for metric in choose_best_new: 
    count_best_new[metric] = dict()
    for name in choose_best_new[metric]:
        if choose_best_new[metric][name][0] + "_" + choose_best_new[metric][name][1] not in count_best_new[metric]:
            count_best_new[metric][choose_best_new[metric][name][0] + "_" + choose_best_new[metric][name][1]] = 0
        count_best_new[metric][choose_best_new[metric][name][0] + "_" + choose_best_new[metric][name][1]] += 1
    print(metric, count_best_new[metric])