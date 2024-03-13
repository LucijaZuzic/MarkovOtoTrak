import os
import numpy as np
import matplotlib.pyplot as plt
from utilities import load_object, save_object, compare_traj_and_sample

def str_convert_new(val):
    new_val = val
    power_to = 0
    while abs(new_val) < 1 and new_val != 0.0:
        new_val *= 10
        power_to += 1 
    rounded = "$" + str(np.round(new_val, 2))
    if rounded[-2:] == '.0':
        rounded = rounded[:-2]
    if power_to != 0:  
        rounded += " \\times 10^{-" + str(power_to) + "}"
    return rounded + "$"

def new_metric_translate(metric_name):
    new_metric_name = {"trapz x": "$x$ integration", 
              "trapz y": "$y$ integration",
              "euclidean": "Euclidean distance"}
    if metric_name in new_metric_name:
        return new_metric_name[metric_name]
    else:
        return metric_name
    
def translate_category(long):
    translate_name = {
        "long no abs": "$x$ and $y$ offset",  
        "long speed dir": "Speed, heading, and time", 
        "long speed ones dir": "Speed, heading, and a 1s time interval", 
    }
    if long in translate_name:
        return translate_name[long]
    else:
        return long
    
def draw_mosaic(rides_actual, rides_predicted, name):
    
    x_dim_rides = int(np.sqrt(len(rides_actual)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides_actual):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides), dpi = 80)

    for ix_ride in range(len(rides_actual)):
 
        x_actual, y_actual = rides_actual[ix_ride]["long"], rides_actual[ix_ride]["lat"]
        x_predicted, y_predicted = rides_predicted[ix_ride]["long"], rides_predicted[ix_ride]["lat"]
            
        plt.subplot(y_dim_rides, x_dim_rides, ix_ride + 1)
        plt.rcParams.update({'font.size': 28}) 
        plt.rcParams['font.family'] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.axis("equal")
        plt.axis("off")

        plt.plot(x_predicted, y_predicted, c = "b", linewidth = 2, label = "Estimated")
    
        plt.plot(x_actual, y_actual, c = "k", linewidth = 2, label = "Original")

    plt.savefig(name, bbox_inches = "tight")
    plt.close()
    
def draw_mosaic_one(x_actual, y_actual, x_predicted, y_predicted, k, model_name, name, dist_name):
     
    plt.figure(figsize = (10, 10), dpi = 80)
    plt.rcParams.update({'font.size': 28}) 
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal")
  
    plt.plot(x_predicted, y_predicted, c = "b", linewidth = 10, label = "Estimated")
  
    plt.plot(x_actual, y_actual, c = "k", linewidth = 10, label = "Original")

    plt.plot(x_actual[0], x_actual[0], marker = "o", label = "Start", color = "k", mec = "k", mfc = "g", ms = 20, mew = 10, linewidth = 10) 
   
    split_file_veh = k.split("/")
    vehicle = split_file_veh[0].replace("Vehicle_", "")
    ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")

    title_new = "Vehicle " + vehicle + " Ride " + ride + "\n" + model_name + " model\n"

    title_new += translate_category(dist_name) + "\n" 
    for metric in distance_predicted_new:
        if "simpson" in metric:
            continue
        title_new += new_metric_translate(metric) + ": " + str_convert_new(distance_predicted_new[metric][model_name][dist_name][k]) + "\n"
    
    plt.title(title_new)
    plt.legend()
    plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
predicted_all = load_object("pytorch_result/predicted_all")
y_test_all = load_object("pytorch_result/y_test_all")
ws_all = load_object("pytorch_result/ws_all")

actual_long = load_object("pytorch_result/actual_long")
actual_lat = load_object("pytorch_result/actual_lat")
predicted_long = load_object("pytorch_result/predicted_long")
predicted_lat = load_object("pytorch_result/predicted_lat")

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

if not os.path.isdir("pytorch_result"):
    os.makedirs("pytorch_result")

save_object("pytorch_result/distance_predicted_new", distance_predicted_new)

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
    for method in count_best[metric]:
        print(method, np.round(count_best[metric][method] / np.sum(list(count_best[metric].values())) * 100, 2))

distance_predicted = load_object("markov_result/distance_predicted")

choose_best_new = dict()
for metric in choose_best:
    choose_best_new[metric] = dict()
    for name in choose_best[metric]:
        choose_best_new[metric][name] = choose_best[metric][name]

for subdir_name in distance_predicted:
    for some_file in distance_predicted[subdir_name]:
        name = subdir_name + "/cleaned_csv/" + some_file
        for metric in choose_best_new:
            for dist_name_half in distance_predicted_new[metric]["RNN"]:
                dist_name = dist_name_half + "-" + dist_name_half.replace("long", "lat")
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
    for method in count_best_new[metric]:
        print(method, np.round(count_best_new[metric][method] / np.sum(list(count_best_new[metric].values())) * 100, 2))

find_best = set()
find_worst = set()

for metric in distance_predicted_new:
    for model_name in distance_predicted_new[metric]:
        for dist_name in distance_predicted_new[metric][model_name]:
            keys_dist_pred = list(distance_predicted_new[metric][model_name][dist_name].keys())
            mini_val = distance_predicted_new[metric][model_name][dist_name][keys_dist_pred[0]]
            mini_traj = keys_dist_pred[0]
            maxi_val = distance_predicted_new[metric][model_name][dist_name][keys_dist_pred[0]]
            maxi_traj = keys_dist_pred[0]
            for name in distance_predicted_new[metric][model_name][dist_name]:
                if distance_predicted_new[metric][model_name][dist_name][name] < mini_val:
                    mini_traj = name
                    mini_val = distance_predicted_new[metric][model_name][dist_name][name]
                if distance_predicted_new[metric][model_name][dist_name][name] > maxi_val:
                    maxi_traj = name
                    maxi_val = distance_predicted_new[metric][model_name][dist_name][name]
            for dist_name2 in distance_predicted_new[metric][model_name]:
                for model_name2 in distance_predicted_new[metric]:
                    find_best.add((model_name2, dist_name2, mini_traj))
                    find_worst.add((model_name2, dist_name2, maxi_traj))

print(len(find_best))
print(len(find_worst))

find_all = find_best.union(find_worst)
print(len(find_all))

for pair_best in find_all:

    model_name, dist_name, k = pair_best

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
    
    split_file_veh = k.split("/")
    vehicle = split_file_veh[0].replace("Vehicle_", "")
    ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")

    if not os.path.isdir("mosaic_pytorch_all"):
        os.makedirs("mosaic_pytorch_all/")

    filename = "mosaic_pytorch_all/Vehicle_" + vehicle + "_events_" + ride + "_" + model_name + "_" + dist_name + "_" + dist_name.replace("long", "lat") + "_test_mosaic.png"
    draw_mosaic_one(actual_long_one, actual_lat_one, predicted_long_one, predicted_lat_one, k, model_name, filename, dist_name)