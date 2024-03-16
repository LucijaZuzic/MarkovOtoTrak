import numpy as np
from utilities import load_object
from sklearn.metrics import mean_absolute_error

predicted_all = load_object("pytorch_result/predicted_all")
y_test_all = load_object("pytorch_result/y_test_all")
ws_all = load_object("pytorch_result/ws_all")

actual_long = load_object("pytorch_result/actual_long")
actual_lat = load_object("pytorch_result/actual_lat")
predicted_long = load_object("pytorch_result/predicted_long")
predicted_lat = load_object("pytorch_result/predicted_lat")

mean_absolute_error_time = dict()
mean_absolute_error_pred = dict()
mean_absolute_error_pred_wt = dict()
mean_absolute_error_long_pred = dict()
mean_absolute_error_long_pred_wt = dict()
mean_absolute_error_lat_pred = dict()
mean_absolute_error_lat_pred_wt = dict()
  
for model_name in predicted_long:

    mean_absolute_error_time[model_name] = dict()
    mean_absolute_error_pred[model_name] = dict()
    mean_absolute_error_pred_wt[model_name] = dict()
    mean_absolute_error_long_pred[model_name] = dict()
    mean_absolute_error_long_pred_wt[model_name] = dict()
    mean_absolute_error_lat_pred[model_name] = dict()
    mean_absolute_error_lat_pred_wt[model_name] = dict()

    for dist_name in predicted_long[model_name]: 

        actual_tm = []
        predicted_tm = []

        actual_long_lat = []
        actual_long_lat_time = []
        predicted_long_lat = []
        predicted_long_lat_time = []

        actual_long_nt = []
        actual_long_time = []
        predicted_long_nt = []
        predicted_long_time = []

        actual_lat_nt = []
        actual_lat_time = []
        predicted_lat_nt = []
        predicted_lat_time = []
        
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

            for ix_use_len in range(use_len_time):

                actual_tm.append([time_actual_cumulative[ix_use_len]])
                predicted_tm.append([time_predicted_cumulative[ix_use_len]])

                actual_long_lat.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len]])
                actual_long_lat_time.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])
                
                actual_long_nt.append([actual_long_one[ix_use_len]])
                actual_long_time.append([actual_long_one[ix_use_len], time_actual_cumulative[ix_use_len]])

                actual_lat_nt.append([actual_lat_one[ix_use_len]])
                actual_lat_time.append([actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])

                predicted_long_lat.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len]])
                predicted_long_lat_time.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

                predicted_long_nt.append([predicted_long_one[ix_use_len]])
                predicted_long_time.append([predicted_long_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

                predicted_lat_nt.append([predicted_lat_one[ix_use_len]])
                predicted_lat_time.append([predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

        mean_absolute_error_time[model_name][dist_name] = mean_absolute_error(actual_tm, predicted_tm)
        mean_absolute_error_pred[model_name][dist_name] = mean_absolute_error(actual_long_lat, predicted_long_lat)
        mean_absolute_error_pred_wt[model_name][dist_name] = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)
        mean_absolute_error_long_pred[model_name][dist_name] = mean_absolute_error(actual_long_nt, predicted_long_nt)
        mean_absolute_error_long_pred_wt[model_name][dist_name] = mean_absolute_error(actual_long_time, predicted_long_time)
        mean_absolute_error_lat_pred[model_name][dist_name] = mean_absolute_error(actual_lat_nt, predicted_lat_nt)
        mean_absolute_error_lat_pred_wt[model_name][dist_name] = mean_absolute_error(actual_lat_time, predicted_lat_time)
 
predicted_all = load_object("attention_result/predicted_all")
y_test_all = load_object("attention_result/y_test_all")
ws_all = load_object("attention_result/ws_all")

actual_long = load_object("attention_result/actual_long")
actual_lat = load_object("attention_result/actual_lat")
predicted_long = load_object("attention_result/predicted_long")
predicted_lat = load_object("attention_result/predicted_lat")

for model_name in predicted_long:

    mean_absolute_error_time[model_name] = dict()
    mean_absolute_error_pred[model_name] = dict()
    mean_absolute_error_pred_wt[model_name] = dict()
    mean_absolute_error_long_pred[model_name] = dict()
    mean_absolute_error_long_pred_wt[model_name] = dict()
    mean_absolute_error_lat_pred[model_name] = dict()
    mean_absolute_error_lat_pred_wt[model_name] = dict()

    for dist_name in predicted_long[model_name]: 

        actual_tm = []
        predicted_tm = []

        actual_long_lat = []
        actual_long_lat_time = []
        predicted_long_lat = []
        predicted_long_lat_time = []

        actual_long_nt = []
        actual_long_time = []
        predicted_long_nt = []
        predicted_long_time = []

        actual_lat_nt = []
        actual_lat_time = []
        predicted_lat_nt = []
        predicted_lat_time = []
        
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

            for ix_use_len in range(use_len_time):

                actual_tm.append([time_actual_cumulative[ix_use_len]])
                predicted_tm.append([time_predicted_cumulative[ix_use_len]])

                actual_long_lat.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len]])
                actual_long_lat_time.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])
                
                actual_long_nt.append([actual_long_one[ix_use_len]])
                actual_long_time.append([actual_long_one[ix_use_len], time_actual_cumulative[ix_use_len]])

                actual_lat_nt.append([actual_lat_one[ix_use_len]])
                actual_lat_time.append([actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])

                predicted_long_lat.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len]])
                predicted_long_lat_time.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

                predicted_long_nt.append([predicted_long_one[ix_use_len]])
                predicted_long_time.append([predicted_long_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

                predicted_lat_nt.append([predicted_lat_one[ix_use_len]])
                predicted_lat_time.append([predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

        mean_absolute_error_time[model_name][dist_name] = mean_absolute_error(actual_tm, predicted_tm)
        mean_absolute_error_pred[model_name][dist_name] = mean_absolute_error(actual_long_lat, predicted_long_lat)
        mean_absolute_error_pred_wt[model_name][dist_name] = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)
        mean_absolute_error_long_pred[model_name][dist_name] = mean_absolute_error(actual_long_nt, predicted_long_nt)
        mean_absolute_error_long_pred_wt[model_name][dist_name] = mean_absolute_error(actual_long_time, predicted_long_time)
        mean_absolute_error_lat_pred[model_name][dist_name] = mean_absolute_error(actual_lat_nt, predicted_lat_nt)
        mean_absolute_error_lat_pred_wt[model_name][dist_name] = mean_absolute_error(actual_lat_time, predicted_lat_time)
 
long_dict = load_object("Markov_result/long_dict")
lat_dict = load_object("Markov_result/lat_dict")
actual_traj = load_object("actual/actual_traj")
actual_time = load_object("actual/actual_time")
predicted_time = load_object("predicted/predicted_time")

mean_absolute_error_time["Markov"] = dict()
mean_absolute_error_pred["Markov"] = dict()
mean_absolute_error_pred_wt["Markov"] = dict()
mean_absolute_error_long_pred["Markov"] = dict()
mean_absolute_error_long_pred_wt["Markov"] = dict()
mean_absolute_error_lat_pred["Markov"] = dict()
mean_absolute_error_lat_pred_wt["Markov"] = dict()

for dist_name in long_dict[list(long_dict.keys())[0]]:  

    actual_tm = []
    predicted_tm = []

    actual_long_lat = []
    actual_long_lat_time = []
    predicted_long_lat = []
    predicted_long_lat_time = []

    actual_long_nt = []
    actual_long_time = []
    predicted_long_nt = []
    predicted_long_time = []

    actual_lat_nt = []
    actual_lat_time = []
    predicted_lat_nt = []
    predicted_lat_time = []

    for longer_file_name in long_dict:
            
        subdir_name = longer_file_name.split("/")[0]
                        
        some_file = longer_file_name.split("/")[-1] 
    
        time_actual_cumulative = [0]
        time_predicted_cumulative = [0]
        
        for ix in range(len(actual_time[longer_file_name])):
            time_actual_cumulative.append(time_actual_cumulative[-1] + actual_time[longer_file_name][ix])
            time_predicted_cumulative.append(time_predicted_cumulative[-1] + predicted_time[longer_file_name][ix])

        for ix_use_len in range(len(lat_dict[longer_file_name][dist_name.replace("long", "lat")])):

            actual_tm.append(time_actual_cumulative[ix_use_len])
            predicted_tm.append(time_predicted_cumulative[ix_use_len])

            actual_long_lat.append([actual_traj[subdir_name][some_file][0][ix_use_len], actual_traj[subdir_name][some_file][1][ix_use_len]])
            actual_long_lat_time.append([actual_traj[subdir_name][some_file][0][ix_use_len], actual_traj[subdir_name][some_file][1][ix_use_len], time_actual_cumulative[ix_use_len]])
            
            actual_long_nt.append([actual_traj[subdir_name][some_file][0][ix_use_len]])
            actual_long_time.append([actual_traj[subdir_name][some_file][0][ix_use_len], time_actual_cumulative[ix_use_len]])

            actual_lat_nt.append([actual_traj[subdir_name][some_file][1][ix_use_len]])
            actual_lat_time.append([actual_traj[subdir_name][some_file][1][ix_use_len], time_actual_cumulative[ix_use_len]])

            predicted_long_lat.append([long_dict[longer_file_name][dist_name][ix_use_len], lat_dict[longer_file_name][dist_name.replace("long", "lat")][ix_use_len]])
            predicted_long_lat_time.append([long_dict[longer_file_name][dist_name][ix_use_len], lat_dict[longer_file_name][dist_name.replace("long", "lat")][ix_use_len], time_predicted_cumulative[ix_use_len]])

            predicted_long_nt.append([long_dict[longer_file_name][dist_name][ix_use_len]])
            predicted_long_time.append([long_dict[longer_file_name][dist_name][ix_use_len], time_predicted_cumulative[ix_use_len]])

            predicted_lat_nt.append([lat_dict[longer_file_name][dist_name.replace("long", "lat")][ix_use_len]])
            predicted_lat_time.append([lat_dict[longer_file_name][dist_name.replace("long", "lat")][ix_use_len], time_predicted_cumulative[ix_use_len]])

    mean_absolute_error_time["Markov"][dist_name] = mean_absolute_error(actual_tm, predicted_tm)
    mean_absolute_error_pred["Markov"][dist_name] = mean_absolute_error(actual_long_lat, predicted_long_lat)
    mean_absolute_error_pred_wt["Markov"][dist_name] = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)
    mean_absolute_error_long_pred["Markov"][dist_name] = mean_absolute_error(actual_long_nt, predicted_long_nt)
    mean_absolute_error_long_pred_wt["Markov"][dist_name] = mean_absolute_error(actual_long_time, predicted_long_time)
    mean_absolute_error_lat_pred["Markov"][dist_name] = mean_absolute_error(actual_lat_nt, predicted_lat_nt)
    mean_absolute_error_lat_pred_wt["Markov"][dist_name] = mean_absolute_error(actual_lat_time, predicted_lat_time)

def_translate = {"long no abs": "$x$ and $y$ offset", "long speed dir": "Speed and heading", "long speed ones dir": "Speed and heading, $1$ $\\mathrm{s}$"}

print("mean_absolute_error_time")
for model_name in mean_absolute_error_time: 
    for dist_name in mean_absolute_error_time[model_name]:
        print(model_name, "&", "$" + str(np.round(mean_absolute_error_time[model_name][dist_name], 6)) + "$ \\\\ \\hline")
        break

print("mean_absolute_error_pred")
for model_name in mean_absolute_error_pred: 
    for dist_name in mean_absolute_error_pred[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(mean_absolute_error_pred[model_name][dist_name], 6)) + "$ \\\\ \\hline")

print("mean_absolute_error_pred_wt")
for model_name in mean_absolute_error_pred_wt: 
    for dist_name in mean_absolute_error_pred_wt[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(mean_absolute_error_pred_wt[model_name][dist_name], 6)) + "$ \\\\ \\hline")

print("mean_absolute_error_long_pred")
for model_name in mean_absolute_error_long_pred: 
    for dist_name in mean_absolute_error_long_pred[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(mean_absolute_error_long_pred[model_name][dist_name], 6)) + "$ \\\\ \\hline")

print("mean_absolute_error_long_pred_wt")
for model_name in mean_absolute_error_long_pred_wt: 
    for dist_name in mean_absolute_error_long_pred_wt[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(mean_absolute_error_long_pred_wt[model_name][dist_name], 6)) + "$ \\\\ \\hline")

print("mean_absolute_error_lat_pred")
for model_name in mean_absolute_error_lat_pred: 
    for dist_name in mean_absolute_error_lat_pred[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(mean_absolute_error_lat_pred[model_name][dist_name], 6)) + "$ \\\\ \\hline")

print("mean_absolute_error_lat_pred_wt")
for model_name in mean_absolute_error_lat_pred_wt: 
    for dist_name in mean_absolute_error_lat_pred_wt[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(mean_absolute_error_lat_pred_wt[model_name][dist_name], 6)) + "$ \\\\ \\hline")