import numpy as np
from utilities import load_object
import math
from sklearn.metrics import mean_squared_error
 
predicted_all = load_object("pytorch_result/predicted_all")
y_test_all = load_object("pytorch_result/y_test_all")
ws_all = load_object("pytorch_result/ws_all")

actual_long = load_object("pytorch_result/actual_long")
actual_lat = load_object("pytorch_result/actual_lat")
predicted_long = load_object("pytorch_result/predicted_long")
predicted_lat = load_object("pytorch_result/predicted_lat")

RMSE_time = dict()
RMSE_pred = dict()
RMSE_pred_wt = dict()
RMSE_long_pred = dict()
RMSE_long_pred_wt = dict()
RMSE_lat_pred = dict()
RMSE_lat_pred_wt = dict()
  
for model_name in predicted_long:

    RMSE_time[model_name] = dict()
    RMSE_pred[model_name] = dict()
    RMSE_pred_wt[model_name] = dict()
    RMSE_long_pred[model_name] = dict()
    RMSE_long_pred_wt[model_name] = dict()
    RMSE_lat_pred[model_name] = dict()
    RMSE_lat_pred_wt[model_name] = dict()

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

        RMSE_time[model_name][dist_name] = math.sqrt(mean_squared_error(actual_tm, predicted_tm))
        RMSE_pred[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_lat, predicted_long_lat))
        RMSE_pred_wt[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_lat_time, predicted_long_lat_time))
        RMSE_long_pred[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_nt, predicted_long_nt))
        RMSE_long_pred_wt[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_time, predicted_long_time))
        RMSE_lat_pred[model_name][dist_name] = math.sqrt(mean_squared_error(actual_lat_nt, predicted_lat_nt))
        RMSE_lat_pred_wt[model_name][dist_name] = math.sqrt(mean_squared_error(actual_lat_time, predicted_lat_time))
 
predicted_all = load_object("attention_result/predicted_all")
y_test_all = load_object("attention_result/y_test_all")
ws_all = load_object("attention_result/ws_all")

actual_long = load_object("attention_result/actual_long")
actual_lat = load_object("attention_result/actual_lat")
predicted_long = load_object("attention_result/predicted_long")
predicted_lat = load_object("attention_result/predicted_lat")

RMSE_time = dict()
RMSE_pred = dict()
RMSE_pred_wt = dict()
RMSE_long_pred = dict()
RMSE_long_pred_wt = dict()
RMSE_lat_pred = dict()
RMSE_lat_pred_wt = dict()
  
for model_name in predicted_long:

    RMSE_time[model_name] = dict()
    RMSE_pred[model_name] = dict()
    RMSE_pred_wt[model_name] = dict()
    RMSE_long_pred[model_name] = dict()
    RMSE_long_pred_wt[model_name] = dict()
    RMSE_lat_pred[model_name] = dict()
    RMSE_lat_pred_wt[model_name] = dict()

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

        RMSE_time[model_name][dist_name] = math.sqrt(mean_squared_error(actual_tm, predicted_tm))
        RMSE_pred[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_lat, predicted_long_lat))
        RMSE_pred_wt[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_lat_time, predicted_long_lat_time))
        RMSE_long_pred[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_nt, predicted_long_nt))
        RMSE_long_pred_wt[model_name][dist_name] = math.sqrt(mean_squared_error(actual_long_time, predicted_long_time))
        RMSE_lat_pred[model_name][dist_name] = math.sqrt(mean_squared_error(actual_lat_nt, predicted_lat_nt))
        RMSE_lat_pred_wt[model_name][dist_name] = math.sqrt(mean_squared_error(actual_lat_time, predicted_lat_time))
 
long_dict = load_object("Markov_result/long_dict")
lat_dict = load_object("Markov_result/lat_dict")
actual_traj = load_object("actual/actual_traj")
actual_time = load_object("actual/actual_time")
predicted_time = load_object("predicted/predicted_time")

RMSE_time["Markov"] = dict()
RMSE_pred["Markov"] = dict()
RMSE_pred_wt["Markov"] = dict()
RMSE_long_pred["Markov"] = dict()
RMSE_long_pred_wt["Markov"] = dict()
RMSE_lat_pred["Markov"] = dict()
RMSE_lat_pred_wt["Markov"] = dict()

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

    RMSE_time["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_tm, predicted_tm))
    RMSE_pred["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_long_lat, predicted_long_lat))
    RMSE_pred_wt["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_long_lat_time, predicted_long_lat_time))
    RMSE_long_pred["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_long_nt, predicted_long_nt))
    RMSE_long_pred_wt["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_long_time, predicted_long_time))
    RMSE_lat_pred["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_lat_nt, predicted_lat_nt))
    RMSE_lat_pred_wt["Markov"][dist_name] = math.sqrt(mean_squared_error(actual_lat_time, predicted_lat_time))

def_translate = {"long no abs": "$x$ and $y$ offset", "long speed dir": "Speed and heading", "long speed ones dir": "Speed and heading, $1$ $\\mathrm{s}$"}

print("RMSE_time")
for model_name in RMSE_time: 
    for dist_name in RMSE_time[model_name]:
        print(model_name, "&", "$" + str(np.round(RMSE_time[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")
        break

print("RMSE_pred")
for model_name in RMSE_pred: 
    for dist_name in RMSE_pred[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(RMSE_pred[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")

print("RMSE_pred_wt")
for model_name in RMSE_pred_wt: 
    for dist_name in RMSE_pred_wt[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(RMSE_pred_wt[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")

print("RMSE_long_pred")
for model_name in RMSE_long_pred: 
    for dist_name in RMSE_long_pred[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(RMSE_long_pred[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")

print("RMSE_long_pred_wt")
for model_name in RMSE_long_pred_wt: 
    for dist_name in RMSE_long_pred_wt[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(RMSE_long_pred_wt[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")

print("RMSE_lat_pred")
for model_name in RMSE_lat_pred: 
    for dist_name in RMSE_lat_pred[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(RMSE_lat_pred[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")

print("RMSE_lat_pred_wt")
for model_name in RMSE_lat_pred_wt: 
    for dist_name in RMSE_lat_pred_wt[model_name]:
        print(model_name, "&", def_translate[dist_name], "&", "$" + str(np.round(RMSE_lat_pred_wt[model_name][dist_name] * 100, 2)) + "\\%$ \\\\ \\hline")