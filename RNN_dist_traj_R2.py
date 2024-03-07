import os
import numpy as np
from utilities import load_object, save_object, compare_traj_and_sample
from sklearn.metrics import r2_score
 
predicted_all = load_object("RNN_result/predicted_all")
y_test_all = load_object("RNN_result/y_test_all")
ws_all = load_object("RNN_result/ws_all")

actual_long = load_object("RNN_result/actual_long")
actual_lat = load_object("RNN_result/actual_lat")
predicted_long = load_object("RNN_result/predicted_long")
predicted_lat = load_object("RNN_result/predicted_lat")

r2_pred = dict()
r2_pred_wt = dict()
  
for model_name in predicted_long:

    r2_pred[model_name] = dict()
    r2_pred_wt[model_name] = dict()

    for dist_name in predicted_long[model_name]: 

        actual_long_lat = []
        actual_long_lat_time = []
        predicted_long_lat = []
        predicted_long_lat_time = []
        
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

                actual_long_lat.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len]])
                actual_long_lat_time.append([actual_long_one[ix_use_len], actual_lat_one[ix_use_len], time_actual_cumulative[ix_use_len]])

                predicted_long_lat.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len]])
                predicted_long_lat_time.append([predicted_long_one[ix_use_len], predicted_lat_one[ix_use_len], time_predicted_cumulative[ix_use_len]])

        r2_pred_wt[model_name][dist_name] = r2_score(actual_long_lat, predicted_long_lat)
        r2_pred[model_name][dist_name] = r2_score(actual_long_lat_time, predicted_long_lat_time)
 
for model_name in r2_pred: 
    for dist_name in r2_pred[model_name]:
        print(model_name, dist_name, r2_pred[model_name][dist_name], r2_pred_wt[model_name][dist_name])