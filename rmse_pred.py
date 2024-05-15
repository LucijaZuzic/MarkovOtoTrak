from utilities import load_object
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import numpy as np
from sklearn.metrics import r2_score

all_subdirs = os.listdir() 

def flatten(all_x, all_mine):
 
    int_veh = sorted([int(v.split("/")[0].split("_")[1]) for v in all_mine.keys()])

    filenamessorted = []
    for i in set(int_veh):
        for filename in all_x:
            if "Vehicle_" + str(i) + "/cle" in filename:
                filenamessorted.append(filename)

    all_flat_x = []
    all_flat_mine = []
    filenames = []
    filenames_length = []
    for filename in filenamessorted:
        for val in all_x[filename]:
            all_flat_x.append(val)
        for val in all_mine[filename]:
            all_flat_mine.append(val)
        filenames.append(filename)
        filenames_length.append(len(all_mine[filename]))
    return all_flat_x, all_flat_mine, filenames, filenames_length

def read_var(varname): 
    all_x = load_object("predicted/predicted_" + varname)
    all_mine = load_object("actual/actual_" + varname)
    return flatten(all_x, all_mine)

def plot_rmse(file_extension, var_name, actual, predictions, filenames, filenames_length):
    if not os.path.isdir("rmse"):
        os.makedirs("rmse")
 
    plt.figure(figsize = (20, 6), dpi = 80)
    plt.rcParams.update({'font.size': 22}) 
    plt.plot(range(len(actual)), actual, color = "b") 
    plt.plot(range(len(predictions)), predictions, color = "orange") 
    plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
    plt.title("Actual and predicted values\n" + var_name )
    plt.xlabel("Point index")
    plt.ylabel(var_name)
    plt.savefig("rmse/" + file_extension + "_all.png", bbox_inches = "tight")
    plt.close()

    total_len_rides = 0 
    for ix_ride in range(len(filenames_length)):
        len_ride = filenames_length[ix_ride]
        actual_ride = actual[total_len_rides:total_len_rides + len_ride]
        predictions_ride = predictions[total_len_rides:total_len_rides + len_ride]
        total_len_rides += len_ride
        split_filename = filenames[ix_ride].split("/")
        vehicle = split_filename[0].replace("Vehicle_", "")
        ride = split_filename[-1].replace("events_", "").replace(".csv", "") 
        plt.figure(figsize = (20, 6), dpi = 80)
        plt.rcParams.update({'font.size': 22}) 
        plt.plot(range(len(actual_ride)), actual_ride, color = "b") 
        plt.plot(range(len(predictions_ride)), predictions_ride, color = "orange") 
        plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
        plt.title("Actual and predicted values\n" + var_name + "\nVehicle " + vehicle + " Ride " + ride)
        plt.xlabel("Point index")
        plt.ylabel(var_name)
        plt.savefig("rmse/" + file_extension + "_Vehicle_" + vehicle + "_Ride_" + ride + "_all.png", bbox_inches = "tight")
        plt.close()
  
all_x_heading, all_mine_heading, filenames_heading, filenames_length_heading = read_var("direction")
all_x_latitude_no_abs, all_mine_latitude_no_abs, filenames_latitude_no_abs, filenames_length_latitude_no_abs = read_var("latitude_no_abs")
all_x_longitude_no_abs, all_mine_longitude_no_abs, filenames_longitude_no_abs, filenames_length_longitude_no_abs = read_var("longitude_no_abs")
all_x_speed, all_mine_speed, filenames_speed, filenames_length_speed = read_var("speed")
all_x_time, all_mine_time, filenames_time, filenames_length_time = read_var("time")

#plot_rmse("heading", "Heading ($\degree$)", all_mine_heading, all_x_heading, filenames_heading, filenames_length_heading)
#plot_rmse("latitude", "x offset ($\degree$ long.)", all_mine_latitude_no_abs, all_x_latitude_no_abs, filenames_latitude_no_abs, filenames_length_latitude_no_abs)
#plot_rmse("longitude", "y offset ($\degree$ lat.)", all_mine_longitude_no_abs, all_x_longitude_no_abs, filenames_longitude_no_abs, filenames_length_longitude_no_abs)
#plot_rmse("speed", "Speed (km/h)", all_mine_speed, all_x_speed, filenames_speed, filenames_length_speed)
#plot_rmse("time", "Time (s)", all_mine_time, all_x_time, filenames_time, filenames_length_time)

print("NRMSE")
print("heading", max(all_mine_heading),min(all_mine_heading), np.round(math.sqrt(mean_squared_error(all_mine_heading, all_x_heading)) / (max(all_mine_heading) - min(all_mine_heading)) * 100, 6))
print("latitude", max(all_mine_latitude_no_abs),min(all_mine_latitude_no_abs), np.round(math.sqrt(mean_squared_error(all_mine_latitude_no_abs, all_x_latitude_no_abs)) / (max(all_mine_latitude_no_abs) - min(all_mine_latitude_no_abs)) * 100, 6))
print("longitude", max(all_mine_longitude_no_abs),min(all_mine_longitude_no_abs), np.round(math.sqrt(mean_squared_error(all_mine_longitude_no_abs, all_x_longitude_no_abs)) / (max(all_mine_longitude_no_abs) - min(all_mine_longitude_no_abs)) * 100, 6))
print("speed", max(all_mine_speed),min(all_mine_speed), np.round(math.sqrt(mean_squared_error(all_mine_speed, all_x_speed)) / (max(all_mine_speed) - min(all_mine_speed)) * 100, 6))
print("time", max(all_mine_time),min(all_mine_time), np.round(math.sqrt(mean_squared_error(all_mine_time, all_x_time)) / (max(all_mine_time) - min(all_mine_time)) * 100, 6))

print("RMSE")
print("heading", max(all_mine_heading),min(all_mine_heading), np.round(math.sqrt(mean_squared_error(all_mine_heading, all_x_heading)), 6))
print("latitude", max(all_mine_latitude_no_abs),min(all_mine_latitude_no_abs), np.round(math.sqrt(mean_squared_error(all_mine_latitude_no_abs, all_x_latitude_no_abs)), 6))
print("longitude", max(all_mine_longitude_no_abs),min(all_mine_longitude_no_abs), np.round(math.sqrt(mean_squared_error(all_mine_longitude_no_abs, all_x_longitude_no_abs)), 6))
print("speed", max(all_mine_speed),min(all_mine_speed), np.round(math.sqrt(mean_squared_error(all_mine_speed, all_x_speed)), 6))
print("time", max(all_mine_time),min(all_mine_time), np.round(math.sqrt(mean_squared_error(all_mine_time, all_x_time)), 6))

print("R2")
print("heading", max(all_mine_heading),min(all_mine_heading), np.round(r2_score(all_mine_heading, all_x_heading) * 100, 6))
print("latitude", max(all_mine_latitude_no_abs),min(all_mine_latitude_no_abs), np.round(r2_score(all_mine_latitude_no_abs, all_x_latitude_no_abs) * 100, 6))
print("longitude", max(all_mine_longitude_no_abs),min(all_mine_longitude_no_abs), np.round(r2_score(all_mine_longitude_no_abs, all_x_longitude_no_abs) * 100, 6))
print("speed", max(all_mine_speed),min(all_mine_speed), np.round(r2_score(all_mine_speed, all_x_speed) * 100, 6))
print("time", max(all_mine_time),min(all_mine_time), np.round(r2_score(all_mine_time, all_x_time) * 100, 6))

print("MAE")
print("heading", max(all_mine_heading),min(all_mine_heading), np.round(mean_absolute_error(all_mine_heading, all_x_heading), 6))
print("latitude", max(all_mine_latitude_no_abs),min(all_mine_latitude_no_abs), np.round(mean_absolute_error(all_mine_latitude_no_abs, all_x_latitude_no_abs), 6))
print("longitude", max(all_mine_longitude_no_abs),min(all_mine_longitude_no_abs), np.round(mean_absolute_error(all_mine_longitude_no_abs, all_x_longitude_no_abs), 6))
print("speed", max(all_mine_speed),min(all_mine_speed), np.round(mean_absolute_error(all_mine_speed, all_x_speed), 6))
print("time", max(all_mine_time),min(all_mine_time), np.round(mean_absolute_error(all_mine_time, all_x_time), 6))

from sacrebleu.metrics import BLEU

def bleuval(varname): 
    all_x = load_object("predicted/predicted_" + varname)
    all_mine = load_object("actual/actual_" + varname)
 
    int_veh = sorted([int(v.split("/")[0].split("_")[1]) for v in all_mine.keys()])

    filenamessorted = []
    for i in set(int_veh):
        for filename in all_x:
            if "Vehicle_" + str(i) + "/cle" in filename:
                filenamessorted.append(filename)

    BLEU_all = []
    for filename in filenamessorted:
        bleu_params = dict(effective_order=True, tokenize=None, smooth_method="floor", smooth_value=0.01)
        bleu = BLEU(**bleu_params)
        pred_str = ""
        actual_str = "" 
        round_val = 10
        if "dir" in varname or "speed" in varname:
            round_val = 0
        if "time" in varname:
            round_val = 3
        for val_ix in range(len(all_mine[filename])):
            pred_str += str(np.round(float(all_x[filename][val_ix]), round_val)) + " "
            actual_str += str(np.round(float(all_mine[filename][val_ix]), round_val)) + " "
        pred_str = pred_str[:-1]
        actual_str = actual_str[:-1]
        blsc = bleu.sentence_score(hypothesis=pred_str, references=[actual_str]).score
        BLEU_all.append(blsc)
    print(varname, np.mean(BLEU_all))

bleuval("direction")
bleuval("latitude_no_abs")
bleuval("longitude_no_abs")
bleuval("speed")
bleuval("time")

long_dict = load_object("markov_result/long_dict")
lat_dict = load_object("markov_result/lat_dict") 
distance_predicted = load_object("markov_result/distance_predicted")
#print(distance_predicted["Vehicle_11"]['events_8354807.csv']['euclidean'].keys())
#print(long_dict['Vehicle_17/cleaned_csv/events_9132790.csv'].keys())
all_longlats = []

for vehicle_event in long_dict:  
    for long in long_dict[vehicle_event]:
        lat = long.replace("long", "lat").replace("x", "y")
        all_longlats.append([long, lat])
    break 

all_subdirs = os.listdir()

if not os.path.isdir("mosaic_all//test"):
    os.makedirs("mosaic_all//test")

actual_traj = load_object("actual/actual_traj")

int_veh = sorted([int(v.split("_")[1]) for v in actual_traj.keys()])

all_time = load_object("predicted/predicted_time")
pred_time = load_object("actual/actual_time")

for ix_longlat in range(len(all_longlats)):

    all_actual = []
    all_predicted = []

    actual_long_lat = []
    actual_long_lat_time = []
    predicted_long_lat = []
    predicted_long_lat_time = []

    vals_avg = []
    
    min_k = ""
    min_dist = 1000000
    max_k = ""
    max_dist = - 1000000

    for i in int_veh:  

        all_actual_vehicle = []
        all_predicted_vehicle = []

        subdir_name = "Vehicle_" + str(i)

        val_rides = set()
        if os.path.isfile(subdir_name + "/val_rides"):
            val_rides = load_object(subdir_name + "/val_rides")
            
        for some_file in actual_traj[subdir_name]: 
            
            longitudes, latitudes, is_test = actual_traj[subdir_name][some_file] 

            if is_test == "test":
                long_pred = long_dict[subdir_name + "/cleaned_csv/" + some_file][all_longlats[ix_longlat][0]]
                lat_pred = lat_dict[subdir_name + "/cleaned_csv/" + some_file][all_longlats[ix_longlat][1]]
                time_actual = all_time[subdir_name + "/cleaned_csv/" + some_file]
                time_pred = pred_time[subdir_name + "/cleaned_csv/" + some_file]
                
                vals_avg.append(distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]])
                
                if distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]] < min_dist:
                    min_k = subdir_name + "/cleaned_csv/" + some_file
                    min_dist = distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]]

                if distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]] > max_dist:
                    max_k = subdir_name + "/cleaned_csv/" + some_file
                    max_dist = distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]]
                
                #print(i, some_file, all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])
                #print(distance_predicted[subdir_name][some_file]['euclidean'][all_longlats[ix_longlat][0] + "-" + all_longlats[ix_longlat][1]])

                for ix_use_len in range(len(time_actual)):

                    actual_long_lat.append([longitudes[ix_use_len], latitudes[ix_use_len]])
                    actual_long_lat_time.append([longitudes[ix_use_len], latitudes[ix_use_len], time_actual[ix_use_len]])
                    
                    predicted_long_lat.append([long_pred[ix_use_len], lat_pred[ix_use_len]])
                    predicted_long_lat_time.append([long_pred[ix_use_len], lat_pred[ix_use_len], time_pred[ix_use_len]])

                all_actual_vehicle.append({"long": longitudes, "lat": latitudes})
                all_predicted_vehicle.append({"long": long_pred, "lat": lat_pred})

                all_actual.append({"long": longitudes, "lat": latitudes})
                all_predicted.append({"long": long_pred, "lat": lat_pred})
    
    print(all_longlats[ix_longlat], np.round(np.average(vals_avg), 6))
    print(min_k, min_dist)
    print(max_k, max_dist)

    r2_pred_wt = r2_score(actual_long_lat_time, predicted_long_lat_time)

    mae_pred_wt = mean_absolute_error(actual_long_lat_time, predicted_long_lat_time)

    rmse_pred_wt = math.sqrt(mean_squared_error(actual_long_lat_time, predicted_long_lat_time))

    r2_pred = r2_score(actual_long_lat, predicted_long_lat)

    mae_pred = mean_absolute_error(actual_long_lat, predicted_long_lat)

    rmse_pred = math.sqrt(mean_squared_error(actual_long_lat, predicted_long_lat))
    
    print("R2", np.round(r2_pred * 100, 2), "MAE", np.round(mae_pred, 6), "RMSE", np.round(rmse_pred, 6), "R2_wt", np.round(r2_pred_wt * 100, 2), "MAE_wt", np.round(mae_pred_wt, 6), "RMSE_wt", np.round(rmse_pred_wt, 6))