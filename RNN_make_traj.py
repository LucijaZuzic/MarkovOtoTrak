import pandas as pd
import os  
from utilities import load_object, save_object, get_sides_from_angle
from RNN_utilities import get_XY
import matplotlib.pyplot as plt

def change_angle(angle, name_file):
    
    file_with_ride = pd.read_csv(name_file) 
    
    x_dir = list(file_with_ride["fields_longitude"])[0] < list(file_with_ride["fields_longitude"])[-1]
    y_dir = list(file_with_ride["fields_latitude"])[0] < list(file_with_ride["fields_latitude"])[-1]

    new_dir = (90 - angle + 360) % 360 
    if not x_dir: 
        new_dir = (180 - new_dir + 360) % 360
    if not y_dir: 
        new_dir = 360 - new_dir 

    return new_dir

predicted_all = dict()
y_test_all = dict()
ws_all = dict() 

for varname in os.listdir("final_train_net"):

    predicted_all[varname] = dict()
    y_test_all[varname] = dict()
    ws_all[varname] = dict() 
    
    for model_name in os.listdir("final_train_net/" + varname + "/predictions/test/"):

        predicted_all[varname][model_name] = dict()
        y_test_all[varname][model_name] = dict() 

        for filename in os.listdir("final_train_net/" + varname + "/predictions/test/" + model_name):
 
            final_test_data = pd.read_csv("final_train_net/" + varname + "/predictions/test/" + model_name + "/" + filename, sep = ";", index_col = False)
  
            file_object_test = load_object("actual/actual_" + varname)

            ws_use = int(filename.replace(".csv", "").split("_")[-4])
            ws_all[varname][model_name] = ws_use
 
            len_total = 0

            for k in file_object_test:

                x_test_part, y_test_part = get_XY(file_object_test[k], ws_use)
                
                y_test_all[varname][model_name][k] = []
                for ix1 in range(len(y_test_part)): 
                    for ix2 in range(len(y_test_part[ix1])): 
                        y_test_all[varname][model_name][k].append(y_test_part[ix1][ix2])
 
                predicted_all[varname][model_name][k] = list(final_test_data["predicted"][len_total:len_total + len(y_test_all[varname][model_name][k])])
                len_total += len(y_test_all[varname][model_name][k])  

if not os.path.isdir("RNN_result"):
    os.makedirs("RNN_result")

save_object("RNN_result/predicted_all", predicted_all)
save_object("RNN_result/y_test_all", y_test_all)
save_object("RNN_result/ws_all", ws_all)

predicted_long = dict()
predicted_lat = dict()

actual_long = dict()
actual_lat = dict()
 
for model_name in predicted_all["speed"]:
    
    print(model_name)
    print(ws_all["longitude_no_abs"][model_name], ws_all["latitude_no_abs"][model_name])
 
    actual_long[model_name] = dict()
    actual_lat[model_name] = dict()

    for k in y_test_all["longitude_no_abs"][model_name]:
        print(model_name, k, "actual")
        actual_long[model_name][k] = [0]
        actual_lat[model_name][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name], ws_all["latitude_no_abs"][model_name])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name]
        range_long = len(y_test_all["longitude_no_abs"][model_name][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            actual_long[model_name][k].append(actual_long[model_name][k][-1] + y_test_all["longitude_no_abs"][model_name][k][ix + long_offset])
            actual_lat[model_name][k].append(actual_lat[model_name][k][-1] + y_test_all["latitude_no_abs"][model_name][k][ix + lat_offset])

    predicted_long[model_name] = dict()
    predicted_lat[model_name] = dict()
        
    predicted_long[model_name]["long no abs"] = dict()
    predicted_lat[model_name]["lat no abs"] = dict()

    for k in predicted_all["longitude_no_abs"][model_name]:
        print(model_name, k, "long no abs")
        predicted_long[model_name]["long no abs"][k] = [0]
        predicted_lat[model_name]["lat no abs"][k] = [0]
        
        max_offset_long_lat = max(ws_all["longitude_no_abs"][model_name], ws_all["latitude_no_abs"][model_name])
        long_offset = max_offset_long_lat - ws_all["longitude_no_abs"][model_name]
        lat_offset = max_offset_long_lat - ws_all["latitude_no_abs"][model_name]
        range_long = len(y_test_all["longitude_no_abs"][model_name][k]) - long_offset
        range_lat = len(y_test_all["latitude_no_abs"][model_name][k]) - lat_offset
        min_range_long_lat = min(range_long, range_lat)

        for ix in range(min_range_long_lat):
            predicted_long[model_name]["long no abs"][k].append(predicted_long[model_name]["long no abs"][k][-1] + predicted_all["longitude_no_abs"][model_name][k][ix + long_offset])
            predicted_lat[model_name]["lat no abs"][k].append(predicted_lat[model_name]["lat no abs"][k][-1] + predicted_all["latitude_no_abs"][model_name][k][ix + lat_offset])

    predicted_long[model_name]["long speed dir"] = dict()
    predicted_lat[model_name]["lat speed dir"] = dict()

    for k in predicted_all["speed"][model_name]:
        print(model_name, k, "long speed dir")
        predicted_long[model_name]["long speed dir"][k] = [0]
        predicted_lat[model_name]["lat speed dir"][k] = [0]
    
        max_offset_speed_dir_time = max(max(ws_all["speed"][model_name], ws_all["direction"][model_name]), ws_all["time"][model_name])
        speed_offset_time = max_offset_speed_dir_time - ws_all["speed"][model_name]
        dir_offset_time = max_offset_speed_dir_time - ws_all["direction"][model_name]
        time_offset_time = max_offset_speed_dir_time - ws_all["time"][model_name]
        range_speed_time = len(y_test_all["speed"][model_name][k]) - speed_offset_time
        range_dir_time = len(y_test_all["direction"][model_name][k]) - dir_offset_time
        range_time_time = len(y_test_all["time"][model_name][k]) - time_offset_time
        min_range_speed_dir_time = min(min(range_speed_time, range_dir_time), range_time_time)

        for ix in range(min_range_speed_dir_time):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][k][ix + speed_offset_time] / 111 / 0.1 / 3600 * predicted_all["time"][model_name][k][ix + time_offset_time], change_angle(predicted_all["direction"][model_name][k][ix + dir_offset_time], k))
            predicted_long[model_name]["long speed dir"][k].append(predicted_long[model_name]["long speed dir"][k][-1] + new_long)
            predicted_lat[model_name]["lat speed dir"][k].append(predicted_lat[model_name]["lat speed dir"][k][-1] + new_lat)
            
    predicted_long[model_name]["long speed ones dir"] = dict()
    predicted_lat[model_name]["lat speed ones dir"] = dict()

    for k in predicted_all["speed"][model_name]:
        print(model_name, k, "long speed ones dir")
        predicted_long[model_name]["long speed ones dir"][k] = [0]
        predicted_lat[model_name]["lat speed ones dir"][k] = [0]
    
        max_offset_speed_dir = max(ws_all["speed"][model_name], ws_all["direction"][model_name])
        speed_offset = max_offset_speed_dir - ws_all["speed"][model_name]
        dir_offset = max_offset_speed_dir - ws_all["direction"][model_name]
        range_speed = len(y_test_all["speed"][model_name][k]) - speed_offset
        range_dir = len(y_test_all["direction"][model_name][k]) - dir_offset
        min_range_speed_dir = min(range_speed, range_dir)

        for ix in range(min_range_speed_dir):
            new_long, new_lat = get_sides_from_angle(predicted_all["speed"][model_name][k][ix + speed_offset] / 111 / 0.1 / 3600, change_angle(predicted_all["direction"][model_name][k][ix + dir_offset], k))
            predicted_long[model_name]["long speed ones dir"][k].append(predicted_long[model_name]["long speed ones dir"][k][-1] + new_long)
            predicted_lat[model_name]["lat speed ones dir"][k].append(predicted_lat[model_name]["lat speed ones dir"][k][-1] + new_lat)

if not os.path.isdir("RNN_result"):
    os.makedirs("RNN_result")

save_object("RNN_result/actual_long", actual_long)
save_object("RNN_result/actual_lat", actual_lat)
save_object("RNN_result/predicted_long", predicted_long)
save_object("RNN_result/predicted_lat", predicted_lat)