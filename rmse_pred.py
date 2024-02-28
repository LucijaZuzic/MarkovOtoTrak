from utilities import load_object
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math

all_subdirs = os.listdir() 

def flatten(all_x, all_mine):
    all_flat_x = []
    all_flat_mine = []
    filenames = []
    filenames_length = []
    for filename in all_x:
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
    plt.title("Actual and predicted values " + var_name) 
    total_len = 0
    plt.axvline(0, c = "g") 
    last_vehicle = "" 
    for ix_val in range(len(filenames_length)):
        split_filename = filenames[ix_val].split("/")
        vehicle = split_filename[0]
        if vehicle == last_vehicle and ix_val != len(filenames_length) - 1:
            plt.axvline(total_len, c = "r")
        else:
            plt.axvline(total_len, c = "g") 
        last_vehicle = vehicle
        total_len += filenames_length[ix_val]
        #plt.text((total_len + total_len - filenames_length[ix_val]) / 2, min(min(actual), min(predictions)), filenames[ix_val], rotation = 90)
    
    
    name_of_vehicle = []
    length_of_vehicle = []
    filenames_of_vehicle = []
    filenames_length_of_vehicle = []

    for ix_file in range(len(filenames)):
        filename = filenames[ix_file]
        split_filename = filename.split("/")
        vehicle = split_filename[0]
        ride = split_filename[-1].replace("events_", "").replace(".csv", "")
        if len(name_of_vehicle) == 0 or name_of_vehicle[-1] != vehicle:
            name_of_vehicle.append(vehicle)
            length_of_vehicle.append(0)
            filenames_of_vehicle.append([])
            filenames_length_of_vehicle.append([])
        length_of_vehicle[-1] += filenames_length[ix_file]
        filenames_of_vehicle[-1].append(ride)
        filenames_length_of_vehicle[-1].append(filenames_length[ix_file])
    
    total_len_of_vehicle = [0]
    for ix_veh in range(len(length_of_vehicle)):
        total_len_of_vehicle.append(total_len_of_vehicle[-1] + length_of_vehicle[ix_veh])

    lab_of_vehicle = []
    pos_of_vehicle = []
    for ix_veh in range(len(total_len_of_vehicle) - 1):
        lab_of_vehicle.append(name_of_vehicle[ix_veh].split("_")[-1])
        pos_of_vehicle.append((total_len_of_vehicle[ix_veh] + total_len_of_vehicle[ix_veh + 1]) / 2)

    plt.xlabel('Vehicle and ride')
    plt.xticks(pos_of_vehicle, lab_of_vehicle)
    plt.ylabel(var_name) 
    plt.savefig("rmse/" + file_extension + "_all.png", bbox_inches = "tight")
    plt.close()

    total_len_all_vehicles = 0
    for ix_vehicle in range(len(name_of_vehicle)):
        len_veh = length_of_vehicle[ix_vehicle]
        actual_vehicle = actual[total_len_all_vehicles:total_len_all_vehicles + len_veh]
        predictions_vehicle = predictions[total_len_all_vehicles:total_len_all_vehicles + len_veh]
        total_len_all_vehicles += len_veh
        plt.figure(figsize = (20, 6), dpi = 80)
        plt.rcParams.update({'font.size': 22})
        plt.plot(range(len(actual_vehicle)), actual_vehicle, color = "b") 
        plt.plot(range(len(predictions_vehicle)), predictions_vehicle, color = "orange") 
        total_len = 0
        plt.axvline(0, c = "r")
        tick_pos = []
        for ix_val in range(len(filenames_of_vehicle[ix_vehicle])):
            total_len += filenames_length_of_vehicle[ix_vehicle][ix_val]
            plt.axvline(total_len, c = "r")
            tick_pos.append((total_len + total_len - filenames_length_of_vehicle[ix_vehicle][ix_val]) / 2)
            #plt.text((total_len + total_len - filenames_length_of_vehicle[ix_vehicle][ix_val]) / 2, min(min(actual_vehicle), min(predictions_vehicle)), filenames_of_vehicle[ix_vehicle][ix_val], rotation = 90)
        plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
        plt.title("Actual and predicted values " + var_name + " " + name_of_vehicle[ix_vehicle].replace("_", " ")) 
        plt.xlabel('Ride')
        plt.xticks(tick_pos, filenames_of_vehicle[ix_vehicle], rotation = 90, ha = 'right')
        plt.ylabel(var_name) 
        plt.savefig("rmse/" + file_extension + "_" + name_of_vehicle[ix_vehicle] + "_all.png", bbox_inches = "tight")
        plt.close()

all_x_heading, all_mine_heading, filenames_heading, filenames_length_heading = read_var("direction")
all_x_latitude_no_abs, all_mine_latitude_no_abs, filenames_latitude_no_abs, filenames_length_latitude_no_abs = read_var("latitude_no_abs")
all_x_longitude_no_abs, all_mine_longitude_no_abs, filenames_longitude_no_abs, filenames_length_longitude_no_abs = read_var("longitude_no_abs")
all_x_speed, all_mine_speed, filenames_speed, filenames_length_speed = read_var("speed")
all_x_time, all_mine_time, filenames_time, filenames_length_time = read_var("time")

plot_rmse("heading", "Heading", all_mine_heading, all_x_heading, filenames_heading, filenames_length_heading)
plot_rmse("latitude", "x offset", all_mine_latitude_no_abs, all_x_latitude_no_abs, filenames_latitude_no_abs, filenames_length_latitude_no_abs)
plot_rmse("longitude", "y offset", all_mine_longitude_no_abs, all_x_longitude_no_abs, filenames_longitude_no_abs, filenames_length_longitude_no_abs)
plot_rmse("speed", "Speed (km/h)", all_mine_speed, all_x_speed, filenames_speed, filenames_length_speed)
plot_rmse("time", "Time (s)", all_mine_time, all_x_time, filenames_time, filenames_length_time)

print("heading", math.sqrt(mean_squared_error(all_mine_heading, all_x_heading)) / (max(all_mine_heading) - min(all_mine_heading)))
print("latitude", math.sqrt(mean_squared_error(all_mine_latitude_no_abs, all_x_latitude_no_abs)) / (max(all_mine_latitude_no_abs) - min(all_mine_latitude_no_abs)))
print("longitude", math.sqrt(mean_squared_error(all_mine_longitude_no_abs, all_x_longitude_no_abs)) / (max(all_mine_longitude_no_abs) - min(all_mine_longitude_no_abs)))
print("speed", math.sqrt(mean_squared_error(all_mine_speed, all_x_speed)) / (max(all_mine_speed) - min(all_mine_speed)))
print("time", math.sqrt(mean_squared_error(all_mine_time, all_x_time)) / (max(all_mine_time) - min(all_mine_time)))