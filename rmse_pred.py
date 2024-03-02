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

def plot_rmse_prrr(file_extension, var_name, actual, predictions, filenames, filenames_length):
    if not os.path.isdir("rmse"):
        os.makedirs("rmse")
 
    plt.figure(figsize = (20, 6 * len(filenames)), dpi = 80)
    plt.rcParams.update({'font.size': 22})
    
    total_len_rides = 0
    last_vehicle = "" 
    length_of_vehicle = []
    for ix_ride in range(len(filenames_length)):
        len_ride = filenames_length[ix_ride]
        actual_ride = actual[total_len_rides:total_len_rides + len_ride]
        predictions_ride = predictions[total_len_rides:total_len_rides + len_ride]
        total_len_rides += len_ride
        split_filename = filenames[ix_ride].split("/")
        vehicle = split_filename[0].replace("Vehicle_", "")
        ride = split_filename[-1].replace("events_", "").replace(".csv", "")
        print(len(filenames_length) // 2 + len(filenames_length) % 2, ix_ride + 1)
        plt.subplot(len(filenames_length) // 2 + len(filenames_length) % 2, 2, ix_ride + 1)
        plt.plot(range(len(actual_ride)), actual_ride, color = "b") 
        plt.plot(range(len(predictions_ride)), predictions_ride, color = "orange") 
        plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
        plt.title("Actual and predicted values " + var_name + "\nVehicle " + vehicle + " Ride " + ride)
        plt.xlabel("Point index")
        plt.ylabel(var_name)
        if vehicle != last_vehicle:
            length_of_vehicle.append(0) 
        length_of_vehicle[-1] += 1
        last_vehicle = vehicle
    plt.savefig("rmse/" + file_extension + "_all.png", bbox_inches = "tight")
    plt.close()

    print(length_of_vehicle)

    total_len_rides = 0
    last_vehicle = filenames[0].split("/")[0].replace("Vehicle_", "")
    ix_vehicle = 0
    ix_ride_zero = 0
    for ix_ride in range(len(filenames_length)):
        len_ride = filenames_length[ix_ride]
        actual_ride = actual[total_len_rides:total_len_rides + len_ride]
        predictions_ride = predictions[total_len_rides:total_len_rides + len_ride]
        total_len_rides += len_ride
        split_filename = filenames[ix_ride].split("/")
        vehicle = split_filename[0].replace("Vehicle_", "")
        ride = split_filename[-1].replace("events_", "").replace(".csv", "")
        if vehicle != last_vehicle:
            if ix_vehicle != 0:
                plt.savefig("rmse/" + file_extension + "_Vehicle_" + vehicle + "_all.png", bbox_inches = "tight")
                plt.close()
            ix_vehicle += 1
            plt.figure(figsize = (20, 6 * length_of_vehicle[ix_vehicle]), dpi = 80)
            plt.rcParams.update({'font.size': 22}) 
            ix_ride_zero = 0
        last_vehicle = vehicle
        print(vehicle, last_vehicle, ix_vehicle, length_of_vehicle[ix_vehicle], length_of_vehicle[ix_vehicle] // 2 + length_of_vehicle[ix_vehicle] % 2, ix_ride_zero + 1)
        plt.subplot(length_of_vehicle[ix_vehicle] // 2 + length_of_vehicle[ix_vehicle] % 2, 2, ix_ride_zero + 1)
        ix_ride_zero += 1
        plt.plot(range(len(actual_ride)), actual_ride, color = "b") 
        plt.plot(range(len(predictions_ride)), predictions_ride, color = "orange") 
        plt.legend(['Actual', 'Predicted'], loc = "upper left", ncol = 2)
        plt.title("Actual and predicted values " + var_name + "\nVehicle " + vehicle + " Ride " + ride)
        plt.xlabel("Point index")
        plt.ylabel(var_name)

all_x_heading, all_mine_heading, filenames_heading, filenames_length_heading = read_var("direction")
all_x_latitude_no_abs, all_mine_latitude_no_abs, filenames_latitude_no_abs, filenames_length_latitude_no_abs = read_var("latitude_no_abs")
all_x_longitude_no_abs, all_mine_longitude_no_abs, filenames_longitude_no_abs, filenames_length_longitude_no_abs = read_var("longitude_no_abs")
all_x_speed, all_mine_speed, filenames_speed, filenames_length_speed = read_var("speed")
all_x_time, all_mine_time, filenames_time, filenames_length_time = read_var("time")

plot_rmse("heading", "Heading ($\degree$)", all_mine_heading, all_x_heading, filenames_heading, filenames_length_heading)
plot_rmse("latitude", "x offset ($\degree$ long.)", all_mine_latitude_no_abs, all_x_latitude_no_abs, filenames_latitude_no_abs, filenames_length_latitude_no_abs)
plot_rmse("longitude", "y offset ($\degree$ lat.)", all_mine_longitude_no_abs, all_x_longitude_no_abs, filenames_longitude_no_abs, filenames_length_longitude_no_abs)
plot_rmse("speed", "Speed (km/h)", all_mine_speed, all_x_speed, filenames_speed, filenames_length_speed)
plot_rmse("time", "Time (s)", all_mine_time, all_x_time, filenames_time, filenames_length_time)

print("heading", max(all_mine_heading),min(all_mine_heading), math.sqrt(mean_squared_error(all_mine_heading, all_x_heading)) / (max(all_mine_heading) - min(all_mine_heading)))
print("latitude", max(all_mine_latitude_no_abs),min(all_mine_latitude_no_abs), math.sqrt(mean_squared_error(all_mine_latitude_no_abs, all_x_latitude_no_abs)) / (max(all_mine_latitude_no_abs) - min(all_mine_latitude_no_abs)))
print("longitude", max(all_mine_longitude_no_abs),min(all_mine_longitude_no_abs), math.sqrt(mean_squared_error(all_mine_longitude_no_abs, all_x_longitude_no_abs)) / (max(all_mine_longitude_no_abs) - min(all_mine_longitude_no_abs)))
print("speed", max(all_mine_speed),min(all_mine_speed), math.sqrt(mean_squared_error(all_mine_speed, all_x_speed)) / (max(all_mine_speed) - min(all_mine_speed)))
print("time", max(all_mine_time),min(all_mine_time), math.sqrt(mean_squared_error(all_mine_time, all_x_time)) / (max(all_mine_time) - min(all_mine_time)))