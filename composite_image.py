from utilities import load_object
import os
import numpy as np
import matplotlib.pyplot as plt
        
long_dict = load_object("markov_result/long_dict")
lat_dict = load_object("markov_result/lat_dict") 

def mosaic(rides, name, method_long = "", method_lat = ""):
    
    x_dim_rides = int(np.sqrt(len(rides)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides))

    for ix_ride in range(len(rides)):

        test_ride = rides[ix_ride]
            
        plt.subplot(y_dim_rides, x_dim_rides, ix_ride + 1)
        plt.plot(test_ride[0], test_ride[1], c = "k", linewidth = 2)

        minlong = min(test_ride[0])
        minlat = min(test_ride[1])

        maxlong = max(test_ride[0])
        maxlat = max(test_ride[1])
    
        if method_long != "":
            plt.plot(long_dict[test_ride[2]][method_long], lat_dict[test_ride[2]][method_lat], c = "b", linewidth = 2)

            minlong = min(minlong, min(long_dict[test_ride[2]][method_long]))
            minlat = min(minlat, min(lat_dict[test_ride[2]][method_lat]))

            maxlong = max(maxlong, max(long_dict[test_ride[2]][method_long]))
            maxlat = max(maxlat, max(lat_dict[test_ride[2]][method_lat]))

        xrange = maxlong - minlong
        yrange = maxlat - minlat

        maxrange = max(xrange, yrange)
        offset_val = 0.01 * maxrange
             
        plt.xlim(minlong - offset_val, minlong + maxrange + offset_val)
        plt.ylim(minlat - offset_val, minlat + maxrange + offset_val)
       
        plt.axis("off")
        
    plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
all_longlats = []

for vehicle_event in long_dict:  
    for long in long_dict[vehicle_event]:
        lat = long.replace("long", "lat").replace("x", "y")
        all_longlats.append([long, lat])
    break 
 
all_subdirs = os.listdir()

if not os.path.isdir("mosaic"):
    os.makedirs("mosaic")

test_rides_all = []
train_rides_all = []
rides_all = []

actual_traj = load_object("actual/actual_traj")

for subdir_name in actual_traj:  

    test_rides_veh = []
    train_rides_veh = []
    rides_veh = []
        
    for some_file in actual_traj[subdir_name]:  

        longitudes, latitudes, is_test = actual_traj[subdir_name][some_file]

        if is_test == "test":
            test_rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
            test_rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
        else:
            train_rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
            train_rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])

        rides_veh.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])
        rides_all.append([longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file])

    if len(test_rides_veh):
        mosaic(test_rides_veh, "mosaic/" + subdir_name + "_test_mosaic.png")

    if len(train_rides_veh):
        mosaic(train_rides_veh, "mosaic/" + subdir_name + "_train_mosaic.png")

    if len(rides_veh):
        mosaic(rides_veh, "mosaic/" + subdir_name + "_all_mosaic.png")

    for ix_longlat in range(len(all_longlats)):

        if len(test_rides_veh):
            mosaic(test_rides_veh, "mosaic/" + subdir_name + "_" + all_longlats[ix_longlat][0] + "_" + all_longlats[ix_longlat][1] + "_test_mosaic.png", all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])

    print(subdir_name, len(test_rides_veh), len(train_rides_veh), len(rides_veh))

if len(test_rides_all):
    mosaic(test_rides_all, "mosaic/all_test_mosaic.png")

if len(train_rides_all):
    mosaic(train_rides_all, "mosaic/all_train_mosaic.png")

if len(rides_all):
    mosaic(rides_all, "mosaic/all_all_mosaic.png")

for ix_longlat in range(len(all_longlats)):
        
    if len(test_rides_all):
        mosaic(test_rides_all, "mosaic/all_" + all_longlats[ix_longlat][0] + "_" + all_longlats[ix_longlat][1] + "_test_mosaic.png", all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])

print(len(test_rides_all), len(train_rides_all), len(rides_all))