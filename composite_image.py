from utilities import load_object
import os
import numpy as np
import matplotlib.pyplot as plt
        
long_dict = load_object("markov_result/long_dict")
lat_dict = load_object("markov_result/lat_dict") 
distance_predicted = load_object("markov_result/distance_predicted")

best_best = dict()
worst_worst = dict()
best_best_ride = dict()
worst_worst_ride = dict()
best_best_score = dict()
worst_worst_score = dict()

best_best_avg_ride = dict()
worst_worst_avg_ride = dict()
best_best_avg_score = dict()
worst_worst_avg_score = dict()

avg_for_ride_longlat = dict()

best_best_xavg_ride = dict()
worst_worst_xavg_ride = dict()
best_best_xavg_score = dict()
worst_worst_xavg_score = dict()

xavg_for_ride_longlat = dict()

best_best_yavg_ride = dict()
worst_worst_yavg_ride = dict()
best_best_yavg_score = dict()
worst_worst_yavg_score = dict()

yavg_for_ride_longlat = dict()

all_metrics = dict()
all_longs = dict()
all_lats = dict()
all_longlats = dict()
for vehicle in distance_predicted:
    avg_for_ride_longlat[vehicle] = dict() 
    xavg_for_ride_longlat[vehicle] = dict() 
    yavg_for_ride_longlat[vehicle] = dict() 
    for event in distance_predicted[vehicle]:  
        avg_for_ride_longlat[vehicle][event] = dict()
        xavg_for_ride_longlat[vehicle][event] = dict()
        yavg_for_ride_longlat[vehicle][event] = dict()
        for metric_name in distance_predicted[vehicle][event]:
            if "ray" in metric_name or "custom" in metric_name:
                continue
            if "no time" in metric_name or "custom" in metric_name:
                continue
            all_metrics[metric_name] = True
            for longlat in distance_predicted[vehicle][event][metric_name]:
                long = longlat.split("-")[0] 
                lat = long.replace("long", "lat").replace("x", "y")
                all_longs[long] = True
                all_lats[lat] = True
                all_longlats[long+"-"+lat] = True

for longlat in all_longlats: 
    best_best_avg_ride[longlat] = ""
    worst_worst_avg_ride[longlat] = ""
    best_best_avg_score[longlat] = 100000
    worst_worst_avg_score[longlat] = -100000
    best_best_xavg_ride[longlat] = ""
    worst_worst_xavg_ride[longlat] = ""
    best_best_xavg_score[longlat] = 100000
    worst_worst_xavg_score[longlat] = -100000
    best_best_yavg_ride[longlat] = ""
    worst_worst_yavg_ride[longlat] = ""
    best_best_yavg_score[longlat] = 100000
    worst_worst_yavg_score[longlat] = -100000
    for vehicle in distance_predicted:
        for event in distance_predicted[vehicle]:  
            avg_score = 0
            xavg_score = 0
            yavg_score = 0
            for metric_name in all_metrics:   
                avg_score += distance_predicted[vehicle][event][metric_name][longlat]
                if " x" in metric_name:
                    xavg_score += distance_predicted[vehicle][event][metric_name][longlat]
                if " y" in metric_name:
                    yavg_score += distance_predicted[vehicle][event][metric_name][longlat]
            avg_score /= len(all_metrics) 
            xavg_score /= 2 
            yavg_score /= 2
            avg_for_ride_longlat[vehicle][event][longlat] = avg_score
            xavg_for_ride_longlat[vehicle][event][longlat] = xavg_score
            yavg_for_ride_longlat[vehicle][event][longlat] = yavg_score
            if avg_score < best_best_avg_score[longlat]:
                best_best_avg_score[longlat] = avg_score
                best_best_avg_ride[longlat] = vehicle + "/" + event  
            if avg_score > worst_worst_avg_score[longlat]:
                worst_worst_avg_score[longlat] = avg_score
                worst_worst_avg_ride[longlat] = vehicle + "/" + event  
            if xavg_score < best_best_xavg_score[longlat]:
                best_best_xavg_score[longlat] = xavg_score
                best_best_xavg_ride[longlat] = vehicle + "/" + event  
            if xavg_score > worst_worst_xavg_score[longlat]:
                worst_worst_xavg_score[longlat] = xavg_score
                worst_worst_xavg_ride[longlat] = vehicle + "/" + event 
            if yavg_score < best_best_yavg_score[longlat]:
                best_best_yavg_score[longlat] = yavg_score
                best_best_yavg_ride[longlat] = vehicle + "/" + event  
            if yavg_score > worst_worst_yavg_score[longlat]:
                worst_worst_yavg_score[longlat] = yavg_score
                worst_worst_yavg_ride[longlat] = vehicle + "/" + event 

count_best_by_avg = dict()
count_worst_by_avg = dict()
for longlat in all_longlats: 
    count_best_by_avg[longlat] = 0
    count_worst_by_avg[longlat] = 0
count_best_by_xavg = dict()
count_worst_by_xavg = dict()
for longlat in all_longlats: 
    count_best_by_xavg[longlat] = 0
    count_worst_by_xavg[longlat] = 0
count_best_by_yavg = dict()
count_worst_by_yavg = dict()
for longlat in all_longlats: 
    count_best_by_yavg[longlat] = 0
    count_worst_by_yavg[longlat] = 0

for vehicle in avg_for_ride_longlat:
    for event in avg_for_ride_longlat[vehicle]:  
        bestlonglat = 100000
        worstlonglat = -100000
        bestlonglatname = ""
        worstlonglatname = "" 
        for longlat in avg_for_ride_longlat[vehicle][event]: 
            if avg_for_ride_longlat[vehicle][event][longlat] < bestlonglat:
                bestlonglat = avg_for_ride_longlat[vehicle][event][longlat]
                bestlonglatname = longlat
            if avg_for_ride_longlat[vehicle][event][longlat] > worstlonglat:
                worstlonglat = avg_for_ride_longlat[vehicle][event][longlat]
                worstlonglatname = longlat 
        count_best_by_avg[bestlonglatname] += 1
        count_worst_by_avg[worstlonglatname] += 1
        
for vehicle in xavg_for_ride_longlat:
    for event in xavg_for_ride_longlat[vehicle]:  
        bestlonglat = 100000
        worstlonglat = -100000
        bestlonglatname = ""
        worstlonglatname = "" 
        for longlat in xavg_for_ride_longlat[vehicle][event]: 
            if xavg_for_ride_longlat[vehicle][event][longlat] < bestlonglat:
                bestlonglat = xavg_for_ride_longlat[vehicle][event][longlat]
                bestlonglatname = longlat
            if xavg_for_ride_longlat[vehicle][event][longlat] > worstlonglat:
                worstlonglat = xavg_for_ride_longlat[vehicle][event][longlat]
                worstlonglatname = longlat 
        count_best_by_xavg[bestlonglatname] += 1
        count_worst_by_xavg[worstlonglatname] += 1

for vehicle in yavg_for_ride_longlat:
    for event in yavg_for_ride_longlat[vehicle]:  
        bestlonglat = 100000
        worstlonglat = -100000
        bestlonglatname = ""
        worstlonglatname = "" 
        for longlat in yavg_for_ride_longlat[vehicle][event]: 
            if yavg_for_ride_longlat[vehicle][event][longlat] < bestlonglat:
                bestlonglat = yavg_for_ride_longlat[vehicle][event][longlat]
                bestlonglatname = longlat
            if yavg_for_ride_longlat[vehicle][event][longlat] > worstlonglat:
                worstlonglat = yavg_for_ride_longlat[vehicle][event][longlat]
                worstlonglatname = longlat 
        count_best_by_yavg[bestlonglatname] += 1
        count_worst_by_yavg[worstlonglatname] += 1

for metric_name in all_metrics:   
    best_best[metric_name] = dict()
    worst_worst[metric_name] = dict()
    best_best_ride[metric_name] = dict()
    worst_worst_ride[metric_name] = dict()
    best_best_score[metric_name] = dict()
    worst_worst_score[metric_name] = dict()
    longlats = set()
    for longlat in all_longlats:   
        best_best[metric_name][longlat] = 0
        worst_worst[metric_name][longlat] = 0
        best_best_ride[metric_name][longlat] = ""
        worst_worst_ride[metric_name][longlat] = ""
        best_best_score[metric_name][longlat] = 100000
        worst_worst_score[metric_name][longlat] = -100000
    for vehicle in distance_predicted:
        for event in distance_predicted[vehicle]:  
            minmetric = 100000
            minname = ""  
            bestride = ""
            maxmetric = -100000
            maxname = ""
            worstride = ""
            for longlat in all_longlats:
                if distance_predicted[vehicle][event][metric_name][longlat] < minmetric:
                    minmetric = distance_predicted[vehicle][event][metric_name][longlat]
                    minname = longlat
                if distance_predicted[vehicle][event][metric_name][longlat] > maxmetric:
                    maxmetric = distance_predicted[vehicle][event][metric_name][longlat]
                    maxname = longlat
                if distance_predicted[vehicle][event][metric_name][longlat] < best_best_score[metric_name][longlat]:
                    best_best_score[metric_name][longlat] = distance_predicted[vehicle][event][metric_name][longlat]
                    best_best_ride[metric_name][longlat] = vehicle + "/" + event
                if distance_predicted[vehicle][event][metric_name][longlat] > worst_worst_score[metric_name][longlat]:
                    worst_worst_score[metric_name][longlat] = distance_predicted[vehicle][event][metric_name][longlat]
                    worst_worst_ride[metric_name][longlat] = vehicle + "/" + event 
            best_best[metric_name][minname] += 1
            worst_worst[metric_name][maxname] += 1

set_drawable = set()

for metric in best_best_ride:
    for method_composite in best_best_ride[metric]:
        set_drawable.add(best_best_ride[metric][method_composite].replace("/", "/cleaned_csv/") + method_composite)
         
for method_composite in best_best_avg_ride:
    set_drawable.add(best_best_avg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in best_best_xavg_ride:
    set_drawable.add(best_best_xavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in best_best_yavg_ride:
    set_drawable.add(best_best_yavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)

for metric in worst_worst_ride:
    for method_composite in worst_worst_ride[metric]:
        set_drawable.add(worst_worst_ride[metric][method_composite].replace("/", "/cleaned_csv/") + method_composite)

for method_composite in worst_worst_avg_ride:
    set_drawable.add(worst_worst_avg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in worst_worst_xavg_ride:
    set_drawable.add(worst_worst_xavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)
 
for method_composite in worst_worst_yavg_ride:
    set_drawable.add(worst_worst_yavg_ride[method_composite].replace("/", "/cleaned_csv/") + method_composite)

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

def mosaic_one(rides, name, method_long = "", method_lat = ""):
    
    x_dim_rides = int(np.sqrt(len(rides)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides), dpi = 80)
    plt.rcParams.update({'font.size': 28}) 
    plt.rcParams['font.family'] = "serif"
    plt.rcParams["mathtext.fontset"] = "dejavuserif"
    plt.axis("equal")

    test_ride = rides[0]
  
    if method_long != "":

        plt.plot(long_dict[test_ride[2]][method_long], lat_dict[test_ride[2]][method_lat], c = "b", linewidth = 10, label = "Estimated")
  
    plt.plot(test_ride[0], test_ride[1], c = "k", linewidth = 10, label = "Original")

    if method_long != "":

        plt.plot(long_dict[test_ride[2]][method_long][0], lat_dict[test_ride[2]][method_lat][0], marker = "o", label = "Start", color = "k", mec = "k", mfc = "g", ms = 20, mew = 10, linewidth = 10) 
   
    split_file_veh = test_ride[2].split("/")
    vehicle = split_file_veh[0].replace("Vehicle_", "")
    ride = split_file_veh[-1].replace("events_", "").replace(".csv", "")

    title_new = "Vehicle " + vehicle + " Ride " + ride + "\n"
    if method_long != "":
        title_new += translate_category(method_long) + "\n" 
        for metric in distance_predicted[split_file_veh[0]][split_file_veh[-1]]:
            if "simpson" in metric:
                continue
            title_new += new_metric_translate(metric) + ": " + str_convert_new(distance_predicted[split_file_veh[0]][split_file_veh[-1]][metric][method_long + "-" + method_lat]) + "\n"
    
    plt.title(title_new) 
    if method_long != "":
        plt.legend()
    if test_ride[2] + method_long + "-" + method_lat in set_drawable:
        plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
def mosaic(rides, name, method_long = "", method_lat = ""):
    
    x_dim_rides = int(np.sqrt(len(rides)))
    y_dim_rides = x_dim_rides
 
    while x_dim_rides * y_dim_rides < len(rides):
        y_dim_rides += 1
    
    plt.figure(figsize = (10, 10 * y_dim_rides / x_dim_rides), dpi = 80)

    for ix_ride in range(len(rides)):

        test_ride = rides[ix_ride]
            
        plt.subplot(y_dim_rides, x_dim_rides, ix_ride + 1)
        plt.rcParams.update({'font.size': 28}) 
        plt.rcParams['font.family'] = "serif"
        plt.rcParams["mathtext.fontset"] = "dejavuserif"
        plt.axis("equal")
        plt.axis("off")

        if method_long != "":

            plt.plot(long_dict[test_ride[2]][method_long], lat_dict[test_ride[2]][method_lat], c = "b", linewidth = 2)
    
        plt.plot(test_ride[0], test_ride[1], c = "k", linewidth = 2)
        
    plt.savefig(name, bbox_inches = "tight")
    plt.close()
 
all_longlats = []

for vehicle_event in long_dict:  
    for long in long_dict[vehicle_event]:
        lat = long.replace("long", "lat").replace("x", "y")
        all_longlats.append([long, lat])
    break 
 
all_subdirs = os.listdir()

if not os.path.isdir("mosaic//test"):
    os.makedirs("mosaic//test")

test_rides_all = []
train_rides_all = []
rides_all = []

actual_traj = load_object("actual/actual_traj")

int_veh = sorted([int(v.split("_")[1]) for v in actual_traj.keys()])

for i in int_veh:  

    subdir_name = "Vehicle_" + str(i)

    test_rides_veh = []
    train_rides_veh = []
    rides_veh = []
        
    for some_file in actual_traj[subdir_name]:  

        longitudes, latitudes, is_test = actual_traj[subdir_name][some_file]

        if is_test == "test":
            for ix_longlat in range(len(all_longlats)):
                mosaic_one([[longitudes, latitudes, subdir_name + "/cleaned_csv/" + some_file]], "mosaic/test/" + subdir_name + "_" + some_file.replace(".csv", "") + "_" + all_longlats[ix_longlat][0] + "_" + all_longlats[ix_longlat][1] + "_test_mosaic.png", all_longlats[ix_longlat][0], all_longlats[ix_longlat][1])
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