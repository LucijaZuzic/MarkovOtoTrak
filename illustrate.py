from utilities import translate_var, translate_method, load_object, fill_gap, load_traj_name, scale_long_lat, process_time, preprocess_long_lat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

actual_traj = load_object("actual/actual_traj")

long_dict = load_object("markov_result/long_dict")
lat_dict = load_object("markov_result/lat_dict") 

predicted_time = load_object("predicted/predicted_time")   
predicted_longitude_no_abs = load_object("predicted/predicted_longitude_no_abs")  
predicted_latitude_no_abs = load_object("predicted/predicted_latitude_no_abs")  
predicted_direction = load_object("predicted/predicted_direction")   
predicted_speed = load_object("predicted/predicted_speed")  

actual_time = load_object("actual/actual_time")   
actual_longitude_no_abs = load_object("actual/actual_longitude_no_abs")  
actual_latitude_no_abs = load_object("actual/actual_latitude_no_abs")  
actual_direction = load_object("actual/actual_direction")   
actual_speed = load_object("actual/actual_speed")  

all_longlats = []

for vehicle_event in long_dict:  
    for long in long_dict[vehicle_event]:
        lat = long.replace("long", "lat").replace("x", "y")
        all_longlats.append([long, lat])
    break 

min_len = 100000
lv = "" 
lr = "" 
for subdir_name in actual_traj:
    for some_file in actual_traj[subdir_name]:
        if len(actual_traj[subdir_name][some_file][0]) < min_len and actual_traj[subdir_name][some_file][2] == "test":
            min_len = len(actual_traj[subdir_name][some_file][0])
            lv = subdir_name
            lr = some_file
            
print(min_len, lv, lr)
start_ix, end_ix = 0, 3
print(actual_traj[lv][lr][0][0], actual_traj[lv][lr][1][0])
print(long_dict[lv + "/cleaned_csv/" + lr]["long no abs"][0], lat_dict[lv + "/cleaned_csv/" + lr]["lat no abs"][0])
print(len(long_dict[lv + "/cleaned_csv/" + lr]["long no abs"]), len(actual_traj[lv][lr][0]))
plt.axis('equal')
plt.plot(actual_traj[lv][lr][0][:10], actual_traj[lv][lr][1][:10], c = "b")
plt.plot(long_dict[lv + "/cleaned_csv/" + lr]["long no abs"][:10], lat_dict[lv + "/cleaned_csv/" + lr]["lat no abs"][:10], c = "g")
plt.plot(long_dict[lv + "/cleaned_csv/" + lr]["long speed dir"][:10], lat_dict[lv + "/cleaned_csv/" + lr]["lat speed dir"][:10], c = "r")
plt.plot(long_dict[lv + "/cleaned_csv/" + lr]["long speed ones dir"][:10], lat_dict[lv + "/cleaned_csv/" + lr]["lat speed ones dir"][:10], c = "k")
plt.show()
plt.close()

plt.figure(figsize=(15, 5))
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.axis('equal')
plt.plot(actual_traj[lv][lr][0][start_ix:end_ix], actual_traj[lv][lr][1][start_ix:end_ix], c = "b")
for xval_ix in range(start_ix, end_ix):
    xval = actual_traj[lv][lr][0][xval_ix]
    yval = actual_traj[lv][lr][1][xval_ix]
    xval_next = actual_traj[lv][lr][0][xval_ix + 1]
    yval_next = actual_traj[lv][lr][1][xval_ix + 1]
    
    yvals = np.arange(min(yval, yval_next), max(yval, yval_next) + 10 ** -20, (abs(yval_next - yval) + 10 ** -20) / 1000)
    xvals_first = [xval for val in yvals]
    xvals_second = [xval_next for val in yvals]
    xvals = np.arange(min(xval, xval_next), max(xval, xval_next) + 10 ** -20, (abs(xval_next - xval) + 10 ** -20) / 1000)
    yvals_first = [yval for val in xvals]
    yvals_second = [yval_next for val in xvals]
 
    if xval_ix == 0:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 12, yval + abs(yval - yval_next) / 6, "$\Delta x_{" + str(xval_ix + 1) + "} = x_{" + str(xval_ix + 2) + "}-x_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 5, yval_next - abs(yval - yval_next) / 3, "$\Delta x_{" + str(xval_ix + 1) + "} = x_{" + str(xval_ix + 2) + "}-x_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.text(xval_next + abs(xval - xval_next) / 40, (yval + yval_next) / 2 - abs(yval - yval_next) / 6, "$\Delta y_{" + str(xval_ix + 1) + "} = y_{" + str(xval_ix + 2) + "}-y_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text(xval - abs(xval - xval_next) / 4, (yval + yval_next) / 2 - abs(yval - yval_next) / 8, "$\Delta y_{" + str(xval_ix + 1) + "} = y_{" + str(xval_ix + 2) + "}-y_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.plot(xvals_second, yvals, c = "r")
        plt.plot(xvals, yvals_first, c = "r")
    else:
        plt.plot(xvals_first, yvals, c = "r")
        plt.plot(xvals, yvals_second, c = "r")

    print(len(xvals_first), yval, yval_next) 

plt.title("Example")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

plt.figure(figsize=(15, 5))
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.axis('equal')
plt.plot(long_dict[lv + "/cleaned_csv/" + lr]["long no abs"][start_ix:end_ix], lat_dict[lv + "/cleaned_csv/" + lr]["lat no abs"][start_ix:end_ix], c = "b")
for xval_ix in range(start_ix, end_ix):
    xval = long_dict[lv + "/cleaned_csv/" + lr]["long no abs"][xval_ix]
    yval = lat_dict[lv + "/cleaned_csv/" + lr]["lat no abs"][xval_ix]
    xval_next = long_dict[lv + "/cleaned_csv/" + lr]["long no abs"][xval_ix + 1]
    yval_next = lat_dict[lv + "/cleaned_csv/" + lr]["lat no abs"][xval_ix + 1]
    
    yvals = np.arange(min(yval, yval_next), max(yval, yval_next) + 10 ** -20, (abs(yval_next - yval) + 10 ** -20) / 1000)
    xvals_first = [xval for val in yvals]
    xvals_second = [xval_next for val in yvals]
    xvals = np.arange(min(xval, xval_next), max(xval, xval_next) + 10 ** -20, (abs(xval_next - xval) + 10 ** -20) / 1000)
    yvals_first = [yval for val in xvals]
    yvals_second = [yval_next for val in xvals]
 
    if xval_ix == 0:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 12, yval + abs(yval - yval_next) / 6, "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} + \Delta x_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 5, yval_next - abs(yval - yval_next) / 3, "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} + \Delta x_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.text(xval_next + abs(xval - xval_next) / 40, (yval + yval_next) / 2 - abs(yval - yval_next) / 6, "$y_{" + str(xval_ix + 2) + "} = y_{" + str(xval_ix + 1) + "} + \Delta y_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text(xval - abs(xval - xval_next) / 4, (yval + yval_next) / 2 - abs(yval - yval_next) / 8, "$y_{" + str(xval_ix + 2) + "} = y_{" + str(xval_ix + 1) + "} + \Delta y_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.plot(xvals_second, yvals, c = "r")
        plt.plot(xvals, yvals_first, c = "r")
    else:
        plt.plot(xvals_first, yvals, c = "r")
        plt.plot(xvals, yvals_second, c = "r")

    print(len(xvals_first), yval, yval_next) 

plt.title("Example")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

plt.figure(figsize=(15, 5))
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.axis('equal')
plt.plot(long_dict[lv + "/cleaned_csv/" + lr]["long speed dir"][start_ix:end_ix], lat_dict[lv + "/cleaned_csv/" + lr]["lat speed dir"][start_ix:end_ix], c = "b")
for xval_ix in range(start_ix, end_ix):
    xval = long_dict[lv + "/cleaned_csv/" + lr]["long speed dir"][xval_ix]
    yval = lat_dict[lv + "/cleaned_csv/" + lr]["lat speed dir"][xval_ix]
    xval_next = long_dict[lv + "/cleaned_csv/" + lr]["long speed dir"][xval_ix + 1]
    yval_next = lat_dict[lv + "/cleaned_csv/" + lr]["lat speed dir"][xval_ix + 1]
    
    yvals = np.arange(min(yval, yval_next), max(yval, yval_next) + 10 ** -20, (abs(yval_next - yval) + 10 ** -20) / 1000)
    xvals_first = [xval for val in yvals]
    xvals_second = [xval_next for val in yvals]
    xvals = np.arange(min(xval, xval_next), max(xval, xval_next) + 10 ** -20, (abs(xval_next - xval) + 10 ** -20) / 1000)
    yvals_first = [yval for val in xvals]
    yvals_second = [yval_next for val in xvals]
 
    if xval_ix == 0:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 12, yval + abs(yval - yval_next) / 6, "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} + \Delta x_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 5, yval_next - abs(yval - yval_next) / 3, "$\Delta x_{" + str(xval_ix + 1) + "} = x_{" + str(xval_ix + 2) + "}-x_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.text(xval_next + abs(xval - xval_next) / 40, (yval + yval_next) / 2 - abs(yval - yval_next) / 6, "$\Delta y_{" + str(xval_ix + 1) + "} = y_{" + str(xval_ix + 2) + "}-y_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text(xval - abs(xval - xval_next) / 4, (yval + yval_next) / 2 - abs(yval - yval_next) / 8, "$\Delta y_{" + str(xval_ix + 1) + "} = y_{" + str(xval_ix + 2) + "}-y_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.plot(xvals_second, yvals, c = "r")
        plt.plot(xvals, yvals_first, c = "r")
    else:
        plt.plot(xvals_first, yvals, c = "r")
        plt.plot(xvals, yvals_second, c = "r")

    print(len(xvals_first), yval, yval_next) 

plt.title("Example")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

plt.figure(figsize=(15, 5))
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = "serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.axis('equal')
plt.plot(long_dict[lv + "/cleaned_csv/" + lr]["long speed ones dir"][start_ix:end_ix], lat_dict[lv + "/cleaned_csv/" + lr]["lat speed ones dir"][start_ix:end_ix], c = "b")
for xval_ix in range(start_ix, end_ix):
    xval = long_dict[lv + "/cleaned_csv/" + lr]["long speed ones dir"][xval_ix]
    yval = lat_dict[lv + "/cleaned_csv/" + lr]["lat speed ones dir"][xval_ix]
    xval_next = long_dict[lv + "/cleaned_csv/" + lr]["long speed ones dir"][xval_ix + 1]
    yval_next = lat_dict[lv + "/cleaned_csv/" + lr]["lat speed ones dir"][xval_ix + 1]
    
    yvals = np.arange(min(yval, yval_next), max(yval, yval_next) + 10 ** -20, (abs(yval_next - yval) + 10 ** -20) / 1000)
    xvals_first = [xval for val in yvals]
    xvals_second = [xval_next for val in yvals]
    xvals = np.arange(min(xval, xval_next), max(xval, xval_next) + 10 ** -20, (abs(xval_next - xval) + 10 ** -20) / 1000)
    yvals_first = [yval for val in xvals]
    yvals_second = [yval_next for val in xvals]
 
    if xval_ix == 0:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 12, yval + abs(yval - yval_next) / 6, "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} + \Delta x_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text((xval + xval_next) / 2 - abs(xval - xval_next) / 5, yval_next - abs(yval - yval_next) / 3, "$\Delta x_{" + str(xval_ix + 1) + "} = x_{" + str(xval_ix + 2) + "}-x_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.text(xval_next + abs(xval - xval_next) / 40, (yval + yval_next) / 2 - abs(yval - yval_next) / 6, "$\Delta y_{" + str(xval_ix + 1) + "} = y_{" + str(xval_ix + 2) + "}-y_{" + str(xval_ix + 1) + "}$")
    else:
        plt.text(xval - abs(xval - xval_next) / 4, (yval + yval_next) / 2 - abs(yval - yval_next) / 8, "$\Delta y_{" + str(xval_ix + 1) + "} = y_{" + str(xval_ix + 2) + "}-y_{" + str(xval_ix + 1) + "}$")

    if xval_ix == 0:
        plt.plot(xvals_second, yvals, c = "r")
        plt.plot(xvals, yvals_first, c = "r")
    else:
        plt.plot(xvals_first, yvals, c = "r")
        plt.plot(xvals, yvals_second, c = "r")

    print(len(xvals_first), yval, yval_next) 

plt.title("Example")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
plt.close()

if False:
    plt.plot(actual_traj[lv][lr][0], actual_traj[lv][lr][1])

    for ix_longlat in range(len(all_longlats)):
        plt.plot(long_dict[lv + "/cleaned_csv/" + lr][all_longlats[ix_longlat][0]], lat_dict[lv + "/cleaned_csv/" + lr][all_longlats[ix_longlat][1]])

    plt.show()
    plt.close()

    plt.plot(predicted_time[lv + "/cleaned_csv/" + lr])
    plt.plot(actual_time[lv + "/cleaned_csv/" + lr])
    plt.show()
    plt.close()

    plt.plot(predicted_longitude_no_abs[lv + "/cleaned_csv/" + lr])
    plt.plot(actual_longitude_no_abs[lv + "/cleaned_csv/" + lr])
    plt.show()
    plt.close()

    plt.plot(predicted_latitude_no_abs[lv + "/cleaned_csv/" + lr])
    plt.plot(actual_latitude_no_abs[lv + "/cleaned_csv/" + lr])
    plt.show()
    plt.close()

    plt.plot(predicted_direction[lv + "/cleaned_csv/" + lr])
    plt.plot(actual_direction[lv + "/cleaned_csv/" + lr])
    plt.show()
    plt.close()

    plt.plot(predicted_speed[lv + "/cleaned_csv/" + lr])
    plt.plot(actual_speed[lv + "/cleaned_csv/" + lr])
    plt.show()
    plt.close()