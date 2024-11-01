from utilities import load_object
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from matplotlib import rc
cm = 1/2.54  # centimeters in inches

def translate_category(long):
    translate_name = {
        "long no abs": "$x$ and $y$ offset",  
        "long speed dir": "speed, heading, and time", 
        "long speed ones dir": "speed, heading, and a 1s time interval", 
    }
    if long in translate_name:
        return translate_name[long]
    else:
        return long

def return_pattern(xval_ix, category):
    if category == "actual":
        return "$\Delta x_{" + str(xval_ix + 1) + "} = x_{" + str(xval_ix + 2) + "}-x_{" + str(xval_ix + 1) + "}$"
    if category == "long no abs":
        return "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} + \Delta x_{" + str(xval_ix + 1) + "}$"
    if category == "long speed dir":
        return "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} + \cos (\\theta_{" + str(xval_ix + 1) + "}) \\times v_{" + str(xval_ix + 1) + "} \\times \Delta t_{" + str(xval_ix + 1) + "}$"
    return "$x_{" + str(xval_ix + 2) + "} = x_{" + str(xval_ix + 1) + "} +  \cos (\\theta_{" + str(xval_ix + 1) + "}) \\times v_{" + str(xval_ix + 1) + "} \\times 1$"

def plot_an_arr(arrx, arry, start_ix, category):
    end_ix = start_ix + 3
    
    plt.rcParams["svg.fonttype"] = "none"
    rc('font',**{'family':'Arial'})
    #plt.rcParams.update({"font.size": 5})
    SMALL_SIZE = 5
    MEDIUM_SIZE = 5
    BIGGER_SIZE = 5

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.figure(figsize=(29.7/3*cm, 29.7/11*cm), dpi = 300)

    plt.axis('equal')
    plt.plot(arrx[start_ix:end_ix], arry[start_ix:end_ix], linewidth = 1, c = "b")
    xoffset = 0.01 * (max(arrx[start_ix:end_ix]) - min(arrx[start_ix:end_ix]))
    yoffset = 0.1 * (max(arry[start_ix:end_ix]) - min(arry[start_ix:end_ix]))
    for xval_ix in range(start_ix, end_ix - 1):
        xval = arrx[xval_ix]
        yval = arry[xval_ix]
        xval_next = arrx[xval_ix + 1]
        yval_next = arry[xval_ix + 1]
        
        yvals = np.arange(min(yval, yval_next), max(yval, yval_next) + 10 ** -20, (abs(yval_next - yval) + 10 ** -20) / 1000)
        xvals_first = [xval for val in yvals]
        xvals_second = [xval_next for val in yvals]
        xvals = np.arange(min(xval, xval_next), max(xval, xval_next) + 10 ** -20, (abs(xval_next - xval) + 10 ** -20) / 1000)
        yvals_first = [yval for val in xvals]
        yvals_second = [yval_next for val in xvals]

        xpattern = return_pattern(xval_ix, category)
        ypattern = xpattern.replace("x", "y").replace("cos", "sin")
    
        if xval_ix == start_ix:
            plt.text(xval + xoffset, yval - 2 * yoffset, xpattern)
        else:
            plt.text(xval + xoffset, yval_next + yoffset, xpattern)
            
        if xval_ix == start_ix:
            plt.text(xval_next + xoffset, (yval + yval_next) / 2, ypattern)
        else:
            plt.text(xval - xoffset * (len(ypattern) / 2 + 4), (yval + yval_next) / 2, ypattern)
 
        if xval_ix == start_ix:
            plt.plot(xvals_second, yvals, linewidth = 1, c = "r") 
            plt.plot(xvals, yvals_first, linewidth = 1, c = "r")
        else:
            plt.plot(xvals_first, yvals, linewidth = 1, c = "r") 
            plt.plot(xvals, yvals_second, linewidth = 1, c = "r")

        radi = np.sqrt((xval - xval_next) ** 2 + (yval - yval_next) ** 2) * 0.95

        if "cos" in xpattern:

            angle_step = 0.001
            angle_diff = np.arctan2(abs(yval - yval_next), abs(xval - xval_next))

            if xval_ix == start_ix:
                cx = xval
                cy = yval
                angle_min = 10 ** -20
                angle_max = angle_diff
            else:
                cx = xval_next
                cy = yval_next
                angle_min = np.pi
                angle_max = angle_diff + np.pi

            xvals_radius = [np.cos(angle) * radi + cx for angle in np.arange(angle_min, angle_max, angle_step)]
            yvals_radius = [np.sin(angle) * radi + cy for angle in np.arange(angle_min, angle_max, angle_step)]
            
            plt.plot(xvals_radius, yvals_radius, linewidth = 1, c = "g")

            if xval_ix == start_ix:
                plt.text(xval_next - 6 * xoffset, yval + yoffset, "$\\theta_{" + str(xval_ix + 1) + "}$")
            else:
                plt.text(xval + 3 * xoffset, yval_next - 2 * yoffset, "$\\theta_{" + str(xval_ix + 1) + "}$")
    if "actual" not in category:
        plt.title("Estimating a trajectory using " + translate_category(category))
    else:
        plt.title("Calculating $x$ and $y$ offset")
    plt.xlabel("x")
    plt.ylabel("y")
    if not os.path.isdir("illustrate2/"):
        os.makedirs("illustrate2")
    plt.scatter(arrx[start_ix:end_ix], arry[start_ix:end_ix], c = "purple", zorder = 3, s = 3)
    plt.savefig("illustrate2/" + category + ".png", bbox_inches = "tight")
    plt.savefig("illustrate2/" + category + ".svg", bbox_inches = "tight")
    plt.savefig("illustrate2/" + category + ".pdf", bbox_inches = "tight")
    plt.close()

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

plot_an_arr(actual_traj[lv][lr][0], actual_traj[lv][lr][1], 3, "actual") 

for longlat in all_longlats:
    plot_an_arr(long_dict[lv + "/cleaned_csv/" + lr][longlat[0]], lat_dict[lv + "/cleaned_csv/" + lr][longlat[1]], 3, longlat[0])

def str_convert(val):
    if val == False:
        return "0"
    if val == True:
        return "1"
    new_val = val
    power_to = 0
    while abs(new_val) < 1 and new_val != 0.0:
        new_val *= 10
        power_to += 1 
    rounded = str(np.round(new_val, 2))
    if rounded[-2:] == '.0':
        rounded = rounded[:-2]
    if power_to != 0:  
        rounded += " \\times 10^{-" + str(power_to) + "}"
    return rounded

strintro = ""
for longlat in all_longlats:
    strintro += translate_category(longlat[0]) + " & "
print(strintro)
maxval = 10 ** 5
for ixnum in range(6):
    strpr = "$" + str_convert(actual_traj[lv][lr][0][ixnum] * maxval) + "$ & "
    for longlat in all_longlats:
        strpr += "$" + str_convert(long_dict[lv + "/cleaned_csv/" + lr][longlat[0]][ixnum] * maxval) + "$ & "
    print(strpr[:-2], "\\\\ \\hline")

for ixnum in range(6):
    strpr = "$" + str_convert(actual_traj[lv][lr][1][ixnum] * maxval) + "$ & "
    for longlat in all_longlats:
        strpr += "$" + str_convert(lat_dict[lv + "/cleaned_csv/" + lr][longlat[1]][ixnum] * maxval) + "$ & "
    print(strpr[:-2], "\\\\ \\hline")

for ixnum in range(6):
    print(actual_longitude_no_abs[lv + "/cleaned_csv/" + lr][ixnum], predicted_longitude_no_abs[lv + "/cleaned_csv/" + lr][ixnum])
for ixnum in range(6):
    print(actual_latitude_no_abs[lv + "/cleaned_csv/" + lr][ixnum], predicted_latitude_no_abs[lv + "/cleaned_csv/" + lr][ixnum])
for ixnum in range(6):
    print(actual_direction[lv + "/cleaned_csv/" + lr][ixnum], predicted_direction[lv + "/cleaned_csv/" + lr][ixnum])
for ixnum in range(6):
    print(actual_speed[lv + "/cleaned_csv/" + lr][ixnum], predicted_speed[lv + "/cleaned_csv/" + lr][ixnum])
for ixnum in range(6):
    print(actual_time[lv + "/cleaned_csv/" + lr][ixnum], predicted_time[lv + "/cleaned_csv/" + lr][ixnum])