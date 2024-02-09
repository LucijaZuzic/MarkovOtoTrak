import matplotlib.pyplot as plt
import os
import math
import pandas as pd
import numpy as np 
import pickle
from scipy.integrate import simpson
import scipy.fft 
from datetime import datetime, timedelta
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.model_selection import train_test_split
#from sympy import Matrix 
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN

def dtw(longitudes1, latitudes1, longitudes2, latitudes2): 
    if len(longitudes1) == 0 and len(latitudes1) == 0 and len(longitudes2) == 0 and len(latitudes2) == 0:
        return 0
    if len(longitudes1) == 0 and len(latitudes1) == 0 and len(longitudes2) > 0 and len(latitudes2) > 0:
        return 10000000
    if len(longitudes1) > 0 and len(latitudes1) > 0 and len(longitudes2) == 0 and len(latitudes2) == 0:
        return 10000000 
    headlong1 = [longitudes1[0]]
    restlong1 = []
    headlong2 = [longitudes2[0]]
    restlong2 = []
    headlat1 = [latitudes1[0]]
    restlat1 = []
    headlat2 = [latitudes2[0]]
    restlat2 = []
    if len(longitudes1) > 1 and len(latitudes1) > 1:
        restlong1 = longitudes1[1:]
        restlat1 = latitudes1[1:]
    if len(longitudes2) > 1 and len(latitudes2) > 1:
        restlong2 = longitudes2[1:]
        restlat2 = latitudes2[1:]
    dtw1 = dtw(headlong1, headlat1, restlong2, restlat2) 
    dtw2 = dtw(restlong1, restlat1, restlong2, restlat2)
    dtw3 = dtw(restlong1, restlat1, headlong2, headlat2)
    return euclidean(headlong1, headlat1, headlong2, headlat2) + min(min(dtw1, dtw2), dtw3)

def decompose_fft(data: list, threshold: float = 0.0):
    fft3 = np.fft.fft(data)
    x = np.arange(0, 10, 10 / len(data))
    freqs = np.fft.fftfreq(len(x), .01)
    recomb = np.zeros((len(x),))
    for i in range(len(fft3)):
        if abs(fft3[i]) / len(x) > threshold:
            sinewave = (
                1 
                / len(x) 
                * (
                    fft3[i].real 
                    * np.cos(freqs[i] * 2 * np.pi * x) 
                    - fft3[i].imag 
                    * np.sin(freqs[i] * 2 * np.pi * x)))
            recomb += sinewave
            plt.plot(x, sinewave)
    plt.title("Sinewave")
    plt.show()

    plt.plot(x, recomb, x, data)
    plt.title("Recomb")
    plt.show()
 
def process_time(time_as_str):
    milisecond = int(time_as_str.split(".")[1]) / 1000
    time_as_str = time_as_str.split(".")[0]
    epoch = datetime(1970, 1, 1)
    return (datetime.strptime(time_as_str, '%Y-%m-%d %H:%M:%S') - epoch).total_seconds() + milisecond
    
def back_to_time_string(time_as_seconds_and_miliseconds):
    seconds = np.round(time_as_seconds_and_miliseconds, 0)
    milisecond = time_as_seconds_and_miliseconds - seconds 
    epoch = datetime(1970, 1, 1) 
    new_datetime = epoch + timedelta(seconds = seconds)
    return new_datetime.strftime('%Y-%m-%d %H:%M:%S') + "." + str(milisecond)

def poly_calc(coeffs, xs):
    ys = []
    for xval in xs:
        yval = 0
        for i in range(len(coeffs)):
            yval += coeffs[i] * (xval ** (len(coeffs) - 1 - i))
        ys.append(yval)
    return ys

def get_fft_xt_yt(longitudes, latitudes, times_ride, deg): 
    xt, yt = get_poly_xt_yt(longitudes, latitudes, times_ride, deg)
    xn = poly_calc(xt, range(len(longitudes)))
    yn = poly_calc(yt, range(len(latitudes)))
    fftx = scipy.fft.fft(xn)
    ffty = scipy.fft.fft(yn)
    return xt, yt, xn, yn, fftx, ffty

def get_poly_xt_yt(longitudes, latitudes, times_ride, deg):
    xt = np.polyfit(times_ride, longitudes, deg)
    yt = np.polyfit(times_ride, latitudes, deg) 
    return xt, yt

def get_surf_xt_yt(longitudes, latitudes, times_ride, metric_used): 
    if metric_used == "trapz":
        return np.trapz(longitudes, times_ride), np.trapz(latitudes, times_ride) 
    if metric_used == "simpson":
        return simpson(longitudes, times_ride), simpson(latitudes, times_ride) 

def transform_time(times_ride): 
    times_ride = [process_time(time_as_str) for time_as_str in times_ride]
    times_ride = [time_one - times_ride[0] for time_one in times_ride] 
    return times_ride
    
def traj_len_offset(longitudes, latitudes):
    sum_dist = 0
    for i in range(len(longitudes) - 1):
        sum_dist += np.sqrt((longitudes[i + 1] - longitudes[i]) ** 2 + (latitudes[i + 1] - latitudes[i]) ** 2)
    offset_total = np.sqrt((longitudes[len(longitudes) - 1] - longitudes[0]) ** 2 + (latitudes[len(latitudes) - 1] - latitudes[0]) ** 2)
    return sum_dist, offset_total
 
def return_speeds_long_lat(longitudes, latitudes, times):
    speeds = []
    for i in range(len(longitudes) - 1):
        if times[i + 1] != times[i]:
            speeds.append(np.sqrt((longitudes[i + 1] - longitudes[i]) ** 2 + (latitudes[i + 1] - latitudes[i]) ** 2) / (times[i + 1] - times[i]))
        else:
            speeds.append(speeds[-1])
    return speeds

def return_speeds_by_axis(longitudes, latitudes, times):
    xspeeds = []
    yspeeds = []
    absxspeeds = []
    absyspeeds = []
    for i in range(len(longitudes) - 1):
        if times[i + 1] != times[i]:
            xspeeds.append((longitudes[i + 1] - longitudes[i]) / (times[i + 1] - times[i]))
            yspeeds.append((latitudes[i + 1] - latitudes[i]) / (times[i + 1] - times[i]))
            absxspeeds.append(abs(longitudes[i + 1] - longitudes[i]) / (times[i + 1] - times[i]))
            absyspeeds.append(abs(latitudes[i + 1] - latitudes[i]) / (times[i + 1] - times[i]))
    return xspeeds, yspeeds, absxspeeds, absyspeeds

def return_steps_by_axis(longitudes, latitudes, times):
    xsteps = []
    ysteps = []
    absxsteps = []
    absysteps = []
    for i in range(len(longitudes) - 1):
        if times[i + 1] != times[i]:
            xsteps.append((longitudes[i + 1] - longitudes[i])) 
            ysteps.append((latitudes[i + 1] - latitudes[i]))
            absxsteps.append(abs(longitudes[i + 1] - longitudes[i]))
            absysteps.append(abs(latitudes[i + 1] - latitudes[i]))
    return xsteps, ysteps, absxsteps, absysteps

def return_euclid_by_axis(longitudes, latitudes):
    euclid = [] 
    for i in range(len(longitudes) - 1):
        euclid.append(np.sqrt((longitudes[i + 1] - longitudes[i]) ** 2 + (latitudes[i + 1] - latitudes[i]) ** 2)) 
    return euclid

def avg_speed_long_lat(longitudes, latitudes, times):
    speeds = return_speeds_long_lat(longitudes, latitudes, times)
    return sum(speeds) / len(speeds)

def return_acceler_long_lat(longitudes, latitudes, times, abs_use = False):
    speeds = return_speeds_long_lat(longitudes, latitudes, times)
    accelers = []
    for i in range(len(speeds) - 1):
        if times[i + 1] - times[i] != 0:
            if not abs_use:
                accelers.append((speeds[i + 1] - speeds[i]) / (times[i + 1] - times[i]))
            else:
                accelers.append(abs(speeds[i + 1] - speeds[i]) / (times[i + 1] - times[i]))
    return accelers

def avg_acceler_long_lat(longitudes, latitudes, times, abs_use = False): 
    accelers = return_acceler_long_lat(longitudes, latitudes, times, abs_use) 
    return sum(accelers) / len(accelers)
 
def return_acceler_speeds(speeds, times, abs_use = False):
    accelers = []
    for i in range(len(speeds) - 1):
        if times[i + 1] - times[i] != 0:
            if not abs_use:
                accelers.append((speeds[i + 1] - speeds[i]) / (times[i + 1] - times[i]))
            else:
                accelers.append(abs(speeds[i + 1] - speeds[i]) / (times[i + 1] - times[i]))
    return accelers

def avg_acceler_speeds(speeds, abs_use = False):
    accelers = return_acceler_speeds(speeds, abs_use)
    return sum(accelers) / len(accelers)

def avg_ratio(s1, s2):
    ratios = []
    for i in range(min(len(s1), len(s2))):
        if s2[i] != 0 and s1[i] != 0:
            ratios.append(s1[i] / s2[i])
    if len(ratios) > 0:
        return sum(ratios) / len(ratios)
    else:
        return 0

def get_vector(x1, x2, y1, y2):
    if x1 == x2:
        return 0, -1, x1 
    else:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1 
        return 1, a, b 
        
def get_intersection(x1, x2, y1, y2, x3, x4, y3, y4):
    yc1, a1, b1 = get_vector(x1, x2, y1, y2)
    yc2, a2, b2 = get_vector(x3, x4, y3, y4)
    if yc1 != 0 and yc2 != 0:
        if a1 != a2:
            xs = (b2 - b1) / (a1 - a2)
            ys = a1 * xs + b1
            return xs, ys
        else:
            return "Nan", "Nan"
    else:
        if yc1 == 0 and yc2 != 0:
            return x1, a2 * x1 + b2
        if yc2 == 0 and yc1 != 0:
            return x3, a1 * x3 + b1
        return "Nan", "Nan"
        
def point_on_line(xs, ys, x1, x2, y1, y2): 
    minx = min(x1, x2)
    maxx = max(x1, x2)
    miny = min(y1, y2)
    maxy = max(y1, y2) 
    return xs >= minx and xs <= maxx and ys >= miny and ys <= maxy
    
def get_surface(x1, x2, x3, y1, y2, y3):
    return 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y2 - y1))
    
def traj_segment_dist(x1, y1, x2, y2, x3, y3, x4, y4):
    xs, ys = get_intersection(x1, x2, y1, y2, x3, x4, y3, y4)
    s1 = get_surface(x2, x3, x4, y2, y3, y4)
    s2 = get_surface(x1, x3, x4, y1, y3, y4)
    s3 = get_surface(x1, x2, x4, y1, y2, y4)
    s4 = get_surface(x1, x2, x3, y1, y2, y3)
    if xs == "Nan":
        return (s1 + s2 + s3 + s4) / 2
    else:
        if point_on_line(xs, ys, x3, x4, y3, y4) or point_on_line(xs, ys, x1, x2, y1, y2):
            s5 = get_surface(x1, x2, xs, y1, y2, ys)
            s6 = get_surface(x3, x4, xs, y3, y4, ys)
            return s5 + s6
        else:
            return (s1 + s2 + s3 + s4) / 2
        
def traj_segment_dist_markings(color_paint, x1, y1, x2, y2, x3, y3, x4, y4):
    xs, ys = get_intersection(x1, x2, y1, y2, x3, x4, y3, y4)
    s1 = get_surface(x2, x3, x4, y2, y3, y4)
    s2 = get_surface(x1, x3, x4, y1, y3, y4)
    s3 = get_surface(x1, x2, x4, y1, y2, y4)
    s4 = get_surface(x1, x2, x3, y1, y2, y3)
    plt.plot([x1, x2], [y1, y2], color_paint)
    plt.plot([x1, x3], [y1, y3], color_paint)
    plt.plot([x1, x4], [y1, y4], color_paint)
    plt.plot([x2, x3], [y2, y3], color_paint)
    plt.plot([x2, x4], [y2, y4], color_paint)
    plt.plot([x3, x4], [y3, y4], color_paint)
    if xs == "Nan":
        return (s1 + s2 + s3 + s4) / 2
    else:
        if point_on_line(xs, ys, x3, x4, y3, y4) or point_on_line(xs, ys, x1, x2, y1, y2):
            s5 = get_surface(x1, x2, xs, y1, y2, ys)
            s6 = get_surface(x3, x4, xs, y3, y4, ys)
            return s5 + s6
        else:
            return (s1 + s2 + s3 + s4) / 2
     
def traj_dist(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    for i in range(len(longitudes1) - 1):
        sum_dist += traj_segment_dist(longitudes1[i], latitudes1[i], longitudes1[i + 1], latitudes1[i + 1], longitudes2[i], latitudes2[i], longitudes2[i + 1], latitudes2[i + 1])
    return sum_dist

def euclidean(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    for i in range(len(longitudes1)):
        sum_dist += np.sqrt((longitudes1[i] - longitudes2[i]) ** 2 + (latitudes1[i] - latitudes2[i]) ** 2)
    return sum_dist / len(longitudes1)

def traj_dist_markings(longitudes1, latitudes1, longitudes2, latitudes2):
    sum_dist = 0
    colors_set = random_colors(len(longitudes1) - 1)
    for i in range(len(longitudes1) - 1):
        sum_dist += traj_segment_dist_markings(colors_set[i], longitudes1[i], latitudes1[i], longitudes1[i + 1], latitudes1[i + 1], longitudes2[i], latitudes2[i], longitudes2[i + 1], latitudes2[i + 1])
    return sum_dist
    
def save_plot_longitude_latitudes_for_ride(longitudes, latitudes, image_name):  
    plt.plot(longitudes, latitudes, color = 'k', linewidth = 10) 
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelbottom = False)
    ax.yaxis.set_tick_params(labelleft = False)  
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')
    plt.savefig(image_name, bbox_inches = 'tight') 
    plt.clf()

def random_colors(num_colors):
    colors_set = []
    for x in range(num_colors):
        string_color = "#"
        while string_color == "#" or string_color in colors_set:
            string_color = "#"
            set_letters = "0123456789ABCDEF"
            for y in range(6):
                string_color += set_letters[np.random.randint(0, 16)]
        colors_set.append(string_color)
    return colors_set
    
def save_object(file_name, std1):       
    with open(file_name, 'wb') as file_object:
        pickle.dump(std1, file_object) 
        file_object.close()

def load_object(file_name): 
    with open(file_name, 'rb') as file_object:
        data = pickle.load(file_object) 
        file_object.close()
        return data
    
def preprocess_long_lat(long_list, lat_list):
    x_dir = long_list[0] < long_list[-1]
    y_dir = lat_list[0] < lat_list[-1]
 
    long_list2 = [x - min(long_list) for x in long_list]
    lat_list2 = [y - min(lat_list) for y in lat_list]
    if x_dir == False: 
        long_list2 = [max(long_list2) - x for x in long_list2]
    if y_dir == False:
        lat_list2 = [max(lat_list2) - y for y in lat_list2]

    return long_list2, lat_list2    
      
def scale_long_lat(long_list, lat_list, xmax = 0, ymax = 0, keep_aspect_ratio = True):
    minx = np.min(long_list)
    maxx = np.max(long_list)
    miny = np.min(lat_list)
    maxy = np.max(lat_list)
    x_diff = maxx - minx
    if x_diff == 0:
        x_diff = 1
    y_diff = maxy - miny 
    if y_diff == 0:
        y_diff = 1
    if xmax == 0 and ymax == 0 and keep_aspect_ratio:
        xmax = max(x_diff, y_diff)
        ymax = max(x_diff, y_diff)
    if xmax == 0 and ymax == 0 and not keep_aspect_ratio:
        xmax = x_diff
        ymax = y_diff
    if xmax == 0 and ymax != 0 and keep_aspect_ratio:
        xmax = ymax 
    if xmax == 0 and ymax != 0 and not keep_aspect_ratio:
        xmax = x_diff 
    if xmax != 0 and ymax == 0 and keep_aspect_ratio:
        ymax = xmax 
    if xmax != 0 and ymax == 0 and not keep_aspect_ratio:
        ymax = y_diff 
    if xmax != 0 and ymax != 0 and keep_aspect_ratio and xmax != ymax:
        ymax = xmax # ymax = xmax or xmax = ymax or keep_aspect_ratio = False or return
    long_list2 = [(x - min(long_list)) / xmax for x in long_list]
    lat_list2 = [(y - min(lat_list)) / ymax for y in lat_list]
    return long_list2, lat_list2  

def total_len(long_list, lat_list):
    total_sum = 0
    for index_coord in range(len(long_list) - 1):
        total_sum += np.sqrt((long_list[index_coord + 1] - long_list[index_coord]) ** 2 + (lat_list[index_coord + 1] - lat_list[index_coord]) ** 2)
    return total_sum

def total_offset(long_list, lat_list):
    return np.sqrt((long_list[0] - long_list[-1]) ** 2 + (lat_list[0] - lat_list[-1]) ** 2)
       
def return_angle_diffs(angles):
    total_angles = []
    for index_coord in range(len(angles) - 1):
        diff = (360 + angles[index_coord + 1] - angles[index_coord]) % 360
        if diff > 180:
            diff = 360 - diff
        total_angles.append(diff)
    return total_angles

def return_angle_diffs_no_abs(angles):
    total_angles = []
    for index_coord in range(len(angles) - 1):
        diff = (360 + angles[index_coord + 1] - angles[index_coord]) % 360
        rotate = False
        if diff > 180:
            rotate = True
            diff = 360 - diff
        if angles[index_coord + 1] < angles[index_coord] and not rotate:
            diff = - diff
        if angles[index_coord + 1] > angles[index_coord] and rotate:
            diff = - diff
        total_angles.append(diff)
    return total_angles

def return_angle_diffs_time(angles, times):
    total_angles = []
    for index_coord in range(len(angles) - 1):
        time_diff = times[index_coord + 1] - times[index_coord]
        diff = (360 + angles[index_coord + 1] - angles[index_coord]) % 360
        if diff > 180:
            diff = 360 - diff
        if time_diff != 0:
            total_angles.append(diff / time_diff)
    return total_angles

def return_angle_diffs_no_abs_time(angles, times):
    total_angles = []
    for index_coord in range(len(angles) - 1):
        time_diff = times[index_coord + 1] - times[index_coord]
        diff = (360 + angles[index_coord + 1] - angles[index_coord]) % 360
        rotate = False
        if diff > 180:
            rotate = True
            diff = 360 - diff
        if angles[index_coord + 1] < angles[index_coord] and not rotate:
            diff = - diff
        if angles[index_coord + 1] > angles[index_coord] and rotate:
            diff = - diff
        if time_diff != 0:
            total_angles.append(diff / time_diff)
    return total_angles

def return_angles(long_list, lat_list):
    total_angles = []
    for index_coord in range(len(long_list) - 1):
        total_angles.append((np.arctan2((lat_list[index_coord + 1] - lat_list[index_coord]), (long_list[index_coord + 1] - long_list[index_coord])) / np.pi * 180 + 360) % 360) 
         
    return total_angles

def return_angles_abs(long_list, lat_list):
    total_angles = []
    for index_coord in range(len(long_list) - 1):
        total_angles.append((np.arctan2(abs(lat_list[index_coord + 1] - lat_list[index_coord]), abs(long_list[index_coord + 1] - long_list[index_coord])) / np.pi * 180 + 360) % 360) 
         
    return total_angles

def mean_speed_len(long_list, lat_list, times_list):
    return total_len(long_list, lat_list) / times_list[-1]

def mean_speed_offset(long_list, lat_list, times_list):
    return total_offset(long_list, lat_list) / times_list[-1]

def SomePolyArea(corners):
    n = len(corners) # of corners 
    area = 0.0
    for i in range(n):
        j = (i + 1) % n 
        area += corners[i][0] * corners[j][1] 
        area -= corners[j][0] * corners[i][1]      
    area = abs(area) / 2.0
    return area

def total_surf(long_list, lat_list):
    points_data = np.column_stack((np.array(long_list), np.array(lat_list)))  
    ch = ConvexHull(points_data)
    corners = []  
    for vx in ch.vertices:
        corners.append((points_data[vx, 0], points_data[vx, 1]))  
    value_ret = SomePolyArea(corners)
    return value_ret
 
def compare_traj_and_sample(sample_x, sample_y, sample_time, t1, metric_used, no_time_t1 = False, no_time_sample = False, scale = True, offset = True, dotsx_original = [], dotsy_original = []): 
    if "simpson " in metric_used:
        if no_time_t1:
            t1["time"] = range(len(t1["long"]))
        if no_time_sample:
            sample_time = range(len(sample_x))
        sample_time_new = [x for x in sample_time]
        for x in range(1, len(sample_time_new)):
            if sample_time_new[x] == sample_time_new[x - 1]:
                sample_time_new[x] = sample_time_new[x - 1] + 10 ** -20
        t1_new = [x for x in t1["time"]]
        for x in range(1, len(t1_new)):
            if t1_new[x] == t1_new[x - 1]:
                t1_new[x] = t1_new[x - 1] + 10 ** -20 
    if metric_used == "custom":
        return traj_dist(t1["long"], t1["lat"], sample_x, sample_y)  
    if metric_used == "dtw":
        return dtw(t1["long"], t1["lat"], sample_x, sample_y)    
    if metric_used == "trapz":
        return abs(np.trapz(t1["lat"], t1["long"]) - np.trapz(sample_y, sample_x)) 
    if metric_used == "simpson": 
        sample_x_sgn = sample_x[1] > sample_x[0]
        for x in range(1, len(sample_x)):
            if sample_x[x] > sample_x[x - 1] != sample_x_sgn: 
                return 1000000 
        t1_sgn = t1["long"][1] > t1["long"][0]
        for x in range(1, len(t1["long"])):
            if t1["long"][x] > t1["long"][x - 1] != t1_sgn: 
                return 1000000  
        return abs(simpson(t1["lat"], t1["long"]) - simpson(sample_y, sample_x))  
    if metric_used == "trapz x":
        return abs(np.trapz(t1["long"], t1["time"]) - np.trapz(sample_x, sample_time))
    if metric_used == "simpson x":  
        return abs(simpson(t1["long"], t1_new) - simpson(sample_x, sample_time_new)) 
    if metric_used == "trapz y":
        return abs(np.trapz(t1["lat"], t1["time"]) - np.trapz(sample_y, sample_time))
    if metric_used == "simpson y": 
        return abs(simpson(t1["lat"], t1_new) - simpson(sample_y, sample_time_new))  
    if metric_used == "euclidean":
        return euclidean(t1["long"], t1["lat"], sample_x, sample_y) 
    if metric_used == "rays":
        if len(dotsx_original) == 0:
            dotsx_original, dotsy_original = make_rays(size)
        else:
            size = len(dotsx_original)
        a1, x1, y1, i1, d1, dx1, dy1, ni1 = compare_traj_ray(dotsx_original, dotsy_original, t1["long"], t1["lat"], scale, offset)
        a2, x2, y2, i2, d2, dx2, dy2, ni2 = compare_traj_ray(dotsx_original, dotsy_original, sample_x, sample_y, scale, offset)
        return abs(a1 - a2)
    
def get_sides_from_angle(longest, angle):
    return longest * np.cos(angle / 180 * np.pi), longest * np.sin(angle / 180 * np.pi)

def count_gap(list_gap): 
    gap_num = 0
    for index_num in range(len(list_gap)):
        if list_gap[index_num] == "undefined":
            gap_num += 1
    return gap_num

def fill_gap(list_gap):
    list_no_gap = []
    last_val = 0
    for index_num in range(len(list_gap)):
        if list_gap[index_num] != "undefined":
            last_val = list_gap[index_num]
        list_no_gap.append(last_val)  
    return list_no_gap

def make_ray(radius, angle, sx, sy):
    return sx + np.cos(angle / 180 * np.pi) * radius, sy + np.sin(angle / 180 * np.pi) * radius
 
def compare_traj_ray(dotsx_original, dotsy_original, test_x, test_y, scale = False, offset = False):
    window_size = len(test_x)
    size = len(dotsx_original)
    num_of_intersection = 0
	
    dotsx = []
    dotsy = []
    for x in range(size):
        dotsx.append(dotsx_original[x])
        dotsy.append(dotsy_original[x])
    scaling_factor = 1
    if scale:
        x_range = max(test_x) - min(test_x)
        y_range = max(test_y) - min(test_y)
        scaling_factor = max(x_range, y_range)
		
    for x1 in range(size): 
        if offset:
            dotsx[x1] = (min(test_x) + max(test_x)) / 2 - scaling_factor / 2 + dotsx[x1] * scaling_factor
            dotsy[x1] = (min(test_y) + max(test_y)) / 2 - scaling_factor / 2 + dotsy[x1] * scaling_factor 
        else:
            dotsx[x1] = min(test_x) + dotsx[x1] * scaling_factor
            dotsy[x1] = min(test_y) + dotsy[x1] * scaling_factor 
			
    '''
	for x1 in range(size):  
		for x2 in range(x1 + 1, size): 
			plt.plot([dotsx[x1], dotsx[x2]], [dotsy[x1], dotsy[x2]], 'g-')	 
	plt.plot(dotsx, dotsy, 'ro')
	if scale:
		if offset:
			plt.plot((min(test_x) + max(test_x)) / 2, (min(test_y) + max(test_y)) / 2, 'bo')
		else:
			plt.plot(min(test_x) + scaling_factor / 2, min(test_y) + scaling_factor / 2, 'bo')
	else:
		plt.plot(0.5, 0.5, 'bo')
	
	test_x = np.random.rand(window_size)  
	test_y = np.random.rand(window_size) 
	plt.plot(test_x, test_y, 'y-o')
	'''
    vectors_from_point = []
    intersections = []
    distances = []
    distancesx = []
    distancesy = []
    for x1 in range(size):
        vectors_from_point.append([])
        intersections.append([])
        distances.append([])
        distancesx.append([])
        distancesy.append([])
        for x2 in range(size):
            vectors_from_point[-1].append(0)
            intersections[-1].append([])
            distances[-1].append([])
            distancesx[-1].append([])
            distancesy[-1].append([])
            for t in range(window_size - 1):
                intersections[-1][-1].append(("Nan", "Nan"))
                distances[-1][-1].append(0)
                distancesx[-1][-1].append(0)
                distancesy[-1][-1].append(0) 
    '''
	vectors_all = []
	for x1 in range(size): 
		for x2 in range(x1 + 1, size):
			vector_new = get_vector(dotsx[x1], dotsx[x2], dotsy[x1], dotsy[x2])
			vectors_from_point[x1][x2] = vector_new
			vectors_from_point[x2][x1] = vector_new 
			vectors_all.append(vector_new)
			plt.plot([dotsx[x1], dotsx[x2]], [dotsy[x1], dotsy[x2]], 'g-')	
	''' 
    for x1 in range(size): 
        for x2 in range(x1 + 1, size): 
            sgnx12 = dotsx[x2] > dotsx[x1] 
            sgny12 = dotsy[x2] > dotsy[x1]  
            for t in range(window_size - 1): 
                xs, ys = get_intersection(test_x[t], test_x[t + 1], test_y[t], test_y[t + 1], dotsx[x1], dotsx[x2], dotsy[x1], dotsy[x2])
                if xs != "Nan":
                    present = point_on_line(xs, ys, test_x[t], test_x[t + 1], test_y[t], test_y[t + 1])
                    if present:
						#plt.plot(xs, ys, 'mo')
						
                        d1x = abs(dotsx[x1] - xs)
                        d1y = abs(dotsy[x1] - ys)
						
                        d2x = abs(dotsx[x2] - xs)
                        d2y = abs(dotsy[x2] - ys)
						
                        d1 = np.sqrt(d1x ** 2 + d1y ** 2)
                        d2 = np.sqrt(d2x ** 2 + d2y ** 2)
						
                        sgnxs1 = dotsx[x1] > xs
                        sgnys1 = dotsy[x1] > ys
						
                        sgnxs2 = dotsx[x2] > xs 
                        sgnys2 = dotsy[x2] > ys
						
                        if sgnxs1 != sgnx12 and sgnys1 != sgny12:
                            num_of_intersection += 1
                            intersections[x1][x2][t] = (xs, ys) 
                            distances[x1][x2][t] = d1 
                            distancesx[x1][x2][t] = d1x
                            distancesy[x1][x2][t] = d1y
							
                        if sgnxs2 == sgnx12 and sgnys2 == sgny12:
                            num_of_intersection += 1
                            intersections[x2][x1][t] = (xs, ys) 
                            distances[x2][x1][t] = d2 
                            distancesx[x2][x1][t] = d2x
                            distancesy[x2][x1][t] = d2y 
    '''
	for t in range(window_size - 1): 			
		for x1 in range(size):
			for x2 in range(size):
				print(intersections[x1][x2][t], distances[x1][x2][t])
			print()
	'''
    all_distances = 0	
    all_distancesx = 0	
    all_distancesy = 0		
    for t in range(window_size - 1): 	
        for x1 in range(size): 
            for x2 in range(size):
                all_distances += distances[x1][x2][t]
                all_distancesx += distancesx[x1][x2][t]
                all_distancesy += distancesy[x1][x2][t]
    all_distances /= (size ** 2) * (window_size - 1) 
    all_distancesx /= (size ** 2) * (window_size - 1) 
    all_distancesy /= (size ** 2) * (window_size - 1) 
    if scale:
        all_distances /= scaling_factor	
        all_distancesx /= scaling_factor	
        all_distancesy /= scaling_factor	
	#print(all_distances, all_distancesx, all_distancesy)
	#plt.show()
    return all_distances, all_distancesx, all_distancesy, intersections, distances, distancesx, distancesy, num_of_intersection / size / size / (window_size - 1)

def process_csv_generic(window_size, all_possible_trajs, save_name):
    header = "window_size,vehicle,ride,start,"
    found = False
    for vehicle1 in all_possible_trajs[window_size].keys():  
        for r1 in all_possible_trajs[window_size][vehicle1]:
            for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
                for feat_name in all_possible_trajs[window_size][vehicle1][r1][x1]:
                    header += feat_name + ","
                    found = True
                if found:
                    break
            if found:
                break
        if found:
            break
    header = header[:-1]
    header += "\n" 
    new_csv_content = header
    for vehicle1 in all_possible_trajs[window_size].keys():  
        for r1 in all_possible_trajs[window_size][vehicle1]:
            for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
                new_csv_content += str(window_size) + "," + str(vehicle1) + "," + str(r1) + "," + str(x1) + ","  
                for feat_name in all_possible_trajs[window_size][vehicle1][r1][x1]: 
                    new_csv_content += str(all_possible_trajs[window_size][vehicle1][r1][x1][feat_name]) + ","
                new_csv_content = new_csv_content[:-1]
                new_csv_content += "\n"  
    csv_file = open(save_name, "w")
    csv_file.write(new_csv_content)
    csv_file.close()
 
def process_csv_ray(window_size, vehicle1, r1, some_dict, save_name, flag_overwrite = False): 
    if not os.path.isfile(save_name) or flag_overwrite:
        new_csv_content = "window_size,vehicle,ride,start,"
        for key in some_dict[0].keys():
            new_csv_content += key + "," + key + " x," + key + " y,"  
        new_csv_content += "\n"
        for x1 in some_dict: 
            new_csv_content += str(window_size) + "," + str(vehicle1) + "," + str(r1) + "," + str(x1) + ","  
            for feat_name in some_dict[x1]:  
                for val in some_dict[x1][feat_name]:  
                    new_csv_content += str(val) + "," 
            new_csv_content += "\n"
        csv_file = open(save_name, "w")
        csv_file.write(new_csv_content)
        csv_file.close()
    else:
        csv_file = open(save_name, "r")
        old_lines = csv_file.readlines()
        csv_file.close()
        new_csv_content = old_lines[0]
        skip_feats = set()
        for key in some_dict[0].keys():
            if key + "," + key + " x," + key + " y," not in new_csv_content:
                new_csv_content = new_csv_content.replace("\n", key + "," + key + " x," + key + " y,\n")  
            else:
                skip_feats.add(key)
        for x1 in some_dict: 
            line_start = str(window_size) + "," + str(vehicle1) + "," + str(r1) + "," + str(x1) + ","  
            index_line = -1  
            for index_some in range(len(old_lines)):
                if line_start in old_lines[index_some]:
                    index_line = index_some
                    break
            new_csv_line = "" 
            for feat_name in some_dict[x1]:  
                if feat_name in skip_feats:
                    continue
                for val in some_dict[x1][feat_name]:  
                    new_csv_line += str(val) + "," 
            new_csv_line += "\n"
            if index_line != -1:
                old_lines[index_line] = old_lines[index_line].replace("\n", new_csv_line)
        for line_old_index in range(1, len(old_lines)):
            new_csv_content += old_lines[line_old_index]
        csv_file = open(save_name, "w")
        csv_file.write(new_csv_content)
        csv_file.close() 
     
def process_csv(trajectory_flags, trajectory_monotonous, window_size, all_possible_trajs, sample_names, metric_names, deg, some_dict, save_name): 
    new_csv_content = "window_size,vehicle,ride,start,max_x,max_y,surf_trapz_x,surf_trapz_y,surf_simpson_x,surf_simpson_y,"
    for d in range(deg + 1):
        new_csv_content += "x_poly_" + str(d + 1) + ","
    for d in range(deg + 1):
        new_csv_content += "y_poly_" + str(d + 1) + ","
    for d in range(deg + 1):
        new_csv_content += "xy_poly_" + str(d + 1) + "," 
    new_csv_content += "duration,len,offset,mean_speed_len,mean_speed_offset,len_vs_offset,total_surf,"
    for sample_name in sample_names:
        for metric_name in metric_names: 
            new_csv_content += sample_name + "_same_" + metric_name + "," 
            new_csv_content += sample_name + "_diff_" + metric_name + ","
    new_csv_content += "monotonous," 
    for flag in flag_names:
        new_csv_content += flag + ","
    new_csv_content += "\n"
    header = new_csv_content
    for vehicle1 in all_possible_trajs[window_size].keys():  
        new_csv_content = header
        for r1 in all_possible_trajs[window_size][vehicle1]:
            for x1 in all_possible_trajs[window_size][vehicle1][r1]: 
                new_csv_content += str(window_size) + "," + str(vehicle1) + "," + str(r1) + "," + str(x1) + ","  
                for feat_name in some_dict[window_size][vehicle1][r1][x1]: 
                    if "poly" in feat_name: 
                        for val in list(some_dict[window_size][vehicle1][r1][x1][feat_name]): 
                            new_csv_content += str(val) + ","
                        if len(list(some_dict[window_size][vehicle1][r1][x1][feat_name])) == 0:
                            for d in range(deg + 1): 
                                new_csv_content += ","
                    else:
                        new_csv_content += str(some_dict[window_size][vehicle1][r1][x1][feat_name]) + ","
                new_csv_content += trajectory_monotonous[window_size][vehicle1][r1][x1] + ","  
                for flag in flag_names:
                     new_csv_content += str(trajectory_flags[flag][window_size][vehicle1][r1][x1]) + ","  
                new_csv_content += "\n"    
        csv_file = open(save_name.replace(".csv", "_" + vehicle1 + ".csv"), "w")
        csv_file.write(new_csv_content)
        csv_file.close()
  
def make_rays(size): 
    dotsx_original = []
    dotsy_original = [] 
    for x in range(size):
        px, py = make_ray(0.5, 360 / size * x, 0.5, 0.5) 
        dotsx_original.append(px)
        dotsy_original.append(py)
    return dotsx_original, dotsy_original

def load_traj(vehicle, ride): 
    name = vehicle + "/cleaned_csv/events_" + str(ride) + ".csv" 
    return load_traj_name(name)

def load_traj_window(vehicle, ride, x, window): 
    longitudes, latitudes, times = load_traj(vehicle, ride) 
    return longitudes[x:x + window], latitudes[x:x + window], times[x:x + window]

def load_traj_window_name(name, x, window): 
    longitudes, latitudes, times = load_traj_name(name) 
    return longitudes[x:x + window], latitudes[x:x + window], times[x:x + window]

def load_traj_name(name): 
    file_with_ride = pd.read_csv(name)
    longitudes = list(file_with_ride["fields_longitude"])
    latitudes = list(file_with_ride["fields_latitude"]) 
    times = list(file_with_ride["time"])  
    return longitudes, latitudes, times
 
def plot_long_lat_dict(title, long_dict, lat_dict, longer_file_name, long_keys, lat_keys, scalex, scaley):
    longitudes, latitudes, times = load_traj_name(longer_file_name)
    longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
    longitudes, latitudes = scale_long_lat(longitudes, latitudes, scalex, scaley, True) 
    plt.title(title + " " + longer_file_name.replace("/cleaned_csv/events_", " ").replace(".csv", ""))
    for latit in lat_keys:
        for longit in long_keys: 
            plt.plot(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], label = longit + " " + latit)
    plt.plot(longitudes, latitudes, label = "original", color = 'blue')
    plt.plot(longitudes[0], latitudes[0], 'ro', label = "start")
    plt.legend(loc = "upper center", bbox_to_anchor = (-0.02, -0.02), ncol = 10)
    plt.show()

def plot_long_lat_pairs(title, long_dict, lat_dict, longer_file_name, long_lat_pairs, scalex, scaley):
    longitudes, latitudes, times = load_traj_name(longer_file_name)
    longitudes, latitudes = preprocess_long_lat(longitudes, latitudes)
    longitudes, latitudes = scale_long_lat(longitudes, latitudes, scalex, scaley, True) 
    plt.title(title + " " + longer_file_name.replace("/cleaned_csv/events_", " ").replace(".csv", ""))
    for latit_longit in long_lat_pairs:
        longit = latit_longit.split("-")[0]
        latit = latit_longit.split("-")[1]
        plt.plot(long_dict[longer_file_name][longit], lat_dict[longer_file_name][latit], label = longit + " " + latit)
    plt.plot(longitudes, latitudes, label = "original", color = 'blue')
    plt.plot(longitudes[0], latitudes[0], 'ro', label = "start")
    plt.legend(loc = "upper center", bbox_to_anchor = (-0.02, -0.02), ncol = 10)
    plt.show()
  
def fix_prob_max(num_occurences, num_occurences_in_next_step, num_occurences_in_next_next_step, minval, maxval, stepv):
    possible_values = (maxval - minval) // stepv + 1  

    probability = dict()
    for distance in num_occurences:
        probability[distance] = num_occurences[distance] / (sum(list(num_occurences.values())) + possible_values - len(num_occurences.keys()))
    probability["undefined"] = (possible_values - len(num_occurences.keys())) / (sum(list(num_occurences.values())) + possible_values - len(num_occurences.keys()))
    
    probability_in_next_step = dict()
    for prev_distance in num_occurences_in_next_step:
        probability_in_next_step[prev_distance] = dict()
        for distance in num_occurences_in_next_step[prev_distance]:
            probability_in_next_step[prev_distance][distance] = num_occurences_in_next_step[prev_distance][distance] / (sum(list(num_occurences_in_next_step[prev_distance].values())) + possible_values - len(num_occurences_in_next_step[prev_distance].keys()))
        probability_in_next_step[prev_distance]["undefined"] = (possible_values - len(num_occurences_in_next_step[prev_distance].keys())) / (sum(list(num_occurences_in_next_step[prev_distance].values())) + possible_values - len(num_occurences_in_next_step[prev_distance].keys()))

    probability_in_next_step["undefined"] = dict()
    for distance in num_occurences:
        probability_in_next_step["undefined"][distance] = 1 / possible_values
    probability_in_next_step["undefined"]["undefined"] = 1 - len(num_occurences.keys()) / possible_values

    probability_in_next_next_step = dict()
    for prev_prev_distance in num_occurences_in_next_next_step:
        probability_in_next_next_step[prev_prev_distance] = dict()
        for prev_distance in num_occurences_in_next_next_step[prev_prev_distance]:
            probability_in_next_next_step[prev_prev_distance][prev_distance] = dict()
            for distance in num_occurences_in_next_next_step[prev_prev_distance][prev_distance]:
                probability_in_next_next_step[prev_prev_distance][prev_distance][distance] = num_occurences_in_next_next_step[prev_prev_distance][prev_distance][distance] / (sum(list(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].values())) + possible_values - len(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].keys()))
            probability_in_next_next_step[prev_prev_distance][prev_distance]["undefined"] = (possible_values - len(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].keys())) / (sum(list(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].values())) + possible_values - len(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].keys()))
 
    probability_in_next_next_step["undefined"] = dict()

    probability_in_next_next_step["undefined"]["undefined"] = dict() 
    for distance in num_occurences: 
        probability_in_next_next_step["undefined"]["undefined"][distance] = 1 / possible_values
    probability_in_next_next_step["undefined"]["undefined"]["undefined"] = 1 - len(num_occurences.keys()) / possible_values

    for prev_distance in num_occurences:
        probability_in_next_next_step["undefined"][prev_distance] = dict() 
        probability_in_next_next_step[prev_distance]["undefined"] = dict() 
        for distance in num_occurences:
            probability_in_next_next_step["undefined"][prev_distance][distance] = 1 / possible_values
            probability_in_next_next_step[prev_distance]["undefined"][distance] = 1 / possible_values
        probability_in_next_next_step["undefined"][prev_distance]["undefined"] = 1 - len(num_occurences.keys()) / possible_values
        probability_in_next_next_step[prev_distance]["undefined"]["undefined"] = 1 - len(num_occurences.keys()) / possible_values

    return probability, probability_in_next_step, probability_in_next_next_step
 
def fix_prob(num_occurences, num_occurences_in_next_step, num_occurences_in_next_next_step, fix = True):
    min_prob = 10 ** -20  
    if not fix:
        min_prob = 0
    probability = dict()
    for distance in num_occurences:
        probability[distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
    if fix:
        probability["undefined"] = min_prob
    
    probability_in_next_step = dict()
    for prev_distance in num_occurences_in_next_step:
        probability_in_next_step[prev_distance] = dict()
        for distance in num_occurences_in_next_step[prev_distance]:
            probability_in_next_step[prev_distance][distance] = num_occurences_in_next_step[prev_distance][distance] / sum(list(num_occurences_in_next_step[prev_distance].values())) - min_prob
        if fix:    
            probability_in_next_step[prev_distance]["undefined"] = min_prob

    if fix:
        probability_in_next_step["undefined"] = dict()
        for distance in num_occurences:
            probability_in_next_step["undefined"][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
        probability_in_next_step["undefined"]["undefined"] = min_prob

    probability_in_next_next_step = dict()
    for prev_prev_distance in num_occurences_in_next_next_step:
        probability_in_next_next_step[prev_prev_distance] = dict()
        for prev_distance in num_occurences_in_next_next_step[prev_prev_distance]:
            probability_in_next_next_step[prev_prev_distance][prev_distance] = dict()
            for distance in num_occurences_in_next_next_step[prev_prev_distance][prev_distance]:
                probability_in_next_next_step[prev_prev_distance][prev_distance][distance] = num_occurences_in_next_next_step[prev_prev_distance][prev_distance][distance] / sum(list(num_occurences_in_next_next_step[prev_prev_distance][prev_distance].values())) - min_prob
            if fix:    
                probability_in_next_next_step[prev_prev_distance][prev_distance]["undefined"] = min_prob
 
    if fix:
        probability_in_next_next_step["undefined"] = dict()

        probability_in_next_next_step["undefined"]["undefined"] = dict() 
        for distance in num_occurences: 
            probability_in_next_next_step["undefined"]["undefined"][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
        probability_in_next_next_step["undefined"]["undefined"]["undefined"] = min_prob

        for prev_distance in num_occurences:
            probability_in_next_next_step["undefined"][prev_distance] = dict()  
            for distance in num_occurences:
                probability_in_next_next_step["undefined"][prev_distance][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
            probability_in_next_next_step["undefined"][prev_distance]["undefined"] = min_prob 

        for prev_distance in probability_in_next_next_step: 
            probability_in_next_next_step[prev_distance]["undefined"] = dict() 
            for distance in num_occurences:
                probability_in_next_next_step[prev_distance]["undefined"][distance] = num_occurences[distance] / sum(list(num_occurences.values())) - min_prob
            probability_in_next_next_step[prev_distance]["undefined"]["undefined"] = min_prob

    return probability, probability_in_next_step, probability_in_next_next_step
 
def predict_prob(probability, probability_in_next_step, probability_in_next_next_step, minval, maxval, stepv):
    roundingval = int(-np.log10(stepv))
    possible_values = (maxval - minval) // stepv + 1 
    x = []
    n = 10000
    prev_distance = 0
    prev_prev_distance = 0
    for i in range(n):
        if i == 0:
            distance = np.random.choice(list(probability.keys()),p=list(probability.values()))  
        if i == 1:
            if prev_distance in probability_in_next_step:
                distance = np.random.choice(list(probability_in_next_step[prev_distance].keys()),p=list(probability_in_next_step[prev_distance].values())) 
            else:
                distance = np.random.choice(list(probability_in_next_step["undefined"].keys()),p=list(probability_in_next_step["undefined"].values())) 
        if i > 1:
            if prev_prev_distance in probability_in_next_next_step and prev_distance in probability_in_next_next_step[prev_prev_distance]:
                distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance][prev_distance].keys()),p=list(probability_in_next_next_step[prev_prev_distance][prev_distance].values())) 
            else:
                if prev_prev_distance in probability_in_next_next_step:
                    distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance]["undefined"].keys()),p=list(probability_in_next_next_step[prev_prev_distance]["undefined"].values())) 
                else:
                    if prev_distance in probability_in_next_next_step["undefined"]:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"][prev_distance].keys()),p=list(probability_in_next_next_step["undefined"][prev_distance].values()))
                    else:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"]["undefined"].keys()),p=list(probability_in_next_next_step["undefined"]["undefined"].values()))
        if distance == "undefined": 
            distance = np.round(minval + np.random.randint(possible_values) * stepv, roundingval) 
        prev_prev_distance = prev_distance
        prev_distance = distance
        x.append(distance)
    return x

def predict_prob_with_array(probability, probability_in_next_step, probability_in_next_next_step, array_vals, minval, maxval, stepv, isangle = False):
    roundingval = int(-np.log10(stepv))
    possible_values = (maxval - minval) // stepv + 1 
    x = []
    n = len(array_vals)
    prev_distance = 0
    prev_prev_distance = 0
    no_empty = 0
    match_score = 0 
    no_empty = 0
    delta_series = [] 
    for i in range(n):
        if i > 1:
            prev_prev_distance = array_vals[i - 2]
        if i > 0:
            prev_distance = array_vals[i - 1]
        if i == 0:
            distance = np.random.choice(list(probability.keys()),p=list(probability.values()))  
        if i == 1:
            if prev_distance in probability_in_next_step:
                distance = np.random.choice(list(probability_in_next_step[prev_distance].keys()),p=list(probability_in_next_step[prev_distance].values())) 
            else:
                distance = np.random.choice(list(probability_in_next_step["undefined"].keys()),p=list(probability_in_next_step["undefined"].values())) 
        if i > 1:
            if prev_prev_distance in probability_in_next_next_step and prev_distance in probability_in_next_next_step[prev_prev_distance]:
                distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance][prev_distance].keys()),p=list(probability_in_next_next_step[prev_prev_distance][prev_distance].values())) 
            else:
                if prev_prev_distance in probability_in_next_next_step:
                    distance = np.random.choice(list(probability_in_next_next_step[prev_prev_distance]["undefined"].keys()),p=list(probability_in_next_next_step[prev_prev_distance]["undefined"].values())) 
                else:
                    if prev_distance in probability_in_next_next_step["undefined"]:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"][prev_distance].keys()),p=list(probability_in_next_next_step["undefined"][prev_distance].values()))
                    else:
                        distance = np.random.choice(list(probability_in_next_next_step["undefined"]["undefined"].keys()),p=list(probability_in_next_next_step["undefined"]["undefined"].values()))
        if distance == "undefined": 
            distance = np.round(minval + np.random.randint(possible_values) * stepv, roundingval)
        else:
            no_empty += 1
        x.append(float(distance)) 
        if float(array_vals[i]) == float(x[i]):
            match_score += 1   
        delta_x = abs(float(array_vals[i]) - float(x[i]))
        if isangle:
            if delta_x > 180:
                delta_x = 360 - delta_x
        delta_series.append(delta_x) 
    #print(match_score / n, match_score / no_empty, min(delta_series), np.quantile(delta_series, 0.25), np.quantile(delta_series, 0.5), np.quantile(delta_series, 0.75), max(delta_series), np.average(delta_series), np.std(delta_series), np.var(delta_series))
    #plt.hist(delta_series)
    #plt.show() 
    return x, n, match_score, no_empty, delta_series
 
def random_sample_of_cluster(subdirname, files_in_cluster, nrow, ncol, filename):
    print(filename)
    for cluster in files_in_cluster:
        if len(files_in_cluster[cluster]) > 0: 
            print(cluster, len(files_in_cluster[cluster]))
            if nrow * ncol > len(files_in_cluster[cluster]):
                ncol = int(np.sqrt(len(files_in_cluster[cluster])))
                nrow = ncol
            indexes = set([x for x in range(len(files_in_cluster[cluster]))])
            while len(indexes) > nrow * ncol:
                index_remove = np.random.randint(0, len(files_in_cluster[cluster]))
                if index_remove in indexes:
                    indexes.remove(index_remove)
            long1 = []
            lat1 = []
            titles = []
            for index in indexes:
                name_file = files_in_cluster[cluster][index]["short_name"]
                name_file_long = name_file.replace("/", "/cleaned_csv/") 
                long, lat, time = load_traj_window_name(name_file_long, files_in_cluster[cluster][index]["start"], files_in_cluster[cluster][index]["window"])
                long, lat = preprocess_long_lat(long, lat) 
                long, lat = scale_long_lat(long, lat, 0.005, 0.005, True) 
                long1.append(long)
                lat1.append(lat)
                titles.append(filename + "_cluster_" + str(cluster) + name_file)
            if not os.path.isdir("all_clus/" + subdirname + "/samples"):
                os.makedirs("all_clus/" + subdirname + "/samples")
            composite_image_random_cluster(long1, lat1, titles, nrow, ncol, "all_clus/" + subdirname + "/samples/" + filename + "_cluster_" + str(cluster))

def random_sample_of_isomap(subdirname, files_in_cluster, nrow, ncol, filename):
    print(filename)
    for cluster in files_in_cluster:
        if len(files_in_cluster[cluster]) > 0:
            print(cluster, len(files_in_cluster[cluster]))
            if nrow * ncol > len(files_in_cluster[cluster]):
                ncol = int(np.sqrt(len(files_in_cluster[cluster])))
                nrow = ncol
            indexes = set([x for x in range(len(files_in_cluster[cluster]))])
            while len(indexes) > nrow * ncol:
                index_remove = np.random.randint(0, len(files_in_cluster[cluster]))
                if index_remove in indexes:
                    indexes.remove(index_remove)
            long1 = []
            lat1 = []
            titles = []
            for index in indexes:
                name_file = files_in_cluster[cluster][index]["short_name"]
                name_file_long = name_file.replace("/", "/cleaned_csv/") 
                long, lat, time = load_traj_window_name(name_file_long, files_in_cluster[cluster][index]["start"], files_in_cluster[cluster][index]["window"])
                long, lat = preprocess_long_lat(long, lat) 
                long, lat = scale_long_lat(long, lat, 0.005, 0.005, True) 
                long1.append(long)
                lat1.append(lat)
                titles.append(filename + "_cluster_" + str(cluster) + name_file)
            if not os.path.isdir("all_isomap/" + subdirname + "/samples"):
                os.makedirs("all_isomap/" + subdirname + "/samples")
            composite_image_random_cluster(long1, lat1, titles, nrow, ncol, "all_isomap/" + subdirname + "/samples/" + filename + "_cluster_" + str(cluster))

def composite_image_random_cluster(long1, lat1, titles, nrow, ncol, filename): 
    plt.rcParams.update({'font.size': 6})
    plt.figure(figsize=(ncol, nrow))
    for row in range(nrow):
        for col in range(ncol):
            ix = row * ncol + col
            plt.subplot(nrow, ncol, ix + 1) 
            plt.axis('off')
            #plt.title(titles[ix])   
            plt.plot(long1[ix], lat1[ix], color = "k")      
    plt.savefig(filename, bbox_inches = "tight")  
    plt.close() 
        
def composite_image(filename, show, long1, lat1, nrow, ncol, long_other = [], lat_other = [], legends = [], mark_start = False, subtitles = []):  
    random_colors_legend = random_colors(len(legends) + 2) 
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(15 * ncol, 15 * nrow)) 
    numseen = 0
    for ix in range(len(show)):  
        if show[ix]:
            numseen += 1
            plt.subplot(nrow, ncol, numseen)  
            new_subtitle = ""
            new_subtitle_separated = subtitles[ix].split("\n")
            segment = ""
            for el in new_subtitle_separated:
                if segment != "": 
                    segment += " "
                segment += el
                if len(segment) > 30:
                    if new_subtitle != "": 
                        new_subtitle += "\n"
                    new_subtitle += segment
                    segment = ""
            if segment != "":
                if new_subtitle != "": 
                    new_subtitle += "\n"
                new_subtitle += segment
                segment = ""
            plt.title(new_subtitle) 
            if len(long_other) >= ix:
                for i in range(len(long_other[ix])): 
                    plt.plot(long_other[ix][i], lat_other[ix][i], label = legends[i], color = random_colors_legend[i + 2], linewidth = 10)  
            plt.plot(long1[ix], lat1[ix], label = "Original", color = random_colors_legend[0], linewidth = 10)     
            if mark_start:
                plt.plot(long1[ix][0], lat1[ix][0], marker = "o", label = "Start", color = random_colors_legend[0], mec = random_colors_legend[0], mfc = random_colors_legend[1], ms = 20, mew = 10, linewidth = 10) 
            if len(legends) > 0:
                plt.legend()  

    plt.savefig(filename, bbox_inches = "tight")
    plt.close() 
    composite_image_reverse(filename, show, long1, lat1, ncol, nrow, long_other, lat_other, legends, mark_start, subtitles)
    
def composite_image_reverse(filename, show, long1, lat1, nrow, ncol, long_other = [], lat_other = [], legends = [], mark_start = False, subtitles = []):  
    random_colors_legend = random_colors(len(legends) + 2) 
    plt.rcParams.update({'font.size': 22})
    plt.figure(figsize=(15 * ncol, 15 * nrow)) 
    numseen = 0 
    for ix in range(len(show)):  
        if show[ix]:
            num_of_row = numseen % nrow
            num_of_col = numseen // nrow
            plt.subplot(nrow, ncol, num_of_row * ncol + num_of_col + 1)  
            numseen += 1
            new_subtitle = ""
            new_subtitle_separated = subtitles[ix].split("\n")
            segment = ""
            for el in new_subtitle_separated:
                if segment != "": 
                    segment += " "
                segment += el
                if len(segment) > 30:
                    if new_subtitle != "": 
                        new_subtitle += "\n"
                    new_subtitle += segment
                    segment = ""
            if segment != "":
                if new_subtitle != "": 
                    new_subtitle += "\n"
                new_subtitle += segment
                segment = ""
            plt.title(new_subtitle) 
            if len(long_other) >= ix:
                for i in range(len(long_other[ix])): 
                    plt.plot(long_other[ix][i], lat_other[ix][i], label = legends[i], color = random_colors_legend[i + 2], linewidth = 10)  
            plt.plot(long1[ix], lat1[ix], label = "Original", color = random_colors_legend[0], linewidth = 10)     
            if mark_start:
                plt.plot(long1[ix][0], lat1[ix][0], marker = "o", label = "Start", color = random_colors_legend[0], mec = random_colors_legend[0], mfc = random_colors_legend[1], ms = 20, mew = 10, linewidth = 10) 
            if len(legends) > 0:
                plt.legend()  

    plt.savefig(filename.replace(".png", "_reverse.png"), bbox_inches = "tight")
    plt.close() 
    
def format_e(n):
    if abs(n) >= 10 ** -2 or n == 0:
        return ("$" + str(np.round(n, 2)) + "$").replace(".0$", "$").replace(".00$", "$")
    a = '%.2E' % n
    return "$" + str(str(a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]) + "}$").replace("E-0", "*10^{-").replace("E+0", "*10^{").replace("$1*", "$")

def format_e2(n):
    if abs(n) >= 10 ** -2 or n == 0:
        return str(np.round(n, 2))  
    else:
        return str(np.round(n, 5))
    
def new_metric(metric_name):
    new_metric_name = {"simpson x": "Simpson x", "trapz x": "Trapez x", 
              "simpson y": "Simpson y", "trapz y": "Trapez y",
              "euclidean": "Euklidska"}
    if metric_name in new_metric_name:
        return new_metric_name[metric_name]
    else:
        return metric_name

def translate_method(longlat):
    translate_name = {
        "long no abs-lat no abs": "Pomak u x i y smjeru",
        "long-lat": "Apsolutna vrijednost pomaka u x i y smjeru",
        "long dist dir abs alt-lat dist dir abs alt": "Euklidska udaljenost i kut s osi x",
        "long dist dir-lat dist dir": "Euklidska udaljenost i odmak od sjevera",
        "long speed ones alt dir abs alt-lat speed ones alt dir abs alt": "Brzina na segmentu i kut s osi x, jedna sekunda",
        "x speed-y speed": "Apsolutna vrijednost brzine u x i y smjeru",
        "x speed no abs-y speed no abs": "Brzina u x i y smjeru",
        "long speed dir-lat speed dir": "Brzina u točki i odmak od sjevera",
        "long speed ones alt dir-lat speed ones alt dir": "Brzina na segmentu i odmak od sjevera, jedna sekunda",
        "long speed alt dir abs alt-lat speed alt dir abs alt": "Brzina na segmentu i kut s osi x",
        "long speed alt dir-lat speed alt dir": "Brzina na segmentu i odmak od sjevera",
        "long speed dir abs alt-lat speed dir abs alt": "Brzina u točki i kut s osi x",
        "x speed ones no abs-y speed ones no abs": "Brzina u x i y smjeru, jedna sekunda",
        "long speed ones dir-lat speed ones dir": "Brzina u točki i odmak od sjevera, jedna sekunda",
        "long speed ones dir abs alt-lat speed ones dir abs alt": "Brzina u točki i kut s osi x, jedna sekunda",
        "x speed ones-y speed ones": "Apsolutna vrijednost brzine u x i y smjeru, jedna sekunda",
        "long dist dir alt-lat dist dir alt": "Euklidska udaljenost i odmak od osi x",
        "long speed ones alt dir alt-lat speed ones alt dir alt": "Brzina na segmentu i odmak od osi x, jedna sekunda",
        "long speed ones dir alt-lat speed ones dir alt": "Brzina u točki i odmak od osi x, jedna sekunda",
        "long speed alt dir alt-lat speed alt dir alt": "Brzina na segmentu i odmak od osi x",
        "long speed dir alt-lat speed dir alt": "Brzina u točki i odmak od osi x",
    }
    if longlat in translate_name:
        return translate_name[longlat]
    else:
        return longlat
    
translate_var = {"Distance": "Euklidska udaljenost",
             "Direction": "Odmak od sjevera", 
             "Direction abs alt": "Kut s osi x",
             "Direction alt": "Odmak od osi x",
             "Latitude": "Apsolutna vrijednost pomaka u y smjeru", 
             "Latitude no abs": "Pomak u y smjeru", 
             "Latitude sgn": "Predznak pomaka u y smjeru", 
             "Longitude": "Apsolutna vrijednost pomaka u x smjeru",
             "Longitude no abs": "Pomak u x smjeru",  
             "Longitude sgn": "Predznak pomaka u x smjeru",
             "Time": "Vrijeme",
             "Speed": "Brzina u točki",
             "Speed alt": "Brzina na segmentu",
             "X speed": "Apsolutna vrijednost brzine u x smjeru",
             "X speed no abs": "Brzina u x smjeru",
             "Y speed": "Apsolutna vrijednost brzine u y smjeru",
             "Y speed no abs": "Brzina u y smjeru",
             }

flag_names = ["key", "flip", "zone", "engine", "in_zone", "ignition", "sleep_mode", "staff_mode", "buzzer_active", "in_primary_zone", "in_restricted_zone", "onboard_geofencing", "speed_limit_active"]

def skip_var(variable_name, subdirname):
 
    if "no_rays" in subdirname: 
        for size in os.listdir("rays"):
            if str(size) + "all_" in variable_name:
                return True

    if "poly" in variable_name and "poly" not in subdirname:
        return True 

    if "flags" not in subdirname:
        for var_skip in flag_names:
            if var_skip in variable_name: 
                return True

    if "same" in variable_name and "no_same" in subdirname:
        return True

    if "xy_poly" in variable_name and "no_xy" in subdirname:
        return True
    if "simpson" in variable_name and "simpson x" not in variable_name and "simpson y" not in variable_name and "simpson_x" not in variable_name and "simpson_y" not in variable_name and "no_xy" in subdirname:
        return True
    if "trapz" in variable_name and "trapz x" not in variable_name and "trapz y" not in variable_name and "trapz_x" not in variable_name and "trapz_y" not in variable_name and "no_xy" in subdirname:
        return True

    if "acceler" in variable_name and "acceler" not in subdirname:
        return True

    if "heading" in variable_name and "heading" not in subdirname:
        return True

    if "only_rays" in subdirname and "size" not in subdirname:
        any_present = False
        for size in os.listdir("rays"): 
            if str(size) + "all_" not in variable_name:
                any_present = True
                break
        if not any_present:
            return True

    if "size_" in subdirname:
        for size in os.listdir("rays"):
            if "size_" + str(size) + "_" in subdirname and  str(size) + "all_" not in variable_name:
                return True
    
    return False 

def divide_train_test(properties, train_set, test_set, subdirname): 
    sd_window_train = []
    sd_subdir_train = []
    sd_ride_train = []
    sd_start_train = []
    sd_x_train = [] 

    sd_window_test = []
    sd_subdir_test = []
    sd_ride_test = []
    sd_start_test = []
    sd_x_test = [] 
    
    for window_size in properties:
        for subdir_name in properties[window_size]:
            for some_file in properties[window_size][subdir_name]:
                for start in properties[window_size][subdir_name][some_file]: 
                    
                    if len(properties[window_size][subdir_name][some_file][start]) == 0:
                        continue

                    if some_file in train_set:
                        sd_window_train.append(window_size)
                        sd_subdir_train.append(subdir_name)
                        sd_ride_train.append(some_file)
                        sd_start_train.append(start)
                        sd_x_train.append([])
                        for variable_name in properties[window_size][subdir_name][some_file][start]:
                            if skip_var(variable_name, subdirname):
                                continue 
 
                            if "monoto" not in variable_name:
                                if math.isnan(properties[window_size][subdir_name][some_file][start][variable_name]):
                                    sd_x_train[-1].append(0)
                                else:
                                    sd_x_train[-1].append(properties[window_size][subdir_name][some_file][start][variable_name]) 
                            else:
                                if properties[window_size][subdir_name][some_file][start] == "I":
                                    sd_x_train[-1].append(3)
                                if properties[window_size][subdir_name][some_file][start] == "D":
                                    sd_x_train[-1].append(2)
                                if properties[window_size][subdir_name][some_file][start] == "NM":
                                    sd_x_train[-1].append(1)
                                if properties[window_size][subdir_name][some_file][start] == "NF":
                                    sd_x_train[-1].append(0)
                    if some_file in test_set:
                        sd_window_test.append(window_size)
                        sd_subdir_test.append(subdir_name)
                        sd_ride_test.append(some_file)
                        sd_start_test.append(start)
                        sd_x_test.append([])
                        for variable_name in properties[window_size][subdir_name][some_file][start]:
                            if skip_var(variable_name, subdirname):
                                continue

                            if "monoto" not in variable_name:
                                if math.isnan(properties[window_size][subdir_name][some_file][start][variable_name]):
                                    sd_x_test[-1].append(0)
                                else:
                                    sd_x_test[-1].append(properties[window_size][subdir_name][some_file][start][variable_name]) 
                            else:
                                if properties[window_size][subdir_name][some_file][start] == "I":
                                    sd_x_test[-1].append(3)
                                if properties[window_size][subdir_name][some_file][start] == "D":
                                    sd_x_test[-1].append(2)
                                if properties[window_size][subdir_name][some_file][start] == "NM":
                                    sd_x_test[-1].append(1)
                                if properties[window_size][subdir_name][some_file][start] == "NF":
                                    sd_x_test[-1].append(0)
    
    for rn in range(len(sd_x_train)):
        for cn in range(len(sd_x_train[rn])):
            sd_x_train[rn][cn] = float(sd_x_train[rn][cn])
     
    for rn in range(len(sd_x_test)):
        for cn in range(len(sd_x_test[rn])):
            sd_x_test[rn][cn] = float(sd_x_test[rn][cn])

    sd_x_train = np.array(sd_x_train)
    sd_x_test = np.array(sd_x_test)

    print(np.shape(sd_x_train), np.shape(sd_x_test))
    return sd_window_train, sd_subdir_train, sd_ride_train, sd_start_train, sd_x_train, sd_window_test, sd_subdir_test, sd_ride_test, sd_start_test, sd_x_test 
    
def read_clusters(subdirname):
    for filename in os.listdir("all_clus/" + subdirname + "/filenames"):
        random_sample_of_cluster(subdirname, load_object("all_clus/" + subdirname + "/filenames/" + filename), 1000, 1000, filename)

def read_isomaps(subdirname):
    for filename in os.listdir("all_isomap/" + subdirname + "/filenames"):
        random_sample_of_isomap(subdirname, load_object("all_isomap/" + subdirname + "/filenames/" + filename), 1000, 1000, filename)
        
def speed_cluster(files_in_cluster, sdn, filename, output): 

    properties = load_object("dict_for_clustering") 
  
    avg_values = dict()
    var_clus = output
    std_clus = output
    varscaled_clus = output
    stdscaled_clus = output
    avg_clus = output
    min_clus = output
    max_clus = output
    range_clus = output
    for cluster in files_in_cluster:
        if len(files_in_cluster[cluster]) > 0:
            avg_values[cluster] = dict()
            var_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            std_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            varscaled_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            stdscaled_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            avg_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            min_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            max_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            range_clus += str(sdn) + "," + str(filename) + "," + str(cluster) + "," + str(len(files_in_cluster[cluster])) + ","
            for index in range(len(files_in_cluster[cluster])): 
                name_file = files_in_cluster[cluster][index]["short_name"]
                vehicle_ride = name_file.split("/") 
                subdir_name = vehicle_ride[0] 
                window_size = files_in_cluster[cluster][index]["window"]
                x = files_in_cluster[cluster][index]["start"] 
    
                for variable_name in properties[window_size][subdir_name][vehicle_ride[1]][x]: 
                    if skip_var(variable_name, sdn):
                        continue
                    if variable_name not in avg_values[cluster]:
                        avg_values[cluster][variable_name] = []
                    avg_values[cluster][variable_name] = add_key(variable_name, properties[window_size][subdir_name][vehicle_ride[1]][x][variable_name], avg_values[cluster][variable_name]) 
                     
            for key_name in avg_values[cluster]:
                for inde in range(len(avg_values[cluster][key_name])):
                    avg_values[cluster][key_name][inde] = float(avg_values[cluster][key_name][inde])

            for key_name in avg_values[cluster]:                   
                var_clus += str(np.var(avg_values[cluster][key_name])) + ","
            var_clus = var_clus[:-1]
            var_clus += "\n"

            for key_name in avg_values[cluster]:                     
                std_clus += str(np.std(avg_values[cluster][key_name])) + ","
            std_clus = std_clus[:-1]
            std_clus += "\n"

            for key_name in avg_values[cluster]:                     
                varscaled_clus += str(np.var(avg_values[cluster][key_name]) / ((max(avg_values[cluster][key_name]) - min(avg_values[cluster][key_name])) ** 2)) + ","
            varscaled_clus = varscaled_clus[:-1]
            varscaled_clus += "\n"

            for key_name in avg_values[cluster]:                     
                stdscaled_clus += str(np.std(avg_values[cluster][key_name]) / (max(avg_values[cluster][key_name]) - min(avg_values[cluster][key_name]))) + ","
            stdscaled_clus = stdscaled_clus[:-1]
            stdscaled_clus += "\n"

            for key_name in avg_values[cluster]:                     
                avg_clus += str(sum(avg_values[cluster][key_name]) / len(avg_values[cluster][key_name])) + ","
            avg_clus = avg_clus[:-1]
            avg_clus += "\n"

            for key_name in avg_values[cluster]:                     
                min_clus += str(min(avg_values[cluster][key_name])) + ","
            min_clus = min_clus[:-1]
            min_clus += "\n"

            for key_name in avg_values[cluster]:                     
                max_clus += str(max(avg_values[cluster][key_name])) + ","
            max_clus = max_clus[:-1]
            max_clus += "\n"

            for key_name in avg_values[cluster]:                     
                range_clus += str(max(avg_values[cluster][key_name]) - min(avg_values[cluster][key_name])) + ","
            range_clus = range_clus[:-1]
            range_clus += "\n"
              
    return var_clus, std_clus, varscaled_clus, stdscaled_clus, avg_clus, min_clus, max_clus, range_clus

def add_key(key, val, wheradd):
    if "monoto" not in key:
        if math.isnan(val):
            wheradd.append(0)
        else:
            wheradd.append(val) 
    else:
        if val == "I":
            wheradd.append(3)
        if val == "D":
            wheradd.append(2)
        if val == "NM":
            wheradd.append(1)
        if val == "NF":
            wheradd.append(0)
    return wheradd