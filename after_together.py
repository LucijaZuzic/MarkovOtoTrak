from utilities import load_object, translate_method, composite_image, load_traj_name, preprocess_long_lat, new_metric, scale_long_lat, format_e2 
import os
import numpy as np
  
all_subdirs = os.listdir()   

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

print("Worst occurence") 
first_row = ""
sum_cols = dict()
for metric_name in worst_worst:
    first_row += metric_name + " & "   
    sum_cols[metric_name] = 0            
print(first_row + "\\\\ \\hline") 
sum_rows = dict()
sum_x = dict()
sum_y = dict()
sum_else = dict()
for longlat in worst_worst["euclidean"]:
    sum_x[longlat] = 0
    sum_y[longlat] = 0
    sum_else[longlat] = 0 
for longlat in worst_worst["euclidean"]:
    sum_row = 0
    row_str = translate_method(longlat) + " & "
    for metric_name in worst_worst:
        row_str += str(translate_method(worst_worst[metric_name][longlat])) + " & "
        sum_row += worst_worst[metric_name][longlat]
        sum_cols[metric_name] += worst_worst[metric_name][longlat]
        if " x" in metric_name:
            sum_x[longlat] += worst_worst[metric_name][longlat]
        if " y" in metric_name:
            sum_y[longlat] += worst_worst[metric_name][longlat]
        if " y" not in metric_name and " x" not in metric_name:
            sum_else[longlat] += worst_worst[metric_name][longlat]
    sum_rows[longlat] = sum_row
    if sum_row > 0:
        print(row_str + "$" + str(sum_row) + "$ \\\\ \\hline")  
last_row = "" 
sum_sum_cols = 0
for metric_name in sum_cols:
    last_row += "$" + str(sum_cols[metric_name]) + "$ & "  
    sum_sum_cols += sum_cols[metric_name]
print(last_row + str(sum_sum_cols) + " \\\\ \\hline")  

print("Worst occurence percent")   
print(first_row + "\\\\ \\hline") 
for longlat in dict(sorted(sum_rows.items(), key=lambda item: item[1])): 
    if sum_rows[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in worst_worst:
            row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        print(row_str + "$" + str(np.round(sum_rows[longlat] / sum_sum_cols * 100, 2)) + "\\%$ \\\\ \\hline")  
print(last_row + str(sum_sum_cols) + " \\\\ \\hline")   

print("Worst occurence percent x y")    
for longlat in dict(sorted(sum_x.items(), key=lambda item: item[1])): 
    if sum_x[longlat] > 0 or sum_y[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in ["simpson x", "trapz x"]:
            row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(sum_x[longlat] / (sum_cols["simpson x"] + sum_cols["trapz x"]) * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(count_worst_by_xavg[longlat] / sum(list(count_worst_by_xavg.values())) * 100, 2)) + "\\%$ & "
        for metric_name in ["simpson y", "trapz y"]:
            row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(sum_y[longlat] / (sum_cols["simpson y"] + sum_cols["trapz y"]) * 100, 2)) + "\\%$ "  
        row_str += "$" + str(np.round(count_worst_by_yavg[longlat] / sum(list(count_worst_by_yavg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round((sum_x[longlat] + sum_y[longlat]) / (sum_cols["simpson x"] + sum_cols["trapz x"] + sum_cols["simpson y"] + sum_cols["trapz y"]) * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round((count_worst_by_xavg[longlat] + count_worst_by_yavg[longlat]) / (sum(list(count_worst_by_xavg.values())) + sum(list(count_worst_by_yavg.values()))) * 100, 2)) + "\\%$ \\\\ \\hline"
        print(row_str)    

print("Worst occurence percent other")    
for longlat in dict(sorted(sum_else.items(), key=lambda item: item[1])): 
    if sum_else[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if " x" in metric_name:
                row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(sum_else[longlat] / sum_sum_cols * 100, 2)) + "\\%$ \\\\ \\hline" 
        print(row_str)

print("Worst occurence percent x")   
for longlat in dict(sorted(sum_x.items(), key=lambda item: item[1])): 
    if sum_x[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if " x" in metric_name:
                row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(count_worst_by_xavg[longlat] / sum(list(count_worst_by_xavg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round(sum_x[longlat] / (sum_cols["simpson x"] + sum_cols["trapz x"]) * 100, 2)) + "\\%$ \\\\ \\hline"  
        print(row_str)    

print("Worst occurence percent y")    
for longlat in dict(sorted(sum_y.items(), key=lambda item: item[1])): 
    if sum_y[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if " y" in metric_name:
                row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & " 
        row_str += "$" + str(np.round(count_worst_by_yavg[longlat] / sum(list(count_worst_by_yavg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round(sum_y[longlat] / (sum_cols["simpson y"] + sum_cols["trapz y"]) * 100, 2)) + "\\%$ \\\\ \\hline"   
        print(row_str)  

print("Worst occurence percent total")    
for longlat in dict(sorted(sum_rows.items(), key=lambda item: item[1])): 
    if sum_rows[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if not " x" in metric_name and not " y" in metric_name:
                row_str += "$" + str(np.round(worst_worst[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(count_worst_by_avg[longlat] / sum(list(count_worst_by_avg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round(sum_rows[longlat] / sum_sum_cols * 100, 2)) + "\\%$ \\\\ \\hline"
        print(row_str)
   
worst_scores_for_longlat_all = []   
worst_longs_for_longlat_all = [] 
worst_lats_for_longlat_all = []    
worst_other_longs_for_longlat_all = [] 
worst_other_lats_for_longlat_all = []  
worst_methods_for_longlat_all = []    
worst_metrics_for_longlat_all = []    
worst_rides_for_longlat_all = []    

for metric_name in range(len(worst_worst_ride)):
    worst_scores_for_longlat_all.append(-1000000) 
    worst_longs_for_longlat_all.append([])
    worst_lats_for_longlat_all.append([])
    worst_other_longs_for_longlat_all.append([])
    worst_other_lats_for_longlat_all.append([])  
    worst_methods_for_longlat_all.append("")    
    worst_metrics_for_longlat_all.append("")      
    worst_rides_for_longlat_all.append("")      

for longlat in worst_worst_ride["euclidean"]:  
    
    row_str = translate_method(longlat) + " & "   
    worst_longs_for_longlat = [] 
    worst_lats_for_longlat = []    
    worst_other_longs_for_longlat = [] 
    worst_other_lats_for_longlat = []  
    worst_methods_for_longlat = []    
    worst_metrics_for_longlat = []    
    worst_rides_for_longlat = []    
    
    ix = 0
    for metric_name in worst_worst_ride: 
        row_str += str(translate_method(worst_worst_ride[metric_name][longlat])) + " & "
        longer_name = worst_worst_ride[metric_name][longlat].replace("/", "/cleaned_csv/")
        long, lat, time = load_traj_name(longer_name)
        long, lat = preprocess_long_lat(long, lat)
        long, lat = scale_long_lat(long, lat, 0.1, 0.1)
        process_ride = worst_worst_ride[metric_name][longlat].replace("/events_", " Ride ").replace(".csv", "").replace("Vehicle_", "Vehicle ")
        
        worst_methods_for_longlat.append(translate_method(longlat))
        worst_metrics_for_longlat.append(new_metric(metric_name) + " = " + format_e2(worst_worst_score[metric_name][longlat]))
        worst_rides_for_longlat.append(process_ride) 
        worst_longs_for_longlat.append(long)
        worst_lats_for_longlat.append(lat) 
        worst_other_longs_for_longlat.append([long_dict[longer_name][longlat.split("-")[0]]])
        worst_other_lats_for_longlat.append([lat_dict[longer_name][longlat.split("-")[1]]])  
      
        if worst_worst_score[metric_name][longlat] > worst_scores_for_longlat_all[ix]:
            worst_methods_for_longlat_all[ix] = translate_method(longlat)
            worst_metrics_for_longlat_all[ix] = new_metric(metric_name) + " = " + format_e2(worst_worst_score[metric_name][longlat])
            worst_rides_for_longlat_all[ix] = process_ride 
            worst_longs_for_longlat_all[ix] = long 
            worst_lats_for_longlat_all[ix] = lat  
            worst_other_longs_for_longlat_all[ix] = [long_dict[longer_name][longlat.split("-")[0]]] 
            worst_other_lats_for_longlat_all[ix] = [lat_dict[longer_name][longlat.split("-")[1]]] 
            worst_scores_for_longlat_all[ix] = worst_worst_score[metric_name][longlat] 
     
        ix += 1
       
    if not os.path.isdir("markov_rides"):
        os.makedirs("markov_rides")
    filename = "markov_rides/" + longlat + "_worst.png"
 
    worst_subs_for_longlat = []
    worst_show_for_longlat = []
    for ix_val1 in range(len(worst_methods_for_longlat)):
        worst_subs_for_longlat.append(worst_methods_for_longlat[ix_val1] + "\n")
        for ix_val2 in range(len(worst_methods_for_longlat)):
            if worst_methods_for_longlat[ix_val1] == worst_methods_for_longlat[ix_val2] and worst_rides_for_longlat[ix_val1] == worst_rides_for_longlat[ix_val2]:
                worst_subs_for_longlat[-1] = worst_subs_for_longlat[-1] + worst_metrics_for_longlat[ix_val2] + "\n" 
        worst_subs_for_longlat[-1] = worst_subs_for_longlat[-1] + worst_rides_for_longlat[ix_val1] + "\n" 
        worst_show_for_longlat.append(worst_subs_for_longlat.index(worst_subs_for_longlat[-1]) == ix_val1)

    composite_image(filename, worst_show_for_longlat, worst_longs_for_longlat, worst_lats_for_longlat, 8, 1, worst_other_longs_for_longlat, worst_other_lats_for_longlat, ["Estimated"], True, worst_subs_for_longlat)
 
worst_subs_for_longlat_all = []
worst_show_for_longlat_all = []
for ix_val1 in range(len(worst_methods_for_longlat_all)):
    worst_subs_for_longlat_all.append(worst_methods_for_longlat_all[ix_val1] + "\n")
    num_match = 0
    for ix_val2 in range(len(worst_methods_for_longlat_all)):
        if worst_methods_for_longlat_all[ix_val1] == worst_methods_for_longlat_all[ix_val2] and worst_rides_for_longlat_all[ix_val1] == worst_rides_for_longlat_all[ix_val2]:
            num_match += 1
            worst_subs_for_longlat_all[-1] = worst_subs_for_longlat_all[-1] + worst_metrics_for_longlat_all[ix_val2] + "\n" 
    worst_subs_for_longlat_all[-1] = worst_subs_for_longlat_all[-1] + worst_rides_for_longlat_all[ix_val1] + "\n" 
    worst_show_for_longlat_all.append(worst_subs_for_longlat_all.index(worst_subs_for_longlat_all[-1]) == ix_val1)

composite_image("markov_rides/A_worst.png", worst_show_for_longlat_all, worst_longs_for_longlat_all, worst_lats_for_longlat_all, 8, 1, worst_other_longs_for_longlat_all, worst_other_lats_for_longlat_all, ["Estimated"], True, worst_subs_for_longlat_all)

print("Best occurence") 
first_row = ""
sum_cols = dict()
for metric_name in best_best:
    first_row += metric_name + " & "   
    sum_cols[metric_name] = 0            
print(first_row + "\\\\ \\hline") 
sum_rows = dict()
sum_x = dict()
sum_y = dict()
sum_else = dict()
for longlat in best_best["euclidean"]:
    sum_x[longlat] = 0
    sum_y[longlat] = 0
    sum_else[longlat] = 0 
for longlat in best_best["euclidean"]:
    sum_row = 0
    row_str = translate_method(longlat) + " & "
    for metric_name in best_best:
        row_str += str(translate_method(best_best[metric_name][longlat])) + " & "
        sum_row += best_best[metric_name][longlat]
        sum_cols[metric_name] += best_best[metric_name][longlat]
        if " x" in metric_name:
            sum_x[longlat] += best_best[metric_name][longlat]
        if " y" in metric_name:
            sum_y[longlat] += best_best[metric_name][longlat]
        if " y" not in metric_name and " x" not in metric_name:
            sum_else[longlat] += best_best[metric_name][longlat]
    sum_rows[longlat] = sum_row
    if sum_row > 0:
        print(row_str + "$" + str(sum_row) + "$ \\\\ \\hline")  
last_row = "" 
sum_sum_cols = 0
for metric_name in sum_cols:
    last_row += "$" + str(sum_cols[metric_name]) + "$ & "  
    sum_sum_cols += sum_cols[metric_name]
print(last_row + str(sum_sum_cols) + " \\\\ \\hline")  

print("Best occurence percent")   
print(first_row + "\\\\ \\hline") 
for longlat in dict(sorted(sum_rows.items(), key=lambda item: item[1], reverse = True)): 
    if sum_rows[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in best_best:
            row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        print(row_str + "$" + str(np.round(sum_rows[longlat] / sum_sum_cols * 100, 2)) + "\\%$ \\\\ \\hline")  
print(last_row + str(sum_sum_cols) + " \\\\ \\hline")   

print("Best occurence percent x y")    
for longlat in dict(sorted(sum_x.items(), key=lambda item: item[1], reverse = True)): 
    if sum_x[longlat] > 0 or sum_y[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in ["simpson x", "trapz x"]:
            row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(sum_x[longlat] / (sum_cols["simpson x"] + sum_cols["trapz x"]) * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(count_best_by_xavg[longlat] / sum(list(count_best_by_xavg.values())) * 100, 2)) + "\\%$ & "
        for metric_name in ["simpson y", "trapz y"]:
            row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(sum_y[longlat] / (sum_cols["simpson y"] + sum_cols["trapz y"]) * 100, 2)) + "\\%$ "  
        row_str += "$" + str(np.round(count_best_by_yavg[longlat] / sum(list(count_best_by_yavg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round((sum_x[longlat] + sum_y[longlat]) / (sum_cols["simpson x"] + sum_cols["trapz x"] + sum_cols["simpson y"] + sum_cols["trapz y"]) * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round((count_best_by_xavg[longlat] + count_best_by_yavg[longlat]) / (sum(list(count_best_by_xavg.values())) + sum(list(count_best_by_yavg.values()))) * 100, 2)) + "\\%$ \\\\ \\hline"
        print(row_str)    

print("Best occurence percent other")    
for longlat in dict(sorted(sum_else.items(), key=lambda item: item[1], reverse = True)): 
    if sum_else[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if " x" in metric_name:
                row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(sum_else[longlat] / sum_sum_cols * 100, 2)) + "\\%$ \\\\ \\hline" 
        print(row_str)

print("Best occurence percent x")   
for longlat in dict(sorted(sum_x.items(), key=lambda item: item[1], reverse = True)): 
    if sum_x[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if " x" in metric_name:
                row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(count_best_by_xavg[longlat] / sum(list(count_best_by_xavg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round(sum_x[longlat] / (sum_cols["simpson x"] + sum_cols["trapz x"]) * 100, 2)) + "\\%$ \\\\ \\hline"  
        print(row_str)    

print("Best occurence percent y")    
for longlat in dict(sorted(sum_y.items(), key=lambda item: item[1], reverse = True)): 
    if sum_y[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if " y" in metric_name:
                row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & " 
        row_str += "$" + str(np.round(count_best_by_yavg[longlat] / sum(list(count_best_by_yavg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round(sum_y[longlat] / (sum_cols["simpson y"] + sum_cols["trapz y"]) * 100, 2)) + "\\%$ \\\\ \\hline"   
        print(row_str)  

print("Best occurence percent total")    
for longlat in dict(sorted(sum_rows.items(), key=lambda item: item[1], reverse = True)): 
    if sum_rows[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in all_metrics:
            if not " x" in metric_name and not " y" in metric_name:
                row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        row_str += "$" + str(np.round(count_best_by_avg[longlat] / sum(list(count_best_by_avg.values())) * 100, 2)) + "\\%$ & "
        row_str += "$" + str(np.round(sum_rows[longlat] / sum_sum_cols * 100, 2)) + "\\%$ \\\\ \\hline"
        print(row_str)
    
best_scores_for_longlat_all = []   
best_longs_for_longlat_all = [] 
best_lats_for_longlat_all = []    
best_other_longs_for_longlat_all = [] 
best_other_lats_for_longlat_all = []  
best_methods_for_longlat_all = []    
best_metrics_for_longlat_all = []    
best_rides_for_longlat_all = []    

for metric_name in range(len(best_best_ride)):
    best_scores_for_longlat_all.append(1000000) 
    best_longs_for_longlat_all.append([])
    best_lats_for_longlat_all.append([])
    best_other_longs_for_longlat_all.append([])
    best_other_lats_for_longlat_all.append([])  
    best_methods_for_longlat_all.append("")    
    best_metrics_for_longlat_all.append("")      
    best_rides_for_longlat_all.append("")      

for longlat in best_best_ride["euclidean"]:  
    
    row_str = translate_method(longlat) + " & "   
    best_longs_for_longlat = [] 
    best_lats_for_longlat = []    
    best_other_longs_for_longlat = [] 
    best_other_lats_for_longlat = []  
    best_methods_for_longlat = []    
    best_metrics_for_longlat = []    
    best_rides_for_longlat = []    
    
    ix = 0
    for metric_name in best_best_ride: 
        row_str += str(translate_method(best_best_ride[metric_name][longlat])) + " & "
        longer_name = best_best_ride[metric_name][longlat].replace("/", "/cleaned_csv/")
        long, lat, time = load_traj_name(longer_name)
        long, lat = preprocess_long_lat(long, lat)
        long, lat = scale_long_lat(long, lat, 0.1, 0.1)
        process_ride = best_best_ride[metric_name][longlat].replace("/events_", " Ride ").replace(".csv", "").replace("Vehicle_", "Vehicle ")
        
        best_methods_for_longlat.append(translate_method(longlat))
        best_metrics_for_longlat.append(new_metric(metric_name) + " = " + format_e2(best_best_score[metric_name][longlat]))
        best_rides_for_longlat.append(process_ride) 
        best_longs_for_longlat.append(long)
        best_lats_for_longlat.append(lat) 
        best_other_longs_for_longlat.append([long_dict[longer_name][longlat.split("-")[0]]])
        best_other_lats_for_longlat.append([lat_dict[longer_name][longlat.split("-")[1]]])  
      
        if best_best_score[metric_name][longlat] < best_scores_for_longlat_all[ix]:
            best_methods_for_longlat_all[ix] = translate_method(longlat)
            best_metrics_for_longlat_all[ix] = new_metric(metric_name) + " = " + format_e2(best_best_score[metric_name][longlat])
            best_rides_for_longlat_all[ix] = process_ride 
            best_longs_for_longlat_all[ix] = long 
            best_lats_for_longlat_all[ix] = lat  
            best_other_longs_for_longlat_all[ix] = [long_dict[longer_name][longlat.split("-")[0]]] 
            best_other_lats_for_longlat_all[ix] = [lat_dict[longer_name][longlat.split("-")[1]]] 
            best_scores_for_longlat_all[ix] = best_best_score[metric_name][longlat] 
     
        ix += 1
       
    if not os.path.isdir("markov_rides"):
        os.makedirs("markov_rides")
    filename = "markov_rides/" + longlat + "_best.png"

    best_subs_for_longlat = []
    best_show_for_longlat = []
    for ix_val1 in range(len(best_methods_for_longlat)):
        best_subs_for_longlat.append(best_methods_for_longlat[ix_val1] + "\n")
        for ix_val2 in range(len(best_methods_for_longlat)):
            if best_methods_for_longlat[ix_val1] == best_methods_for_longlat[ix_val2] and best_rides_for_longlat[ix_val1] == best_rides_for_longlat[ix_val2]:
                best_subs_for_longlat[-1] = best_subs_for_longlat[-1] + best_metrics_for_longlat[ix_val2] + "\n" 
        best_subs_for_longlat[-1] = best_subs_for_longlat[-1] + best_rides_for_longlat[ix_val1] + "\n" 
        best_show_for_longlat.append(best_subs_for_longlat.index(best_subs_for_longlat[-1]) == ix_val1)

    composite_image(filename, best_show_for_longlat, best_longs_for_longlat, best_lats_for_longlat, 8, 1, best_other_longs_for_longlat, best_other_lats_for_longlat, ["Estimated"], True, best_subs_for_longlat)
 
best_subs_for_longlat_all = []
best_show_for_longlat_all = []
for ix_val1 in range(len(best_methods_for_longlat_all)):
    best_subs_for_longlat_all.append(best_methods_for_longlat_all[ix_val1] + "\n")
    num_match = 0
    for ix_val2 in range(len(best_methods_for_longlat_all)):
        if best_methods_for_longlat_all[ix_val1] == best_methods_for_longlat_all[ix_val2] and best_rides_for_longlat_all[ix_val1] == best_rides_for_longlat_all[ix_val2]:
            num_match += 1
            best_subs_for_longlat_all[-1] = best_subs_for_longlat_all[-1] + best_metrics_for_longlat_all[ix_val2] + "\n" 
    best_subs_for_longlat_all[-1] = best_subs_for_longlat_all[-1] + best_rides_for_longlat_all[ix_val1] + "\n" 
    best_show_for_longlat_all.append(best_subs_for_longlat_all.index(best_subs_for_longlat_all[-1]) == ix_val1)

composite_image("markov_rides/A_best.png", best_show_for_longlat_all, best_longs_for_longlat_all, best_lats_for_longlat_all, 8, 1, best_other_longs_for_longlat_all, best_other_lats_for_longlat_all, ["Estimated"], True, best_subs_for_longlat_all)

print("Best occurence percent")   
print(first_row + "\\\\ \\hline") 
for longlat in dict(sorted(sum_else.items(), key=lambda item: item[1], reverse = True)): 
    if sum_rows[longlat] > 0:
        row_str = translate_method(longlat) + " & "
        for metric_name in best_best:
            row_str += "$" + str(np.round(best_best[metric_name][longlat] / sum_cols[metric_name] * 100, 2)) + "\\%$ & "  
        print(row_str + "\\\\ \\hline")  
print(last_row + str(sum_sum_cols) + " \\\\ \\hline")   