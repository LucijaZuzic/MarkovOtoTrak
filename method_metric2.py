from utilities import *  


distance_predicted = load_object("markov_result/distance_predicted")
 
all_metrics = dict()
all_longs = dict()
all_lats = dict()
all_longlats = dict()
for vehicle in distance_predicted: 
    for event in distance_predicted[vehicle]:   
        for metric_name in distance_predicted[vehicle][event]: 
            all_metrics[metric_name] = True
            for longlat in distance_predicted[vehicle][event][metric_name]:  
                long = longlat.split("-")[0] 
                lat = long.replace("long", "lat").replace("x", "y")
                all_longs[long] = True
                all_lats[lat] = True
                all_longlats[long+"-"+lat] = True

lines_ret = dict()
sorty_by = dict()
for metric_name in all_metrics:
    lines_ret[metric_name] = dict()
    sorty_by[metric_name] = dict()
    for longlat in all_longlats:
        series = []
        for vehicle in distance_predicted: 
            for event in distance_predicted[vehicle]:  
                series.append(distance_predicted[vehicle][event][metric_name][longlat]) 
        sorty_by[metric_name][longlat] = np.quantile(series, 0.50)
        lines_ret[metric_name][longlat] = [ 
            [translate_method(longlat), format_e(min(series)), format_e(np.average(series)), format_e(np.std(series)), format_e(np.var(series))],
            [translate_method(longlat), format_e(np.quantile(series, 0.50))] ]
    
for metric_name in lines_ret:
    print(metric_name)
    for table_number in range(len(lines_ret[metric_name]["long no abs-lat no abs"])):
        if table_number != 1:
            continue
        strpr = "" 
        for longlat in dict(sorted(sorty_by[metric_name].items(), key=lambda item: item[1])): 
            for val_index in range(len(lines_ret[metric_name][longlat][table_number])):
                if val_index != len(lines_ret[metric_name][longlat][table_number]) - 1:
                    strpr += lines_ret[metric_name][longlat][table_number][val_index] + " & "
                else:
                    strpr += lines_ret[metric_name][longlat][table_number][val_index] + " \\\\ \\hline\n"
        print(strpr)
 
strpr = "name"
for metric_name in lines_ret: 
    strpr += " & " + metric_name
strpr += " \\\\ \\hline"
print(strpr)
for longlat in dict(sorted(sorty_by["euclidean"].items(), key=lambda item: item[1])): 
    strpr = lines_ret["euclidean"][longlat][1][0]
    for metric_name in lines_ret: 
        strpr += " & " + lines_ret[metric_name][longlat][1][1]
    strpr += " \\\\ \\hline"
    print(strpr)
        
for longlat in all_longlats:
    ix = 1
    plt.rcParams.update({'font.size': 12})
    plt.figure(figsize=(40, 5))
    for metric_name in all_metrics: 
        series = []
        for vehicle in distance_predicted: 
            for event in distance_predicted[vehicle]:  
                series.append(distance_predicted[vehicle][event][metric_name][longlat])
        plt.subplot(1, len(all_metrics), ix)
        plt.title(translate_method(longlat) + "\n" + new_metric(metric_name))
        plt.hist(series)
        ix += 1 
    plt.close()