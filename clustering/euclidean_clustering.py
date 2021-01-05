from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import json
import pandas as pd
import sys
import argparse
import os

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='seq2seq_d64',
                 type=str, help="data name")
par.add_argument("-t", "--threshold",
                 type=float, help="Set threshold")
args = par.parse_args()

names = json.loads(open("../data/" + args.data_name +
                   "/" + args.data_name + '_names.json').read())
data = np.load("../data/" + args.data_name +
               "/" + args.data_name + '_data.npy')
name_list = names['name']
threshold = args.threshold

result = dict()
result["etc"] = []
cluster_label = 1
max_size = len(data)

def printProgress (iteration, total, prefix = '', suffix = '', decimals = 1, barLength = 50):
    formatStr = "{0:." + str(decimals) + "f}"
    percent = formatStr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix))
    if(iteration == total):
        sys.stdout.write('\n')
    sys.stdout.flush()

inp = 0
for k in range(len(data)):
    printProgress(max_size-len(data), max_size, 'Progress', 'Complete')
    if len(data) == 0:
        break
    df = pd.DataFrame(data)
    euclidean_distance_result = dict()
    for i in range(len(data)):
        euclidean_distance_result[str(i)] = euclidean_distances(data[inp:inp+1], data[i:i+1]).tolist()[0][0]
    euclidean_distance_result_sort = sorted(euclidean_distance_result.items(), key=lambda x: x[1])
    cnt = 0
    del_d = []
    res = []
    for i in range((len(data))):
        if euclidean_distance_result_sort[cnt][1] >= threshold:
            break
        res.append(name_list[int(euclidean_distance_result_sort[cnt][0])])
        del_d.append(int(euclidean_distance_result_sort[cnt][0]))
        cnt += 1
    if cnt == 1:
        result["etc"].append(res[0])
        df = df.drop(del_d)
        for d in res:
            name_list.remove(d)
    else:
        result[str(cluster_label)] = res
        df = df.drop(del_d)
        for d in res:
            name_list.remove(d)
        cluster_label += 1
    data = df.values

print(result.keys())

result_path = "result/" + args.data_name
if not os.path.exists(result_path):
    os.mkdir(result_path)

result_json = json.dumps(result, indent=4)
f = open(result_path + "/" + args.data_name +
         "_t" + str(args.threshold) + ".json", "w")
f.write(result_json)
f.close()
