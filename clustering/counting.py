import json
import argparse
import os
from tqdm import tqdm

par = argparse.ArgumentParser()
par.add_argument("-d", "--data_name", default='seq2seq_d64',
                 type=str, help="data name")
par.add_argument("-t", "--threshold",
                 type=float, help="Set threshold")
args = par.parse_args()

data_path = ("result/" + args.data_name  + "/" + args.data_name +
             "_t" + str(args.threshold) + ".json")

jsonData = json.loads(open(data_path).read())
print("Number of class : {}".format(len(jsonData)))
print("Number of etc : {}".format(len(jsonData['etc'])))

data_sort = sorted(jsonData.items(), key=lambda item: len(item[1]), reverse = True)

total = 0
cnt = 1
for d in tqdm(data_sort):
    if d[0] == 'etc':
        continue
    elif len(d[1]) > 500:
        continue
    elif len(d[1]) > 30:
        print("Top{} = {}".format(cnt, len(d[1])))
        total += len(d[1])
        cnt += 1

print("Total = {}".format(total))
