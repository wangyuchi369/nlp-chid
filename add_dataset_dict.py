import json
import random
from copy import deepcopy

with open("idioms_dict.json", "r") as f:
    idiom_exam = json.load(f)

f = open("add_dataset_dict.json", "w")
with open("train_data_10w.json", "r") as f2:
    train_data = f2.read().split("\n")[:-1]

for line in train_data:
    f.write(json.dumps(json.loads(line), ensure_ascii=False) + '\n')

res = deepcopy(idiom_exam)

for key, value in res.items():
    if (value == "无"):
        del idiom_exam[key]

res = deepcopy(idiom_exam)
for key, value in res.items():
    if(len(key)!=4):
        del idiom_exam[key]

for key in idiom_exam.keys():
    idiom_exam[key] = idiom_exam[key].replace('～', '#idiom#')

other_keys=idiom_exam.keys()

for key in idiom_exam.keys():
    it={}
    it["groundTruth"]=[key]
    bx=random.sample(other_keys,6)
    it["candidates"]=bx
    it["candidates"].append(key)
    random.shuffle(it["candidates"])
    it["content"]=idiom_exam[key]
    it["realCount"]=1
    line2 = json.dumps(it, ensure_ascii=False) + '\n'
    f.write(line2)

