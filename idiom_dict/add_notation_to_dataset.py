import json

with open("idioms_dict.json", "r") as f:
    idiom_dict = json.load(f)

with open("test_data.json", "r") as f:
    train_data = f.read().split("\n")[:-1]

f = open("test_data_with_sep.json", "w")

for line in train_data:
    tmp = json.loads(line)
    string = ""
    for candidate in tmp['candidates'][0]:
        string += candidate + ":" + idiom_dict[candidate] +" "
    tmp['content'] = tmp['content'] +" 注：" +string 
    line2 = json.dumps(tmp, ensure_ascii=False) + '\n'
    f.write(line2)

f.close()

    
