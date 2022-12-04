import json
from pylangtools.langconv import Converter

with open("idioms_dict.json", "r") as f:
    idiom_dict = json.load(f)

with open("train_data_10w.json", "r") as f:
    train_data = f.read().split("\n")[:-1]

f = open("train_data_with_truthsep.json", "w")

for line in train_data:
    tmp = json.loads(line)
    string = ""
    for candidate in tmp['groundTruth']:
        try:
            string += idiom_dict[Converter('zh-hans').convert(candidate)] +" "
        except:
            continue
    tmp['content'] = tmp['content'] +" 注：" +string 
    line2 = json.dumps(tmp, ensure_ascii=False) + '\n'
    f.write(line2)

f.close()

    
