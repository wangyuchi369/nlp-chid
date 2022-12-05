# -*- coding:utf-8 -*-
import json
import numpy as np
with open('idiom.json', 'rb') as f:
    idioms = json.load(f)
print(idioms[0])
print(len(idioms))
idioms_list = [(item['word'], item['explanation']) for item in idioms]
idioms_dict = dict(idioms_list)
print(idioms_dict['顶天立地'])
info_json = json.dumps(idioms_dict,sort_keys=False, indent=4, separators=(',', ': '), ensure_ascii=False)

with open('idioms_dict.json', 'w', encoding='utf-8') as f:
    f.write(info_json)
