import json
import math
import tqdm
import numpy  as np

from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-cluecorpussmall")
model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-cluecorpussmall").to('cuda')
def cal_ppl(text):
    input_ids = tokenizer(text, return_tensors='pt')
    input_tensor = input_ids['input_ids'].to('cuda')
    outputs = model(input_tensor, labels=input_tensor)
    ppl = math.exp(outputs.loss.item())
    return ppl

with open('CHID_dataset/dev_data.json', 'r') as f:
    eval_data = f.read().split("\n")[:-1]


truth, count = 0, 0
for item in tqdm.tqdm(eval_data):
    # item = eval_data[4]
    example = json.loads(item)
    if len(example['groundTruth']) == 1:
        res = []
        for each_idiom in example['candidates'][0]:
            new_content = example['content'].replace('#idiom#', each_idiom)
            res.append(cal_ppl(new_content))
        idx = np.argmin(np.array(res))
        if example['groundTruth'][0] == example['candidates'][0][idx]:
            truth += 1
        count += 1
    assert len(example['groundTruth'])
# if len(example['groundTruth']) > 1:
print(truth/count)