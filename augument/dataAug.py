import json
import random

with open('CHID_dataset/train_data_1w.json', 'r') as f:
    train_data = f.read().split("\n")[:-1]
with open('idiom_dict/idioms_dict.json') as f:
    idioms = json.load(f)
with open('idiom_dict/idiom_syno.json') as f:
    idioms_syn = json.load(f)


def rand_insert(example):
    example = json.loads(example)
    rand_idiom = random.choice(list(idioms))
    if len(rand_idiom)==4:
        idx = random.randint(0, len(example['candidates'][0]))
        example['candidates'][0].insert(idx, rand_idiom)
    return json.dumps(example, ensure_ascii=False)


def rand_delete(example):
    example = json.loads(example)
    # rand_idiom = random.choice(list(idioms))
    to_delete_idiom = random.choice(example['candidates'][0])
    if to_delete_idiom not in example['groundTruth']:
        example['candidates'][0].remove(to_delete_idiom)
    return json.dumps(example, ensure_ascii=False)


def swap(example):
    example = json.loads(example)
    # rand_idiom = random.choice(list(idioms))
    rand_idiom = random.choice(list(idioms))
    if len(rand_idiom)==4:
        to_replace_idiom = random.randint(0, len(example['candidates'][0]) - 1)
        if example['candidates'][0][to_replace_idiom] not in example['groundTruth']:
            example['candidates'][0][to_replace_idiom] = rand_idiom
    return json.dumps(example, ensure_ascii=False)


def synony(example):
    example = json.loads(example)
    # rand_idiom = random.choice(list(idioms))
    # rand_idiom = random.choice(list(idioms))
    to_replace_idiom_idx = random.randint(0, len(example['candidates'][0]) - 1)
    to_replace_idiom = example['candidates'][0][to_replace_idiom_idx]
    if to_replace_idiom not in example['groundTruth']:
        if to_replace_idiom in idioms_syn:
            syn = idioms_syn[to_replace_idiom]['近义词']
            if syn:
                syn_idiom = random.choice(syn)
                if len(syn_idiom)==4:
                    example['candidates'][0][to_replace_idiom_idx] = syn_idiom
    return json.dumps(example, ensure_ascii=False)

ratio = 1
augment_style = swap
augument = []
augument2 = []
for example in train_data:
    if random.random() < ratio:
        augument.append(augment_style(example))
    if random.random() < ratio:
        augument2.append(synony(example))
train_data.extend(augument)
train_data.extend(augument2)

with open(f'CHID_dataset/train_combine_3w.json', 'w') as f:
    for example in train_data:
        f.write(example + '\n')


