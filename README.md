## 关于处理成语为成语词典
在idioms.py中将成语词典处理成[成语:释义]的形式

## 数据集？

## 关于获得预测错的例子

1. 修改huggingface trainer的eval代码，一般在/anaconda/envs/**/lib/python/site-packages/transformers/trainer.py
   将evaluate（）最后一行改为return output
2. 将run.sh中--model_name_or_path hfl/chinese-roberta-wwm-ext改为要评估的checkpoint位置，默认为run.sh中的tmp文件夹

   ```
   --output_dir ./tmp
   ```
3. 去掉run.sh的--dotrain运行，结果文件为case.csv
