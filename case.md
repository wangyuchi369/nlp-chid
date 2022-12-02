1. 修改huggingface trainer的eval代码，一般在/anaconda/envs/**/lib/python/site-packages/transformers/trainer.py
   将evaluate（）最后一行改为return output
2. 将run.sh中--model_name_or_path hfl/chinese-roberta-wwm-ext改为要评估的checkpoint位置，默认为run.sh中的tmp文件夹

   ```
   --output_dir ./tmp
   ```
3. 去掉run.sh的--dotrain运行，结果文件为case.csv
