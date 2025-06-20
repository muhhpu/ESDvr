import torch
from PIL import Image
from modelscope import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import csv
import sys

df = pd.read_csv('./MicroLens-50k-Dataset/MicroLens-50k_titles.csv')
df2 = pd.read_csv('./MicroLens-50k-Dataset/MicroLens-100k_title_en.csv' , header=None)



with open('./i_id_mapping.csv', 'r', encoding='utf-8') as file:
    reader = csv.reader(file, delimiter='\t')

    # 读取每行并分成两列
    data = [row for row in reader]





data = data[1:]

need_to_do = []
titles = []

for row in data:
    # print(type(row[0]))
    need_to_do.append(str(row[0]))
    filtered_row = df[(df['item'] == int(row[0]))]

    if not filtered_row.empty:
        titles.append(filtered_row['title'].values[0])

    else:
        filtered_row = df2[df2[0] == int(row[0])]
        titles.append(filtered_row[1].values[0])


# need_to_do = [ str(i) for i in df['item'].tolist()]
# titles = [ str(i) for i in df['title'].tolist()]
gap = len(need_to_do)//2

# button = int(sys.argv[1])
button = 1

if button == 1:
    title_token = 'text_glm_feat_1.npy'
    this_do = need_to_do[:1]
    this_titles = titles[:1]
    device = "cuda:7"

elif button == 2:
    title_token = 'text_glm_feat_2.npy'
    this_do = need_to_do[gap:]
    this_titles = titles[gap:]
    device = "cuda:1"




tokenizer = AutoTokenizer.from_pretrained("ZhipuAI/glm-4v-9b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "ZhipuAI/glm-4v-9b",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to(device).eval()


title_token_save = np.zeros((len(this_do),4096), dtype=np.float32)

for i in range(len(this_do)):
    print(this_do[i],this_titles[i])
    query = "I will now provide you with a sentence that has been translated from Chinese to English, but the translation is not very good. Many of the words are regional dialects or Chinese slang or internet jargon. The sentence contains two parts: a title and tags (after #), and some tags are also transliterations or slang. I will also include an image that may help you understand the sentence. I would like to ask you to provide the most accurate English translation based on the original Chinese meaning. Please give me the most accurate translation directly, without explaining the content of the image.:{}".format(this_titles[i])
    image = Image.open("/home/team/caoziyi/MicroLens-50k-Dataset/MicroLens-50k_covers/{}.jpg".format(this_do[i])).convert('RGB')
    inputs = tokenizer.apply_chat_template([{"role": "user", "image": image, "content": query}],
                                           add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                           return_dict=True,output_hidden_states=True)  # chat mode
    inputs = inputs.to(device)
    # gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1,}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

        
        resp = model.transformer(**inputs, output_hidden_states=True)
        y = resp.last_hidden_state
        y_mean = torch.mean(y.squeeze(), dim=0, keepdim=True)
        title_token_save[i] = y_mean.detach().to(torch.float).cpu().numpy()
        # print(y_mean.shape)
        # print(outputs.shape)
        # print(outputs[0].shape)
        print(tokenizer.decode(outputs[0]))
# np.save(title_token, title_token_save)
