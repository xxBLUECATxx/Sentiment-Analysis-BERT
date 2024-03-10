from transformers import pipeline
import matplotlib.pyplot as plt
from transformers import BertTokenizer
import pandas as pd
import torch
"""class Net(nn.Module):

    model = Bert()
    torch.optim.SGD(lr=0.001) #According to your own Configuration.
    checkpoint = torch.load(pytorch_model)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['opt']) 
"""

pipe = pipeline("text-classification", model="Wakaka/bert-finetuned-imdb")



# Load the dataset
df = pd.read_excel('paratext corpus pt en ch.xlsx', sheet_name='Synopsis')
ZH = df['ZH'].values.tolist()
EN = df['EN'].values.tolist()
PT = df['PT'].values.tolist()
max_length = 512
processed_texts_ZH = [text[:max_length-1] if len(str(text)) > max_length else text for text in ZH]
processed_texts_EN = [text[:max_length-1] if len(str(text)) > max_length else text for text in EN]
processed_texts_PT = [text[:max_length-1] if len(str(text)) > max_length else text for text in PT]

#for labelled data
"""
df['ENLabel'] = [pipe(str(text))[0]['label'] for text in processed_texts_EN]   #not a good practice cuz repeated called pipe, 2 times slower.
df['ENScore'] = [pipe(str(text))[0]['score'] for text in processed_texts_EN]
df.to_excel('BERTEN.xlsx', index=False)
print('EN done')
df['PTLabel'] = [pipe(str(text))[0]['label'] for text in processed_texts_PT]
df['PTScore'] = [pipe(str(text))[0]['score'] for text in processed_texts_PT]
df.to_excel('BERTPT.xlsx', index=False)
print('PT done')
df['ZHLabel'] = [pipe(str(text))[0]['label'] for text in processed_texts_ZH]
df['ZHScore'] = [pipe(str(text))[0]['score'] for text in processed_texts_ZH]
"""
#for unlabelled data

df['ENScore'] = [pipe(str(text))[0]['score'] if pipe(str(text))[0]['label'] == 'LABEL_1' else -(pipe(str(text))[0]['score'])  for text in processed_texts_EN]
df.to_excel('BERTEN.xlsx', index=False)
print('EN done')
df['PTScore'] = [pipe(str(text))[0]['score'] if pipe(str(text))[0]['label'] == 'LABEL_1' else -(pipe(str(text))[0]['score'])  for text in processed_texts_PT]
df.to_excel('BERTPT.xlsx', index=False)
print('PT done')
df['ZHScore'] = [pipe(str(text))[0]['score'] if pipe(str(text))[0]['label'] == 'LABEL_1' else -(pipe(str(text))[0]['score'])  for text in processed_texts_ZH]

col=['ZHBN','EN','PT','ZH','ENScore','PTScore','ZHScore']
df = df[col]

df.to_excel('BERT_Unlabelled.xlsx', index=False)
