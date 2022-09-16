# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/3/29 11:01
@Author     : Danke Wu
@File       : Bert_sentences.py
"""
import torch
import os
from transformers import BertTokenizer
from pytorch_pretrained_bert import BertModel

MODELNAME ='bert-base-chinese'
# url =r'D:\数据集\2020-5-18\AllRumorDetection-master-data\PCL_process'
url = r'D:\数据集\2020-5-18\weiboRumor-dataset-master\PCL_process'
# MODELNAME = 'bert-base-uncased'
# url = r'D:\数据集\2020-5-18\rumor_detection_acl2017\pheme'
tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
model = BertModel.from_pretrained(MODELNAME)  # 模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

if __name__ == '__main__':
    max_len = 0
    for _, _, filenames in os.walk(os.path.join(url, 'pro_content')):
        for file in filenames:
            f = open(os.path.join(url, 'pro_content', file), 'r', encoding='utf-8')
            f_w = open(os.path.join(url, 'pro_content_emb', str(file)), 'a')
            idx_bertsent ={}
            sent_bertsent = {}
            for line in f.readlines():
                line = line.strip('\n').split('\t')
                if line[0] in idx_bertsent.keys():
                    f_w.write(line[0] + '\t' + line[1] + '\t' + idx_bertsent[line[0]] + '\n')
                elif line[2] in sent_bertsent.keys():
                    idx_bertsent[line[0]] = sent_bertsent[line[2]]
                    f_w.write(line[0] + '\t' + line[1] + '\t' + sent_bertsent[line[2]] + '\n')
                else:
                    with torch.no_grad():
                        input_ids = tokenizer.encode(
                            line[2],
                            add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
                            # max_length=114,  # 设定最大文本长度
                            # padding = 'max_length',   # pad到最大的长度
                            return_tensors='pt'  # 返回的类型为pytorch tensor
                        )
                        encoded_layers, _ = model(input_ids.to(device))
                    sentence_vec = torch.mean(encoded_layers[10], 1).squeeze()
                    sentence_vec = sentence_vec.cpu().numpy().tolist()

                    input_ids = [str(x) for x in sentence_vec]
                    if len(input_ids) > max_len:
                        max_len = len(input_ids)
                    f_w.write(line[0] + '\t' + line[1] + '\t' + ' '.join(input_ids) + '\n')
                    idx_bertsent[line[0]] = ' '.join(input_ids)
                    sent_bertsent[line[2]] = ' '.join(input_ids)

    print( '%d', max_len)
    max_len = 0
#
# #l
# MODELNAME ='bert-base-chinese'
# url =r'D:\数据集\2020-5-18\AllRumorDetection-master-data\PCL_process'
#
# # MODELNAME = 'bert-base-uncased'
# # url = r'D:\数据集\2020-5-18\rumor_detection_acl2017\pheme'
# tokenizer = BertTokenizer.from_pretrained(MODELNAME)  # 分词词
# model = BertModel.from_pretrained(MODELNAME)  # 模型
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)
# model.eval()
#
# if __name__ == '__main__':
#     max_len = 0
#     for _, _, filenames in os.walk(os.path.join(url, 'pro_content')):
#         for file in filenames:
#             f = open(os.path.join(url, 'pro_content', file), 'r', encoding='utf-8')
#             f_w = open(os.path.join(url, 'pro_content_emb', str(file)), 'a')
#             idx_bertsent = {}
#             for line in f.readlines():
#                 line = line.strip('\n').split('\t')
#                 if line[1] in idx_bertsent.keys():
#                     f_w.write(line[0] + '\t' + line[1] + '\t' + idx_bertsent[line[1]] + '\n')
#                 else:
#                     with torch.no_grad():
#                         input_ids = tokenizer.encode(
#                             line[2],
#                             add_special_tokens=True,  # 添加special tokens， 也就是CLS和SEP
#                             # max_length=114,  # 设定最大文本长度
#                             # padding = 'max_length',   # pad到最大的长度
#                             return_tensors='pt'  # 返回的类型为pytorch tensor
#                         )
#                         encoded_layers, _ = model(input_ids.to(device))
#                     sentence_vec = torch.mean(encoded_layers[10], 1).squeeze()
#                     sentence_vec = sentence_vec.cpu().numpy().tolist()
#
#                     input_ids = [str(x) for x in sentence_vec]
#                     if len(input_ids) > max_len:
#                         max_len = len(input_ids)
#                     f_w.write(line[0] + '\t' + line[1] + '\t' + ' '.join(input_ids) + '\n')
#                     idx_bertsent[line[1]] = ' '.join(input_ids)
#
#     print( '%d', max_len)
#     max_len = 0