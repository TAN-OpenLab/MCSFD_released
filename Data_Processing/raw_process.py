# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/3/30 9:03
@Author     : Danke Wu
@File       : raw_process.py
"""
##encoding= 'utf-8'
#weibo-dataset
import json
import pprint
import os
import glob
import pickle
import random
from datetime import datetime
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import re
import sys
sys.setrecursionlimit(10000)


def add(structure,content_dict, replies, s):
    for t in replies[s]:
        if t in structure.keys():
            structure = add(structure, replies, t)
            structure[s]['cascade'].extend(structure[t]['cascade'])
            content_dict[s].extend(content_dict[t])
            del structure[t]
            del content_dict[t]
    return structure, content_dict

if __name__  == '__main__':

    url = r'D:\数据集\2020-5-18\AllRumorDetection-master-data'
    cas_data = open(os.path.join(url,'Weibo.txt'))
    structure = {}
    content_dict = {}
    Posts = {}

    for line in cas_data.readlines():
        line = line.strip('\n').split('\t')
        wid = line[0].split(':')[1]
        label = line[1].split(':')[1]

        cont_data = open(os.path.join(url,'Weibo', wid +'.json'),'r',encoding='utf-8')
        users = json.load(cont_data)


        for user in users:

            if user == users[0] :

                sourceWId = user['mid']
                Posts[sourceWId] = {}
                Posts[sourceWId]['uid'] = str(user['uid'])

                structure[sourceWId] = {}
                content_dict[sourceWId] = []
                # content
                txt = user['text']
                # Remove '@name'
                txt = re.sub(r'(@.*?)[\s]', ' ', txt)
                # Replace '&amp;' with '&'
                txt = re.sub(r'&amp;', '&', txt)
                txt = re.sub(r'\n+', ' ', txt)
                # Remove trailing whitespace 删除空格
                txt = re.sub(r'\s+', ' ', txt).strip()
                # remove http
                p = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', re.S)
                txt = re.sub(p, '', txt)
                Posts[sourceWId]['text'] = txt
                content_dict[sourceWId].append(str(sourceWId) + '\t' + str(Posts[sourceWId]['uid']) + '\t' + Posts[sourceWId]['text'])

                # publication time
                time_pub = int(user['t'])

                structure[sourceWId]['time'] = time_pub
                structure[sourceWId]['userId'] = Posts[sourceWId]['uid']
                structure[sourceWId]['cascade'] = []
                structure[sourceWId]['label'] = label
            else:

                reactionWId = user['mid']
                if reactionWId == sourceWId:
                    continue
                replyTo = user["parent"]
                if replyTo not in Posts.keys():
                    replyTo = sourceWId
                Posts[reactionWId] ={}
                Posts[reactionWId]['uid'] = str(user['uid'])
                # content process
                txt = user['text']
                if txt == '"转发微博"' or txt == "轉發微博。" or txt== "转发微博。" or txt =='转' or txt=='转！':
                    if replyTo not in Posts.keys():
                        txt = Posts[sourceWId]['text']
                    else:
                        txt = Posts[replyTo]['text']

                txt = re.sub(r'(@.*?)[\s]+', ' ', txt)
                # Replace '&amp;' with '&'
                txt = re.sub(r'&amp;', '&', txt)
                txt = re.sub(r'\n+', ' ', txt)
                # Remove trailing whitespace 删除空格
                txt = re.sub(r'\s+', ' ', txt).strip()
                txt = re.sub(p, '', txt)
                Posts[reactionWId]['text'] = txt

                # public time
                time = int(user['t'])  # Sun Nov 06 21:21:26 +0000 2011
                structure[sourceWId]['cascade'].append([Posts[replyTo]['uid'], Posts[reactionWId]['uid'], time])
                content_dict[sourceWId].append(str(reactionWId) + '\t' + str(Posts[reactionWId]['uid']) + '\t' + Posts[reactionWId]['text'])

    result_dir = os.path.join(url, 'PCL_process','pro_cascades', "Weibo_cascade.txt")

    n=0

    with open(result_dir, 'a', encoding='utf-8') as fp:
        for key, value in structure.items():
            c = []
            if len(structure[key]['cascade']) == 0:
                continue
            for i in range(len(structure[key]['cascade'])):
                c.append(structure[key]['cascade'][i][0] + '/' + structure[key]['cascade'][i][1] + ':' + str(
                    structure[key]['cascade'][i][2] - structure[key]['time']))

            cascade = str(n) + '\t' + str(key) + '\t' + str(structure[key]['userId']) + '\t' + str(
                structure[key]['time']) + '\t' + str(
                len(c) + 1) + '\t' + str(structure[key]['userId']) + ':' + str(0) + ' ' + ' '.join(
                c) + '\t' + structure[key]['label'] + '\n'  # str(node_time.get(node)
            fp.write(cascade)
            content_dir = os.path.join(url, 'PCL_process','pro_content', str(key) + ".txt")
            con_f = open(content_dir, 'a', encoding='utf-8')
            con_f.write('\n'.join(content_dict[key]))
            n += 1




