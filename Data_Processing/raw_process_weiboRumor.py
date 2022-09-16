# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/4/8 20:01
@Author     : Danke Wu
@File       : raw_process_weiboRumor.py
"""
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
import re
import sys
sys.setrecursionlimit(10000)



url_root = r'D:\数据集\2020-5-18\weiboRumor-dataset-master'
c_typos = ['rumor','non_rumor']
url_dst =''

for c_typo in c_typos:
    i = 1
    if c_typo == 'rumor':
        label = '1'
    else:
        label = '0'

    for root, dirs, files in os.walk(os.path.join(url_root, 'original-microblog', c_typo)):
        structure = {}
        content_dict = {}
        Posts = {}

        for file in files:
            ids = file.strip('.josn').split('_')
            sourceData = json.load(open(os.path.join(root, file),encoding= 'utf-8'))
            sourceWId = ids[1]
            sourceUId = ids[2]
            structure[sourceWId] = {}
            content_dict[sourceWId] = []
            Posts[sourceWId] = {}
            Posts[sourceWId]['uid'] = str(sourceUId)

            # content
            txt = sourceData['text']
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
            content_dict[sourceWId].append(
                str(sourceWId) + '\t' + str(Posts[sourceWId]['uid']) + '\t' + Posts[sourceWId]['text'])
            # publication time
            if isinstance(sourceData['time'], str):
                if len(sourceData['time'].split(' ')) > 4:
                    time_pub = datetime.strptime(sourceData['time'], '%a %b %d  %H:%M:%S +0800 %Y')#Sat Jun 29 12:40:09 +0800 2013
                    time_pub = int(datetime.timestamp(time_pub))
                else:
                    time_pub = datetime.strptime(sourceData['time'], '%Y %m %d  %H:%M:%S')  #2013 01 25 01:48:43
                    time_pub = int(datetime.timestamp(time_pub))
            else:
                time_pub = int(sourceData['time']) #1347334462

            structure[sourceWId]['time'] = time_pub
            structure[sourceWId]['userId'] = sourceUId
            structure[sourceWId]['cascade'] = []
            structure[sourceWId]['label'] = label

            if c_typo == 'rumor':
                dir = 'rumor-repost'
            else:
                dir = 'non-rumor-repost'
            reactions = json.load(open(os.path.join(url_root, dir, file),encoding= 'utf-8'))

            for i in range(len(reactions)):
                reactionData = reactions[len(reactions) - 1 - i]
                reactionWId = reactionData["mid"]
                if reactionWId == sourceWId:
                    continue
                replyTo = reactionData["parent"]
                if replyTo == '':
                    replyTo = sourceWId
                reactionUId = reactionData['uid']

                Posts[reactionWId] = {}
                Posts[reactionWId]['uid'] = str(reactionData['uid'])
                # content process
                txt = reactionData['text']
                if txt == '"转发微博"' or txt == "轉發微博。" or txt == "转发微博。" or txt == '转' or txt == '转！' or txt == '':
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
                time_pub = reactionData['date']
                date = time_pub.split(' ')[0].split('-')
                if len(date)< 3:
                    print(sourceWId)
                    continue
                times =  time_pub.split(' ')[1]
                t = date[0] + " " + date[1] + " " + date[2] + " " + times
                time_pub = datetime.strptime(t, '%Y %m %d  %H:%M:%S')  # 2012-09-11 11:35:48
                time_pub = int(datetime.timestamp(time_pub))

                structure[sourceWId]['cascade'].append([Posts[replyTo]['uid'], Posts[reactionWId]['uid'], time_pub])
                content_dict[sourceWId].append(
                    str(reactionWId) + '\t' + str(Posts[reactionWId]['uid']) + '\t' + Posts[reactionWId]['text'])

        result_dir = os.path.join(url_root, 'PCL_process', 'pro_cascades', "Weibo_cascade.txt")

        num_post = 0

        with open(result_dir, 'a', encoding='utf-8') as fp:
            for key, value in structure.items():
                c = []
                if len(structure[key]['cascade']) == 0:
                    continue
                for i in range(len(structure[key]['cascade'])):
                    c.append(
                        structure[key]['cascade'][i][0] + '/' + structure[key]['cascade'][i][1] + ':' + str(
                            structure[key]['cascade'][i][2] - structure[key]['time']))

                cascade = str(num_post) + '\t' + str(key) + '\t' + str(structure[key]['userId']) + '\t' + str(
                    structure[key]['time']) + '\t' + str(
                    len(c) + 1) + '\t' + str(structure[key]['userId']) + ':' + str(0) + ' ' + ' '.join(
                    c) + '\t' + structure[key]['label'] + '\n'  # str(node_time.get(node)
                fp.write(cascade)
                content_dir = os.path.join(url_root, 'PCL_process', 'pro_content', str(key) + ".txt")
                con_f = open(content_dir, 'a', encoding='utf-8')
                con_f.write('\n'.join(content_dict[key]))
                num_post += 1













