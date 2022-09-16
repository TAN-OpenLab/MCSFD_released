# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/3/29 10:58
@Author     : Danke Wu
@File       : data_generator.py
"""
# import os
# import networkx as nx
# import numpy as np
# import math, pickle, scipy,random
#
# #生成谣言数据 weibo
# # user_content: bert_sentence_embedding, (N, 768)
# # XT : temporal interval matrix, time interval between posts, (N, 1)
# # XS : node matrix contains the structure feature of node, (N,3), i.e.  one-hot matrix, in-degree ==out-degree or in-degree>out-degree or in-degree < out-degree
#
# NUM_NODES = 50
# MAX_TXT_LEN =768
# url = r'D:\数据集\2020-5-18\AllRumorDetection-master-data'
# url_target = r'E:\WDK_workshop\PCL_rumor_detection\data'
# #url = r'D:\数据集\2020-5-18\rumor_detection_acl2017\twitter15'
# dataset ='weibo'
#
# newwid = [i for i in range(1490)]
# len_data =0
# random.shuffle(newwid)
#
# batch_x_len={}
#
# # 保留长度靠前的数据
# f = open(os.path.join(url, 'pro_cascades', "weibo_cascades.txt"), 'r', encoding='utf-8')
# for line in f.readlines():
#     g = nx.DiGraph()
#     node_time = {}
#     cascade = {}
#     line = line.strip('\n').split('\t')
#     wid = line[1]
#     paths = line[5].split(' ')
#     label = int(line[6])
#
#     if len(paths)<10:
#         continue
#
#     # 节点的对应新id
#     node_new_id = {}
#     n = 0
#     i = 0
#     cascade[wid] = {}
#     xT = np.zeros((1,NUM_NODES),dtype=float)
#
#
#     for path in paths:
#         nodes = path.split(':')[0].split('/')
#         t = float(path.split(":")[1])
#
#         if i < NUM_NODES:
#             xT[1, i] = t
#             i += 1
#
#         # 控制节点数量
#         if n >= NUM_NODES:
#             for i in range(len(nodes) - 1):
#                 if nodes[i + 1] not in node_new_id.keys():
#                     continue
#                 else:
#                     if nodes[i] in node_new_id.keys():
#                         g.add_edge(nodes[i], nodes[i + 1])
#         else:
#             if len(nodes) == 1:
#                 if nodes[0] not in node_time.keys():
#                     node_new_id[nodes[0]] = n
#                     n += 1
#                     source = nodes[0]
#             else:
#                 for i in range(len(nodes) - 1):
#
#                     if nodes[i] not in node_time.keys():
#                         node_new_id[nodes[i]] = n
#                         n += 1
#                         g.add_edge(source, nodes[i])
#
#                     if n >= NUM_NODES:
#                         continue
#                     else:
#                         g.add_edge(nodes[i], nodes[i + 1])
#
#                     if nodes[i + 1] not in node_time.keys():
#                         node_new_id[nodes[i + 1]] = n
#                         n += 1
#
#
#     bert_f = open(os.path.join(url, 'pro_content_emb', wid + '.txt'), 'r', encoding='utf-8')
#     act_num_node = len(node_new_id)
#     user_content = np.zeros((act_num_node, MAX_TXT_LEN), dtype=float)
#     xS = np.zeros((act_num_node,3),dtype=int)
#
#     # user_time_content darry
#     node_persence= {}
#     for line in bert_f.readlines():
#         line = line.strip('\n').split('\t')
#         node = line[0]
#         txt = line[2].split(' ')
#         if node not in node_new_id.keys():
#             continue
#         if node not in node_persence.keys():
#             node_persence[node] = 0
#         node_persence[node] += 1
#
#         if txt == ['']:
#             user_content[ node_new_id[node], :] = user_content[0, :]
#         else:
#             txt = list(map(float, txt))
#             txt_len = len(txt)
#             id = node_new_id[node]
#             user_content[node_new_id[node], :txt_len] = np.array(txt)
#
#
#     # user without content = source content
#     for node in list(nx.nodes(g)):
#         if node not in node_persence.keys():
#             user_content[node_new_id[node], :] = user_content[0, :]
#         if node.in_degree() == node.out_degree():
#             xS[node_new_id[node]][0] = 1
#         elif node.in_degree() > node.out_degree():
#             xS[node_new_id[node]][1] = 1
#         else:
#             xS[node_new_id[node]][2] = 1
#
#     g_new = nx.DiGraph()
#     for (s, t) in list(nx.edges(g)):
#         g_new.add_edge(node_new_id[s], node_new_id[t])
#     g_new.remove_edges_from(list(g_new.selfloop_edges()))
#     g_adj = nx.adj_matrix(g_new).todense()
#     N = nx.number_of_nodes(g_new)
#     rows, cols = np.nonzero(g_adj)
#     A = [list(rows), list(cols)]
#     if len(node_new_id) != g_new.number_of_nodes():
#         print('wrong!')
#     else:
#         file = open(os.path.join(url_target, 'weibo_all_bertemb50', wid + '_' + str(label) + '_' + dataset + '_N50.pkl'),'wb')
#         pickle.dump((user_content.tolist(), A, label), file)
#         len_data += 1
#
# import os
# import networkx as nx
# import numpy as np
# import math, pickle, scipy,random
#
# #生成谣言数据 weibo
# # user_content: bert_sentence_embedding, (N, 768)
# # XT : temporal interval matrix, time interval between posts, (N, 1)
# # XS : node matrix contains the structure feature of node, (N,3), i.e.  one-hot matrix, in-degree ==out-degree or in-degree>out-degree or in-degree < out-degree
#
# NUM_NODES = 50
# MAX_TXT_LEN =768
# # url = r'D:\数据集\2020-5-18\AllRumorDetection-master-data\PCL_process'
# url = r'D:\数据集\2020-5-18\weiboRumor-dataset-master\PCL_process'
# url_dst = r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data'
#
#
# dataset ='weibo2021'
# len_data = 0
#
# f = open(os.path.join(url, 'pro_cascades', "weibo_cascades.txt"), 'r', encoding='utf-8')
# for line in f.readlines():
#     g = nx.DiGraph()
#     cascade = {}
#     line = line.strip('\n').split('\t')
#     wid = line[1]
#     paths = line[5].split(' ')
#     label = int(line[6])
#
#     if len(paths) < 10:
#         continue
#
#     # 节点的对应新id
#     node_new_id = {}
#     nn = 0
#     tn = 0
#     cascade[wid] = {}
#     xT = np.zeros((1, NUM_NODES), dtype=float)
#
#     for path in paths:
#         nodes = path.split(':')[0].split('/')
#         t = float(path.split(":")[1])
#
#         if tn < NUM_NODES:
#             xT[0, tn] = t
#             tn += 1
#
#         # 控制节点数量
#         if nn >= NUM_NODES:
#             for i in range(len(nodes) - 1):
#                 if nodes[i + 1] not in node_new_id.keys():
#                     continue
#                 else:
#                     if nodes[i] in node_new_id.keys():
#                         g.add_edge(nodes[i], nodes[i + 1])
#         else:
#             if len(nodes) == 1:
#                 if nodes[0] not in node_new_id.keys():
#                     node_new_id[nodes[0]] = nn
#                     nn += 1
#                     source = nodes[0]
#             else:
#                 for i in range(len(nodes) - 1):
#
#                     if nodes[i] not in node_new_id.keys():
#                         node_new_id[nodes[i]] = nn
#                         nn += 1
#                         g.add_edge(source, nodes[i])
#
#                     if nn >= NUM_NODES:
#                         continue
#                     else:
#                         if nodes[i + 1] not in node_new_id.keys():
#                             node_new_id[nodes[i + 1]] = nn
#                             nn += 1
#                         g.add_edge(nodes[i], nodes[i + 1])
#
#     bert_f = open(os.path.join(url, 'pro_content_bertemb', wid + '.txt'), 'r', encoding='utf-8')
#     act_num_node = len(node_new_id)
#     user_content = np.zeros((NUM_NODES, MAX_TXT_LEN), dtype=float)
#     xS = np.zeros((NUM_NODES, 3), dtype=int)
#
#     # user_time_content darry
#     node_persence = {}
#     for line in bert_f.readlines():
#         line = line.strip('\n').split('\t')
#         node = line[1]
#         txt = line[2].split(' ')
#         if node not in node_new_id.keys():
#             continue
#         if node not in node_persence.keys():
#             node_persence[node] = 0
#         node_persence[node] += 1
#
#         if txt == '':
#             user_content[node_new_id[node], :] = user_content[0, :]
#         else:
#             txt = list(map(float, txt))
#             txt_len = len(txt)
#             user_content[node_new_id[node], :txt_len] = np.array(txt)
#
#     # user without content = source content
#     for node in list(nx.nodes(g)):
#         if node not in node_persence.keys():
#             user_content[node_new_id[node], :] = user_content[0, :]
#         if g.in_degree(node) == g.out_degree(node):
#             xS[node_new_id[node]][0] = 1
#         elif g.in_degree(node) > g.out_degree(node):
#             xS[node_new_id[node]][1] = 1
#         else:
#             xS[node_new_id[node]][2] = 1
#
#     g_new = nx.DiGraph()
#     for (s, t) in list(nx.edges(g)):
#         g_new.add_edge(node_new_id[s], node_new_id[t])
#     g_new.remove_edges_from(list(g_new.selfloop_edges()))
#     g_adj = nx.adj_matrix(g_new).todense()
#     N = nx.number_of_nodes(g_new)
#     if N < NUM_NODES:
#         col_padding_L = np.zeros(shape=(N, NUM_NODES - N))
#         L_col_padding = np.column_stack((g_adj, col_padding_L))
#         row_padding = np.zeros(shape=(NUM_NODES - N, NUM_NODES))
#         L_col_row_padding = np.row_stack((L_col_padding, row_padding))
#         A = scipy.sparse.coo_matrix(L_col_row_padding, dtype=np.int)
#     else:
#         A = scipy.sparse.coo_matrix(g_adj, dtype=np.int)
#     xT_base = np.zeros_like(xT)
#     xT_base[1:] = xT[0:-1]
#     xT = np.log(xT - xT_base)
#     xT = np.where(np.isinf(xT), 0, xT)
#     xT = np.where(np.isnan(xT), 0, xT)
#     if len(node_new_id) != g_new.number_of_nodes():
#         print('wrong!')
#     else:
#         file = open(os.path.join(url_dst, dataset, wid + '_' + str(label) + '_' + '_N50.pkl'), 'wb')
#         pickle.dump((user_content.tolist(), xS.tolist(), A, xT.tolist(), label), file)
#         len_data += 1
#




##pheme
import os
import networkx as nx
import numpy as np
import math, pickle, scipy,random


#一个数据一个文件
#TIME_WINDOWS =6
NUM_NODES =50
MAX_TXT_LEN =768
url = r'D:\数据集\2020-5-18\pheme-rnr-dataset'
url_dst = r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data_withdomain'
datasets = ['ferguson','germanwings-crash', 'charliehebdo','ottawashooting', 'sydneysiege']  #
types = ['rumours', 'non-rumours']
# x_len = {}
# x_len['germanwings-crash']= 78
# x_len['charliehebdo']= 274
# x_len['ferguson']= 204
# x_len['sydneysiege']= 365
# x_len['ottawashooting'] = 232

type2id = {}
type2id['rumours'] = '1'
type2id['non-rumours'] ='0'



for dataset in datasets:
    if not os.path.exists(os.path.join(url_dst, dataset)):
        os.mkdir(os.path.join(url_dst, dataset))

    for type in types:
        len_data = 0
        batch_x_len={}

        f = open(os.path.join(url, 'pro_cascades', dataset + '_' + type + '_cascade.txt'), 'r', encoding='utf-8')
        for line in f.readlines():
            g = nx.DiGraph()
            cascade = {}
            line = line.strip('\n').split('\t')
            wid = line[1]
            paths = line[5].split(' ')
            label = int(line[6])

            if len(paths) < 10:
                continue

            # 节点的对应新id
            node_new_id = {}
            nn = 0
            tn = 0
            cascade[wid] = {}
            xT = np.zeros((1, NUM_NODES), dtype=float)

            for path in paths:
                nodes = path.split(':')[0].split('/')
                t = float(path.split(":")[1])

                if tn < NUM_NODES:
                    xT[0, tn] = t
                    tn += 1

                # 控制节点数量
                if nn >= NUM_NODES:
                    for i in range(len(nodes) - 1):
                        if nodes[i + 1] not in node_new_id.keys():
                            continue
                        else:
                            if nodes[i] in node_new_id.keys():
                                g.add_edge(nodes[i], nodes[i + 1])
                else:
                    if len(nodes) == 1:
                        if nodes[0] not in node_new_id.keys():
                            node_new_id[nodes[0]] = nn
                            nn += 1
                            source = nodes[0]
                    else:
                        for i in range(len(nodes) - 1):

                            if nodes[i] not in node_new_id.keys():
                                node_new_id[nodes[i]] = nn
                                nn += 1
                                g.add_edge(source, nodes[i])

                            if nn >= NUM_NODES:
                                continue
                            else:
                                if nodes[i + 1] not in node_new_id.keys():
                                    node_new_id[nodes[i + 1]] = nn
                                    nn += 1
                                g.add_edge(nodes[i], nodes[i + 1])



            bert_f = open(os.path.join(url, 'pro_content_bertemb', dataset, type, wid + '.txt'), 'r', encoding='utf-8')
            act_num_node = len(node_new_id)
            user_content = np.zeros((NUM_NODES, MAX_TXT_LEN), dtype=float)

            # user_time_content darry
            node_persence = {}
            for line in bert_f.readlines():
                line = line.strip('\n').split('\t')
                node = line[1]
                txt = line[2].split(' ')
                if node not in node_new_id.keys():
                    continue
                if node not in node_persence.keys():
                    node_persence[node] = 0
                node_persence[node] += 1

                if txt == '':
                    user_content[node_new_id[node], :] = user_content[0, :]
                else:
                    txt = list(map(float, txt))
                    txt_len = len(txt)
                    user_content[node_new_id[node], :txt_len] = np.array(txt)

            # user without content = source content
            for node in list(nx.nodes(g)):
                if node not in node_persence.keys():
                    user_content[node_new_id[node], :] = user_content[0, :]


            g_new = nx.DiGraph()
            for (s, t) in list(nx.edges(g)):
                g_new.add_edge(node_new_id[s], node_new_id[t])
            g_new.remove_edges_from(list(g_new.selfloop_edges()))
            g_adj = nx.adj_matrix(g_new).todense()
            N = nx.number_of_nodes(g_new)
            if N < NUM_NODES:
                col_padding_L = np.zeros(shape=(N, NUM_NODES - N))
                L_col_padding = np.column_stack((g_adj, col_padding_L))
                row_padding = np.zeros(shape=(NUM_NODES - N, NUM_NODES))
                L_col_row_padding = np.row_stack((L_col_padding, row_padding))
                A = scipy.sparse.coo_matrix(L_col_row_padding, dtype=np.int)
            else:
                A = scipy.sparse.coo_matrix(g_adj, dtype=np.int)

            if len(node_new_id) != g_new.number_of_nodes():
                print('wrong!')
            else:
                file = open(os.path.join(url_dst, dataset, wid + '_' + str(label) + '_' + '_N50.pkl'), 'wb')
                pickle.dump((user_content.tolist(), A, label, dataset), file)
                len_data += 1

#
