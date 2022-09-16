# -*-coding:utf-8-*-
"""
@Project    : MCSFD
@Time       : 2022/3/31 20:09
@Author     : Danke Wu
@File       : data_split.py
"""
# ##划分数据集
# import os,re,math,random
#
#
# root_source= r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data'
# root_dest = r'E:\WDK_workshop\PCL_rumor_detection\data'
# datasets = ['ferguson', 'germanwings-crash', 'charliehebdo','ottawashooting', 'sydneysiege']  #
#
# # root_source= r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data'
# # root_dest = r'E:\WDK_workshop\PCL_rumor_detection\data'
# # datasets = ['weibo2021']  #
# for dataset in datasets:
#     widlist = set()
#     filename_list = {}
#     for _, _, files in os.walk(os.path.join(root_source, dataset)):
#         file_num = len(files)
#         for file in files:
#             wid = re.sub('.pkl', '', file)
#             wid = wid.split('_')
#             wid = wid[0]
#             widlist.add(wid)
#             filename_list[wid] = file
#     train_idx = random.sample(widlist, math.floor(0.8 * file_num))
#     val_idx = widlist - set(train_idx)
#     test_idx = random.sample(val_idx, math.floor(0.1 * file_num))
#     val_idx = val_idx - set(test_idx)
#
#     if not os.path.exists(os.path.join(root_dest, dataset)):
#         os.mkdir(os.path.join(root_dest, dataset))
#         os.mkdir(os.path.join(root_dest, dataset, 'train'))
#         os.mkdir(os.path.join(root_dest, dataset, 'val'))
#         os.mkdir(os.path.join(root_dest, dataset, 'test'))
#
#     f_trian = open(os.path.join(root_dest, dataset, 'train_id_list.txt'), 'w')
#     f_trian.write('\n'.join(list(train_idx)))
#     f_test = open(os.path.join(root_dest, dataset, 'test_id_list.txt'), 'w')
#     f_test.write('\n'.join(list(test_idx)))
#     f_val = open(os.path.join(root_dest, dataset, 'val_id_list.txt'), 'w')
#     f_val.write('\n'.join(list(val_idx)))
#
#     import shutil
#
#     for idx in train_idx:
#         shutil.copy(os.path.join(root_source, dataset, filename_list[idx]),
#                     os.path.join(root_dest, dataset, 'train', filename_list[idx]))
#     for idx in test_idx:
#         shutil.copy(os.path.join(root_source,dataset, filename_list[idx]),
#                     os.path.join(root_dest, dataset, 'test', filename_list[idx]))
#     for idx in val_idx:
#         shutil.copy(os.path.join(root_source, dataset, filename_list[idx]),
#                     os.path.join(root_dest, dataset, 'val', filename_list[idx]))

##4合一
import os,re,math,random

root_source= r'E:\WDK_workshop\PCL_rumor_detection\data\raw_data_withdomain'
root_dest = r'E:\WDK_workshop\PCL_rumor_detection\data\data_withdomain'
datasets = [ 'ferguson','germanwings-crash', 'charliehebdo','ottawashooting', 'sydneysiege']#
#with domian label [0,1,2,3]
for dataset in datasets:
    dataset_dst = '4' + dataset
    domain = {}
    d_idx = 0
    others = set(datasets) - set([dataset])
    for other in others:
        widlist = set()
        filename_list = {}
        for _, _, files in os.walk(os.path.join(root_source, other)):
            file_num = len(files)
            for file in files:
                wid = re.sub('.pkl', '', file)
                wid = wid.split('_')
                wid = wid[0]
                widlist.add(wid)
                filename_list[wid] = file
            train_idx = random.sample(widlist, math.floor(0.8 * file_num))
            val_idx = widlist - set(train_idx)
            test_idx = random.sample(val_idx, math.floor(0.1 * file_num))
            val_idx = val_idx - set(test_idx)

        if not os.path.exists(os.path.join(root_dest, dataset_dst)):
            os.mkdir(os.path.join(root_dest, dataset_dst))
            os.mkdir(os.path.join(root_dest, dataset_dst, 'train'))
            os.mkdir(os.path.join(root_dest, dataset_dst, 'train', 'rumor'))
            os.mkdir(os.path.join(root_dest, dataset_dst, 'train', 'nonrumor'))
            os.mkdir(os.path.join(root_dest, dataset_dst, 'val'))
            os.mkdir(os.path.join(root_dest, dataset_dst, 'test'))

        f_trian = open(os.path.join(root_dest, dataset_dst, 'train_id_list.txt'), 'w')
        f_trian.write('\n'.join(list(train_idx)))
        f_test = open(os.path.join(root_dest, dataset_dst, 'test_id_list.txt'), 'w')
        f_test.write('\n'.join(list(test_idx)))
        f_val = open(os.path.join(root_dest, dataset_dst, 'val_id_list.txt'), 'w')
        f_val.write('\n'.join(list(val_idx)))

        import shutil

        for idx in train_idx:
            type = int(filename_list[idx].split('_')[1])
            if type ==1:
                shutil.copy(os.path.join(root_source, other,  filename_list[idx]),
                            os.path.join(root_dest, dataset_dst, 'train', 'rumor', filename_list[idx]))
            else:
                shutil.copy(os.path.join(root_source, other,  filename_list[idx]),
                            os.path.join(root_dest, dataset_dst, 'train', 'nonrumor',filename_list[idx]))

        for idx in test_idx:
            shutil.copy(os.path.join(root_source, other, filename_list[idx]),
                        os.path.join(root_dest, dataset_dst, 'test', filename_list[idx]))
        for idx in val_idx:
            shutil.copy(os.path.join(root_source, other, filename_list[idx]),
                        os.path.join(root_dest, dataset_dst, 'val', filename_list[idx]))


