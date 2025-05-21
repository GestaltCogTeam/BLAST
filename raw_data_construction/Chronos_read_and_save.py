import os
import pyarrow as pa
import numpy as np
import json
from datetime import datetime
import pandas as pd
import pyarrow.parquet as pq

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集文件夹路径
datasets_folder = os.path.join(current_dir, '..', 'datasets/raw_datasets/chronos_datasets')
output_folder = os.path.join(current_dir, '..', 'datasets/processed_datasets/Chronos')

import re


def format_dataset_name(name):
    # 移除包含"monash"或"Monash"的字段
    name = re.sub(r'monash', '', name, flags=re.IGNORECASE)

    # 去除开头的下划线
    name = re.sub(r'^_+', '', name)
    # 处理连续大写的情况，将其转换为首字母大写，其余小写
    name = re.sub(r'([A-Z]+)', lambda m: m.group(0).capitalize(), name)

    # 使用正则表达式在每个大写字母前添加下划线，但不在数字之间添加下划线
    name = re.sub(r'(?<!^)(?=[A-Z])', '_', name)

    # 将所有字母转换为小写
    name = name.lower()

    # 将首字母和每个下划线后的第一个字母转换为大写
    formatted_name = '_'.join([word.capitalize() if not word.isdigit() else word for word in name.split('_')])

    # 处理数字之间不添加下划线的情况
    #formatted_name = re.sub(r'(\d)_+(\d)', r'\1\2', formatted_name)
    formatted_name = re.sub(r'_{2,}', '_', formatted_name)

    return formatted_name

def extract_keys(schema, prefix=''):
    keys = []
    for field in schema:
        if isinstance(field.type, pa.lib.ListType):
            keys.extend(extract_keys(field.type.value_type, prefix + field.name + '.'))
        else:
            keys.append(prefix + field.name)
    return keys

def time_stamp_check(timestamp):
    # 检查所有数据集的
    if len(set([i.shape[0] for i in timestamp])) != 1:
        time_diff = [seq[i] - seq[i + 1] for seq in timestamp for i in range(len(seq) - 1)]
        time_diff = set(time_diff)
        if len(time_diff) != 1:
            print(dataset_name, [pd.to_timedelta(i) for i in time_diff])

def get_key(table):
    # 获取parquet文件的所有key以及对应的type, 划分为序列key 和 meta data key
    # 获取所有列的元数据
    key = table.schema.names
    types = table.schema.types
    TS_key = []
    meta_key = []
    for i in range(len(types)):
        if 'list' in str(types[i]):
            TS_key.append(key[i])
        else:
            meta_key.append(key[i])
    return TS_key, meta_key

def replace_edge_zeros_with_nan(arr):
    # 每个输入都是一个list, 将首尾的0替换为nan

    l_array = arr
    # 找到 非 0 值的索引
    non_zero_indices = np.where(l_array != 0)[0]
    # print(non_zero_indices)
    if non_zero_indices.size > 0:
        first_non_zero = non_zero_indices[0]
        last_non_zero = non_zero_indices[-1]
        # print(first_non_zero, last_non_zero, l_array.shape)
        arr[0: first_non_zero] = np.nan
        arr[last_non_zero:] = np.nan

    return arr

def get_valid_indicies_and_length(N, C, data):
    valid_lengths = np.zeros((N, C, 1), dtype=int)
    valid_indices = np.zeros((N, C, 2), dtype=int)

    # 计算有效长度和索引
    for n in range(N):
        for c in range(C):
            non_nan_indices = np.where(~np.isnan(data[n, c]))[0]
            if non_nan_indices.size > 0:
                first_non_nan = non_nan_indices[0]
                last_non_nan = non_nan_indices[-1]
                valid_lengths[n, c, 0] = last_non_nan - first_non_nan + 1
                valid_indices[n, c, 0] = first_non_nan
                valid_indices[n, c, 1] = last_non_nan
    return valid_indices, valid_lengths

def save_data(data, file_path):
    np.save(file_path, data) # TODO: save as memmap

def unique_freq_transform(freq):
    # 将以days 为技术的频率转为相应的单位，由于pandas没有相应的转换方法，需要手动转换。
    if 28 <= freq.days <= 31:
        freq_label = 'Month'
    elif 88 <= freq.days <= 93:
        freq_label = 'Quarter'
    elif 364 <= freq.days:
        freq_label = 'Year'
    else:
        freq_label = None
    return freq_label


def construct_and_padding_data(min_start, starts, freq, data, TS_key, N, C):

    series = []
    # 1. 完成读取，并同时处理掉首尾的0
    for key in TS_key:
        if key in data.columns:
            arr = data[key]
            series.append(arr)

    # 2. 计算每个序列的左边padding
    freq_label = unique_freq_transform(freq)
    padding_dict = {}
    for i in range(len(starts)):
        start = pd.to_datetime(starts[i])
        if freq_label == 'Month':
            delta = (start.year - min_start.year) * 12 + (start.month - min_start.month)
        elif freq_label == 'Quarter':
            delta = (start.year - min_start.year) * 4 + (start.quarter - min_start.quarter)
        elif freq_label == 'Year':
            delta = start.year - min_start.year
        else:
            delta = (start - min_start) // pd.Timedelta(freq)
        padding_dict[i] = delta

    target_left = []
    # 3. 遍历 series,在左侧添加padding
    for j in series:
        ts = []
        for i, seq in enumerate(j):
            padding = [np.nan] * padding_dict[i]
            padded_seq = padding + list(seq)
            ts.append(padded_seq)
        target_left.append(ts)

    target = []
    L = max([len(seq) for channel_list in target_left for seq in channel_list])
    # 4. 根据最长的序列，在右侧添加padding
    for j in series:
        ts = []
        for i, seq in enumerate(j):
            padding = [np.nan] * (L - len(seq))
            padded_seq = list(seq) + padding
            ts.append(np.array(padded_seq))
        target.append(np.array(ts)) # target : C x N x L
    # 5. padding完成, reshape
    target = np.stack(target, axis=0).transpose(1, 0, 2)  # target: N x C x L

    return target

count = 0
folder_name = datasets_folder.split('\\')[-1]
total = len(os.listdir(datasets_folder))
for dataset_name in os.listdir(datasets_folder):
    count += 1
    dataset_path = os.path.join(datasets_folder, dataset_name)
    print(f"Processing dataset {count}/{total}: {dataset_name}")
    if os.path.isdir(dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file_name in files:
                if file_name.endswith('.parquet'):

                    file_path = os.path.join(root, file_name)
                    table = pq.read_table(file_path)
                    dataset_name = "_".join(root.split(folder_name)[-1].split('/')[1:])
                    dataset_name = format_dataset_name(dataset_name)
                    print(dataset_name)
                    output_dataset_folder = os.path.join(output_folder, dataset_name)
                    os.makedirs(output_dataset_folder, exist_ok=True)

                    TS_key, meta_key = get_key(table) # 获取所有列的元数据

                    data = table.to_pandas()
                    timestamp = [np.array(seq) for seq in data['timestamp']]

                    TS_key.remove('timestamp') # 筛选出所有是sequence的key

                    # 计算每个序列的频率和第一个时间戳
                    # 计算第一个序列的频率
                    if len(timestamp[0]) > 1:
                        # time_stamp_check(timestamp) # 检查时间戳是否一致，发现monash的不一致，故将monash单独处理
                        time_diff = timestamp[0][1] - timestamp[0][0] # 已经事先检查完了，monash的数据被剔除在外
                        freq = pd.Timedelta(time_diff)


                    else:
                        freq = None

                    # 获取每个序列的第一个时间戳
                    starts = [seq[0] if len(seq) > 0 else None for seq in timestamp]
                    min_start = min(pd.to_datetime(starts))


                    # 获取 节点数量与通道维度
                    N = len(timestamp)
                    C = len(TS_key)

                    # 开始转换数据集
                    target = construct_and_padding_data(min_start, starts, freq, data, TS_key, N, C)
                    L = target.shape[2]
                    # 保存数据
                    part_number = str(int(file_name.split('-')[1]))
                    target_file_name = f"{dataset_name}_{part_number}.npy"
                    target_file_path = os.path.join(output_dataset_folder, target_file_name)
                    save_data(target, target_file_path)

                    # 收集target列的元数据
                    target_count_non_nan = np.sum(~np.isnan(target))
                    valid_indices, valid_lengths = get_valid_indicies_and_length(N, C, target)

                    shapes_info = {}
                    shapes_info['data_shape'] = (N, C, L)
                    shapes_info['data_valid_value'] = int(target_count_non_nan)
                    shapes_info["min_length"] = int(valid_lengths.min())
                    shapes_info['max_length'] = int(valid_lengths.max())
                    shapes_info["mean_length"] = float(valid_lengths.mean())
                    shapes_info["valid_lengths"] = valid_lengths.tolist()
                    shapes_info["valid_indices"] = valid_indices.tolist()

                    other_info = {}
                    other_info['freq'] = str(freq)
                    other_info['start_timestamp'] = [str(i) for i in starts]
                    other_info['TS_keys'] = [str(i) for i in TS_key]
                    other_info['meta-keys'] = [str(i) for i in meta_key]
                    # 遍历meta_key中的每个key
                    for key in meta_key:
                        # 将data中的内容转换为字符串并存储在字典中
                        other_info[key] = data[key].astype(str).tolist()

                    other_info.update(shapes_info)
                    json_file_name = f"{dataset_name}.json"
                    if '-' in file_name:
                        json_file_name = f"{dataset_name}_{part_number}.json"
                    # print(json_file_name)
                    with open(os.path.join(output_dataset_folder, json_file_name), 'w') as json_file:
                        json.dump(other_info, json_file, ensure_ascii=False, indent=4)

