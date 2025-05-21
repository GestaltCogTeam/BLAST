# import packages
import os
import re
import json
import numpy as np
import pandas as pd
import pyarrow as pa
from datetime import datetime


current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集文件夹路径
datasets_folder = os.path.join(current_dir, '..', 'datasets/raw_datasets/lotsa_data')
output_folder = os.path.join(current_dir, '..', 'datasets/processed_datasets/LOTSA')

def format_dataset_name(name):
    """
    格式化数据集名称的函数。通过删除特定字段、调整大小写和添加下划线等操作，将原始数据集名称转化为符合命名规范的格式。

    参数:
    name (str): 原始数据集名称，可能包含大小写不一致、下划线或特定关键词的字符串。

    示例:
    >>> format_dataset_name("monash_m3_monthly")
    'M3_Monthly'

    """
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

def conver_freq(freq):
    """将旧版本的频率字符串转换为新版本的频率字符串。
        "H" -> "h"
        "T" -> "min"
        "S" -> "s"
        "L" -> "ms"
        "U" -> "us"
        "N" -> "ns"

    Args:
        freq (str): 旧版本的频率字符串。
    
    示例：
    >>> convert_freq("15H")
    "15h"
    """
    mapping_dict = {
        "H": "h",
        "T": "min",
        "S": "s",
        "L": "ms",
        "U": "us",
        "N": "ns",
        "A": "Y",
        "Ms": "M"
    }
    for key, value in mapping_dict.items():
        freq = freq.replace(key, value)
    return freq

def parse_custom_timedelta(freq):
    if 'M' in freq:
        return pd.DateOffset(months=int(freq[:-1]))
    elif 'Y' in freq:
        return pd.DateOffset(years=int(freq[:-1]))
    elif 'Q' in freq:
        return pd.DateOffset(months=int(freq[:-1]) * 3)
    else:
        return pd.to_timedelta(freq)


def construct_and_padding_data(starts, freq, data):
    """
        将时间序列数据填充到相同长度的张量中。对于每个时间序列，将其填充到与最长时间序列相同的长度，以便在训练模型时能够使用批处理。

        参数:
            starts (list): 包含每个时间序列的开始时间的列表。
            freq (str): 时间序列的频率。
            data (list): 包含每个时间序列数据的列表。
    """
    min_start = min(pd.to_datetime(starts))
    #print(freq)
    freq_offset = parse_custom_timedelta(freq)
    target = []
    for i, seq in enumerate(data):
        if not isinstance(seq[0], (list, np.ndarray)):
            seq = [seq]  # C = 1
        seq = np.stack(seq, axis=0)  # seq: C x L
        seq = replace_edge_zeros_with_nan(seq)

        start_date = pd.to_datetime(starts[i])
        if isinstance(freq_offset, pd.DateOffset):
            steps = 0
            current_date = min_start
            while current_date < start_date:
                current_date += freq_offset
                #print(current_date)
                steps += 1
            padding_steps = steps
        else:
            padding_steps = (start_date - min_start) // freq_offset

        padded_seq = np.pad(seq, ((0, 0), (padding_steps, 0)), 'constant', constant_values=np.nan)
        target.append(padded_seq)

    # 3. 右侧padding
    N = len(target)
    C = len(target[0]) # 假设所有序列的C相同
    L = max([item.shape[1] for item in target]) # 找到最长的L
    for i, seq in enumerate(target):
        target[i] = np.pad(seq, ((0, 0), (0, L - seq.shape[1])), 'constant', constant_values=np.nan) # 行维度不padding，列维度左侧不padding，右侧padding(L - seq.shape[1])列

    # 4. padding完成
    target = np.stack(target, axis=0) # target: N x C x L

    return target

def replace_edge_zeros_with_nan(arr):
    # 遍历每个N和每个C
    for n in range(arr.shape[0]):
        l_array = arr[n, :]
        # 找到 非 0 值的索引
        non_zero_indices = np.where(l_array != 0)[0]
        # print(non_zero_indices)
        if non_zero_indices.size > 0:
            first_non_zero = non_zero_indices[0]
            last_non_zero = non_zero_indices[-1]
            # print(first_non_zero, last_non_zero, l_array.shape)
            arr[n, 0: first_non_zero] = np.nan
            arr[n, last_non_zero:] = np.nan
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

def get_folder_list(folder_path):
    return [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

def check_same_freq(freq):
    # TODO: check if all freq are the same
    pass

count = 0
total = len(get_folder_list(datasets_folder))

for dataset_name in get_folder_list(datasets_folder):
    shapes_info = {}
    count += 1
    print(f"Processing dataset {count}/{total}: {dataset_name}")
    dataset_path = os.path.join(datasets_folder, dataset_name)
    # 逐个读取dataset_path下的所有文件
    for i, file_name in enumerate(os.listdir(dataset_path)):
        # if i == 1: break # TODO: only for test. comment this line
        if not file_name.endswith('.arrow'): continue
        file_path = os.path.join(dataset_path, file_name)

        # 读取arrow文件
        with open(file_path, 'rb') as f:
            table = pa.ipc.RecordBatchStreamReader(f).read_all()
            data = table.to_pandas()

        # 转换数据集名称
        dataset_name = format_dataset_name(dataset_name)
        output_dataset_folder = os.path.join(output_folder, dataset_name)
        os.makedirs(output_dataset_folder, exist_ok=True)

        starts = data['start']
        freq = data['freq'][0]  # 假设所有序列的频率相同
        if '-' in freq:
            freq = freq.split('-')[0]
        if not any(char.isdigit() for char in freq): freq = '1' + freq  # 如果freq没有数字，添加一个默认的数字1
        freq = conver_freq(freq)
        
        # 开始转换数据集
        target = construct_and_padding_data(starts, freq, data['target'])
        N, C, L = target.shape

        # 保存数据
        part_number = str(int(file_name.split('-')[1]))
        target_file_name = f"{dataset_name}_{part_number}.npy"
        target_file_path = os.path.join(output_dataset_folder, target_file_name)
        save_data(target, target_file_path)

        # 收集target列的元数据
        target_count_non_nan = np.sum(~np.isnan(target))
        valid_indices, valid_lengths = get_valid_indicies_and_length(N, C, target)

        shapes_info['data_shape'] = (N, C, L)
        shapes_info['data_valid_value'] = int(target_count_non_nan)
        shapes_info["min_length"] = int(valid_lengths.min())
        shapes_info['max_length'] = int(valid_lengths.max())
        shapes_info["mean_length"] = float(valid_lengths.mean())
        shapes_info["valid_lengths"] = valid_lengths.tolist()
        shapes_info["valid_indices"] = valid_indices.tolist()
        
        if 'past_feat_dynamic_real' in data.columns:
            # 开始转换协变量
            past_feat_dynamic_real = construct_and_padding_data(starts, freq, data['past_feat_dynamic_real'])
            N, C, L = past_feat_dynamic_real.shape

            # 保存数据
            past_feat_file_name = f"{dataset_name}_past_feat_dynamic_real_{part_number}.npy"
            past_feat_file_path = os.path.join(output_dataset_folder, past_feat_file_name)
            save_data(past_feat_dynamic_real, past_feat_file_path)
            
            # 收集协变量列的元数据
            past_feat_dynamic_real_count_non_nan = np.sum(~np.isnan(past_feat_dynamic_real))
            valid_indices, valid_lengths = get_valid_indicies_and_length(N, C, past_feat_dynamic_real)
            
            shapes_info['past_feat_dynamic_real_shape'] = (N, C, L)
            shapes_info['past_feat_dynamic_real_valid_value'] = int(past_feat_dynamic_real_count_non_nan)
            shapes_info["past_feat_dynamic_real_min_length"] = int(valid_lengths.min())
            shapes_info['past_feat_dynamic_real_max_length'] = int(valid_lengths.max())
            shapes_info["past_feat_dynamic_real_mean_length"] = float(valid_lengths.mean())
            shapes_info["past_feat_dynamic_real_valid_lengths"] = valid_lengths.tolist()
            shapes_info["past_feat_dynamic_real_valid_indices"] = valid_indices.tolist()
        
        if 'feat_dynamic_real' in data.columns:
            # 开始转换协变量
            feat_dynamic_real = construct_and_padding_data(starts, freq, data['feat_dynamic_real'])
            N, C, L = past_feat_dynamic_real.shape
            
            # 保存数据
            feat_file_name = f"{dataset_name}_feat_dynamic_real_{part_number}.npy"
            feat_file_path = os.path.join(output_dataset_folder, feat_file_name)
            save_data(feat_dynamic_real, feat_file_path)
            
            # 收集协变量列的元数据
            feat_dynamic_real_count_non_nan = np.sum(~np.isnan(feat_dynamic_real))
            valid_indices, valid_lengths = get_valid_indicies_and_length(N, C, past_feat_dynamic_real)

            shapes_info['feat_dynamic_real_shape'] = (N, C, L)
            shapes_info['feat_dynamic_real_valid_value'] = int(feat_dynamic_real_count_non_nan)
            shapes_info["feat_dynamic_real_min_length"] = int(valid_lengths.min())
            shapes_info['feat_dynamic_real_max_length'] = int(valid_lengths.max())
            shapes_info["feat_dynamic_real_mean_length"] = float(valid_lengths.mean())
            shapes_info["feat_dynamic_real_valid_lengths"] = valid_lengths.tolist()
            shapes_info["feat_dynamic_real_valid_indices"] = valid_indices.tolist()
            
        # 保存元数据
        ## 其他列保留，作为其余信息
        other_info = data.drop(columns=['target', 'past_feat_dynamic_real', 'feat_dynamic_real'],
                                    errors='ignore').to_dict(orient='list')
        if 'start' in other_info: # 将时间戳转换为字符串
            other_info['start'] = [dt.isoformat() if isinstance(dt, datetime) else dt for dt in other_info['start']]
        ## 将start字段转换为字符串
        other_info.update(shapes_info)
        ## 存储
        json_file_name = f"{dataset_name}_{part_number}.json"
        json_file_path = os.path.join(output_dataset_folder, json_file_name)
        with open(json_file_path, 'w') as f:
            json.dump(other_info, f, ensure_ascii=False, indent=4)


# 处理结果说明：
# TODO