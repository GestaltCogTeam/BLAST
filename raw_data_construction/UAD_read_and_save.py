# Description: For the UAD dataset.


import os
import pyarrow as pa
import numpy as np
import json
from datetime import datetime
import pandas as pd
import re
# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义数据集文件夹路径
datasets_folder = os.path.join(current_dir, '..', 'datasets/raw_datasets/Timeseries-PILE/anomaly_detection/TSB-UAD-Public')
output_folder = os.path.join(current_dir, '..', 'datasets/processed_datasets/UAD')
interpolate = False


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
    formatted_name = re.sub(r'(\d)_+(\d)', r'\1\2', formatted_name)
    formatted_name = re.sub(r'_{2,}', '_', formatted_name)

    return formatted_name

count = 0
total = len(os.listdir(datasets_folder))
# 遍历数据集文件夹
for dataset_name in os.listdir(datasets_folder):
    count += 1
    dataset_path = os.path.join(datasets_folder, dataset_name)
    if os.path.isdir(dataset_path):
        # 创建输出文件夹
        dataset_name = format_dataset_name(dataset_name)
        print(f"Processing dataset {count}/{total}: {dataset_name}")
        output_dataset_folder = os.path.join(output_folder, dataset_name)
        os.makedirs(output_dataset_folder, exist_ok=True)

        # 初始化shape信息
        shapes_info = {}

        # 遍历Arrow文件
        for file_name in os.listdir(dataset_path):
            if file_name.endswith('.out'):
                file_path = os.path.join(dataset_path, file_name)
                df = pd.read_csv(file_path)
                if interpolate:
                    df.interpolate(inplace=True, method="cubic")
                # 分离时序数据和标签
                #print(df.shape)
                #print(df.iloc[:, 0].to_numpy().shape)
                time_series = df.iloc[:, 0].to_numpy().reshape(1, 1, -1)
                labels = df.iloc[:, 1].to_numpy().reshape(1, 1, -1)

                N, C, L = time_series.shape

                # 初始化结果数组
                valid_lengths = np.zeros((N, C, 1), dtype=int)
                valid_indices = np.zeros((N, C, 2), dtype=int)

                # 计算有效长度和索引
                for n in range(N):
                    for c in range(C):
                        non_nan_indices = np.where(~np.isnan(time_series[n, c]))[0]
                        if non_nan_indices.size > 0:
                            first_non_nan = non_nan_indices[0]
                            last_non_nan = non_nan_indices[-1]
                            valid_lengths[n, c, 0] = last_non_nan - first_non_nan + 1
                            valid_indices[n, c, 0] = first_non_nan
                            valid_indices[n, c, 1] = last_non_nan

                # 去除文件名前缀
                head_info = output_dataset_folder.split("/")[-1]
                new_file_name = head_info + '@@' + file_name.replace('anomaly_detection_TSB-UAD-Public_', '')
                npy_file_path = os.path.join(output_dataset_folder, new_file_name.replace('.out', '.npy'))
                label_file_path = os.path.join(output_dataset_folder, new_file_name.replace('.out', '_label.npy'))



                #print(target)
                data_count_non_nan = np.sum(~np.isnan(time_series))

                # 保存时序数据和标签为npy文件
                np.save(npy_file_path, time_series)
                np.save(label_file_path, labels)

                # 保存shape信息和标签范围到json文件
                shapes_info = {
                    'phase': 'TRAIN' if 'train' in file_name else 'TEST' if 'test' in file_name else 'NONE',
                    'data_shape': time_series.shape,
                    'label_min': int(labels.min()),
                    'label_max': int(labels.max()),
                    'data_valid_value': int(data_count_non_nan),
                    "min_length": int(valid_lengths.min()),
                    'max_length':int(valid_lengths.max()),
                    "mean_length":float(valid_lengths.mean()),
                    "valid_lengths": valid_lengths.tolist(),
                    "valid_indices": valid_indices.tolist()
                }
                json_file_path = os.path.join(output_dataset_folder, new_file_name.replace('.out', '.json'))
                with open(json_file_path, 'w') as json_file:
                    json.dump(shapes_info, json_file, indent=4)
