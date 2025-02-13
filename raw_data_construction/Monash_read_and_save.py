import os
import json
import numpy as np
import pandas as pd
from Monash_utils import get_monash_time_stamsp_and_values
import re


current_dir = os.path.dirname(os.path.abspath(__file__))
datasets_folder = os.path.join(current_dir, '..', 'datasets/raw_datasets/Timeseries-PILE/forecasting/monash')
output_folder = os.path.join(current_dir, '..', 'datasets/processed_datasets/Monash')
#print(os.path.abspath(datasets_folder))
# 读取project_dir/raw_data下的文件
file_names = os.listdir(datasets_folder)
dataset_names = []
for name in file_names:
    #print(name)
    if name.endswith(".tsf"):
        dataset_names.append(name[:-4])

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

print("converting {0} datasets from .tsf to pandas dataframe".format(len(dataset_names)))
count = 0
total = len(os.listdir(datasets_folder))
for dataset_name in dataset_names:
    count += 1
    print(f"Processing dataset {count}/{total}: {dataset_name}")
    #print('------A total of ' + str(total) + ' files, with the ' + str(count) + '-th currently being processed------')
    print("converting the {0} dataset".format(dataset_name))

    dataset_path = os.path.join(datasets_folder, dataset_name)
    shapes_info = {}
    # get data
    dataset_path += '.tsf'

    if dataset_name.startswith("forecasting_monash_"):
        data_real_name = dataset_name.replace("forecasting_monash_", "")
    else:
        data_real_name = dataset_name
    data_real_name = format_dataset_name(data_real_name)
    #print(data_real_name)
    #print(data_real_name, ' is Loading.')
    output_dataset_folder = os.path.join(output_folder, data_real_name)
    os.makedirs(output_dataset_folder, exist_ok=True)


    time_stamps, values, meta = get_monash_time_stamsp_and_values(dataset_path)

    shapes_info = {
        'freq': meta[0],
        'forecast_horizon' : meta[1],
        'contain_missing_values' : meta[2],
        'contain_equal_length': meta[3],
    }

    values = values.swapaxes(0, -1)
    if time_stamps is not None:
        time_stamps = time_stamps.swapaxes(0, -1)
        if time_stamps.ndim == 2:
            time_stamps = np.expand_dims(time_stamps, axis=1)
    if values.ndim == 2:
        values = np.expand_dims(values, axis=1)
    N, C, L = values.shape
    '''if True in np.isnan(values):
        pass
    else:
        values[values == 0] = np.nan'''
    values_count_non_nan = np.sum(~np.isnan(values))

    # 初始化结果数组
    valid_lengths = np.zeros((N, C, 1), dtype=int)
    valid_indices = np.zeros((N, C, 2), dtype=int)

    # 计算有效长度和索引
    for n in range(N):
        for c in range(C):
            non_nan_indices = np.where(~np.isnan(values[n, c]))[0]
            if non_nan_indices.size > 0:
                first_non_nan = non_nan_indices[0]
                last_non_nan = non_nan_indices[-1]
                valid_lengths[n, c, 0] = last_non_nan - first_non_nan + 1
                valid_indices[n, c, 0] = first_non_nan
                valid_indices[n, c, 1] = last_non_nan

    shapes_info['data_shape'] = (N, C, L)
    shapes_info['data_valid_value'] = int(values_count_non_nan)
    shapes_info["min_length"] = int(valid_lengths.min())
    shapes_info['max_length'] = int(valid_lengths.max())
    shapes_info["mean_length"] = float(valid_lengths.mean())
    shapes_info["valid_lengths"] =  valid_lengths.tolist()
    shapes_info["valid_indices"] = valid_indices.tolist()
    # save data
    np.save(os.path.join(output_dataset_folder, "{0}_values.npy".format(data_real_name)), values)
    if time_stamps is not None:
        np.savez(os.path.join(output_dataset_folder,"{0}_time_stamps.npz".format(data_real_name)), time_stamps)
    with open(os.path.join(output_dataset_folder, f'{data_real_name}.json'), 'w') as json_file:
        json.dump(shapes_info, json_file, indent=4)
