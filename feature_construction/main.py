# %%
import os

project_dir = os.path.abspath("")
metrics_dir = project_dir + '/metrics_calculation/output/'
save_dir = project_dir + '/feature_construction/output/'

os.makedirs(save_dir, exist_ok=True)

# 获取metrics_dir下的所有文件名
file_list = os.listdir(metrics_dir)
file_list = [_ for _ in file_list if _.endswith(".json")]

# %%
import json

def read_json(file_path):
    """读取数据集Part对应的json文件, 返回list of dict"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def extract_metrics(data, path=None, result=None):
    if path is None:
        path = []
    if result is None:
        result = {}

    for key, value in data.items():
        new_path = path + [key]
        if isinstance(value, dict):
            extract_metrics(value, new_path, result)
        else:
            path_str = '-'.join(new_path)
            # print(path_str)
            if path_str not in result:
                result[path_str] = []
            result[path_str].append(value)
    return result


# %%
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from construct_feature_vector import transform_metrics_to_vector
import os

# 假设read_json, extract_metrics, transform_metrics_to_vector等函数已经定义
def process_file(file_name, metrics_dir, save_dir):
    shape_file_name = file_name.split(".json")[0] + "_shape.npy"
    shape = np.load(metrics_dir + shape_file_name)
    n, c, l = shape
    
    vector_tensor = np.full((n, c, 61), np.nan)  # N, C, L
    
    data = read_json(metrics_dir + file_name)

    for _ in data:
        current_node, current_channel = list(_.keys())[0].split('_')[-2:]
        metrics = extract_metrics(_[list(_.keys())[0]])
        vector = transform_metrics_to_vector(metrics)
        vector_tensor[int(current_node), int(current_channel)] = vector
    # save the feature vector
    np.save(save_dir + file_name.replace("json", "npy"), vector_tensor)

def parallel_process(file_list, metrics_dir, save_dir, num_workers=None):
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_file, file_name, metrics_dir, save_dir) for file_name in file_list]
        for _ in tqdm(futures):
            _.result()  # 等待所有的进程完成

# 定义文件列表、路径等
parallel_process(file_list, metrics_dir, save_dir, num_workers=96)  # 使用8个进程并行处理
