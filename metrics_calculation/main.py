# %%
import os
import numexpr as ne
ne.set_num_threads(64)  # 设置为 64 线程

# 根据__file__获取当前文件的路径，然后取上一级目录
project_dir = os.path.abspath("")
processed_datasets_dir = project_dir + '/datasets/processed_datasets'
save_dir = project_dir + '/metrics_calculation/output/'
os.makedirs(save_dir, exist_ok=True)

NUM_JOBS = -1
MIN_LEN = 512

# %%
import logging

logging.basicConfig(filename=save_dir + 'output.log', level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler())

# %%
data_path_list = []
# 递归地读取 processed_datasets_dir下面的所有文件，并找到以npy结尾的文件
for root, dirs, files in os.walk(processed_datasets_dir):
    for file in files:
        if file.endswith(".npy"):
            if 'TEST' not in file and 'test' not in file and 'label' not in file:
                if 'ETTh' not in file and 'ETTm' not in file and 'Weather_autoformer' not in file: ### 11.26改--li
                    data_path_list.append(os.path.join(root, file))
                else:
                    logger.info('skip ' + file)
            else:
                # data_path_list.append(os.path.join(root, file))
                pass
logger.info('total datasets: ' + str(len(data_path_list)))

# %%
import numpy as np
from tqdm import tqdm

def get_npy_shape(file_path):
    with open(file_path, 'rb') as f:
        # 从文件头解析 shape
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape


n_list = []
c_list = []
l_list = []
path_list = []

for i, path in enumerate(tqdm(data_path_list)):
    dataset_part_name = path.split("/")[-1].split(".npy")[0]
    n, c, l = get_npy_shape(path)
    n_list.append(n)
    c_list.append(c)
    l_list.append(l)
    path_list.append(path)

nc_list = [n*c for n, c in zip(n_list, c_list)]
# 将 nc_list 和 path_list 组合在一起
combined_list = list(zip(nc_list, path_list, l_list))

# 对组合后的列表按 nc_list 的值进行排序
combined_list.sort(key=lambda x: x[0])

# 将排序后的列表解压回 nc_list 和 path_list
nc_list_sorted, path_list_sorted, l_list_sorted = zip(*combined_list)

# 如果需要将它们转换回列表
nc_list_sorted = list(nc_list_sorted)
l_list_sorted = list(l_list_sorted)
data_path_list = list(path_list_sorted) 

# %%

data_path_list_filtered = []
nc_list_sorted_filtered = []
l_list_sorted_filtered = []
for nc, l, path in zip(nc_list_sorted, l_list_sorted, data_path_list):
    if l >= MIN_LEN: # 初筛
        data_path_list_filtered.append(path)
        nc_list_sorted_filtered.append(nc)
        l_list_sorted_filtered.append(l)

data_path_list = data_path_list_filtered
nc_list_sorted = nc_list_sorted_filtered
l_list_sorted = l_list_sorted_filtered

len(data_path_list)


# %%

import json
import numpy as np
from joblib import Parallel, delayed
from compute_ts_indicators import process_time_series # 核心接口


def evaluate_dataset_part(path, save_dir):
    """评估一个数据集Part中的所有序列, 并将其存储下来。

    Args:
        path (str): 数据集Part的路径
        name (str): 数据集Part的名称
        save_dir (str): 保存文件夹路径
    """
    dataset_part_name = path.split("/")[-1].split(".npy")[0]
    result_path = os.path.join(save_dir, f"{dataset_part_name}.json")
    result_shape_path = os.path.join(save_dir, f"{dataset_part_name}_shape.npy")

    if os.path.exists(result_path) and os.path.exists(result_shape_path):
        print(f"{dataset_part_name} and its shape already exist.")
        return

    # load data
    data = np.load(path).astype(np.float32)
    n, c, l = data.shape
    if not os.path.exists(result_shape_path):
        np.save(result_shape_path, np.array([n, c, l]))
    if os.path.exists(result_path):
        print(f"{result_path} already exists.")
        return
    delayed_tasks = []
    for node_id in range(n):
        for channel in range(c):
            delayed_tasks.append(delayed(process_time_series)(data[node_id, channel], dataset_part_name, node_id, channel))
    # process
    results = Parallel(n_jobs=NUM_JOBS)(delayed_tasks) # list of dicts. key: {dataset_part_name}_{node_id}_{channel}, value: dict
    # save each element of results to json file
    with open(result_path, "w", encoding='utf-8') as f:
        for _ in results:
            if list(_.values())[0] is None:
                logger.info(f"Skipping {list(_.keys())[0]}, since it is too short.")
                continue
            json.dump(_, f)
            f.write("\n")

# %%

import time
total = len(data_path_list)
for i, data_path in enumerate(data_path_list[::-1]):
    logger.info(f"{time.strftime('%Y-%m-%d %H:%M:%S')} Processing {i+1}/{total}: {data_path}")
    evaluate_dataset_part(data_path, save_dir)

# %%
