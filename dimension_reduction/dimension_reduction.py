# %%
import os
import numpy as np
import matplotlib.pyplot as plt

project_dir = os.path.abspath("")
vector_dir = project_dir + '/feature_construction/output/'
save_dir = project_dir + '/dimension_reduction/output/'
save_scaled_reduced_data_dir = save_dir + 'scaled_reduced_data/'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_scaled_reduced_data_dir, exist_ok=True) # 存储降维之后数据的目录

RESOLUTION = 0.01
SAMPLING_RATE = 0.1
GRID_SIZE = int(1 / RESOLUTION)
MARK = f"0{len(str(GRID_SIZE))}"

umap_args = {'n_components': 2, 'n_neighbors': 200, 'min_dist': 0.9, 'metric': 'hamming', 'n_jobs': -1, 'init': 'random', 'verbose': True, 'low_memory': False}

suffix = '_'.join([f"{key}={value}" for key, value in umap_args.items()])
suffix += f"_sample_rate={SAMPLING_RATE}"

# %%
# load model
import pickle
model_path = save_dir + f"umap_model_{suffix}.pkl"
with open(model_path, 'rb') as f:
    model = pickle.load(f)
    print(f"Model loaded from {model_path}")
# save scaler
scaler_path = save_dir + f"scaler_{suffix}.pkl"
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)
    print(f"Scaler loaded from {scaler_path}")

# %%
import json

# 遍历数据集Part的向量，并把降维之后的数据写入栅格中
def init_grid():
    grid = {}
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            key = f"{i:{MARK}d}__{j:{MARK}d}"
            grid[key] = []
    return grid

def process_dataset_part_vector(dataset_part_path, model, scaler):
    # 读取数据并降维
    dataset_part_vector = np.load(dataset_part_path)
    n, c, l = dataset_part_vector.shape
    scaled_reduced_data_all = np.zeros((n*c, 2)) * np.nan

    dataset_part_vector = dataset_part_vector.reshape(n*c, l)
    rows_with_nan = np.any(np.isnan(dataset_part_vector), axis=1)
    nan_rows_indices = np.where(rows_with_nan)[0] # nan的行索引
    non_nan_rows_indices = np.where(~rows_with_nan)[0] # 非nan的行索引
    dataset_part_vector = dataset_part_vector[non_nan_rows_indices]
    if dataset_part_vector.shape[0] == 0:
        print(f"{dataset_part_path} has no data, skip")
        return None
    reduced_data = model.transform(dataset_part_vector)
    scaled_reduced_data = scaler.transform(reduced_data)
    scaled_reduced_data_all[non_nan_rows_indices] = scaled_reduced_data
    
    scaled_reduced_data_all = scaled_reduced_data_all.reshape(n, c, 2)
    save_path = save_scaled_reduced_data_dir + dataset_part_path.split("/")[-1]
    np.save(save_path, scaled_reduced_data_all)

    # 构建该数据集Part的栅格
    # 初始化
    dataset_part_grid = init_grid()
    # 统计grid中的数据
    for i in range(n):
        for j in range(c):
            x, y = scaled_reduced_data_all[i, j] # 降维之后的坐标
            assert (np.isnan(x) and np.isnan(y)) or (not np.isnan(x) and not np.isnan(y))
            if np.isnan(x) and np.isnan(y): continue
            x_idx = int(x * (GRID_SIZE - 1)) # x栅格坐标
            y_idx = int(y * (GRID_SIZE - 1)) # y栅格坐标
            # 处理超出边界的情况
            if x_idx >= GRID_SIZE: x_idx = GRID_SIZE - 1
            if y_idx >= GRID_SIZE: y_idx = GRID_SIZE - 1
            if x_idx < 0: x_idx = 0
            if y_idx < 0: y_idx = 0
            key = f"{x_idx:{MARK}d}__{y_idx:{MARK}d}"
            dataset_part_name = dataset_part_path.split("/")[-1].split(".npy")[0]
            value = f"{dataset_part_name}_{i}_{j}"
            dataset_part_grid[key].append(value)

    return dataset_part_grid

# 顺序读取所有的数据集Part向量
# 单线程即可，因为UMap是多线程的，多线程+多线程会导致开销很大，反而不如单线程
from tqdm import tqdm
dataset_part_grids = []
for dataset_part_vector_path in tqdm(os.listdir(vector_dir)):
    dataset_part_grid = process_dataset_part_vector(os.path.join(vector_dir, dataset_part_vector_path), model, scaler)
    dataset_part_grids.append(dataset_part_grid)

dataset_part_grids = [x for x in dataset_part_grids if x is not None]

# 合并字典字典
from tqdm import tqdm
grid_mapping = init_grid()
for dataset_part_grid in tqdm(dataset_part_grids):
    for key, value in dataset_part_grid.items():
        grid_mapping[key].extend(value)

# 保存grid_mapping
grid_mapping_path = os.path.join(save_dir, "grid.json")
with open(grid_mapping_path, "w") as f:
    print(f"Grid mapping saved to {grid_mapping_path}")
    json.dump(grid_mapping, f)
