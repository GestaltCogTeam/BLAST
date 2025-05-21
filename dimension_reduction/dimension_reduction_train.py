# %%
import os
import numpy as np
import matplotlib.pyplot as plt

project_dir = os.path.abspath("")
vector_dir = project_dir + '/feature_construction/output/'
save_dir = project_dir + '/dimension_reduction/grid_mapping/'
save_scaled_reduced_data_dir = save_dir + 'scaled_reduced_data/'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_scaled_reduced_data_dir, exist_ok=True) # 存储降维之后数据的目录

RESOLUTION = 0.01
SAMPLING_RATE = 0.1
GRID_SIZE = int(1 / RESOLUTION)
MARK = f"0{len(str(GRID_SIZE))}"

# %%
# 按照数据比例在不同的数据集中等比例抽样
from tqdm import tqdm
current_index = 0
dataset_range = {}
all_data = []
for dataset_part in tqdm(os.listdir(vector_dir)):
    data = np.load(os.path.join(vector_dir, dataset_part)).astype(np.float32)
    n, c, l = data.shape
    dataset_range[dataset_part] = (current_index, current_index + n * c)
    all_data.append(data.reshape(n * c, l))
    current_index += n * c
all_data_np = np.concatenate(all_data, axis=0)
rows_with_nan = np.any(np.isnan(all_data_np), axis=1)
nan_rows_indices = np.where(rows_with_nan)[0]
all_data_np = np.delete(all_data_np, nan_rows_indices, axis=0)
del all_data

# %%
# 采样
NUM_SAMPLES = int(all_data_np.shape[0] * SAMPLING_RATE) # 原始数据4100w左右，采样2%，大概80w
sample_idx = np.random.choice(all_data_np.shape[0], NUM_SAMPLES, replace=False)
sample_data = all_data_np[sample_idx]

print(f"sampling {NUM_SAMPLES} data from {all_data_np.shape[0]} data, {NUM_SAMPLES/all_data_np.shape[0]*100:.2f}%")

del all_data_np


# %%
import umap
from sklearn.preprocessing import MinMaxScaler

umap_args = {'n_components': 2, 'n_neighbors': 200, 'min_dist': 0.9, 'metric': 'hamming', 'n_jobs': -1, 'init': 'random', 'verbose': True, 'low_memory': False}

suffix = '_'.join([f"{key}={value}" for key, value in umap_args.items()])
suffix += f"_sample_rate={SAMPLING_RATE}"

# %%
model = umap.UMAP(**umap_args)
scaler = MinMaxScaler()

print("Dimension reduction...")
reduced_data = model.fit_transform(sample_data)
normalized_data = scaler.fit_transform(reduced_data)
np.save(save_dir + f"normalized_data_{suffix}.npy", normalized_data)


# %%
# save model
import pickle
model_path = save_dir + f"umap_model_{suffix}.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
    print(f"Model saved to {model_path}")
# save scaler
scaler_path = save_dir + f"scaler_{suffix}.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
    print(f"Scaler saved to {scaler_path}")
