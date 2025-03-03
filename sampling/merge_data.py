# %%
import os
project_dir = os.path.abspath("")
output_dir = project_dir + "/sampling/output"
save_dir = project_dir + "/sampling/output"


# %%
# 获取output下的文件
file_list = os.listdir(output_dir)
file_list = [_ for _ in file_list if _.endswith(".npy") and "grids" not in _ and "pretrain" in _]
file_list = [_.split(".")[0] for _ in file_list]
file_list = [_.split("_")[-1] for _ in file_list]
max_data_part_number_train = max([int(_) for _ in file_list])

file_list = os.listdir(output_dir)
file_list = [_ for _ in file_list if _.endswith(".npy") and "grids" not in _ and "pretrain" not in _]
file_list = [_.split(".")[0] for _ in file_list]
file_list = [_.split("_")[-1] for _ in file_list]
max_data_part_number_validation = max([int(_) for _ in file_list])


# %%
import numpy as np

def get_npy_shape(file_path):
    with open(file_path, 'rb') as f:
        # 从文件头解析 shape
        version = np.lib.format.read_magic(f)
        shape, _, _ = np.lib.format._read_array_header(f, version)
    return shape

# %%
# merge train data
from tqdm import tqdm
num_samples = 0
for i in tqdm(range(max_data_part_number_train)):
    file_name = f"{output_dir}/pretrain_4096_{i}.npy"
    shape = get_npy_shape(file_name)
    num_samples += shape[0]

# create memmap
data = np.memmap(f"{save_dir}/train.dat", dtype='float32', mode='w+', shape=(num_samples, 4096))
current_index = 0

# merge train data
for i in tqdm(range(max_data_part_number_train)):
    file_name = f"{output_dir}/pretrain_4096_{i}.npy"
    data_i = np.load(file_name)
    shape = data_i.shape
    data[current_index:current_index + shape[0]] = data_i
    current_index += shape[0]

np.save(f"{save_dir}/train_shape.npy", (num_samples, 4096))

# %%
# merge validation data
num_samples = 0
for i in tqdm(range(max_data_part_number_validation)):
    file_name = f"{output_dir}/validation_4096_{i}.npy"
    shape = get_npy_shape(file_name)
    num_samples += shape[0]

# create memmap
data = np.memmap(f"{save_dir}/valid.dat", dtype='float32', mode='w+', shape=(num_samples, 4096))
current_index = 0

# merge validation data
for i in tqdm(range(max_data_part_number_validation)):
    file_name = f"{output_dir}/validation_4096_{i}.npy"
    data_i = np.load(file_name)
    shape = data_i.shape
    data[current_index:current_index + shape[0]] = data_i
    current_index += shape[0]

np.save(f"{save_dir}/valid_shape.npy", (num_samples, 4096))
