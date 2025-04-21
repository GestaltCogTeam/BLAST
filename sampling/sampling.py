# %%

import os
import random
import numpy as np

project_dir = os.path.abspath("")
processed_datasets_dir = project_dir + '/datasets/processed_datasets/'
grid_mapping_dir = project_dir + '/dimension_reduction/output/'

NUM_SAMPLES_ALL = 20_000_000 # 20_000_000
CONTEXT_LENGTH = 4096

save_dir = project_dir + '/sampling/output/' # 不区分长度，直接填充到最大长度

os.makedirs(save_dir, exist_ok=True)

final_result = np.memmap(save_dir + 'final_result.dat', dtype='float32', mode='w+', shape=(NUM_SAMPLES_ALL, CONTEXT_LENGTH))

import json

print("Loading grid.json...")
with open(grid_mapping_dir + 'grid.json', 'r') as f:
    grid_data_mapping = json.load(f)


# crop t
import numpy as np

num_samples = []
for key in list(grid_data_mapping.keys()):
    num_samples.append(len(grid_data_mapping[key]))

num_samples = np.array(num_samples)
num_samples_prob = num_samples / np.sum(num_samples)

grid_prob_mapping = {}
for i, key in enumerate(list(grid_data_mapping.keys())):
    grid_prob_mapping[key] = num_samples_prob[i]


# %%
placed_to_grid_zero = True

grid_prob_array = np.array(list(grid_prob_mapping.values()))

if placed_to_grid_zero:
    assert grid_prob_array[0] == 0

def get_long_tail_point(eps, grid_prob_array):
    grid_prob_array_sort = np.sort(grid_prob_array)
    accumulate = 0
    for i, prob in enumerate(grid_prob_array_sort):
        accumulate += prob
        if accumulate > eps:
            index = len(grid_prob_array) - i - 1
            prob = prob
            return index, prob

index, minimal_prob = get_long_tail_point(0.001, grid_prob_array)

cropped_grids = []
for i, key in enumerate(list(grid_data_mapping.keys())):
    grid_prob = grid_prob_mapping[key]
    if grid_prob < minimal_prob:
        cropped_grids.append(key)

# placed_to_grid_zero
cropped_data = []
cropped_prob = 0
for key in cropped_grids:
    cropped_data.extend(grid_data_mapping[key])
    cropped_prob += grid_prob_mapping[key]
    grid_data_mapping[key] = []
    grid_prob_mapping[key] = 0

grid_data_mapping['000__000'] = cropped_data
grid_prob_mapping['000__000'] = cropped_prob


# %%
def pad_sequence(sequence, target_length):
    length = sequence.shape[0]
    return np.pad(sequence, (0, target_length - length), constant_values=np.nan)

def get_data_part_name_to_path_mapping():

    def name_exclusion(file):
        if file.endswith(".npy") and 'TEST' not in file and 'test' not in file and 'label' not in file and 'ETTh' not in file and 'ETTm' not in file:
            return True
        else:
            return False

    data_path_list = []
    # 递归地读取 processed_datasets_dir下面的所有文件，并找到以npy结尾的文件
    for root, dirs, files in os.walk(processed_datasets_dir):
        for file in files:
            if name_exclusion(file):
                data_path_list.append(os.path.join(root, file))

    data_part_name_to_path_mapping = {}
    for _ in data_path_list:
        data_part_name = _.split('/')[-1].split(".npy")[0]
        data_part_name_to_path_mapping[data_part_name] = _
    return data_part_name_to_path_mapping

data_part_name_to_path_mapping = get_data_part_name_to_path_mapping()

# %%
def read_valid_indices(data_path):
    def load_indices(desc_path, mode):
        with open(desc_path, "r") as f:
            desc = json.load(f)
        if mode is None:
            return desc["valid_indices"]
        elif '_past_feat_dynamic_real' == mode:
            return desc['past_feat_dynamic_real_valid_indices']
        elif '_feat_dynamic_real' == mode:
            return desc['feat_dynamic_real_valid_indices']
        else:
            raise NotImplementedError

    dataset_part_name = data_path.split('/')[-1].split(".npy")[0] # linux
    if os.path.exists(data_path.replace(".npy", ".json")):
        # 一般来说，json的name等于npy的name
        desc_path = data_path.replace(".npy", ".json")
        return load_indices(desc_path, None)
    elif '_past_feat_dynamic_real' in dataset_part_name: # LOTSA的past_feat_dynamic_real读取
        desc_name = dataset_part_name.split('_past_feat_dynamic_real')[0] + dataset_part_name.split('_past_feat_dynamic_real')[1] + ".json"
        # 获取data_path上一级目录
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, '_past_feat_dynamic_real')
    elif '_feat_dynamic_real' in dataset_part_name: # LOTSA的'_feat_dynamic_real'读取
        desc_name = dataset_part_name.split('_feat_dynamic_real')[0] + dataset_part_name.split('_past_feat_dynamic_real')[1] + ".json"
        # 获取data_path上一级目录
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, '_feat_dynamic_real')
    elif '_values' in dataset_part_name:
        desc_name = dataset_part_name.split('_values')[0] + ".json"
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, None)
    elif '_data' in dataset_part_name: # UCR和UAD
        desc_name = dataset_part_name.split('_data')[0] + ".json"
        desc_path = os.path.join(os.path.dirname(data_path), desc_name)
        return load_indices(desc_path, None)
    else:
        assert False, "dataset_part_name: {}".format(dataset_part_name)

# %%
# grid_data_mapping: grid id -> dataset
# grid_prob_mapping: grid id -> prob

# %%
valid_data = [grid_data_mapping[key] for key in grid_data_mapping if grid_data_mapping[key]]
valid_prob = [grid_prob_mapping[key] for key in grid_prob_mapping if grid_data_mapping[key]]
avg_prob = [1/len(valid_data) for _ in valid_prob]
valid_size = [len(_) for _ in valid_data]

# %%
def get_current_prob(p_start, p_end, t, T=NUM_SAMPLES_ALL):
    assert t <= T and t >= 0
    # linear interpolation
    p_start = np.array(p_start)
    p_end = np.array(p_end)
    return p_start + (p_end - p_start) * t / T

# %%
# batch size

sample_list = []

def get_a_sample(t):
    # 获取当前的概率
    # current_prob = get_current_prob(p_start=avg_prob, p_end=valid_prob, t=t, T=NUM_SAMPLES_ALL)
    current_prob = avg_prob
    current_prob = np.array(current_prob)
    # 以current_prob有放回地随机采样
    selected_grid = np.random.choice(range(len(avg_prob)), p=current_prob)
    selected_idx = np.random.choice(valid_size[selected_grid])
    selected_data = valid_data[selected_grid][selected_idx]
    return selected_data

from tqdm import tqdm

sampled_time_series = []
for t in tqdm(range(NUM_SAMPLES_ALL)):
    sampled_time_series.append(get_a_sample(t))

# %%
# save sampled_time_series
import pickle
with open(save_dir + 'sampled_time_series.pkl', 'wb') as f:
    pickle.dump(sampled_time_series, f)

# # load sampled_time_series
# import pickle
# with open(save_dir + 'sampled_time_series.pkl', 'rb') as f:
#     sampled_time_series = pickle.load(f)

# %%
from tqdm import tqdm
data_part_name_list = set()

for _ in tqdm(sampled_time_series):
    data_part_name, node_id, channel = _.rsplit('_', 2)
    data_part_name_list.add(data_part_name)

result_mapping = {}
for _ in data_part_name_list:
    result_mapping[_] = {}

for idx, _ in enumerate(tqdm(sampled_time_series)):
    data_part_name, node_id, channel = _.rsplit('_', 2)
    if node_id + '_' + channel not in result_mapping[data_part_name]:
        result_mapping[data_part_name][node_id + '_' + channel] = []
    result_mapping[data_part_name][node_id + '_' + channel].append(idx)

# %%
data_part_names = list(result_mapping.keys())

# %%
for i, data_part_name in enumerate(tqdm(data_part_names)):
    # print(f"Processing {i+1}/{len(data_part_names)}: {data_part_name}")
    
    nc_mapping = result_mapping[data_part_name]
    
    # read data
    selected_data_path = data_part_name_to_path_mapping[data_part_name]
    data = np.load(selected_data_path).astype(np.float32)
    indices = read_valid_indices(selected_data_path)
    mean = np.nanmean(data, axis=2, keepdims=True)
    std = np.nanstd(data, axis=2, keepdims=True)
    std[std == 0] = 1
    data = (data - mean) / std
    
    # read data
    for key, value in nc_mapping.items():
        node_id, channel = key.split('_')
        node_id = int(node_id)
        channel = int(channel)
        start_idx, end_idx = indices[node_id][channel]
        time_series = data[node_id, channel, start_idx:end_idx]
        for idx in value:
            if len(time_series) < CONTEXT_LENGTH:
                context = pad_sequence(time_series, CONTEXT_LENGTH)
            else:
                start_index = random.randint(0, len(time_series) - CONTEXT_LENGTH)
                context = time_series[start_index:start_index + CONTEXT_LENGTH]
            final_result[idx] = context
