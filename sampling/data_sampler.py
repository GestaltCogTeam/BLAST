# %% [markdown]
# **名词**：
# - 归档数据集：LOTSA、Chronos、Monash、UAD、UCR等数据集
# - 小数据集：每一个归档数据集里面包含很多个子数据集
# - 数据集Part：每一个小数据集都可能是分Part存储的，每一个Part都是一个N, C, L的张量
# 
# **流程**：
# - 读取`grid_mapping`下的`grid.json`文件
# - 等概率地从不同的grid下采样时间序列，总时间序列数量为NUM_SAMPLE。产生一个List of Str，每一个元素是 DatasetPartName_Node_Channel。
# - 把上述的List转化成每个数据集的采样结果。格式是Dict，keys=DatasetPartName，values=", ".join(list of Node_Channel)。
# - 顺序读取所有数据集Part，并取出对应地被采到的那条时间序列，然后根据该时间序列的有效长度进行保存。
# - 分Part保存训练数据，避免出现内存溢出。
# - 最终效果：等价于均匀采样所有的grid, 然后均匀采样grid里面的时间序列。

# %%
import os

project_dir = os.path.abspath("")
processed_datasets_dir = project_dir + '/datasets/processed_datasets/'
grid_mapping_dir = project_dir + '/dimension_reduction/output/'

NUM_SAMPLES_ALL = 20_000_000
NUM_SAMPLES_EACH_DATA_PART = 20_000

save_dir = project_dir + '/sampling/output/' # 不区分长度，直接填充到最大长度

# %%
# 询问是否创建保存文件夹
if os.path.exists(save_dir):
    # 警告要删除文件夹
    _input = input("The directory already exists. Do you want to delete it? (y/n)")
    # 如果y
    if _input.lower() == 'y' or _input.lower() == 'yes':
        # 删除文件夹
        os.system('rm -r ' + save_dir)
    else:
        # 退出程序
        exit()

os.makedirs(save_dir, exist_ok=True)


# %% [markdown]
# ## 加载Grid Mapping

# %%
import json

print("Loading grid.json...")
with open(grid_mapping_dir + 'grid.json', 'r') as f:
    grid_data_mapping = json.load(f)


# %%
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


# %% [markdown]
# ## Cropping

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


# %%
cropped_grids = []
for i, key in enumerate(list(grid_data_mapping.keys())):
    grid_prob = grid_prob_mapping[key]
    if grid_prob < minimal_prob:
        cropped_grids.append(key)

if placed_to_grid_zero:
    cropped_data = []
    for key in cropped_grids:
        cropped_data.extend(grid_data_mapping[key])
        grid_data_mapping[key] = []
    grid_data_mapping['000__000'] = cropped_data


# %% [markdown]
# ## Sampling

# %%
import random
from tqdm import tqdm

print("Sampling data...")
valid_keys = [key for key in grid_data_mapping if grid_data_mapping[key]]
random_grids = np.random.choice(valid_keys, size=NUM_SAMPLES_ALL, replace=True)
current_index = 0
with tqdm(total=NUM_SAMPLES_ALL) as pbar:
    sampled_time_series = []
    sampled_grids = []  # grid
    while len(sampled_time_series) != NUM_SAMPLES_ALL:
        # 所有栅格等比例采样
        random_grid = random_grids[current_index]
        random_time_series = random.choice(grid_data_mapping[random_grid])
        sampled_time_series.append(random_time_series)
        sampled_grids.append(random_grid) # grid
        pbar.update(1)
        current_index += 1

# %%
# 把上述的List转化成每个数据集的采样结果。格式是Dict，keys=DatasetPartName，values=", ".join(list of Node_Channel)。
dataset_part_sampled_dict = {}
for num, item in enumerate(sampled_time_series):  # grid
    grids = sampled_grids[num] # grid
    grid_x, grid_y = grids.split('__')  # grid
    dataset_part_name, node_id, channel = item.rsplit('_', 2)
    if dataset_part_name not in dataset_part_sampled_dict: dataset_part_sampled_dict[dataset_part_name] = list()
    dataset_part_sampled_dict[dataset_part_name].append(node_id + '_' + channel + '_' + grid_x + '_' + grid_y) 

# 排序
for key in dataset_part_sampled_dict:
    if all('_' not in x for x in dataset_part_sampled_dict[key]):
        # 处理只有单纯数值的情况
        sorted_numbers = sorted(dataset_part_sampled_dict[key], key=int)
    else:
        # 处理只有数值_数值的情况
        sorted_numbers = sorted(dataset_part_sampled_dict[key], key=lambda x: (int(x.split('_')[0]), int(x.split('_')[1]), int(x.split('_')[2]), int(x.split('_')[3])))
    dataset_part_sampled_dict[key] = ','.join(sorted_numbers)

# 保存
with open(os.path.join(save_dir, 'dataset_part_sampled_dict.json'), 'w') as f:
    json.dump(dataset_part_sampled_dict, f)


# %%
import json

with open(os.path.join(save_dir, 'dataset_part_sampled_dict.json'), 'r') as f:
    dataset_part_sampled_dict = json.load(f)


# %%
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
print("number of raw datasets found: ", len(data_path_list))

data_path_list = [path for path in data_path_list if path.split('/')[-1].split(".npy")[0] in dataset_part_sampled_dict] # linux
print("number of datasets sampled: ", len(data_path_list))

print("If the number of datasets sampled is not equal to the number of datasets found, please check it.")


# %%
def read_valid_indices(data_path, dataset_part_sampled_dict):
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
import random

from tqdm import tqdm
import numpy as np

sampled_results_dict = {
    4096: []
}
sampled_results_dict_grid = {
    4096: []
}
current_part_dict = {key: 0 for key in sampled_results_dict}
num_samples_dict = {key: 0 for key in sampled_results_dict}

sampled_results_dict_validation = {
    4096: []
}
sampled_results_dict_validation_grid = {
    4096: []
}
current_part_dict_validation = {key: 0 for key in sampled_results_dict_validation}
num_samples_dict_validation = {key: 0 for key in sampled_results_dict_validation}

# Helper function to pad a sequence.
def pad_sequence(sequence, target_length):
    length = sequence.shape[0]
    return np.pad(sequence, (0, target_length - length), constant_values=np.nan)


# %%
# Helper function to pad a sequence.
def pad_sequence(sequence, target_length):
    length = sequence.shape[0]
    return np.pad(sequence, (0, target_length - length), constant_values=np.nan)

def save_data_to_file_and_clear_dict(sampled_results_dict, sampled_results_dict_grid):
    for key, value in sampled_results_dict.items():
        if len(value) > NUM_SAMPLES_EACH_DATA_PART:
            current_part = current_part_dict[key]
            value_grid = sampled_results_dict_grid[key]# grid
            np.save(os.path.join(save_dir, f'pretrain_{key}_{current_part}.npy'), np.array(value))
            np.save(os.path.join(save_dir, f'pretrain_{key}_{current_part}_grids.npy'), np.array(value_grid))# grid
            sampled_results_dict[key] = []
            sampled_results_dict_grid[key] = []
            current_part_dict[key] += 1
            # print(f"Saved pretrain_{key}_{current_part}.npy")

def save_data_to_file_and_clear_dict_validation(sampled_results_dict, sampled_results_dict_grid):
    for key, value in sampled_results_dict.items():
        if len(value) > NUM_SAMPLES_EACH_DATA_PART:
            current_part = current_part_dict_validation[key]
            value_grid = sampled_results_dict_grid[key]# grid
            np.save(os.path.join(save_dir, f'validation_{key}_{current_part}.npy'), np.array(value))
            np.save(os.path.join(save_dir, f'validation_{key}_{current_part}_grids.npy'), np.array(value_grid))# grid
            sampled_results_dict[key] = []
            sampled_results_dict_grid[key] = []
            current_part_dict_validation[key] += 1
            # print(f"Saved validation_{key}_{current_part}.npy")


# %%
pretrain_datasets_part_nc = set()
validation_datasets_part_nc = set()

for data_path in tqdm(data_path_list):
    dataset_part_name = data_path.split('/')[-1].split(".npy")[0] # linux
    if dataset_part_name not in dataset_part_sampled_dict:
        print(f"WARNING: {dataset_part_name} not in dataset_part_sampled_dict")
    indices = read_valid_indices(data_path, dataset_part_sampled_dict)
    data = np.load(data_path).astype(np.float32) # N, C, L
    # normalize
    mean = np.nanmean(data, axis=2, keepdims=True)
    std = np.nanstd(data, axis=2, keepdims=True)
    std[std == 0] = 1
    data = (data - mean) / std
    
    # 读取每一条时间序列并保存
    for time_series_nc in dataset_part_sampled_dict[dataset_part_name].split(','):
        n, c, gridx, gridy = time_series_nc.split('_')  # grid
        n = int(n)  # grid
        c = int(c)  # grid
        grid_id = gridx + '__' + gridy # grid
        start, end = indices[n][c]
        if start == end: continue  # 跳过空序列

        time_series = data[n, c, start:end + 1]
        length = time_series.shape[0]
        
        nan_count = np.isnan(time_series).sum()
        if nan_count > 5:
            continue
        
        select_datasets_part_nc = ",".join([dataset_part_name, str(n), str(c)])
        choices = [0, 1]
        weights = [0.98, 0.02] # train, validation, test
        if select_datasets_part_nc in pretrain_datasets_part_nc:
            random_choice = 0
        elif select_datasets_part_nc in validation_datasets_part_nc:
            random_choice = 1
        else:
            # 如果不在任何一个集合中，随机选择一个集合
            random_choice = random.choices(choices, weights, k=1)[0]
            if random_choice == 0:
                pretrain_datasets_part_nc.add(select_datasets_part_nc)
            elif random_choice == 1:
                validation_datasets_part_nc.add(select_datasets_part_nc)
            else:
                raise NotImplementedError
        
        choice_2_sampled_results = {0: sampled_results_dict, 1: sampled_results_dict_validation}
        choice_2_sampled_results_grid = {0: sampled_results_dict_grid, 1: sampled_results_dict_validation_grid} # grid
        choice_2_num_samples = {0: num_samples_dict, 1: num_samples_dict_validation}
        
        if length < 4096:
            # padding
            context = pad_sequence(time_series, 4096)
        else:
            start_index = random.randint(0, length - 4096)
            context = time_series[start_index:start_index + 4096]
        assert context.shape[0] == 4096
        choice_2_sampled_results[random_choice][4096].append(context)
        choice_2_sampled_results_grid[random_choice][4096].append(grid_id)
        choice_2_num_samples[random_choice][4096] += 1

        save_data_to_file_and_clear_dict(sampled_results_dict, sampled_results_dict_grid)
        save_data_to_file_and_clear_dict_validation(sampled_results_dict_validation, sampled_results_dict_validation_grid)
    del data

# %%
# 保存剩余的数据
for key, value in sampled_results_dict.items():
    current_part = current_part_dict[key]
    value_grid = sampled_results_dict_grid[key]
    np.save(os.path.join(save_dir, f'pretrain_{key}_{current_part}.npy'), np.array(value))
    np.save(os.path.join(save_dir, f'pretrain_{key}_{current_part}_grids.npy'), np.array(value_grid))
    sampled_results_dict[key] = []
    current_part_dict[key] += 1

for key, value in sampled_results_dict_validation.items():
    current_part = current_part_dict_validation[key]
    value_grid = sampled_results_dict_validation_grid[key]
    np.save(os.path.join(save_dir, f'validation_{key}_{current_part}.npy'), np.array(value))
    np.save(os.path.join(save_dir, f'validation_{key}_{current_part}_grids.npy'), np.array(value_grid))
    sampled_results_dict_validation[key] = []
    current_part_dict_validation[key] += 1

# %%
# save pretrain_datasets_part_nc, validation_datasets_part_nc, test_datasets_part_nc as pickle
import pickle
with open(os.path.join(save_dir, 'pretrain_datasets_part_nc.pkl'), 'wb') as f:
    pickle.dump(pretrain_datasets_part_nc, f)
with open(os.path.join(save_dir, 'validation_datasets_part_nc.pkl'), 'wb') as f:
    pickle.dump(validation_datasets_part_nc, f)

# %%



