# %%
import numpy as np
from tqdm import tqdm


# %%

# 超参数
k_max = 3  # 用于生成的时间序列数量
alpha = 1.5
num_samples = 30_000_000
context_length = 4096
batch_size = 10_000  # 每批生成的样本数量
raw_data_dir = "/PATH/TO/DATA/"
save_data_dir = "/PATH/TO/SAVE/"
save_path = save_data_dir + "data.dat"


# %%
# 预分配内存映射文件
augmented_data = np.memmap(
    save_path, dtype='float32', mode='w+', shape=(num_samples, context_length)
)


# %% [markdown]
# ## Read Raw Sampled Data

# %%
# # 加载原始数据
shape = np.load(raw_data_dir + "shape.npy")
data = np.memmap(raw_data_dir + "data.dat", mode="r", shape=tuple(shape), dtype=np.float32) # N, L


# %%
# valid index: 0 ~ N - 1
N = shape[0]
valid_idx = np.arange(N)

# %% [markdown]
# ## TSMixup (from Chronos)

# %%

# 批量生成数据
num_batches = num_samples // batch_size
remaining_samples = num_samples % batch_size

def generate_batch(batch_size):
    # 随机生成所需索引和权重
    k = np.random.randint(1, k_max + 1)
    sampled_indices = np.random.choice(valid_idx, size=(batch_size, k), replace=True)
    weights = np.random.dirichlet([alpha] * k, size=batch_size)

    # 提取时间序列并计算加权和
    ## 采样时间序列
    time_series_batch = data[sampled_indices]  # Shape: (batch_size, k, context_length)
    ## 归一化
    mean = np.nanmean(time_series_batch, axis=2, keepdims=True)
    std = np.nanstd(time_series_batch, axis=2, keepdims=True)
    std[std == 0] = 1
    time_series_batch = (time_series_batch - mean) / std
    ## 加权求和
    augmented_batch = np.einsum("bk, bkl -> bl", weights, time_series_batch)
    return augmented_batch

# 主循环
for batch_idx in tqdm(range(num_batches)):
    start_idx = batch_idx * batch_size
    end_idx = start_idx + batch_size
    augmented_batch = generate_batch(batch_size)
    augmented_data[start_idx:end_idx] = augmented_batch

# 处理剩余样本
if remaining_samples > 0:
    start_idx = num_batches * batch_size
    augmented_batch = generate_batch(remaining_samples)
    augmented_data[start_idx:] = augmented_batch

# 确保数据写入磁盘
augmented_data.flush()


# %%
shape = (num_samples, context_length)
with open(save_data_dir + "shape.npy", "wb") as f:
    np.save(f, shape)


# %%
import numpy as np
shape = np.load(save_data_dir + "shape.npy")
memmap_data = np.memmap(save_path, dtype=np.float32, shape=tuple(shape), mode='r')

# %%
import random
random_idx = random.randint(0, shape[0])
random_sample = memmap_data[random_idx]
import matplotlib.pyplot as plt
plt.plot(random_sample)
plt.show()


# %%


# %%



