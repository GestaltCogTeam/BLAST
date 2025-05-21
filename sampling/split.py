
import os

import numpy as np

project_dir = os.path.abspath("")

NUM_SAMPLES_ALL = 20_000_000 # 20_000_000
CONTEXT_LENGTH = 4096

final_result = np.memmap(project_dir + '/sampling/output/final_result.dat', dtype='float32', mode='r', shape=(NUM_SAMPLES_ALL, CONTEXT_LENGTH))

# split final_result
train_size = int(0.99 * NUM_SAMPLES_ALL)
valid_size = int(0.01 * NUM_SAMPLES_ALL)
# random index
random_index = np.random.permutation(NUM_SAMPLES_ALL)
train_index = random_index[:train_size]
valid_index = random_index[train_size:train_size + valid_size]

print(f"train size: {train_size}, valid size: {valid_size}")
# train set
train_set = final_result[train_index]
# save to memmap
train_set_memmap = np.memmap(project_dir + '/sampling/train/data.dat', dtype='float32', mode='w+', shape=(train_size, CONTEXT_LENGTH))
train_set_memmap[:] = train_set[:]
train_set_memmap.flush()
# save shape
np.save(project_dir + '/sampling/train/shape.npy', train_set.shape)
print(f"train set shape: {train_set.shape}")
# valid set
valid_set = final_result[valid_index]
# save to memmap
valid_set_memmap = np.memmap(project_dir + '/sampling/valid/data.dat', dtype='float32', mode='w+', shape=(valid_size, CONTEXT_LENGTH))
valid_set_memmap[:] = valid_set[:]
valid_set_memmap.flush()
# save shape
np.save(project_dir + '/sampling/valid/shape.npy', valid_set.shape)
print(f"valid set shape: {valid_set.shape}")