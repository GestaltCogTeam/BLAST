# Run a baseline model in BasicTS framework.
import os
import sys
from argparse import ArgumentParser
import torch.multiprocessing as mp

# sys.path.append(os.path.abspath(__file__ + '/..'))
# os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['NCCL_SOCKET_IFNAME'] = 'bond1'
os.environ['USE_FLASH_ATTENTION'] = '1'

import torch
import torch._dynamo

torch.set_float32_matmul_precision('high')
torch._dynamo.config.accumulated_cache_size_limit = 256
torch._dynamo.config.cache_size_limit = 256
torch._dynamo.config.capture_scalar_outputs = True
torch._dynamo.config.optimize_ddp = False


import basicts

torch.set_num_threads(8) # aviod high cpu avg usage

def parse_args():
    parser = ArgumentParser(description='Run time series forecasting model in BasicTS framework!')
    parser.add_argument('-c', '--cfg', default='baselines/timemoe/config/timemoe_base_x1_4095_adamw.py', help='training config')
    parser.add_argument('-c', '--cfg', default='baselines/chronos/config/chronos_base_x1.py', help='training config')
    # parser.add_argument('-c', '--cfg', default='tsfm/config/tsfm_pretrain_config.py', help='training config')
    parser.add_argument('-g', '--gpus', default='1', help='visible gpus')
    parser.add_argument('-r', '--node_rank', default='0', help='visible gpus')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    mp.set_start_method("fork", force=True)  # 避免多进程冲突
    basicts.launch_training(args.cfg, args.gpus, node_rank=int(args.node_rank))
