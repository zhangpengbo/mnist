import os
import torch
import torch.distributed as dist
import torch.distributed.rpc as rpc
from torch.nn.parallel import DistributedDataParallel as DDP

def init_ddp():
    dist.init_process_group(
        backend='nccl',
        init_method=f'tcp://{os.getenv("MASTER_ADDR")}:{os.getenv("MASTER_PORT")}',
        world_size=int(os.getenv('WORLD_SIZE')),
        rank=int(os.getenv('RANK'))
    )
    model = YourModel().to(device)
    ddp_model = DDP(model, device_ids=[int(os.getenv('RANK'))])
    # 继续 DDP 训练逻辑

def init_rpc():
    rpc.init_rpc(
        name=f'worker{os.getenv("RANK")}',
        rank=int(os.getenv('RANK')),
        world_size=int(os.getenv('WORLD_SIZE')),
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            init_method=f'tcp://{os.getenv("MASTER_ADDR")}:{os.getenv("RPC_PORT")}'
        )
    )
    # 继续 RPC 训练逻辑
# just test
if __name__ == "__main__":
    if os.getenv('RANK') == '0':
        init_ddp()
    else:
        init_rpc()
