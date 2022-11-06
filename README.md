pytorch:1.7.1
cuda:10.2
DistributeDataParallel: python -m torch.distributed.launch --nproc_per_node=2 train.py
