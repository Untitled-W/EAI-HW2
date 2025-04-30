# CUDA_VISIBLE_DEVICES=2 python train.py --model_type=est_pose --checkpoint=/home/liyicheng/code/EAI-HW2/exps/21-10/checkpoint/checkpoint_40000.pth --max_iter=50000
# CUDA_VISIBLE_DEVICES=2 python test.py --mode=val --checkpoint=/home/liyicheng/code/EAI-HW2/exps/21-10/checkpoint/checkpoint_40000.pth
CUDA_VISIBLE_DEVICES=2 python test.py --mode=val --checkpoint=exps/20-59/checkpoint/checkpoint_10000.pth
