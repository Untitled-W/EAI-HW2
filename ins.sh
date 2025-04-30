# CUDA_VISIBLE_DEVICES=2 python train.py --model_type=est_pose --checkpoint=exps/22-16_/checkpoint/checkpoint_10000.pth --learning_rate=1e-4 --max_iter=50000
CUDA_VISIBLE_DEVICES=2 python test.py --mode=val --checkpoint=exps/23-15_/checkpoint/checkpoint_30000.pth
# CUDA_VISIBLE_DEVICES=2 python train.py \
#     --model_type=est_pose \
#     --max_iter=40000 \
#     --trans_loss=0.5 \
#     --rot_loss=1.5 \
#     --learning_rate=1e-3 \
