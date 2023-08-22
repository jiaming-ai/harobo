#! /bin/bash

####
# best: batch size 16, lr 1e-4, c16, l1 loss, i_s_weight 10
# python train_igp.py --exp_name unet_c32 --options net.c0=32
# python train_igp.py --exp_name unet_c16 --options net.c0=16
# python train_igp.py --exp_name unet_16_lr1e4_ep100 --options net.c0=16,train.lr=1e-4,train.epoch_num=100
# python train_igp.py --exp_name unet_16_lr1e5_ep100 --options net.c0=16,train.lr=1e-5,train.epoch_num=100
# python train_igp.py --exp_name unet_c16_lr1e4_B32 --options net.c0=16,train.batch_size=32,train.lr=1e-4

# python train_igp.py --exp_name unet_c8_lr1e4 --options net.c0=8,train.lr=1e-4

# python train_igp.py --exp_name unet_c16_is10 --options net.c0=16,net.i_s_weight=10
# python train_igp.py --exp_name unet_c16_is20 --options net.c0=16,net.i_s_weight=20

# python train_igp.py --exp_name resnet_c16_l4 --options net.c0=16,net.backbone=resnet,net.resnet_depth=4
# python train_igp.py --exp_name resnet_c8_l4 --options net.c0=8,net.backbone=resnet,net.resnet_depth=4
# python train_igp.py --exp_name resnet_c8_l5 --options net.c0=8,net.backbone=resnet,net.resnet_depth=5

# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name rl_detic --eval_policy rl --gpu_id 1



# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name rl_gtsm --eval_policy rl --gt_semantic --gpu_id 1



# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 20 \
#                         --exp_name ur_detic_igp --eval_policy ur --gpu_id 1

trap 'kill 0' SIGINT
exp_name=("0.01" "0.05")

for exn in "${exp_name[@]}"
do
    python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name igp_util_$exn --save_video \
                        --eval_policy ur --gpu_id 1 +AGENT.IG_PLANNER.util_lambda=$exn &
done



# exp_name=(3 2)

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_planner_dialate_$exn --save_video \
#                         --eval_policy ur --gpu_id 1 AGENT.PLANNER.obs_dilation_selem_radius=$exn &
# done
wait