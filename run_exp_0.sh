#! /bin/bash

# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name fbe_detic --eval_policy fbe --gpu_id 0

# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name fbe_gtsm --eval_policy fbe --gt_semantic --gpu_id 0



trap 'kill 0' SIGINT

# exp_name=("unet_c16_is1_dlis10_more" "unet_c16_is10_dlis10_more")

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_model_$exn --save_video \
#                         --eval_policy ur --gpu_id 0 +AGENT.IG_PLANNER.igp_model_dir=data/checkpoints/igp/$exn &
# done

# dialate size 3 is default setting
exp_name=(3 1)

for exn in "${exp_name[@]}"
do
    python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name igp_map_dialate_$exn --save_video \
                        --eval_policy ur --gpu_id 0 AGENT.SEMANTIC_MAP.dilate_size=$exn &
done
wait