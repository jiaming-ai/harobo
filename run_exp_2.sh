#! /bin/bash
GPU_ID=0

trap 'kill 0' SIGINT

# exp_name=("ur") # add habitat web
# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ub_can_slide_$exn --save_video \
#                         --eval_policy $exn --gpu_id $GPU_ID --allow_sliding &
# done

# python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                     --exp_name ablation_argmax  --skip_existing \
#                     --eval_policy ur --gpu_id $GPU_ID AGENT.IG_PLANNER.ig_predictor_type=argmax &


# python eval_agent.py --no_render --no_interactive \
#                     --exp_name collect_data --collect_data \
#                     --eval_policy ur --gpu_id $GPU_ID &

exp_name=(0.01 0.0075 0.005)

for exn in "${exp_name[@]}"
do
    python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name refact2_alpha10_lambda_$exn --save_video \
                        --eval_policy ur --gpu_id $GPU_ID AGENT.IG_PLANNER.util_lambda=$exn AGENT.IG_PLANNER.info_gain_alpha=10 &
    GPU_ID=$((GPU_ID+1))
done

# python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                     --exp_name replan --save_video \
#                     --eval_policy ur --gpu_id $GPU_ID AGENT.IG_PLANNER.replan_ur_goal_if_stuck=True &

# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ur_detic --eval_policy ur --gpu_id 2
# exp_name=(13 15)

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_IG_dialate_$exn --skip_existing \
#                         --eval_policy ur --gpu_id 2 AGENT.IG_PLANNER.ur_obstacle_dialate_radius=$exn &
# done
# wait
# map_dilate_size=(1 2 3 4)
# close_range=(100 150 200 250)
# planner_obs_dilation_selem_radius=(1 2 3)

# for i in "${map_dilate_size[@]}"
# do
#     for j in "${close_range[@]}"
#     do
#         for k in "${planner_obs_dilation_selem_radius[@]}"
#         do
#             python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ur_detic_md$i_cr$j_pr$k \
#                         --eval_policy ur --gt_semantic --gpu_id 2 \
#                         config.AGENT.SEMANTIC_MAP.dilate_size=$i config.AGENT.SEMANTIC_MAP.close_range$j \
#                         config.AGENT.PLANNER.obs_dilation_selem_radius=$k
#         done
#     done
# done

# test config.AGENT.SEMANTIC_MAP.dilate_size=1 # default is 3

# python interactive.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ur_gtsm --eval_policy ur --gt_semantic --gpu_id 2

wait