#! /bin/bash

trap 'kill 0' SIGINT

# GPU_ID=0
# #############################################
# # baseline experiments
# #############################################
# exp_name=("fbe" "ur" "rl") # add habitat web
# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name baseline_$exn --save_video \
#                         --eval_policy $exn --gpu_id $GPU_ID &
# done
# #############################################


# #############################################
# # ablation experiments
# #############################################
# # 1. IG by rendering
# python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                     --exp_name ablation_ig_rendering --save_video \
#                     --eval_policy ur --gpu_id $GPU_ID AGENT.IG_PLANNER.use_ig_predictor=False,AGENT.IG_PLANNER.other_ig_type=rendering &

# # 2. IG by ray casting
# python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                     --exp_name ablation_ig_ray_casting --save_video \
#                     --eval_policy ur --gpu_id $GPU_ID AGENT.IG_PLANNER.use_ig_predictor=False,AGENT.IG_PLANNER.other_ig_type=ray_casting &

# # 3. IG with no probability map
# python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                     --exp_name ablation_ig_no_prob --save_video \
#                     --eval_policy ur --gpu_id $GPU_ID AGENT.SEMANTIC_MAP.use_probability_map=False &
# #############################################


# #############################################
# # Upper bound experiments
# #############################################
# # 1. GT semantic perception
# exp_name=("fbe" "ur" "rl") # add habitat web
# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ub_gtsm_$exn --save_video \
#                         --eval_policy $exn --gpu_id $GPU_ID GROUND_TRUTH_SEMANTICS=1 &
# done


# # 2. Allow sliding
# exp_name=("fbe" "ur" "rl") # add habitat web
# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ub_can_slide_$exn --save_video \
#                         --eval_policy $exn --gpu_id $GPU_ID habitat.simulator.habitat_sim_v0.allow_sliding=True &
# done
# #############################################




# #############################################
# # IGP experiments
# #############################################
exp_name=("unet_c16_lossis1_dlis5_more" "unet_c16_lossis1_dlis10_more")
for exn in "${exp_name[@]}"
do
    python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name igp_model_$exn --save_video \
                        --eval_policy ur --gpu_id 0 +AGENT.IG_PLANNER.igp_model_dir=data/checkpoints/igp/$exn &
done

# dialate size 3 is default setting
# exp_name=(3 1)

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_map_dialate_$exn --save_video \
#                         --eval_policy ur --gpu_id 0 AGENT.SEMANTIC_MAP.dilate_size=$exn &
# done

# default is 1
# exp_name=(3 2)

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_planner_dialate_$exn --save_video \
#                         --eval_policy ur --gpu_id 0 AGENT.PLANNER.obs_dilation_selem_radius=$exn &
# done
#############################################

wait