#! /bin/bash
GPU_ID=1
trap 'kill 0' SIGINT

####
exp_name=(0.0075)

for exn in "${exp_name[@]}"
do
    python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name refact2_alpha10_sd_lambda_$exn --save_video \
                        --eval_policy ur --gpu_id $GPU_ID AGENT.IG_PLANNER.util_lambda=$exn AGENT.IG_PLANNER.info_gain_alpha=10 &
    GPU_ID=$((GPU_ID+1))
done
# exp_name=(3 2)

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_planner_dialate_$exn --save_video \
#                         --eval_policy ur --gpu_id 1 AGENT.PLANNER.obs_dilation_selem_radius=$exn &
# done
wait