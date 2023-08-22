#! /bin/bash

trap 'kill 0' SIGINT

# python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name ur_detic --eval_policy ur --gpu_id 2

exp_name=(5 3)

for exn in "${exp_name[@]}"
do
    python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name igp_dist_obdiasize_$exn --save_video \
                        --eval_policy ur --gpu_id 3 +AGENT.IG_PLANNER.ur_dist_obstacle_dialate_radius=$exn &
done

# exp_name=(150 200)

# for exn in "${exp_name[@]}"
# do
#     python eval_agent.py --no_render --no_interactive --eval_eps_total_num 200 \
#                         --exp_name igp_close_range_$exn --save_video \
#                         --eval_policy ur --gpu_id 0 AGENT.SEMANTIC_MAP.close_range=$exn &
# done

wait
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