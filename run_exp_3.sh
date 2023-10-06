#! /bin/bash

GPU_ID=3
trap 'kill 0' SIGINT


exp_name=(0.0075)

for exn in "${exp_name[@]}"
do
    python eval_nav.py --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name refact3_default --save_video \
                        --eval_policy ur --gpu_id $GPU_ID &
    # GPU_ID=$((GPU_ID+1))
done
wait
