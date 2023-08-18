#! /bin/bash

python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name fbe_detic --eval_policy fbe --gpu_id 0

python eval_agent.py --save_video --no_render --no_interactive --eval_eps_total_num 200 \
                        --exp_name fbe_gtsm --eval_policy fbe --gt_semantic --gpu_id 0

