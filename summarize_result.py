import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

result_dir = 'datadump/exp_results/'

def process_individual_result(exp_name, file):
    with open(result_dir+exp_name+'/'+file, 'r') as f:
        print(file)
        print(f.read())
        print('----------------')
        
        
EXP = ["ur_ig_alpha_4","rl_detic","fbe_detic", 
       "ablation_ig_no_prob_run2","ablation_ig_rendering","newlc_ur_gtsm1_sliding1"]
LABEL_MAP = {
    "igp_IG_dialate_5": "OIG",
    "rl_detic": "RL",
    "fbe_detic": "FBE"
}
# EXP = None
def summarize_result():
    
    exp_results = {}
    
    for exp_name in os.listdir(result_dir):
        if os.path.isdir(result_dir+exp_name):
            if EXP is not None and exp_name not in EXP:
                continue
            exp_results[exp_name] = {}
            total_count = 0
            success_count = 0
            spl = 0
            exp_coverage_all = np.zeros((200,1250))
            close_coverage_all = np.zeros((200,1250))
            check_coverage_all = np.zeros((200,1250))
            entropy_all = np.zeros((200,1250))
            total_steps = 0
            total_dist_travelled = 0
            total_time = 0
            i = 0
            for file in os.listdir(result_dir+exp_name):
                if file.endswith('.json'):
                    with open(result_dir+exp_name+'/'+file, 'r') as f:
                        result = json.load(f)
                        
                    total_count += 1
                    if result['distance_to_goal'] < 0.5:
                        success_count += 1
                        spl += result['spl']
                       
                    # total steps count
                    total_steps += result['steps']
                    total_dist_travelled += result['travelled_distance']
                    total_time += result.get('total_time',0)

                    # coverage
                    ec = result['exp_coverage']
                    exp_coverage_all[i,:len(ec)] = ec
                    exp_coverage_all[i,len(ec):] = ec[-1]
                    # close coverage
                    cc = result['close_coverage']
                    close_coverage_all[i,:len(cc)] = cc
                    close_coverage_all[i,len(cc):] = cc[-1]
                    # check coverage
                    cc = result['checking_area']
                    check_coverage_all[i,:len(cc)] = cc
                    check_coverage_all[i,len(cc):] = cc[-1]
                    # entropy
                    e = result['entropy']
                    entropy_all[i,:len(e)] = e
                    entropy_all[i,len(e):] = e[-1]
                    
                    i += 1
                    if i == 200:
                        break
            if total_count > 0:
                spl /= total_count
                success_rate = success_count/total_count
                print(f'---------- {exp_name} ----------')
                print(f'Success rate: {success_count}/{total_count} = {success_count/total_count:.2%}')
                print(f'SPL: {spl}')
                
                print(f'Explored area per step: {exp_coverage_all[:,-1].sum()/total_steps}')
                print(f'Explored area per meter: {exp_coverage_all[:,-1].sum()/total_dist_travelled}')
                print(f'Checked area per meter: {check_coverage_all[:,-1].sum()/total_dist_travelled}')
                print(f'Total time: {total_time/total_steps:.2f}')
                
                print(f'Distance travelled per step: {total_dist_travelled/total_steps:.2f}')
                print(f'-------------------------')
            else:
                spl = 0
                success_rate = 0

            exp_results[exp_name]['total_count'] = total_count
            exp_results[exp_name]['success_count'] = success_count
            exp_results[exp_name]['success_rate'] = success_rate
            exp_results[exp_name]['spl'] = spl
            exp_results[exp_name]['exp_coverage'] = exp_coverage_all
            exp_results[exp_name]['close_coverage'] = close_coverage_all
            exp_results[exp_name]['check_coverage'] = check_coverage_all
            exp_results[exp_name]['entropy'] = entropy_all
            
    return exp_results
                    
results = summarize_result()


def plot(results,item):
    plt.subplots(1,3)
    for k,v in results.items():
        data = v[item]
        plt.plot(v[item].mean(axis=0),label=k)
    plt.legend()
    plt.show()
    
    
    
def plot_all():
    items = ['exp_coverage','close_coverage','check_coverage']
    titles = ['(a) Total explored area','(b) Closely checked area','(c) Checked promising area']
    fig, axes = plt.subplots(1,3)
    for i,item in enumerate(items):
        for k,v in results.items():
            data = v[item]
            axes[i].plot(v[item].mean(axis=0),label=LABEL_MAP[k])
            
        axes[i].legend()
        axes[i].set_xlabel('Step')
        # axes[i].xaxis.tick_top()
        # axes[i].xaxis.set_label_position('top')
        axes[i].set_ylabel('Area')
        axes[i].set_title(titles[i],y=-0.15)

        
        
    plt.show()
    
    
# plot_all()

# plot(results,'exp_coverage')
# plot(results,'close_coverage')
# plot(results,'check_coverage')
# plot(results,'entropy')
    