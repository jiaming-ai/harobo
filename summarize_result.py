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

# EXP = ['timing_ig_rendering','video_sum_prob_map_ur',] # for ig timing

# EXP = ['baseline_rl','baseline_fbe', "ur_ig_alpha_4",'ablation_argmax']
# EXP = ["ur_ig_alpha_4","rl_detic","fbe_detic", ] # for plotting
    #    "ablation_ig_no_prob_run2","ablation_ig_rendering","ablation_ig_ray_casting"]
LABEL_MAP = {
    "igp_IG_dialate_5": "OIG",
    "rl_detic": "RL",
    "fbe_detic": "FBE",
    "ur_ig_alpha_4": "UR",
    "ablation_ig_no_prob_run2": "UR-IG w/o prob",
    "ablation_ig_rendering": "UR-IG rendering",
    "ablation_ig_ray_casting": "UR-IG ray casting",
    "baseline_rl": "RL",
    "baseline_fbe": "FBE",
    "sum_prob_map_ur": "UR",
    "ablation_argmax": "UR argmax",
}
# EXP = ['refact_lambda_0.01','refact_lambda_0.005','refact_lambda_0.0075',
#         'refact_alpha10_lambda_0.01','refact_alpha10_lambda_0.005','refact_alpha10_lambda_0.0075',
#        ]
EXP = [
        'refact2_alpha10_lambda_0.01','refact2_alpha10_lambda_0.005','refact2_alpha10_lambda_0.0075',
        'refact2_alpha10_sd_lambda_0.0075',
        'refact2_alpha15_lambda_0.01','refact2_alpha15_lambda_0.005','refact2_alpha15_lambda_0.0075',
       ]
def summarize_result():
    
    exp_results = {}
    results_df = pd.DataFrame()
    
    for exp_name in os.listdir(result_dir):
        if os.path.isdir(result_dir+exp_name):
            if EXP is not None and exp_name not in EXP:
                continue
            exp_results[exp_name] = {}
            total_count = 0
            success_count = 0
            spl = 0
            total_num = 1000
            exp_coverage_all = np.zeros((total_num,1250))
            close_coverage_all = np.zeros((total_num,1250))
            check_coverage_all = np.zeros((total_num,1250))
            entropy_all = np.zeros((total_num,1250))
            total_steps = 0
            total_dist_travelled = 0
            total_time = 0
            ig_time_all = []
            i = 0
            for file in os.listdir(result_dir+exp_name):
                if file.endswith('.json'):
                    with open(result_dir+exp_name+'/'+file, 'r') as f:
                        result = json.load(f)
                        
                    total_count += 1
                    if result['distance_to_goal'] < 0.5:
                        success_count += 1
                        spl += result['spl']
                        
                        # rename video to success
                        vfn = result_dir+exp_name+'/'+file.replace('.json','.mp4')
                        vsfn = result_dir+exp_name+'/success_'+file.replace('.json','.mp4')
                        vfn = vfn.replace(' ','_')
                        vsfn = vsfn.replace(' ','_').replace('False','True')
                        if os.path.exists(vfn):
                            print(f'rename {vfn} to {vfn.replace("False","True")}')
                            os.rename(vfn,vsfn)
                       
                    # total steps count
                    total_steps += result['steps']
                    total_dist_travelled += result['travelled_distance']
                    total_time += result.get('total_time',0)

                    # # coverage
                    # ec = result['exp_coverage']
                    # exp_coverage_all[i,:len(ec)] = ec
                    # exp_coverage_all[i,len(ec):] = ec[-1]
                    # # close coverage
                    # cc = result['close_coverage']
                    # close_coverage_all[i,:len(cc)] = cc
                    # close_coverage_all[i,len(cc):] = cc[-1]
                    # # check coverage
                    # cc = result['checking_area']
                    # check_coverage_all[i,:len(cc)] = cc
                    # check_coverage_all[i,len(cc):] = cc[-1]
                    # # entropy
                    # e = result['entropy']
                    # entropy_all[i,:len(e)] = e
                    # entropy_all[i,len(e):] = e[-1]

                    # # timeing
                    # if "ig_times" in result:
                    #     ig_time_all += result["ig_times"]
                        
                    
                    i += 1
                    # if i == 200:
                    #     break
            if total_count > 0:
                spl /= total_count
                success_rate = success_count/total_count
                print(f'---------- {exp_name} ----------')
                print(f'Success rate: {success_count}/{total_count} = {success_count/total_count:.2%}')
                print(f'SPL: {spl}')
                
                # print(f'Explored area per step: {exp_coverage_all[:,-1].sum()/total_steps}')
                # print(f'Explored area per meter: {exp_coverage_all[:,-1].sum()/total_dist_travelled}')
                # print(f'Checked area per meter: {check_coverage_all[:,-1].sum()/total_dist_travelled}')
                # print(f'Total time: {total_time/total_steps:.2f}')
                
                # print(f'Distance travelled per step: {total_dist_travelled/total_steps:.2f}')

                if len(ig_time_all) > 0:
                    avg_ig_time = np.mean(ig_time_all)
                    std_ig_time = np.std(ig_time_all)
                    print(f'IG time: {avg_ig_time:.2f} +/- {std_ig_time:.2f}')
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

            if exp_name in LABEL_MAP:
                col_name = LABEL_MAP[exp_name]
                results_df[col_name+'_E'] = exp_coverage_all.mean(axis=0)
                results_df[col_name+'_C'] = check_coverage_all.mean(axis=0)
            
    results_df.to_csv('datadump/results.csv')
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
    items = ['exp_coverage','check_coverage']
    ylabel = ['Total explored area','Total checked promising area']
    fig, axes = plt.subplots(1,2)
    for i,item in enumerate(items):
        for k,v in results.items():
            data = v[item]
            axes[i].plot(v[item].mean(axis=0),label=LABEL_MAP[k])
            
        axes[i].legend( prop={'size': 18})
        axes[i].set_xlabel('Number of step',fontsize=20)
        # axes[i].xaxis.tick_top()
        # axes[i].xaxis.set_label_position('top')
        axes[i].set_ylabel(ylabel[i], fontsize=20)

        
        
    plt.show()
    
    
# plot_all()

# plot(results,'exp_coverage')
# plot(results,'close_coverage')
# plot(results,'check_coverage')
# plot(results,'entropy')
    

"""

---------- refact2_alpha15_lambda_0.005 ----------
Success rate: 9/48 = 18.75%
SPL: 0.08448997472675052
Explored area per step: 0.23741349063080017
Explored area per meter: 1.870021187872523
Checked area per meter: 0.030418327215768905
Total time: 0.37
Distance travelled per step: 0.13
---------- refact2_alpha10_lambda_0.01 ----------
Success rate: 8/47 = 17.02%
SPL: 0.07390060751842606
Explored area per step: 0.22167141800658427
Explored area per meter: 1.9025198730480148
Checked area per meter: 0.030734159527008774
Total time: 0.35
Distance travelled per step: 0.12
rename datadump/exp_results/refact2_alpha10_lambda_0.0075/Move_apple_from_cabinet_to_couch_True.mp4 to datadump/exp_results/refact2_alpha10_lambda_0.0075/Move_apple_from_cabinet_to_couch_True.mp4
rename datadump/exp_results/refact2_alpha10_lambda_0.0075/Move_book_from_couch_to_chair_True.mp4 to datadump/exp_results/refact2_alpha10_lambda_0.0075/Move_book_from_couch_to_chair_True.mp4
---------- refact2_alpha10_lambda_0.0075 ----------
Success rate: 10/55 = 18.18%
SPL: 0.06728791468660994
Explored area per step: 0.24172400834604688
Explored area per meter: 1.8345356415412892
Checked area per meter: 0.030935514395193226
Total time: 0.32
Distance travelled per step: 0.13
---------- refact2_alpha15_lambda_0.0075 ----------
Success rate: 9/48 = 18.75%
SPL: 0.06569313189143251
Explored area per step: 0.2457468513517127
Explored area per meter: 1.9254890411418601
Checked area per meter: 0.03102087686185808
Total time: 0.39
Distance travelled per step: 0.13
rename datadump/exp_results/refact2_alpha10_lambda_0.005/Move_toy_bee_from_chest_of_drawers_to_chair_True.mp4 to datadump/exp_results/refact2_alpha10_lambda_0.005/Move_toy_bee_from_chest_of_drawers_to_chair_True.mp4
rename datadump/exp_results/refact2_alpha10_lambda_0.005/Move_book_from_table_to_counter_True.mp4 to datadump/exp_results/refact2_alpha10_lambda_0.005/Move_book_from_table_to_counter_True.mp4
---------- refact2_alpha10_lambda_0.005 ----------
Success rate: 10/48 = 20.83%
SPL: 0.10357162883732417
Explored area per step: 0.22894810309586722
Explored area per meter: 1.7689488902433794
Checked area per meter: 0.028410737350105464
Total time: 0.37
Distance travelled per step: 0.13
rename datadump/exp_results/refact2_alpha15_lambda_0.01/Move_apple_from_cabinet_to_couch_True.mp4 to datadump/exp_results/refact2_alpha15_lambda_0.01/Move_apple_from_cabinet_to_couch_True.mp4
---------- refact2_alpha15_lambda_0.01 ----------
Success rate: 11/49 = 22.45%
SPL: 0.08214342766489076
Explored area per step: 0.23070939760139159
Explored area per meter: 1.8676553972076104
Checked area per meter: 0.03151758980695174
Total time: 0.32
Distance travelled per step: 0.12
"""