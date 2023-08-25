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
        
        
# EXP = ['ur_detic','rl_detic','fbe_detic']
EXP = None
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
            i = 0
            for file in os.listdir(result_dir+exp_name):
                if file.endswith('.json'):
                    i += 1
                    with open(result_dir+exp_name+'/'+file, 'r') as f:
                        result = json.load(f)
                        
                    total_count += 1
                    if result['distance_to_goal'] < 0.5:
                        success_count += 1
                        spl += result['spl']
                       
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
                    
            
            if total_count > 0:
                spl /= total_count
                success_rate = success_count/total_count
                print(f'EXP: {exp_name} Success rate: {success_count}/{total_count} = {success_count/total_count:.2%}')
                print(f'EXP: {exp_name} SPL: {spl}')
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
    for k,v in results.items():
        data = v[item]
        plt.plot(v[item].mean(axis=0),label=k)
    plt.legend()
    plt.show()
    
# plot(results,'exp_coverage')
# plot(results,'close_coverage')
# plot(results,'check_coverage')
# plot(results,'entropy')
    