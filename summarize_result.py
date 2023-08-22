import os
import json

result_dir = 'datadump/exp_results/'

def process_individual_result(exp_name, file):
    with open(result_dir+exp_name+'/'+file, 'r') as f:
        print(file)
        print(f.read())
        print('----------------')
        
        
def summarize_result():
    
    exp_results = {}
    for exp_name in os.listdir(result_dir):
        if os.path.isdir(result_dir+exp_name):
            exp_results[exp_name] = {}
            total_count = 0
            success_count = 0
            for file in os.listdir(result_dir+exp_name):
                if file.endswith('.json'):
                    with open(result_dir+exp_name+'/'+file, 'r') as f:
                        result = json.load(f)
                        exp_results[exp_name] = result
                    total_count += 1
                    if result['distance_to_goal'] < 0.5:
                        success_count += 1
            if total_count > 0:
                print(f'EXP: {exp_name} Success rate: {success_count}/{total_count} = {success_count/total_count:.2%}')

                    
summarize_result()