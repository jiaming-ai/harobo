from habitat.utils.visualizations.utils import images_to_video
import os
import cv2

STAGES = ['nav_to_obj','gaze_obj','pick','nav_to_rec','gaze_rec','place']

def make_result_videos(log_dir,prefix='snapshot',output_dir='videos'):
    
    os.makedirs(output_dir,exist_ok=True)
    
    for dir in os.listdir(log_dir):
        full_dir = os.path.join(log_dir,dir)
        if os.path.isdir(full_dir):
            print(f'Processing {dir}')

            try:
                images = []
                shape = None
                cur_stage = 0
                for step in range(1,5000):
                    full_file = os.path.join(full_dir,f'{prefix}_{step:03d}.png')
                    if os.path.isfile(full_file):
                        image = cv2.imread(full_file)
                        if shape is None:
                            shape = image.shape
                        if shape != image.shape:
                            images_to_video(images,output_dir,dir+'_'+STAGES[cur_stage])
                            images = []
                            cur_stage += 1
                            shape = image.shape
                        
                        images.append(image)
                    else:
                        break
            
                images_to_video(images,output_dir,dir+'_'+STAGES[cur_stage])
            except KeyboardInterrupt:
                break
            except:
                print(f'Error processing {dir}')
                continue


if __name__ == '__main__':
    make_result_videos('datadump/images/eval_hssd',output_dir='videos_ur_10alpha')