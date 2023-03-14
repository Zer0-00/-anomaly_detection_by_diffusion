import csv
from collections import defaultdict
import tqdm
from matplotlib import pyplot as plt
import numpy as np

def evaluate_training(progress_dir, save_dir):
    """
    Visualize the change of metrics in the training process.
    Args:
        progress_dir: progress csv file path
    """
    #figure configuration
    num_columns = 4
    figsize = (12,10)
    warm_up_steps = 4000        #metrics of steps before warmup will not be showed in figure if data shifts too much from center
    skip_metric = ["step", "samples"]        #metrics that are skipped
    
    #reading data
    warm_up_idx = 0
    metrics = defaultdict(list)
    with open(progress_dir, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm.tqdm(reader):
            if int(row["step"]) < warm_up_steps:
                warm_up_idx += 1
                
            for k, v in row.items():
                metrics[k].append(float(v))
    
    #visualize
    num_rows = int(np.ceil((len(metrics) - len(set(skip_metric) & set(metrics.keys()))) / num_columns))
    plt.figure(figsize=figsize)
    
    idx = 0
    for metric_name in metrics:
        if metric_name in skip_metric:
            continue
        else:
            idx += 1
            plt.subplot(num_rows, num_columns, idx)
            plt.plot(metrics["step"], metrics[metric_name])
            plt.title(metric_name)
            plt.xlabel("step")
            plt.ylim((min(metrics[metric_name][warm_up_idx:])), max(metrics[metric_name][warm_up_idx:]))        
            
    plt.savefig(save_dir, dpi=600)
    plt.close()
    
    
if __name__ == '__main__':
    progress_dir = "output/classfier/progress.csv"
    output_dir = "output/classfier/progress.png"
    evaluate_training(progress_dir,output_dir)