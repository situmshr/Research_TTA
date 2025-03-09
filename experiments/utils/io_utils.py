import os
import csv

def generate_path(cfg):
    base_dir = os.getcwd()
    path = os.path.join(base_dir, cfg.dataset, f"{cfg.algorithm}_{cfg.act_type}")
    if cfg.ext_flag:
        path += "_ext"
    if cfg.bn_flag:
        path += "_bn"
    if cfg.act_flag:
        path += "_act"
    return path


def write_results_csv(filepath, common_corruptions, accuracy_dict, comp_time_dict):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Corruption', 'Accuracy', 'Computation Time'])
        for corruption in common_corruptions:
            writer.writerow([corruption, accuracy_dict[corruption], comp_time_dict[corruption]])


def save_results(cfg, common_corruptions, acc_dict, comp_time_dict,
                 readapt_acc_dict=None, readapt_comp_time_dict=None):
    """
    Save evaluation results to CSV files.
    
    Parameters:
      - cfg: Hydra configuration object.
      - common_corruptions: list of corruption types.
      - acc_dict: dictionary with accuracy results.
      - comp_time_dict: dictionary with computation time results.
      - readapt_acc_dict: (optional) dictionary with readapted accuracy results.
      - readapt_comp_time_dict: (optional) dictionary with readapted computation times.
      
    Returns:
      - output_file: path of the main CSV file.
      - readapt_output_file: path of the readapt CSV file (or None if not used).
    """
    output_file = generate_path(cfg) + ".csv"
    write_results_csv(output_file, common_corruptions, acc_dict, comp_time_dict)

    readapt_output_file = None
    if readapt_acc_dict is not None and readapt_comp_time_dict is not None:
        readapt_output_file = generate_path(cfg) + "_readapt.csv"
        write_results_csv(readapt_output_file, common_corruptions, readapt_acc_dict, readapt_comp_time_dict)
                
    return output_file, readapt_output_file
