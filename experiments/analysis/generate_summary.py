import os
import glob
import itertools
import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path

def generate_all_patterns(cfg: DictConfig) -> list:
    """
    Generate all combinations of target file patterns based on the lists defined in the configuration.
    
    The four lists are:
      - algorithm_types: e.g., ["source", "tent"]
      - activations: e.g., ["relu", "prelu", "frelu"]
      - update_params: e.g., ["", "bn", "act", "bn_act", "ext"]
      - readapt_or_not: e.g., ["", "readapt"]
      
    For each combination, non-empty parts are concatenated with underscores.
    For example, one generated pattern might be "tent_prelu_bn_readapt".
    """
    algorithm_types = cfg.algorithm_types
    activations = cfg.activations
    update_params = cfg.update_params
    readapt_or_not = cfg.readapt_or_not
    all_patterns = []
    for a, b, c, d in itertools.product(algorithm_types, activations, update_params, readapt_or_not):
        pattern = f"{a}_{b}"
        if c:
            pattern += f"_{c}"
        if d:
            pattern += f"_{d}"
        all_patterns.append(pattern)
    return all_patterns

def combine_csv_files(cfg: DictConfig, target_file_patterns: list, top_k: int = 1, target_dir_pattern: str = None) -> None:
    """
    Combine CSV files located under the '/output' directory that match any of the provided target_file_patterns.
    For each pattern, the files are sorted by their parent directory name (assumed to be a timestamp)
    and only the earliest top_k files are selected.
    The combined data (with an extra 'source' column indicating the parent directory) is saved as a summary CSV
    in the summary folder under the output directory.
    """
    
    output_dir = cfg.get("output_dir", "../output")
    output_root = to_absolute_path(output_dir)
    csv_files = glob.glob(os.path.join(output_root, '**', '*.csv'), recursive=True)
    
    if target_dir_pattern:
        csv_files = [f for f in csv_files if target_dir_pattern in os.path.basename(os.path.dirname(f))]
    
    selected_files = []
    for pattern in target_file_patterns:
        pattern_files = [f for f in csv_files if pattern in os.path.basename(f)]
        if not pattern_files:
            print(f"No CSV files found matching pattern '{pattern}'.")
            continue
        # Sort files by the name of their parent directory (assumed to reflect the timestamp)
        pattern_files_sorted = sorted(pattern_files, key=lambda f: os.path.basename(os.path.dirname(f)))
        selected_files.extend(pattern_files_sorted[:top_k])
    
    if not selected_files:
        print("No CSV files found matching the given filters.")
        return
    
    df_list = []
    for csv_file in selected_files:
        try:
            df = pd.read_csv(csv_file)
            source_dir = os.path.basename(os.path.dirname(csv_file))
            df['source'] = source_dir
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    summary_dir = os.path.join(to_absolute_path(output_dir), "summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_filename = cfg.get("summary_filename", "summary.csv")
    summary_path = os.path.join(summary_dir, summary_filename)
    combined_df.to_csv(summary_path, index=False)
    print(f"Combined summary CSV saved to {summary_path}")

@hydra.main(config_path="../conf", config_name="target_patterns")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # Generate all target file patterns from the configuration.
    target_patterns = generate_all_patterns(cfg)
    top_k = cfg.get("top_k", 1)
    combine_csv_files(cfg, target_file_patterns=target_patterns, top_k=top_k)

if __name__ == '__main__':
    main()
