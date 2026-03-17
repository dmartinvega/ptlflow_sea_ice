import os
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from ptlflow.utils import flow_utils
import subprocess
from PIL import Image
from omegaconf import DictConfig, OmegaConf
import struct
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="debugpy._vendored.force_pydevd")
    
    
def get_inference_batch(cfg, image_folder):                        
    for model_name, ckpt_paths in cfg["models"].items():
        for ckpt_path in ckpt_paths:
            logger.info(f"Starting with model: {model_name} - ckpt_path: {ckpt_path}")
            try:
                args = [cfg["python_exec"], 'infer.py', 
                        '--model', model_name, '--ckpt_path', ckpt_path, 
                        '--input_path', image_folder,
                        '--output_path', f"{cfg["output_path"]}{cfg["name"]}"]
                p = subprocess.check_call(args)                
            except:
                logger.exception("Unable to do inference with this pretrained model.")
            logger.success(f"Finished model: {model_name} - ckpt_path: {ckpt_path}")

def check_or_create_gt(cfg, image_folder, images):
    gt_path_general = f'{cfg["input_path"]}{cfg["name"]}/{cfg["gt_folder"]}' if cfg["gt_folder"] is not None else None
    ground_truth_available = False
    for gt_source in cfg["gt_source_list"]:
        gt_path = f'{gt_path_general}{gt_source}/'
        gt_raw_path = Path(f'{gt_path}{cfg["gt_folder_raw"]}')
        gt_processed_path = Path(f'{gt_path}{cfg["gt_folder_processed"]}')
        if gt_processed_path.is_dir():
            gt_processed_files_num = len([f for f in gt_processed_path.iterdir() if f.is_file()])
        else:
            gt_processed_files_num = 0
        if gt_raw_path.exists() and cfg["gt_type"] == "sparse" and ((not gt_processed_path.exists()) or (gt_processed_path.is_dir() and (gt_processed_files_num < len(images)-1 or not any(gt_processed_path.iterdir())))):
            ground_truth_input_generation(gt_raw_path, gt_processed_path, images[0])
        ground_truth_available = True
    return image_folder, gt_path_general, ground_truth_available
            

def ground_truth_input_generation(gt_path_raw, gt_path_processed, image_reference_path):
    """
    This is needed when having sparse ground truth with the format used in SIDEx project.
    """
    x_all_buoys = pd.read_csv(f"{gt_path_raw}/buoys_x.csv", index_col=0)
    y_all_buoys = pd.read_csv(f"{gt_path_raw}/buoys_y.csv", index_col=0)
    
    x_flow_gt_all_buoys = pd.DataFrame(index=x_all_buoys.index[:-1])
    y_flow_gt_all_buoys = pd.DataFrame(index=y_all_buoys.index[:-1])
    
    for col in x_all_buoys.columns:
        x_flow_gt_all_buoys[col] = x_all_buoys[col].iloc[1:].values - x_all_buoys[col].iloc[:-1].values
        y_flow_gt_all_buoys[col] = y_all_buoys[col].iloc[1:].values - y_all_buoys[col].iloc[:-1].values
    
    img = Image.open(image_reference_path)
    width = img.size[0]
    height = img.size[1]
    
    if (x_all_buoys < 0).any().any() or (y_all_buoys < 0).any().any():
        img = Image.open(image_reference_path)
        x_all_buoys_corner_coords = x_all_buoys.copy().apply(lambda x: np.round(width // 2 + x))
        x_all_buoys_corner_coords_int = x_all_buoys_corner_coords.astype("Int64")
        y_all_buoys_corner_coords = y_all_buoys.copy().apply(lambda x: np.round(width // 2 + x))
        y_all_buoys_corner_coords_int = y_all_buoys_corner_coords.astype("Int64")
    else:
        x_all_buoys_corner_coords_int = x_all_buoys.astype("Int64")
        y_all_buoys_corner_coords_int = y_all_buoys.astype("Int64")

    out_dir = Path(gt_path_processed)
    out_dir.mkdir(exist_ok=True)
    
    for idx in x_flow_gt_all_buoys.index:
        flow_gt = np.full((height, width, 2), np.nan)

        x_pixel = x_all_buoys_corner_coords_int.loc[idx]
        y_pixel = y_all_buoys_corner_coords_int.loc[idx]
        
        valid_coords = pd.DataFrame({
            "x_valid": (x_pixel >= 0) & (x_pixel <= width),
            "y_valid": (y_pixel >= 0) & (y_pixel <= height)
            })
        
        valid_idx = valid_coords.index[valid_coords["x_valid"] & valid_coords["y_valid"]]
        x_pixel = x_pixel.loc[valid_idx]
        y_pixel = y_pixel.loc[valid_idx]
        x_flow_gt_all_buoys_local = x_flow_gt_all_buoys[valid_idx]
        x_flow_gt_all_buoys_local = y_flow_gt_all_buoys[valid_idx]
        
        flow_gt[y_pixel, x_pixel, 0] = x_flow_gt_all_buoys_local.loc[idx]
        flow_gt[y_pixel, x_pixel, 1] = x_flow_gt_all_buoys_local.loc[idx]

        filename = out_dir / f"{idx}.flo"
        flow_utils.flow_write(output_file = filename, flow = flow_gt)


def compute_metrics(cfg, gt_path_general):
    experiment_outputs_path = f"{cfg["output_path"]}{cfg["name"]}"
    experiment_output_folders = [d for d in os.listdir(experiment_outputs_path) if os.path.isdir(os.path.join(experiment_outputs_path, d))]
    for gt_source in cfg["gt_source_list"]:
        metrics_summary_list = []
        gt_path = f'{gt_path_general}{gt_source}/'
        gt_processed_path = Path(f'{gt_path}{cfg["gt_folder_processed"]}')
        gt_files_list = sorted([f for f in gt_processed_path.iterdir() if f.is_file()])
        
        for model_name in experiment_output_folders:
            model_output_path = Path(f"{experiment_outputs_path}/{model_name}")
            preds_path_list = sorted(list(model_output_path.rglob("*.flo")))
            if len(gt_files_list) == len(preds_path_list):
                logger.info(f"GT source: {gt_source} - Computing metrics for: {model_name}") 
                metrics_experiment_list = []
                for i in range(len(gt_files_list)):
                    gt = flow_utils.flow_read(input_data = gt_files_list[i], format = 'flo')
                    preds = flow_utils.flow_read(input_data = preds_path_list[i], format = 'flo')
                    epe_mean, flall_mean = compute_metrics_per_image(gt, preds)
                    metrics_experiment_list.append({"epe": epe_mean, "flall": flall_mean})
                metrics_experiment_df = pd.DataFrame(metrics_experiment_list)
                metrics_experiment_df.to_csv(f"{model_output_path}/metrics_{gt_source}.csv", index=False)
                metrics_summary_list.append({
                    "model_name": model_name,
                    "epe_mean": metrics_experiment_df["epe"].mean(),
                    "epe_std": metrics_experiment_df["epe"].std(),
                    "flall_mean": metrics_experiment_df["flall"].mean(),
                    "flall_std": metrics_experiment_df["flall"].std()
                })
            else:
                logger.exception(f"Unable to get metrics for model {model_name} since the number of prediction files differs from the ground truth")
        metrics_summary_df = pd.DataFrame(metrics_summary_list)
        metrics_summary_df.to_csv(f"{experiment_outputs_path}/metrics_{gt_source}.csv", index=False)


def compute_metrics_per_image(gt, preds):
    valid = ~np.isnan(gt[..., 0])

    sq_dist = np.power(preds - gt, 2).sum(2)
    epe = np.sqrt(sq_dist[valid])

    gt_sq_dist = np.power(gt, 2).sum(2)
    gt_dist_valid = np.sqrt(gt_sq_dist[valid])
    flall = (epe > 3) & (epe > 0.05 * gt_dist_valid)
    epe_mean = epe.mean()
    flall_mean = 100*flall.mean()
    return epe_mean,flall_mean


def main(cfg):
    image_folder = Path(f'{cfg["input_path"]}{cfg["name"]}/{cfg["images_path"]}')
    images = sorted([f for f in image_folder.iterdir() if f.suffix.lower() in [".tif", ".jpg", ".png", ".jpeg"]])
    get_inference_batch(cfg, image_folder)
    if cfg["gt_folder"] is not None and cfg["gt_source_list"] is not None:
        image_folder, gt_path_general, ground_truth_available = check_or_create_gt(cfg, image_folder, images)
        if ground_truth_available:
            compute_metrics(cfg, gt_path_general)

if __name__ == "__main__":
    cfg = OmegaConf.load("config.yaml")
    main(cfg)
    