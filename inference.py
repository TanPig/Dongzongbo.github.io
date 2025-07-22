import logging
import os
import argparse
from pathlib import Path
from PIL import Image
from contextlib import nullcontext
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import torch
from tqdm.auto import tqdm
from diffusers.utils import check_min_version
from diffusers import AutoencoderTiny
from safetensors.torch import load_file

from pipeline import LotusGPipeline, LotusDPipeline
# from utils.image_utils import colorize_depth_map
from utils.seed_all import seed_all
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# from depthseg.hair_set import *
from depthseg.hecheng import *
from depthseg.inferer import Inferer
from depthseg.unetcondition import UNet2DConditionModel_self
from tqdm import tqdm
# import matplotlib.pyplot as plt
import cv2

check_min_version('0.28.0.dev0')


def parse_args():
    '''Set the Args'''
    parser = argparse.ArgumentParser(
        description="Run Lotus..."
    )
    # model settings
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        help="pretrained model path from hugging face or local dir",
    )
    parser.add_argument(
        "--prediction_type",
        type=str,
        default="sample",
        help="The used prediction_type. ",
    )
    parser.add_argument(
        "--timestep",
        type=int,
        default=999,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="regression", # "generation"
        help="Whether to use the generation or regression pipeline."
    )
    parser.add_argument(
        "--unet_path",
        type=str,
        help="Whether to use funtune U-Net."
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="depth", # "normal"
    )
    parser.add_argument(
        "--disparity",
        action="store_true",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    # inference settings
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Output directory."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True, help="Input directory."
    )
    parser.add_argument(
        "--half_precision",
        action="store_true",
        help="Run with half-precision (16-bit float), might lead to suboptimal result.",
    )

    args = parser.parse_args()

    return args


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Run inference...")

    args = parse_args()

    # -------------------- Preparation --------------------
    # Random seed
    if args.seed is not None:
        seed_all(args.seed)

    # Output directories
    save_dir = args.output_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    logging.info(f"Output dir = {args.output_dir}")

    # output_dir_color = os.path.join(args.output_dir, f'{args.task_name}_vis')
    # output_dir_npy = os.path.join(args.output_dir, f'{args.task_name}')
    # if not os.path.exists(output_dir_color): os.makedirs(output_dir_color)
    # if not os.path.exists(output_dir_npy): os.makedirs(output_dir_npy)

    # half_precision
    if args.half_precision:
        dtype = torch.float16
        logging.info(f"Running with half precision ({dtype}).")
    else:
        dtype = torch.float32

    # -------------------- Device --------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        logging.warning("CUDA is not available. Running on CPU will be slow.")
    logging.info(f"Device = {device}")

    # -------------------- Data --------------------
    # root_dir = Path(args.input_dir)
    # test_images = list(root_dir.rglob('*.png')) + list(root_dir.rglob('*.jpg'))
    # test_images = sorted(test_images)
    # print('==> There are', len(test_images), 'images for validation.')
    #train_dataset = SemanticSegmentationDataset('../data/oppodata/train.txt')
    test_dataset = ValDataset('/home/notebook/data/group/dzb/test_500.txt',mode = 'val')
    # test_dataset = ValDataset('/home/notebook/data/group/dzb/lotus/test_zed.txt',mode = 'val')
    batch_size = 1
    #max_accumulation_steps = 16/batch_size
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=14)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=14)
    
    # -------------------- Model --------------------

    model = LotusGPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
    )
    
    # 设置tinyvae模块
    # model.vae = AutoencoderTiny.from_pretrained("/home/notebook/data/group/dzb/lotus/taesd", torch_dtype=dtype)


    # 单独加载unet权重（需要保持与原UNet结构相同）
    if args.unet_path:
        unet_self = UNet2DConditionModel_self().to(device=device)
        unet_weights = load_file(args.unet_path)
        # unet_self.load_state_dict(unet_weights, strict=True)
        model.unet.load_state_dict(unet_weights, strict=True)  # strict=True 必须完全加载
    # model.unet.load_state_dict(unet_weights, strict=False)  # strict=False 允许部分加载

    logging.info(f"Successfully loading pipeline from {args.pretrained_model_name_or_path}.")

    model = model.to(device)
    model.set_progress_bar_config(disable=True)#关闭进度条

    if args.enable_xformers_memory_efficient_attention:
        model.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=device).manual_seed(args.seed)
    
    # -------------------- Training --------------------
    inferer = Inferer(
        model = model,
        unet = unet_self,
        test_dataloader=test_loader,
        device=device,
        timestep=[999],
        output_dir=save_dir,
        generator = generator,
    )
    inferer.run()


if __name__ == '__main__':
    main()