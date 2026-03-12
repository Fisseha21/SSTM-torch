# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:23:35 2026

@author: Fisseha
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import argparse

import cv2
import numpy as np
import torch

from .core.sstm import SSTM
from .core.utils import frame_utils, flow_viz
from .core.utils.utils import InputPadder


ImageInput = Union[str, Path, np.ndarray, torch.Tensor]


def _default_model_args() -> argparse.Namespace:
    return argparse.Namespace(
        mixed_precision=False,
        alternate_corr=False,
        num_heads=1,
        position_only=False,
        position_and_content=False,
    )


def read_image(img: ImageInput) -> torch.Tensor:
    if isinstance(img, (str, Path)):
        img_path = str(img)
        arr = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if arr is None:
            raise ValueError(f"Could not read image: {img_path}")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(arr).permute(2, 0, 1).float()

    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)

        if img.ndim != 3:
            raise ValueError(f"Unsupported numpy image shape: {img.shape}")

        if img.shape[0] == 3 and img.shape[-1] != 3:
            return torch.from_numpy(img).float()

        if img.shape[-1] == 3:
            return torch.from_numpy(img).permute(2, 0, 1).float()

        raise ValueError(f"Unsupported numpy image shape: {img.shape}")

    if isinstance(img, torch.Tensor):
        if img.ndim == 2:
            return img.unsqueeze(0).repeat(3, 1, 1).float()

        if img.ndim != 3:
            raise ValueError(f"Unsupported tensor image shape: {tuple(img.shape)}")

        if img.shape[0] == 3:
            return img.float()

        if img.shape[-1] == 3:
            return img.permute(2, 0, 1).float()

        raise ValueError(f"Unsupported tensor image shape: {tuple(img.shape)}")

    raise TypeError(f"Unsupported input type: {type(img)}")


def check_same_shape(*imgs: torch.Tensor) -> None:
    if len(imgs) < 2:
        return

    ref_shape = imgs[0].shape
    for idx, img in enumerate(imgs[1:], start=2):
        if img.shape != ref_shape:
            raise ValueError(
                f"All input images must have the same shape. "
                f"Image 1 has shape {tuple(ref_shape)}, "
                f"but image {idx} has shape {tuple(img.shape)}"
            )


def load_model(
    checkpoint_path: Union[str, Path],
    model_args: Optional[argparse.Namespace] = None,
    device: Optional[torch.device] = None,
) -> torch.nn.Module:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_args is None:
        model_args = _default_model_args()

    model = torch.nn.DataParallel(SSTM(model_args))
    state_dict = torch.load(str(checkpoint_path), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def _infer_flow_triplet_core(
    model: torch.nn.Module,
    img1: ImageInput,
    img2: ImageInput,
    img3: ImageInput,
    iters: int = 24,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t1 = read_image(img1)
    t2 = read_image(img2)
    t3 = read_image(img3)

    check_same_shape(t1, t2, t3)

    image1 = t1.unsqueeze(0).to(device)
    image2 = t2.unsqueeze(0).to(device)
    image3 = t3.unsqueeze(0).to(device)

    padder = InputPadder(image1.shape)
    image1, image2, image3 = padder.pad(image1, image2, image3)

    model = model.to(device)
    model.eval()

    run_model = model.module if hasattr(model, "module") else model

    _, _, flow_pr1, flow_pr2 = run_model(
        image1, image2, image3, iters=iters, test_mode=True
    )

    f12 = padder.unpad(flow_pr1[0]).permute(1, 2, 0).cpu().numpy()
    f23 = padder.unpad(flow_pr2[0]).permute(1, 2, 0).cpu().numpy()

    return f12, f23


@torch.no_grad()
def infer_flow_pair(
    model: torch.nn.Module,
    img1: ImageInput,
    img2: ImageInput,
    iters: int = 24,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    f12, _ = _infer_flow_triplet_core(
        model=model,
        img1=img1,
        img2=img2,
        img3=img2,
        iters=iters,
        device=device,
    )
    return f12


@torch.no_grad()
def infer_flow_triplet(
    model: torch.nn.Module,
    img1: ImageInput,
    img2: ImageInput,
    img3: ImageInput,
    iters: int = 24,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    return _infer_flow_triplet_core(
        model=model,
        img1=img1,
        img2=img2,
        img3=img3,
        iters=iters,
        device=device,
    )


def save_flow(flow: np.ndarray, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_utils.writeFlow(str(output_path), flow)


def save_vis(flow: np.ndarray, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis = flow_viz.flow_to_image(flow)
    vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), vis_bgr)


def estimate_flow(
    model_path: Union[str, Path],
    img1: ImageInput,
    img2: ImageInput,
    img3: Optional[ImageInput] = None,
    *,
    iters: int = 24,
    output_dir: Optional[Union[str, Path]] = None,
    save_flo: bool = False,
    save_vis_outputs: bool = False,
    model_args: Optional[argparse.Namespace] = None,
    device: Optional[torch.device] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    High-level entry point for PyPI users.

    Pair input:
        estimate_flow(model_path, img1, img2) -> f12

    Triplet input:
        estimate_flow(model_path, img1, img2, img3) -> (f12, f23)
    """
    model = load_model(model_path, model_args=model_args, device=device)

    if img3 is None:
        f12 = infer_flow_pair(
            model=model,
            img1=img1,
            img2=img2,
            iters=iters,
            device=device,
        )

        if output_dir is not None and (save_flo or save_vis_outputs):
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            if save_flo:
                save_flow(f12, output_dir / "flow0001.flo")

            if save_vis_outputs:
                save_vis(f12, output_dir / "flow0001.png")

        return f12

    f12, f23 = infer_flow_triplet(
        model=model,
        img1=img1,
        img2=img2,
        img3=img3,
        iters=iters,
        device=device,
    )

    if output_dir is not None and (save_flo or save_vis_outputs):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if save_flo:
            save_flow(f12, output_dir / "flow0001.flo")
            save_flow(f23, output_dir / "flow0002.flo")

        if save_vis_outputs:
            save_vis(f12, output_dir / "flow0001.png")
            save_vis(f23, output_dir / "flow0002.png")

    return f12, f23