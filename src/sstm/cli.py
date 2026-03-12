# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 19:24:08 2026

@author: Fisseha
"""

from pathlib import Path
import argparse
import torch

from .inference import load_model, infer_flow_triplet, save_flow, save_vis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSTM optical flow inference")
    parser.add_argument("--model", required=True, help="Path to checkpoint")
    parser.add_argument("--img1", required=True, help="Path to first image")
    parser.add_argument("--img2", required=True, help="Path to second image")
    parser.add_argument("--img3", default=None, help="Optional third image")
    parser.add_argument("--outdir", default="outputs", help="Output directory")
    parser.add_argument("--iters", default=24, type=int, help="Number of refinement iterations")

    parser.add_argument("--save-flo", action="store_true", help="Save .flo output(s)")
    parser.add_argument("--save-vis", action="store_true", help="Save flow visualization image(s)")
    parser.add_argument("--save-f2", action="store_true", help="Also save/display f2 when img3 is not provided")

    parser.add_argument("--mixed-precision", action="store_true", help="Use mixed precision")
    parser.add_argument("--alternate-corr", action="store_true", help="Use efficient correlation implementation")
    parser.add_argument("--num-heads", default=1, type=int, help="Number of attention heads")
    parser.add_argument("--position-only", action="store_true", help="Use position-only attention")
    parser.add_argument("--position-and-content", action="store_true", help="Use position and content attention")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args, device=device)

    f1, f2 = infer_flow_triplet(
        model=model,
        i1=args.img1,
        i2=args.img2,
        i3=args.img3,
        iters=args.iters,
        device=device,
    )

    save_second = args.img3 is not None or args.save_f2
    outdir = Path(args.outdir)

    print(f"f1 shape: {f1.shape}")
    print(f"f2 shape: {f2.shape}")

    if args.save_flo:
        outdir.mkdir(parents=True, exist_ok=True)
        f1_path = outdir / "flow0001.flo"
        save_flow(f1, f1_path)
        print(f"Saved f1 to: {f1_path}")

        if save_second:
            f2_path = outdir / "flow0002.flo"
            save_flow(f2, f2_path)
            print(f"Saved f2 to: {f2_path}")

    if args.save_vis:
        outdir.mkdir(parents=True, exist_ok=True)
        f1_vis_path = outdir / "flow0001.png"
        save_vis(f1, f1_vis_path)
        print(f"Saved f1 visualization to: {f1_vis_path}")

        if save_second:
            f2_vis_path = outdir / "flow0002.png"
            save_vis(f2, f2_vis_path)
            print(f"Saved f2 visualization to: {f2_vis_path}")