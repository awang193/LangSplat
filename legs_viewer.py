#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import traceback
from gaussian_renderer import render, network_gui
import sys
from scene import GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams


def viewing(dataset, opt, pipe, checkpoint):
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.training_setup(opt)

    if checkpoint:
        (model_params, _) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    print("Listening")

    while True:
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, _, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer, show_legs_image = network_gui.receive2()
                if custom_cam != None:
                    # CUSTOM CODE
                    render_pkg = render(custom_cam, gaussians, pipe, background, opt, scaling_modifer)
                    if show_legs_image:
                        net_image = render_pkg["language_feature_image"]
                    else:
                        net_image = render_pkg["render"]

                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())

                network_gui.send(net_image_bytes, dataset.source_path)
                if not keep_alive:
                    break
            except Exception:
                print(traceback.format_exc())
                network_gui.conn = None
                return

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)

    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path + f"_{str(args.feature_level)}"
    print("Viewing " + args.model_path)

    # Start GUI server, configure and run data exchange code
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    viewing(
        lp.extract(args), 
        op.extract(args),
        pp.extract(args), 
        args.start_checkpoint, 
    )
