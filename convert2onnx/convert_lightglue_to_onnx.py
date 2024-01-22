import argparse
import os

import numpy as np
import onnx
import torch
from ops import register_aten_sdpa
from lightglue import LightGlue


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def reduce_l2(desc):
    dn = np.linalg.norm(desc, ord=2, axis=1)  # Compute the norm.
    desc = desc / np.expand_dims(dn, 1)  # Divide by norm to normalize.
    return desc


def main():
    parser = argparse.ArgumentParser(
        description='script to convert superpoint model from pytorch to onnx')
    parser.add_argument('--weight_file', default="./weights/superpoint_lightglue.pth",
                        help="pytorch weight file (.pth)")
    parser.add_argument('--output_dir', default="output", help="output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    weight_file = args.weight_file

    # load model
    extractor_type = "superpoint"
    flash = False
    lightglue = LightGlue(extractor_type, flash=flash)
    pytorch_total_params = sum(p.numel() for p in lightglue.parameters())
    print('total number of params: ', pytorch_total_params)

    # initialize model with the pretrained weights
    map_location = lambda storage, loc: storage
    if torch.cuda.is_available():
        map_location = None
    lightglue.load_state_dict(torch.load(weight_file, map_location=map_location), strict=False)
    lightglue.eval()

    # if (
    #     hasattr(torch.nn.functional, "scaled_dot_product_attention")
    #     and torch.__version__ < "2.1"
    # ):
    register_aten_sdpa(opset_version=14)

    # create input to the model for onnx trace
    x0 = torch.from_numpy(np.random.randint(low=0, high=751, size=(1, 512)))
    y0 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, 512)))
    kpts0 = torch.stack((x0, y0), 2).float()
    scores0 = torch.randn(1, 512)
    desc0 = torch.randn(1, 512, 256)
    x1 = torch.from_numpy(np.random.randint(low=0, high=751, size=(1, 512)))
    y1 = torch.from_numpy(np.random.randint(low=0, high=479, size=(1, 512)))
    kpts1 = torch.stack((x1, y1), 2).float()
    scores1 = torch.randn(1, 512)
    desc1 = torch.randn(1, 512, 256)
    onnx_filename = os.path.join(output_dir,
                                 # `weight_file` is a command-line argument that specifies the path to
                                 # the PyTorch weight file (.pth) for the superpoint model.
                                 weight_file.split("/")[-1].split(".")[0] + ".onnx")


    # Export the model
    with torch.autocast("cuda", enabled=False):
        torch.onnx.export(lightglue,  # model being run
                        (kpts0, 
                         kpts1, 
                         desc0, 
                         desc1),  # model input (or a tuple for multiple inputs), kpts1, desc0, desc1),  # model input (or a tuple for multiple inputs)
                        onnx_filename,  # where to save the model (can be a file or file-like object)
                        export_params=True,  # store the trained parameter weights inside the model file
                        opset_version=16,  # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names=["kpts0",  # batch x feature_number0 x 2 (1,736,2)
                                    "kpts1",  # (1,858,2)
                                    "desc0",  # (1,736,256)
                                    "desc1",  # (1,858,256)
                                    ],  # the model input names
                        output_names=["scores"],  # the model output names
                        dynamic_axes={
                                "kpts0": {1: "num_keypoints0"},
                                "kpts1": {1: "num_keypoints1"},
                                "desc0": {1: "num_keypoints0"},
                                "desc1": {1: "num_keypoints1"},
                                "scores": {1: "num_keypoints0",2:"num_keypoints1"},
                               
                            },
                        )

    # check onnx model
    onnx_model = onnx.load(onnx_filename)
    onnx.checker.check_model(onnx_model)

    print("Exported model has been checked with ONNXRuntime.")


if __name__ == '__main__':
    main()
