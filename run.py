import argparse
import tempfile
from pathlib import Path

import onnx_graphsurgeon as gs
import torch

import onnx
from tools import AdditionNet, buildEngine, check_gs, registerNMS


def main(opt):
    onnxfile = Path(opt.onnxfile)
    onnxsave = Path(opt.onnxsave)
    enginesave = Path(opt.enginesave)
    temp_onnx = tempfile.NamedTemporaryFile(suffix=".onnx")
    temp = None
    opset = opt.opset
    verbose = opt.verbose
    detections_per_img = opt.maxObj
    score_thresh = opt.score
    nms_thresh = opt.nms_score
    fp16 = opt.fp16
    workspace = opt.GiB
    saveTemp = opt.saveTemp
    if saveTemp:
        temp = onnxfile.parent / (onnxfile.stem + "_step")
        temp = temp.with_suffix(".onnx")
    in_name = ["tmpInput"]
    out_name = ["boxes", "scores"]
    onnx_graph = onnx.load(str(onnxfile))
    gs_graph = gs.import_onnx(onnx_graph)
    im = torch.randn(gs_graph.outputs[0].shape)
    net = AdditionNet()
    net.eval()
    torch.onnx.export(net,
                      im,
                      temp_onnx.name,
                      verbose=verbose,
                      opset_version=opset,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=in_name,
                      output_names=out_name,
                      dynamic_axes=None)
    tmp_graph = gs.import_onnx(onnx.load(temp_onnx.name))
    for node in tmp_graph.nodes:
        node.name = "Addition_" + node.name
        gs_graph.nodes.append(node)
    for node in tmp_graph.nodes:
        try:
            name = node.inputs[0].name
        except:
            pass
        else:
            if name == in_name[0]:
                node.inputs[0] = gs_graph.outputs[0]
    gs_graph.outputs = tmp_graph.outputs
    gs_graph.cleanup().toposort()
    onnx_model = check_gs(gs_graph)
    if saveTemp:
        onnx.save(onnx_model, str(temp))
    gs_graph = registerNMS(gs_graph, detections_per_img, score_thresh,
                           nms_thresh)
    onnx.save(gs.export_onnx(gs_graph), str(onnxsave))
    if not enginesave.exists():
        status = buildEngine(onnxsave,
                             enginesave,
                             workspace,
                             verbose,
                             fp16,
                             noTF32=True)
        if status:
            print(f"Build {enginesave.name} finish")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnxfile',
                        type=str,
                        default="onnx/yolov5s.onnx",
                        help='origin onnx path')
    parser.add_argument('--onnxsave',
                        type=str,
                        default="onnx/yolov5s_nms.onnx",
                        help='register onnx path')
    parser.add_argument('--enginesave',
                        type=str,
                        default="engine/yolov5s.engine",
                        help='save engine path')

    parser.add_argument('--opset', type=int, default=13, help='opset version')
    parser.add_argument('--GiB',
                        type=int,
                        default=8,
                        help='workspace for engine builder')
    parser.add_argument('--maxObj',
                        type=int,
                        default=100,
                        help='max object in one image')
    parser.add_argument('--score',
                        type=float,
                        default=0.25,
                        help='min score for nms attrs')
    parser.add_argument('--nms_score',
                        type=float,
                        default=0.25,
                        help='min mns_score for nms attrs')

    parser.add_argument('--verbose',
                        action='store_true',
                        help='verbose log print')
    parser.add_argument('--fp16', action='store_true', help='fp16 exporter')
    parser.add_argument('--saveTemp',
                        action='store_true',
                        help='save temp onnx or not')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
