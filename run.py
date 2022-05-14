import argparse
from pathlib import Path

import onnx
from tools import GraphFix, buildEngine


def main(opt):
    onnxfile = Path(opt.onnxfile)
    onnxsave = Path(opt.onnxsave)
    enginesave = Path(opt.enginesave)
    verbose = opt.verbose
    detections_per_img = opt.maxObj
    score_thresh = opt.score
    nms_thresh = opt.nms_score
    fp16 = opt.fp16
    workspace = opt.GiB
    graphFix = GraphFix(onnxfile,
                        topk_all=detections_per_img,
                        conf_thres=score_thresh,
                        iou_thres=nms_thresh)
    onnx_graph = graphFix.registerNMS()
    onnx.save(onnx_graph, onnxsave)
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
                        default=0.45,
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
