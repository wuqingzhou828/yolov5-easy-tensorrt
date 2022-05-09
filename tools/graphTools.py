import numpy as np
import onnx
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn
from onnx import shape_inference


def check_gs(gs_graph):
    onnx_model = gs.export_onnx(gs_graph)
    onnx_model = shape_inference.infer_shapes(onnx_model)
    onnx.checker.check_model(onnx_model)
    return onnx_model


def findNode(gs_graph, name):
    Node = None
    for node in gs_graph.nodes:
        if node.name == name:
            Node = node
    return Node


class AdditionNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.convert_matrix = torch.tensor(
            [[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
            dtype=torch.float32)

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        return box, score


def registerNMS(gs_graph,
                detections_per_img=100,
                score_thresh=0.25,
                nms_thresh=0.45):
    op_inputs = gs_graph.outputs
    op = "EfficientNMS_TRT"
    attrs = {
        "plugin_version": "1",
        "background_class": -1,  # no background class
        "max_output_boxes": detections_per_img,
        "score_threshold": score_thresh,
        "iou_threshold": nms_thresh,
        "score_activation": False,
        "box_coding": 0,
    }
    # NMS Outputs
    output_num_detections = gs.Variable(
        name="num_detections",
        dtype=np.int64,
        shape=[1, 1],
    )  # A scalar indicating the number of valid detections per batch image.
    output_boxes = gs.Variable(
        name="detection_boxes",
        dtype=np.float32,
        shape=[1, detections_per_img, 4],
    )
    output_scores = gs.Variable(
        name="detection_scores",
        dtype=np.float32,
        shape=[1, detections_per_img],
    )
    output_labels = gs.Variable(
        name="detection_classes",
        dtype=np.int64,
        shape=[1, detections_per_img],
    )
    op_outputs = [
        output_num_detections, output_boxes, output_scores, output_labels
    ]
    gs_graph.layer(op=op,
                   name="batched_nms",
                   inputs=op_inputs,
                   outputs=op_outputs,
                   attrs=attrs)
    gs_graph.outputs = op_outputs
    gs_graph.cleanup().toposort()
    return gs_graph
