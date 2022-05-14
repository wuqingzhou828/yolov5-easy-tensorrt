from collections import OrderedDict

import numpy as np
import onnx_graphsurgeon as gs
import torch
import torch.nn as nn

import onnx
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


class GraphFix():
    def __init__(self, file, topk_all=100, conf_thres=0.25, iou_thres=0.45):
        self.file = file
        gs_graph = gs.import_onnx(onnx.load(self.file))
        self.gs_graph = gs_graph
        self.topk_all = topk_all if topk_all else 100
        self.conf_thres = conf_thres if conf_thres else 0.25
        self.iou_thres = iou_thres if iou_thres else 0.45
        self.Attrs = [[5, 9999, 2, 1], [4, 5, 2, 1], [0, 4, 2, 1]]
        self.Matrix = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0],
                                [-0.5, 0.0, 0.5, 0.0], [0.0, -0.5, 0.0, 0.5]],
                               dtype=np.float32)

    def registerNMS(self):
        s = 'Addition_'
        lastNode = [
            node for node in self.gs_graph.nodes
            if node.outputs and node.outputs[0].name == 'output'
        ][0]
        out_shape = lastNode.outputs[0].shape
        mul_inputs = []
        matmul_inputs = [
            None,
            gs.Constant(name=f'{s}MatMul_inp_1', values=self.Matrix)
        ]
        for i, attr in enumerate(self.Attrs):
            Slice_inp = [lastNode.outputs[0]] + [
                gs.Constant(name=f'{s}Slice_{i}_inp_{j}',
                            values=np.array([val]))
                for j, val in enumerate(attr)
            ]
            Slice_out = gs.Variable(name=f'{s}Slice_{i}_out')
            Slice = gs.Node(name=f'{s}Slice_{i}',
                            op='Slice',
                            inputs=Slice_inp,
                            outputs=[Slice_out])
            self.gs_graph.nodes.append(Slice)
            if i < 2:
                mul_inputs.append(Slice_out)
            elif i == 2:
                matmul_inputs[0] = Slice_out
        mut_output = gs.Variable(name='NMS_input_0',
                                 shape=out_shape[:2] + [out_shape[2] - 5],
                                 dtype=np.float32)
        matmut_output = gs.Variable(name='NMS_input_1',
                                    shape=out_shape[:2] + [4],
                                    dtype=np.float32)
        Mul = gs.Node(name=f'{s}Mul_0',
                      op='Mul',
                      inputs=mul_inputs,
                      outputs=[mut_output])
        self.gs_graph.nodes.append(Mul)
        MatMul = gs.Node(name=f'{s}MatMul_0',
                         op='MatMul',
                         inputs=matmul_inputs,
                         outputs=[matmut_output])
        self.gs_graph.nodes.append(MatMul)
        op = 'EfficientNMS_TRT'
        attrs = OrderedDict({
            'plugin_version': '1',
            'background_class': -1,  # no background class
            'max_output_boxes': self.topk_all,
            'score_threshold': self.conf_thres,
            'iou_threshold': self.iou_thres,
            'score_activation': False,
            'box_coding': 0
        })
        output_num_detections = gs.Variable(name='num_detections',
                                            dtype=np.int32,
                                            shape=[out_shape[0], 1])
        output_boxes = gs.Variable(name='detection_boxes',
                                   dtype=np.float32,
                                   shape=[out_shape[0], self.topk_all, 4])
        output_scores = gs.Variable(name='detection_scores',
                                    dtype=np.float32,
                                    shape=[out_shape[0], self.topk_all])
        output_labels = gs.Variable(name='detection_classes',
                                    dtype=np.int32,
                                    shape=[out_shape[0], self.topk_all])
        op_inputs = [matmut_output, mut_output]
        op_outputs = [
            output_num_detections, output_boxes, output_scores, output_labels
        ]
        TRT = gs.Node(name='batchedNMS',
                      op=op,
                      inputs=op_inputs,
                      outputs=op_outputs,
                      attrs=attrs)
        self.gs_graph.nodes.append(TRT)
        self.gs_graph.outputs = op_outputs
        self.gs_graph.cleanup().toposort()
        return gs.export_onnx(self.gs_graph)
