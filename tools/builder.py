import torch
import tensorrt as trt
import numpy as np
from collections import OrderedDict,namedtuple
from tools import clip_coords


def buildEngine(onnxsave,enginesave,workspace=8,verbose=False,fp16=False,dynamic=False,noTF32=False):
    logger = trt.Logger(trt.Logger.INFO)
    if verbose:
        logger.min_severity = trt.Logger.Severity.VERBOSE
    trt.init_libnvinfer_plugins(logger, namespace="")
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnxsave)):
        raise RuntimeError(f'failed to load ONNX file: {str(onnxsave)}')
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if noTF32:
        config.clear_flag(trt.BuilderFlag.TF32)
    config.set_flag(trt.BuilderFlag.STRICT_TYPES)
    with builder.build_serialized_network(network, config) as engine, open(str(enginesave), 'wb') as t:
        t.write(engine)
    return enginesave.stat().st_size >= 1024

def loadEngine(engine,device=torch.device("cuda")):
    Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, namespace="")
    with open(str(engine), 'rb') as f, trt.Runtime(logger) as runtime:
        model = runtime.deserialize_cuda_engine(f.read())
    bindings = OrderedDict()
    for index in range(model.num_bindings):
        name = model.get_binding_name(index)
        dtype = trt.nptype(model.get_binding_dtype(index))
        shape = tuple(model.get_binding_shape(index))
        data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
        bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
    binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
    context = model.create_execution_context()
    return bindings,binding_addrs,context

def warm(imsz,binding_addrs,context,device=torch.device("cuda"),times=10):
    for i in range(times):
        tmp = torch.randn(imsz).to(device)
        binding_addrs['images'] = int(tmp.data_ptr())
        context.execute_v2(list(binding_addrs.values()))
    return True

def inferImg(im,bindings,binding_addrs,context):
    result = []
    binding_addrs['images'] = int(im.data_ptr())
    context.execute_v2(list(binding_addrs.values()))
    num_detections = bindings['num_detections'].data
    detection_boxes = bindings['detection_boxes'].data
    detection_scores = bindings['detection_scores'].data
    detection_classes = bindings['detection_classes'].data
    for i in range(num_detections[0][0]):
        box = clip_coords(detection_boxes[0][i],im.shape[:2])
        score = detection_scores[0][i]
        cls = detection_classes[0][i]
        result.append((box,score,cls))
    return result

