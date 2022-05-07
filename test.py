import argparse
import time
from pathlib import Path

import cv2
import torch

from tools import inferImg, loadEngine, preprocess, warm


def main(opt):
    image = Path(opt.image)
    engine = Path(opt.engine)
    show = opt.show
    device = torch.device(opt.device)
    imgsz = opt.imgsz
    warmup = opt.warmup
    bindings, binding_addrs, context = loadEngine(engine, device)
    if warmup:
        status = warm([1, 3, *imgsz], binding_addrs, context, times=warmup)
        if status:
            print(f"Finish warmup {warmup} times")
    image = cv2.imread(str(image))
    image_copy = image.copy()
    image, ratio, dwdh = preprocess(image, imgsz, lb=True, device=device)
    start = time.perf_counter()
    result = inferImg(image, bindings, binding_addrs, context)
    print(f"Infer image finish use {time.perf_counter()-start} s")
    for box, score, cls in result:
        box = box.round().int()
        cv2.rectangle(image_copy,
                      box[:2],
                      box[2:4],
                      color=(0, 255, 255),
                      thickness=3)
    if show:
        cv2.imshow("result", image_copy)
        cv2.waitKey(0)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        type=str,
                        default="engine/yolov5s.engine",
                        help='save engine path')
    parser.add_argument('--engine',
                        type=str,
                        default="images/bus.jpg",
                        help='test image path')
    parser.add_argument('--device',
                        default='cuda',
                        help='cuda device, i.e. 0 or 0,1,2,3')
    parser.add_argument('--imgsz',
                        '--img',
                        '--img-size',
                        nargs='+',
                        type=int,
                        default=[640, 640],
                        help='image (h, w)')

    parser.add_argument('--warmup', type=int, default=10, help='warm up times')
    parser.add_argument('--show', action='store_true', help='show infer image')

    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
