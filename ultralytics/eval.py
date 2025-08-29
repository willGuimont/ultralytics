import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils
import torch
import yaml
from torchmetrics.detection import MeanAveragePrecision

from ultralytics import YOLO

STREAM = False
RETINA_MASK = False
CONF = 0.25
IOU = 0.7
TTAUGMENT = False
# Enable class-agnostic NMS (useful if class overlap is common)
AGNOSTIC_NMS = False
SMALL_SIZE = False

RUN_NAME_REGEX = re.compile(r'[a-z0-9.-]*\.json_([a-z-0-9]*)_kfold_([0-9])\.yaml')


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation information")
    parser.add_argument('--run-path', required=True, help='Path to result folder')
    args = parser.parse_args()

    if SMALL_SIZE:
        logging.warning('PUT SMALL_SIZE TO False FOR REAL EVAL')

    run_path = Path(args.run_path)
    args_path = run_path / 'args.yaml'
    weights_path = run_path / 'weights'
    run_name = run_path.name
    res = re.search(RUN_NAME_REGEX, run_name)
    split_num, kfold_index = res[1], res[2]

    with open(args_path, 'r') as f:
        args = yaml.safe_load(f)

    model = weights_path / 'best.pt'
    imgsz = 640 if SMALL_SIZE else args['imgsz']
    split_path = Path(f'/datasets/vhr-silva/subsets/{split_num}/split_{kfold_index}.json')

    with open(split_path, 'r') as f:
        data = yaml.safe_load(f)

    model = YOLO(model)
    model.eval()

    anns_per_img = defaultdict(list)
    for ann in data['annotations']:
        anns_per_img[ann['image_id']].append(ann)

    preds = []
    targets = []
    for img in sorted(data['images'], key=lambda d: d['id']):
        res = model(img['file_name'], imgsz=imgsz, device='cuda', stream=STREAM, retina_masks=RETINA_MASK, conf=CONF,
                    iou=IOU, augment=TTAUGMENT, agnostic_nms=AGNOSTIC_NMS)[0]
        masks = [] if res.masks is None else res.masks
        masks = [m.data for m in masks]
        masks = torch.vstack(masks) if len(masks) > 0 else torch.tensor([])
        preds.append(dict(
            masks=masks.to(torch.uint8).cpu(),
            scores=res.boxes.conf.cpu(),
            labels=res.boxes.cls.detach().clone().cpu().to(torch.uint8),
            speed=res.speed
        ))

        anns = anns_per_img[img['id']]
        masks = [mask_utils.decode(ann['segmentation']) for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        targets.append(dict(
            masks=torch.tensor(np.array(masks)),
            labels=torch.tensor(labels, dtype=torch.long)
        ))

    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, targets)
    metrics = metric.compute()

    # Pretty print
    print('Segmentation mAP metrics:')
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
