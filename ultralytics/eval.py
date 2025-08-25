import argparse
import numpy as np
import json
from collections import defaultdict

import torch
from pycocotools import mask as mask_utils
from torchmetrics.detection import MeanAveragePrecision

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation mAP (segm) using torchmetrics MeanAveragePrecision")
    parser.add_argument('--model', type=str, default='yolo11s-seg.pt', help='Path to YOLO segmentation model weights')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset YAML or directory/txt of images')
    parser.add_argument('--json', type=str, required=True, help='Path to dataset JSON (from the split to evaluate on)')
    parser.add_argument('--split', type=int, required=True, help='Image split identifier')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for predictions')
    args = parser.parse_args()

    model_pt = args.model
    data_yaml_path = args.data
    data_json_path = args.json
    split = args.split
    conf = args.conf

    with open(data_json_path, 'r') as f:
        data = json.load(f)

    model = YOLO(args.model)

    imgsz = [1456, 1092]
    if split == 16:
        imgsz = [728, 546]

    all_images = [img['file_name'].replace('tif', f'tif-{split}', 1) for img in data['images']]
    results = model(all_images, imgsz=imgsz, device='cuda', retina_masks=True)

    preds = []
    for img, res in zip(data['images'], results):
        masks = [] if res.masks is None else res.masks
        masks = [m.data for m in masks]
        masks = torch.vstack(masks) if len(masks) > 0 else torch.tensor([])
        preds.append(dict(
            masks=masks.to(torch.uint8).cpu(),
            scores=res.boxes.conf.cpu(),
            labels=res.boxes.cls.detach().clone().cpu().to(torch.uint8),
            image_id=img['id']
        ))

    anns_per_img = defaultdict(list)
    for ann in data['annotations']:
        anns_per_img[ann['image_id']].append(ann)
    targets = []
    for img in data['images']:
        anns = anns_per_img[img['id']]
        masks = [mask_utils.decode(ann['segmentation']) for ann in anns]
        labels = [ann['category_id'] for ann in anns]
        targets.append(dict(
            masks=torch.tensor(np.array(masks)),
            labels=torch.tensor(labels, dtype=torch.long)
        ))

    # data['preds'] = preds
    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, targets)
    metrics = metric.compute()

    # Pretty print
    print('Segmentation mAP metrics:')
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
