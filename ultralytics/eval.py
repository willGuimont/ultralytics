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

# Added plotting imports
import matplotlib.pyplot as plt
import cv2

STREAM = False
RETINA_MASK = False
CONF = 0.25
IOU = 0.7
TTAUGMENT = False
# Enable class-agnostic NMS (useful if class overlap is common)
AGNOSTIC_NMS = False
SMALL_SIZE = False

RUN_NAME_REGEX = re.compile(r'[a-z0-9.-]*\.json_([a-z-0-9]*)_kfold_([0-9])\.yaml')


def overlay_masks(base_img, masks_tensor, alpha=0.4, seed=0):
    """Return image with colored masks overlay. base_img expected BGR or RGB (we keep channels)."""
    if masks_tensor.numel() == 0:
        return base_img.copy()
    if isinstance(masks_tensor, torch.Tensor):
        m = masks_tensor.cpu().numpy()
    else:
        m = masks_tensor
    out = base_img.copy()
    rng = np.random.default_rng(seed)
    h, w = out.shape[:2]
    for i, mask in enumerate(m):
        if mask.shape != (h, w):  # skip size mismatch
            logging.warning("Could not overlay mask, size missmatch")
            continue
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        color_mask = np.zeros_like(out)
        color_mask[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, color_mask, alpha, 0)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate evaluation information")
    parser.add_argument('--run-path', required=True, help='Path to result folder')
    # New plotting args
    parser.add_argument('--plot-first', action='store_true', help='Plot first image pred vs gt')
    parser.add_argument('--plot-first-path', default='first_pred_gt.png', help='Output path for first plot')
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
    for idx, img in enumerate(sorted(data['images'], key=lambda d: d['id'])):
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

        # Plot first sample
        if idx == 0:
            # Load image (BGR), convert to RGB for matplotlib
            bgr = cv2.imread(img['file_name'])
            if bgr is None:
                logging.warning('Could not read first image for plotting.')
            else:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                pred_overlay = overlay_masks(rgb, preds[-1]['masks'])
                gt_overlay = overlay_masks(rgb, targets[-1]['masks'], seed=42)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb); axes[0].set_title('Image'); axes[0].axis('off')
                axes[1].imshow(gt_overlay); axes[1].set_title('GT Masks'); axes[1].axis('off')
                axes[2].imshow(pred_overlay); axes[2].set_title('Pred Masks'); axes[2].axis('off')
                fig.tight_layout()
                fig.show()
                plt.close(fig)

    metric = MeanAveragePrecision(iou_type='segm')
    metric.update(preds, targets)
    metrics = metric.compute()

    print('Segmentation mAP metrics:')
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
