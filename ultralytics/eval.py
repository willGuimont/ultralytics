import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path

import cv2
# Added plotting imports
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask_utils
import torch
import yaml
from torchmetrics.detection import MeanAveragePrecision

from ultralytics import YOLO

STREAM = False
RETINA_MASK = True
CONF = 0.25
IOU = 0.5
TTAUGMENT = False
# Enable class-agnostic NMS (useful if class overlap is common)
AGNOSTIC_NMS = False

SMALL_SIZE = False
REVERSE_IMGSZ = True
HARDCODED_IMGSZ = (1092, 1456)

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
            logging.warning(f'Could not overlay mask, size miss match (mask={mask.shape} != {(h, w)}=img')
            exit(1)
        color = rng.integers(0, 255, size=3, dtype=np.uint8)
        color_mask = np.zeros_like(out)
        color_mask[mask.astype(bool)] = color
        out = cv2.addWeighted(out, 1.0, color_mask, alpha, 0)
    return out


def remove_padding(masks, imgsz):
    # See Letterbox.__call__ for values
    top, bottom = 8, -8
    left, right = None, None
    masks = masks[:, top:bottom, left:right]
    masks = [cv2.resize(mask.cpu().numpy().data.squeeze(), imgsz[::-1], interpolation=cv2.INTER_NEAREST) for mask in
             masks]
    return masks


def compute_map(run_path):
    if SMALL_SIZE:
        logging.warning('PUT SMALL_SIZE TO False FOR REAL EVAL')

    if REVERSE_IMGSZ:
        logging.warning('imgsz is reversed!')

    run_path = Path(run_path)
    args_path = run_path / 'args.yaml'
    weights_path = run_path / 'weights'
    run_name = run_path.name
    res = re.search(RUN_NAME_REGEX, run_name)
    split_num, kfold_index = res[1], res[2]

    with open(args_path, 'r') as f:
        args = yaml.safe_load(f)

    model = weights_path / 'best.pt'

    imgsz = 640 if SMALL_SIZE else args['imgsz'][::-1]
    if HARDCODED_IMGSZ is not None:
        logging.warning(f'Using hardcoded value for imgsz: {HARDCODED_IMGSZ}')
        imgsz = HARDCODED_IMGSZ

    split_path = Path(f'/datasets/vhr-silva/subsets/{split_num}/split_{kfold_index}.json')

    with open(split_path, 'r') as f:
        data = yaml.safe_load(f)

    try:
        model = YOLO(model)
    except FileNotFoundError:
        print(f'Error loading {run_path}')
        return None
    model.eval()

    model.val(data=args['data'], split='test')

    anns_per_img = defaultdict(list)
    for ann in data['annotations']:
        anns_per_img[ann['image_id']].append(ann)

    preds = []
    targets = []
    for idx, img in enumerate(sorted(data['images'], key=lambda d: d['id'])):
        res = model(img['file_name'], imgsz=imgsz, device='cuda', stream=STREAM, retina_masks=RETINA_MASK, conf=CONF,
                    iou=IOU, augment=TTAUGMENT, agnostic_nms=AGNOSTIC_NMS, verbose=False)[0]
        masks = [] if res.masks is None else remove_padding(res.masks, imgsz)
        masks = [torch.tensor(m).unsqueeze(0) for m in masks]
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
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                axes[0].imshow(gt_overlay)
                axes[0].set_title('GT Masks')
                axes[0].axis('off')
                axes[1].imshow(pred_overlay)
                axes[1].set_title('Pred Masks')
                axes[1].axis('off')
                fig.tight_layout()
                fig.show()
                plt.close(fig)

    output = dict()

    map = MeanAveragePrecision(iou_type='segm')
    map.update(preds, targets)
    metrics = map.compute()
    output['map'] = metrics

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate evaluation information")
    parser.add_argument('--run-path', required=True, help='Path to result folder')
    args = parser.parse_args()

    metrics = compute_map(args.run_path)

    for metric_name, metric in metrics.items():
        print('-' * 20)
        print(metric_name)
        for k, v in metric.items():
            print(f"{k}: {v}")
