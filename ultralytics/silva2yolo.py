import ast
import json
import logging
import math
import operator as op
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import yaml
from PIL import Image
from pycocotools import mask as maskUtils

from ultralytics.data.utils import polygon2mask
from ultralytics.utils import LOGGER
from ultralytics.utils import TQDM
from ultralytics.utils.plotting import colors

operators = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.BitXor: op.xor,
    ast.USub: op.neg,
}

functions = {
    "sqrt": math.sqrt,
}


def eval_expr(expr):
    return eval_(ast.parse(expr, mode='eval').body)


def eval_(node):
    match node:
        case ast.Constant(value) if isinstance(value, (int, float)):
            return value
        case ast.BinOp(left, op_node, right):
            return operators[type(op_node)](eval_(left), eval_(right))
        case ast.UnaryOp(op_node, operand):
            return operators[type(op_node)](eval_(operand))
        case ast.Call(func, args, keywords) if isinstance(func, ast.Name):
            if func.id in functions and not keywords:  # allow only whitelisted functions
                return functions[func.id](*(eval_(arg) for arg in args))
            raise TypeError(f"Unsupported function: {func.id}")
        case _:
            raise TypeError(node)


def ann_to_contours(ann, orig_w: int, orig_h: int):
    """Return list of external contours (np.ndarray Nx2) for a COCO annotation (polygon list or RLE).

    Uses the same contour extraction strategy as convert_segment_masks_to_yolo_seg: builds a binary mask per instance
    and extracts cv2.RETR_EXTERNAL / cv2.CHAIN_APPROX_SIMPLE contours. Minimum of 3 points retained.
    """
    seg = ann.get("segmentation")
    if seg is None:
        return []
    mask = None
    try:
        # RLE dict
        if isinstance(seg, dict) and "counts" in seg:
            mask = maskUtils.decode(seg).astype(np.uint8)
        # Polygon(s) list
        elif isinstance(seg, list):
            mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
            # COCO stores either list of poly lists, each poly list is flat coordinates
            for poly in seg:
                if isinstance(poly, (list, tuple)):
                    if len(poly) < 6:  # fewer than 3 points
                        continue
                    arr = np.asarray(poly, dtype=np.int32).reshape(-1, 2)
                    cv2.fillPoly(mask, [arr], 1)
        else:
            return []
    except Exception as e:
        LOGGER.warning(f"Failed to decode segmentation ann_id={ann.get('id')}: {e}")
        return []

    if mask is None:
        return []
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        if c.shape[0] < 3:  # need at least 3 points
            continue
        c = c.squeeze(1)
        if c.ndim != 2 or c.shape[0] < 3:
            continue
        out.append(c)
    return out


def convert_coco(
        subsets_dir: Path,
        save_dir: Path,
        use_segments: bool = False,
        use_keypoints: bool = False,
        filetype: str = 'tif-8',
        flatten_cnts: bool = True,
):
    # Create dataset directory
    save_dir = Path(save_dir)
    for p in save_dir / "labels", save_dir / "images":
        shutil.rmtree(p, ignore_errors=True)
        p.mkdir(parents=True, exist_ok=True)  # make dir

    for subset_path in sorted(d for d in Path(subsets_dir).resolve().iterdir() if d.is_dir()):
        subset_name = subset_path.name
        subset_downsample = eval_expr(subset_name.split('-')[1].replace('_', '/'))
        # Import json
        for json_file in sorted(Path(subset_path).resolve().glob("split_*.json")):
            lname = f"{subset_name}_{json_file.stem}"
            fn = Path(save_dir) / "labels" / lname  # folder name
            fn.mkdir(parents=True, exist_ok=True)
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            images = {f"{x['id']:d}": x for x in data["images"]}
            annotations = defaultdict(list)
            for ann in data["annotations"]:
                annotations[ann["image_id"].__int__()].append(ann)

            image_txt = []
            for img_id, anns in TQDM(annotations.items(), desc=f"Annotations {json_file}"):
                img = images[f"{img_id:d}"]
                # NOTE: images appear downscaled
                orig_h, orig_w = img["height"], img["width"]
                h, w = orig_h // subset_downsample, orig_w // subset_downsample

                f_rel = img["file_name"]

                bboxes = []
                segments = []
                keypoints = []
                orig_shapes = []
                for ann in anns:
                    if ann.get("iscrowd", False):
                        continue
                    cls = ann["category_id"]
                    # Derive contours (list of Nx2 arrays) using unified approach
                    contours = ann_to_contours(ann, orig_w, orig_h) if use_segments else []

                    if use_segments and contours:
                        if flatten_cnts:
                            shape = [len(x) for x in contours]
                            orig_shapes.append(shape)

                            c = np.vstack(contours)
                            c = c.reshape(-1, 2)
                            flattened = (c / np.array([w, h])).reshape(-1).tolist()
                            segments.append([cls] + flattened)

                            x_min, y_min = c.min(axis=0)
                            x_max, y_max = c.max(axis=0)
                            bw = x_max - x_min
                            bh = y_max - y_min
                            cx = x_min + bw / 2
                            cy = y_min + bh / 2
                            bboxes.append([cls, cx / w, cy / h, bw / w, bh / h])
                        else:
                            for c in contours:
                                # Normalize polygon coordinates (still dividing by w,h as in original implementation)
                                poly_norm = (c / np.array([w, h])).reshape(-1).tolist()
                                segments.append([cls] + poly_norm)
                                # Bounding box from contour
                                x_min, y_min = c.min(axis=0)
                                x_max, y_max = c.max(axis=0)
                                bw = x_max - x_min
                                bh = y_max - y_min
                                cx = x_min + bw / 2
                                cy = y_min + bh / 2
                                bboxes.append([cls, cx / w, cy / h, bw / w, bh / h])
                    else:
                        # Fallback to annotation bbox (COCO format tlx,tly,w,h)
                        box = np.array(ann["bbox"], dtype=np.float64)
                        box[:2] += box[2:] / 2  # to center
                        box[[0, 2]] /= w
                        box[[1, 3]] /= h
                        if box[2] > 0 and box[3] > 0:
                            bboxes.append(
                                [cls] + box.tolist()[1:])  # exclude cls duplication pattern; keep cls + normalized box
                            if use_segments:
                                segments.append([])  # placeholder to align indices

                    if use_keypoints and ann.get("keypoints") is not None:
                        kps = (np.array(ann["keypoints"]).reshape(-1, 3) / np.array([w, h, 1])).reshape(-1).tolist()
                        # Attach keypoints to each added instance (bbox count since segments may be empty placeholders)
                        instance_count = len(bboxes) - len(keypoints)
                        for i in range(instance_count):
                            # Associate with most recent bbox instance(s)
                            keypoints.append([cls] + bboxes[-instance_count + i][1:] + kps)

                img_path = Path(f_rel)
                image_out = save_dir / "images" / f"{subset_name}_{json_file.stem}"
                image_out.mkdir(parents=True, exist_ok=True)
                # Copy corresponding image (adjust tif -> tif-8 path as before)
                try:
                    size = subset_name.split('-')[1]
                    shutil.copy((save_dir / str(img_path).replace('tif/', f'tif-{size}/')).with_suffix('.tif'),
                                image_out)
                except Exception as e:
                    LOGGER.warning(f"Image copy failed for {img_path}: {e}")
                with open((fn / img_path.name).with_suffix(".txt"), "w", encoding="utf-8") as file:
                    for i in range(len(bboxes)):
                        if use_keypoints:
                            line = (*(keypoints[i]),) if i < len(keypoints) else (*(bboxes[i],),)
                        else:
                            chosen = segments[i] if use_segments and i < len(segments) and len(segments[i]) > 0 else \
                                bboxes[
                                    i]
                            line = (*chosen,)
                        file.write(("%g " * len(line)).rstrip() % line + "\n")
                with open((fn / img_path.name).with_suffix(".shapes"), "w", encoding="utf-8") as file:
                    for i in range(len(orig_shapes)):
                        line = (*orig_shapes[i],)
                        file.write(("%g " * len(line)).rstrip() % line + "\n")

    LOGGER.info(f"COCO data converted successfully.\nResults saved to {save_dir.resolve()}")


def parse_yolov8_seg_txt(label_path):
    """
    Returns:
      class_ids: List[int] per instance
      segments_norm: List[np.ndarray] of shape (K,2) with normalized coords in [0,1]
    NOTE: Assumes one polygon per line (standard YOLOv8-seg). If you store multi-polygons per obj,
          split accordingly before converting.
    """
    class_ids, segments_norm = [], []
    with open(label_path, "r") as f:
        for line in f:
            p = line.strip().split()
            if not p:
                continue
            cls = int(float(p[0]))
            coords = np.asarray(list(map(float, p[1:])), dtype=np.float32).reshape(-1, 2)
            class_ids.append(cls)
            segments_norm.append(coords)
    return class_ids, segments_norm


def gen_yaml(path, outpath):
    with open(path, 'r') as f:
        data = json.load(f)
    cats = data['categories']
    output = dict(
        path='vhr-silva',
        train='images/split_1',
        val='images/split_2',
        # test='images/test',
        names=dict({c['id']: c['name'] for c in cats})
    )
    with open(outpath, 'w') as f:
        yaml.dump(output, f)
        print(f'Written to {outpath}')


def viz_masks(image_path, txt_path, id2label):
    img = np.array(Image.open(image_path))
    img_height, img_width = img.shape[:2]
    annotations = []
    with open(txt_path, encoding="utf-8") as file:
        for line in file:
            content = list(map(float, line.split()))
            class_id = int(content[0])
            poly = np.array(content[1:]).reshape(-1, 2)
            annotations.append((class_id, poly))
    display = img.copy()
    for label, poly in annotations:
        abs_poly = (poly * [img_width, img_height]).astype(np.int32)
        mask = polygon2mask((img_height, img_width), [abs_poly], color=1)
        col = np.array(colors(label, True), dtype=np.uint8)
        if display.ndim == 2:  # grayscale to RGB
            display = np.repeat(display[..., None], 3, axis=2)
        display[mask == 1] = (0.5 * display[mask == 1] + 0.5 * col).astype(np.uint8)
    plt.figure()
    plt.imshow(display)
    plt.axis('off')
    plt.show()


def gen_kfold_yaml(json_path, out_path, kfold_name, cats=None):
    if cats is None:
        with open(json_path, 'r') as f:
            data = json.load(f)
        cats = data['categories']
    output = dict(
        path='vhr-silva',
        train=f'images/{kfold_name}_train',
        val=f'images/{kfold_name}_val',
        test=f'images/{kfold_name}_test',
        names=dict({c['id']: c['name'] for c in cats})
    )
    with open(out_path, 'w') as f:
        yaml.dump(output, f)


def prepare_k_folds(root, export_root):
    export_root = Path(export_root)
    subsets = set(f.name.split('_')[0] for f in (export_root / "labels").iterdir() if f.is_dir())

    for subset in subsets:
        image_folder = export_root / 'images'
        label_folder = export_root / 'labels'
        splits = sorted(
            [name for f in label_folder.iterdir() if f.is_dir() and (name := f.name).startswith(f'{subset}_split_')])
        n_splits = len(splits)
        split_indices = set(range(n_splits))

        for k_fold_idx, test_idx in enumerate(range(len(split_indices))):
            val_idx = (test_idx + 1) % n_splits
            test_split = [splits[test_idx]]
            val_split = [splits[val_idx]]
            train_splits = [splits[si] for si in split_indices - {test_idx, val_idx}]

            kfold_name = f"{subset}_kfold_{k_fold_idx + 1}"
            gen_kfold_yaml(root / "subsets" / subset / f"split_1.json", (export_root / kfold_name).with_suffix('.yaml'),
                           kfold_name)

            kfold_splits = dict(train=train_splits, val=val_split, test=test_split)
            for mode, splits_in_kfold in kfold_splits.items():
                kfold_image_path = image_folder / f"{kfold_name}_{mode}"
                kfold_label_path = label_folder / f"{kfold_name}_{mode}"

                for p in kfold_image_path, kfold_label_path:
                    shutil.rmtree(p, ignore_errors=True)
                    p.mkdir(parents=True, exist_ok=True)

                for split_name in splits_in_kfold:
                    for path in [f for f in (image_folder / split_name).iterdir() if f.is_file()]:
                        shutil.copy(path, kfold_image_path)
                    for path in [f for f in (label_folder / split_name).iterdir() if f.is_file()]:
                        shutil.copy(path, kfold_label_path)


def generate_binary_segmentation(root: Path):
    img_path = root / "images"
    label_path = root / "labels"

    fold_names = sorted(f.name for f in img_path.iterdir() if f.is_dir())
    for name in fold_names:
        new_name = f"binary_{name}"
        img_folder = img_path / name
        to_img = img_path / new_name
        shutil.rmtree(to_img, ignore_errors=True)
        shutil.copytree(img_folder, to_img)

        label_folder = label_path / name
        to_lab = label_path / new_name
        shutil.rmtree(to_lab, ignore_errors=True)
        shutil.copytree(label_folder, to_lab)

        name_no_mode = '_'.join(new_name.split('_')[:-1])
        gen_kfold_yaml(root, (root / name_no_mode).with_suffix('.yaml'),
                       name_no_mode, cats=[dict(id=0, name='Tree')])

        labels = sorted(f for f in to_lab.iterdir() if f.is_file() and f.suffix == '.txt')
        for label in labels:
            with open(label, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                values = line.split()
                values[0] = '0'
                new_lines.append(' '.join(values) + '\n')
            with open(label, 'w') as f:
                f.writelines(new_lines)


if __name__ == '__main__':
    root = Path('/datasets/vhr-silva/')
    export_root = Path('/datasets/vhr-silva-yolo')
    convert_coco(
        root / 'subsets',
        export_root,
        use_segments=True,
    )
    prepare_k_folds(root, export_root)
    generate_binary_segmentation(export_root)
