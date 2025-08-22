from ultralytics import YOLO

if __name__ == '__main__':
    # Or n, s, m , l, x
    model = YOLO('yolo11n-seg.pt')
    result_grid = model.tune(iterations=10, data='/home/william/Documents/datasets/vhr-silva/kfold_1.yaml', use_ray=True, epochs=10, grace_period=2)
    print(result_grid)

    # model = YOLO("yolo11x.pt")
    # results = model.tune(data="coco8.yaml", iterations=5)
    # print(results)
