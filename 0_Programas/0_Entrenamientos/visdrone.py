from ultralytics import YOLO

# Carga un modelo pre-entrenado. Se recomienda para un mejor rendimiento.
# Puedes elegir otras versiones de YOLOv8 como yolov8s.pt, yolov8m.pt, etc.
model = YOLO("/home/tdelorenzi/1_yolo/yolo11m.pt")

# Ejemplo con parámetros ajustados para mayor velocidad
results = model.train(data="VisDrone.yaml", epochs=100, imgsz=640, batch=4)

"""
(env1) tdelorenzi@rog:~/1_yolo$ /home/tdelorenzi/1_yolo/env1/bin/python /home/tdelorenzi/1_yolo/2_CITTR/0_Programas/0_Entrenamientos/visdrone.py
New https://pypi.org/project/ultralytics/8.3.176 available 😃 Update with 'pip install -U ultralytics'
Ultralytics 8.3.170 🚀 Python-3.12.3 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 7806MiB)
engine/trainer: agnostic_nms=False, amp=True, augment=False, auto_augment=randaugment, batch=4, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=VisDrone.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dnn=False, dropout=0.0, dynamic=False, embed=None, epochs=100, erasing=0.4, exist_ok=False, fliplr=0.5, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, half=False, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, imgsz=640, int8=False, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=/home/tdelorenzi/1_yolo/yolo11m.pt, momentum=0.937, mosaic=1.0, multi_scale=False, name=train8, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=None, rect=False, resume=False, retina_masks=False, save=True, save_conf=False, save_crop=False, save_dir=runs/detect/train8, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.5, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=botsort.yaml, translate=0.1, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=10

                   from  n    params  module                                       arguments                     
  0                  -1  1      1856  ultralytics.nn.modules.conv.Conv             [3, 64, 3, 2]                 
  1                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]               
  2                  -1  1    111872  ultralytics.nn.modules.block.C3k2            [128, 256, 1, True, 0.25]     
  3                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
  4                  -1  1    444928  ultralytics.nn.modules.block.C3k2            [256, 512, 1, True, 0.25]     
  5                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  6                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  7                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
  8                  -1  1   1380352  ultralytics.nn.modules.block.C3k2            [512, 512, 1, True]           
  9                  -1  1    656896  ultralytics.nn.modules.block.SPPF            [512, 512, 5]                 
 10                  -1  1    990976  ultralytics.nn.modules.block.C2PSA           [512, 512, 1]                 
 11                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 12             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 13                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 14                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 15             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 16                  -1  1    542720  ultralytics.nn.modules.block.C3k2            [1024, 256, 1, True]          
 17                  -1  1    590336  ultralytics.nn.modules.conv.Conv             [256, 256, 3, 2]              
 18            [-1, 13]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 19                  -1  1   1511424  ultralytics.nn.modules.block.C3k2            [768, 512, 1, True]           
 20                  -1  1   2360320  ultralytics.nn.modules.conv.Conv             [512, 512, 3, 2]              
 21            [-1, 10]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 22                  -1  1   1642496  ultralytics.nn.modules.block.C3k2            [1024, 512, 1, True]          
 23        [16, 19, 22]  1   1418734  ultralytics.nn.modules.head.Detect           [10, [256, 512, 512]]         
YOLO11m summary: 231 layers, 20,060,718 parameters, 20,060,702 gradients, 68.2 GFLOPs

Transferred 643/649 items from pretrained weights
Freezing layer 'model.23.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed ✅
train: Fast image access ✅ (ping: 0.0±0.0 ms, read: 6721.3±1963.5 MB/s, size: 261.9 KB)
train: Scanning /home/tdelorenzi/1_yolo/datasets/VisDrone/labels/train.cache... 6471 images, 0 backgrounds, 0 corrupt: 100%|██████████| 6471/6471 
train: /home/tdelorenzi/1_yolo/datasets/VisDrone/images/train/0000137_02220_d_0000163.jpg: 1 duplicate labels removed
train: /home/tdelorenzi/1_yolo/datasets/VisDrone/images/train/0000140_00118_d_0000002.jpg: 1 duplicate labels removed
train: /home/tdelorenzi/1_yolo/datasets/VisDrone/images/train/9999945_00000_d_0000114.jpg: 1 duplicate labels removed
train: /home/tdelorenzi/1_yolo/datasets/VisDrone/images/train/9999987_00000_d_0000049.jpg: 1 duplicate labels removed
val: Fast image access ✅ (ping: 0.0±0.0 ms, read: 2092.2±1435.4 MB/s, size: 131.6 KB)
val: Scanning /home/tdelorenzi/1_yolo/datasets/VisDrone/labels/val.cache... 548 images, 0 backgrounds, 0 corrupt: 100%|██████████| 548/548 [00:00<
Plotting labels to runs/detect/train8/labels.jpg... 
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: SGD(lr=0.01, momentum=0.9) with parameter groups 106 weight(decay=0.0), 113 weight(decay=0.0005), 112 bias(decay=0.0)
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to runs/detect/train8
Starting training for 100 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      1/100      3.61G      1.482      1.812      1.017        332        640:  34%|███▎      | 546/1618 [01:05<02:59,  5.98it/s]

....     

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
     99/100       3.8G      1.009     0.5685     0.8462        123        640: 100%|██████████| 1618/1618 [02:33<00:00, 10.57it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [00:03<00:00, 19.23it/s]
                   all        548      38759      0.557      0.436      0.452      0.278

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
    100/100       3.8G      1.002     0.5644     0.8463        143        640: 100%|██████████| 1618/1618 [02:33<00:00, 10.56it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [00:03<00:00, 19.20it/s]
                   all        548      38759      0.551      0.439      0.452      0.278

100 epochs completed in 4.744 hours.
Optimizer stripped from runs/detect/train8/weights/last.pt, 40.5MB
Optimizer stripped from runs/detect/train8/weights/best.pt, 40.5MB

Validating runs/detect/train8/weights/best.pt...
Ultralytics 8.3.170 🚀 Python-3.12.3 torch-2.7.1+cu128 CUDA:0 (NVIDIA GeForce RTX 4070 Laptop GPU, 7806MiB)
YOLO11m summary (fused): 125 layers, 20,037,742 parameters, 0 gradients, 67.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 69/69 [00:04<00:00, 17.00it/s]
                   all        548      38759      0.575      0.431      0.459      0.284
            pedestrian        520       8844      0.637      0.441      0.503      0.244
                people        482       5125      0.648      0.299      0.379      0.155
               bicycle        364       1287      0.345      0.205        0.2     0.0947
                   car        515      14064      0.793      0.787      0.828      0.606
                   van        421       1975      0.576        0.5      0.504      0.364
                 truck        266        750      0.585      0.459      0.461      0.318
              tricycle        337       1045      0.487      0.362      0.356      0.204
       awning-tricycle        220        532      0.358      0.216      0.222       0.14
                   bus        131        251      0.731      0.541      0.614      0.468
                 motor        485       4886      0.593      0.499      0.522      0.247
Speed: 0.1ms preprocess, 3.2ms inference, 0.0ms loss, 1.6ms postprocess per image
Results saved to runs/detect/train8
(env1) tdelorenzi@rog:~/1_yolo$ 

"""