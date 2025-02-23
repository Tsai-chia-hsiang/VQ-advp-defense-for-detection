# Ultralytics YOLO format INIRIA Dataset
Put Ultralytics YOLO format INIRIA Dataset here

```
.
└── yolodata/
    ├── train/
    │   ├── images/
    │   │   └── ...
    │   └── labels/
    │       └── ...
    └── val/
        ├── images/
        │   └── ...  
        └── labels/
            └── ...
```

## Tools:
- to_yolo.py : Convert original INRIA person annotations to Ultralytics YOLO format Dataset
- view_yolo_label.py: Visualization tool to plot bboxes for a image according its Ultralytics YOLO label annotation