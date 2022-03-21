try:
    from .rbbox_overlap import rbbox_overlaps as rbbox_iou
    from .rbbox_overlap import rotate_gpu_nms as rbbox_nms
except ImportError:
    import os
    import sys

    sys.path.insert(1, os.path.dirname(__file__))

    from rbbox_overlap_gpu.rbbox_overlap import rbbox_overlaps as rbbox_iou
    from rbbox_overlap_gpu.rbbox_overlap import rotate_gpu_nms as rbbox_nms
