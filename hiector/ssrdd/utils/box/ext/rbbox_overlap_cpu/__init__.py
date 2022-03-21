try:
    from .rbbox_overlap import rbbox_iou, rbbox_iou_1x1, rbbox_iou_nxn, rbbox_nms
except ImportError:
    # A hack to make rbbox_overlap work,
    # see https://git.sinergise.com/clients/esa/query-planet-ccn3/-/issues/13#note_220460
    import os
    import sys

    sys.path.insert(1, os.path.dirname(__file__))
    from rbbox_overlap_cpu.rbbox_overlap import rbbox_iou, rbbox_iou_1x1, rbbox_iou_nxn, rbbox_nms
