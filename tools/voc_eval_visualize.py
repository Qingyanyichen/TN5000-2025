from argparse import ArgumentParser

import mmcv
import mmengine
import numpy as np

from mmdet import datasets
###from mmdet.core.evaluation.mean_ap_visualize import eval_map
from mmdet.evaluation.mean_ap_visualize import eval_map
        
def voc_eval(result_file, dataset, iou_thr=0.5, nproc=4):
    det_results = mmcv.load(result_file)
    annotations = [dataset.get_ann_info(i) for i in range(len(dataset))]
    if hasattr(dataset, 'year') and dataset.year == 2007:
        dataset_name = 'voc07'
    else:
        dataset_name = dataset.CLASSES
    print(len(det_results),len(annotations))
    mmap = eval_map(
        det_results,
        annotations,
        scale_ranges=None,
        iou_thr=iou_thr,
        dataset=dataset_name,
        logger='print',
        nproc=nproc)
    print('mAP: ',mmap[0])


def main():
    parser = ArgumentParser(description='VOC Evaluation')
    parser.add_argument('result', help='result file path')
    parser.add_argument('config', help='config file path')
    parser.add_argument(
        '--iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold for evaluation')
    args = parser.parse_args()
    cfg = mmengine.Config.fromfile(args.config)
    test_dataset = mmengine.runner.obj_from_dict(cfg.data.test, datasets)
    voc_eval(args.result, test_dataset, args.iou_thr)


if __name__ == '__main__':
    main()
