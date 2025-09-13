from __future__ import annotations
import torch
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

def build_rgb_only_mapper(dataset_name: str, is_train: bool = True):
    """
    RGB-only mapper.
    - When is_train=True: applies light augs and builds dd["instances"] from annotations.
    - When is_train=False: no augs, no instances (evaluation/inference mode).
    """
    if is_train:
        augs = T.AugmentationList([
            T.RandomFlip(horizontal=True, vertical=False),
            T.RandomRotation(angle=[-5, 5], expand=False, sample_style="range"),
        ])
    else:
        augs = T.AugmentationList([])

    def _mapper(d):
        dd = d.copy()

        # Load RGB as float32 in [0,1]
        img = utils.read_image(dd["file_name"], format="RGB").astype("float32") / 255.0

        # Apply augs
        aug_in = T.AugInput(img)
        tfm = augs(aug_in)
        img = aug_in.image  # possibly augmented

        # To CHW tensor
        dd["image"] = torch.as_tensor(img.transpose(2, 0, 1).copy())  # (3,H,W)

        # Build Instances only for training
        if is_train and "annotations" in dd:
            annos = [
                utils.transform_instance_annotations(obj, tfm, img.shape[:2])
                for obj in dd["annotations"]
            ]
            instances = utils.annotations_to_instances(annos, img.shape[:2])
            instances = utils.filter_empty_instances(instances)
            dd["instances"] = instances

        # Remove raw annotations to save memory
        dd.pop("annotations", None)
        return dd

    return _mapper