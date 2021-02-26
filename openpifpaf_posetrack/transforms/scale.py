import logging

import numpy as np
import PIL

import openpifpaf
from openpifpaf.transforms.scale import _scale

LOG = logging.getLogger(__name__)


class ScaleMix(openpifpaf.transforms.Preprocess):
    def __init__(self, scale_threshold, *,
                 upscale_factor=2.0,
                 downscale_factor=0.5,
                 resample=PIL.Image.BILINEAR):
        self.scale_threshold = scale_threshold
        self.upscale_factor = upscale_factor
        self.downscale_factor = downscale_factor
        self.resample = resample

    def __call__(self, images, all_anns, metas):
        scales = np.array([
            np.sqrt(ann['bbox'][2] * ann['bbox'][3])
            for anns in all_anns
            for ann in anns if (not getattr(ann, 'iscrowd', False)
                                and np.any(ann['keypoints'][:, 2] > 0.0))
        ])
        LOG.debug('scale threshold = %f, scales = %s', self.scale_threshold, scales)
        if not scales.shape[0]:
            return images, all_anns, metas

        all_above_threshold = np.all(scales > self.scale_threshold)
        all_below_threshold = np.all(scales < self.scale_threshold)
        if not all_above_threshold and \
           not all_below_threshold:
            return images, all_anns, metas

        new_images = []
        new_all_anns = []
        new_metas = []
        for image, anns, meta in zip(images, all_anns, metas):
            w, h = image.size

            if all_above_threshold:
                target_w, target_h = int(w / 2), int(h / 2)
            else:
                target_w, target_h = int(w * 2), int(h * 2)

            new_image, new_anns, new_meta = \
                _scale(image, anns, meta, target_w, target_h, self.resample)
            new_images.append(new_image)
            new_all_anns.append(new_anns)
            new_metas.append(new_meta)

        return new_images, new_all_anns, new_metas
