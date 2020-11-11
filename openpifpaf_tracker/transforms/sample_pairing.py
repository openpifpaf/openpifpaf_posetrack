import copy
import logging
import random

import PIL

import openpifpaf

LOG = logging.getLogger(__name__)


class SamplePairing(openpifpaf.transforms.Preprocess):
    def __init__(self):
        self.previous_images = None
        self.previous_all_annotations = []

    def __call__(self, original_images, original_all_anns, metas):
        images = copy.deepcopy(original_images)
        all_anns = copy.deepcopy(original_all_anns)

        if self.previous_images is not None:
            # image
            for current_image, previous_image in zip(images, self.previous_images):
                current_image *= 0.5
                current_image += 0.5 * previous_image

            # annotations
            for current_anns, previous_anns in zip(all_anns, self.previous_all_annotations):
                current_anns += previous_anns

            # meta untouched

        self.previous_images = original_images
        self.previous_all_annotations = original_all_anns
        return images, all_anns, metas
