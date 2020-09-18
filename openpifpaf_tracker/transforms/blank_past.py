import copy
import logging

import PIL

import openpifpaf

LOG = logging.getLogger(__name__)


class BlankPast(openpifpaf.transforms.Preprocess):
    def __call__(self, images, all_anns, metas):
        all_anns = copy.deepcopy(all_anns)
        metas = copy.deepcopy(metas)

        for i, _ in enumerate(images[1:], start=1):
            images[i] = PIL.Image.new('RGB', (320, 240), (127, 127, 127))

        for i, _ in enumerate(all_anns[1:], start=1):
            all_anns[i] = []

        for meta in metas[1:]:
            meta['image'] = {'frame_id': -1, 'file_name': 'blank'}
            assert 'annotations' not in meta

        return images, all_anns, metas


class PreviousPast(openpifpaf.transforms.Preprocess):
    def __init__(self):
        self.previous_image = PIL.Image.new('RGB', (320, 240), (127, 127, 127))
        self.previous_meta = {'frame_id': -1, 'file_name': 'blank'}
        self.previous_annotations = []

    def __call__(self, images, all_anns, metas):
        all_anns = copy.deepcopy(all_anns)
        metas = copy.deepcopy(metas)

        LOG.debug('replacing %s with %s', metas[1], self.previous_meta)

        for i, _ in enumerate(images[1:], start=1):
            images[i] = self.previous_image

        for i, _ in enumerate(all_anns[1:], start=1):
            all_anns[i] = []  # TODO assumes previous image has nothing to do with current

        for meta in metas[1:]:
            meta['image'] = self.previous_meta
            assert 'annotations' not in meta

        self.previous_image = images[0]
        self.previous_annotations = all_anns[0]
        self.previous_meta = metas[0]
        return images, all_anns, metas
