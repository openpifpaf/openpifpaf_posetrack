import logging
import numpy as np

import openpifpaf.encoder

LOG = logging.getLogger(__name__)


class TrackingAnnRescaler(openpifpaf.encoder.annrescaler.AnnRescaler):
    def bg_mask(self, anns, width_height, *, crowd_margin):
        """Create background mask taking crowd annotations into account."""
        anns1, anns2 = anns

        mask = np.ones((
            (width_height[1] - 1) // self.stride + 1,
            (width_height[0] - 1) // self.stride + 1,
        ), dtype=np.bool)
        crowd_bbox = [np.inf, np.inf, 0, 0]
        for ann in anns1 + anns2:
            if not ann['iscrowd']:
                valid_keypoints = 'keypoints' in ann and np.any(ann['keypoints'][:, 2] > 0)
                if valid_keypoints:
                    continue

            if 'mask' not in ann:
                bb = ann['bbox'].copy()
                bb /= self.stride
                bb[2:] += bb[:2]  # convert width and height to x2 and y2

                # left top
                left = np.clip(int(bb[0] - crowd_margin), 0, mask.shape[1] - 1)
                top = np.clip(int(bb[1] - crowd_margin), 0, mask.shape[0] - 1)

                # right bottom
                # ceil: to round up
                # +1: because mask upper limit is exclusive
                right = np.clip(int(np.ceil(bb[2] + crowd_margin)) + 1,
                                left + 1, mask.shape[1])
                bottom = np.clip(int(np.ceil(bb[3] + crowd_margin)) + 1,
                                 top + 1, mask.shape[0])

                crowd_bbox[0] = min(crowd_bbox[0], left)
                crowd_bbox[1] = min(crowd_bbox[1], top)
                crowd_bbox[2] = max(crowd_bbox[2], right)
                crowd_bbox[3] = max(crowd_bbox[3], bottom)
                continue

            assert False  # because code below is not tested
            mask[ann['mask'][::self.stride, ::self.stride]] = 0

        if crowd_bbox[1] < crowd_bbox[3] and crowd_bbox[0] < crowd_bbox[2]:
            LOG.debug('crowd_bbox: %s', crowd_bbox)
            mask[crowd_bbox[1]:crowd_bbox[3], crowd_bbox[0]:crowd_bbox[2]] = 0

        return mask

    def keypoint_sets(self, anns):
        """Ignore annotations of crowds."""
        anns1, anns2 = anns

        anns1_by_trackid = {ann['track_id']: ann for ann in anns1}
        keypoint_sets = [
            np.concatenate((
                anns1_by_trackid[ann2['track_id']]['keypoints'],
                ann2['keypoints'],
            ), axis=0)
            for ann2 in anns2
            if (not ann2['iscrowd']
                and ann2['track_id'] in anns1_by_trackid)
        ]
        if not keypoint_sets:
            return []

        for keypoints in keypoint_sets:
            keypoints[:, :2] /= self.stride
        return keypoint_sets
