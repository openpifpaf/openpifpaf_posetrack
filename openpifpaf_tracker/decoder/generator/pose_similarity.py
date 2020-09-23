import argparse
import logging
import time

import numpy as np
import openpifpaf
import scipy.optimize

from ... import headmeta
from ..euclidean import Euclidean
from ..oks import Oks
from ..track_annotation import TrackAnnotation
from .track_base import TrackBase

LOG = logging.getLogger(__name__)


class PoseSimilarity(TrackBase):
    distance_function = None

    def __init__(self, cif_meta, caf_meta, *, pose_generator=None):
        super().__init__()
        self.cif_meta = cif_meta
        self.caf_meta = caf_meta

        self.pose_generator = pose_generator or openpifpaf.decoder.CifCaf([cif_meta], [caf_meta])

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('PoseSimilarity')
        group.add_argument('--posesimilarity-distance',
                           default='euclidean', choices=('euclidean', 'oks'))
        group.add_argument('--posesimilarity-oks-inflate', default=Oks.inflate, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        if args.posesimilarity_distance == 'euclidean':
            cls.distance_function = Euclidean()
        elif args.posesimilarity_distance == 'oks':
            cls.distance_function = Oks()

        Oks.inflate = args.posesimilarity_oks_inflate

    @classmethod
    def factory(cls, head_metas):
        if len(head_metas) < 2:
            return []
        return [
            cls(cif_meta, caf_meta)
            for cif_meta, caf_meta
            in zip(head_metas, head_metas[1:])
            if (isinstance(cif_meta, headmeta.TBaseCif)
                and isinstance(caf_meta, headmeta.TBaseCaf))
        ]

    def __call__(self, fields, *, initial_annotations=None):
        # set valid_mask
        self.distance_function.valid_mask = [
            1 if kp not in ('left_ear', 'right_ear') else 0
            for kp in self.cif_meta.keypoints
        ]
        # set sigmas
        self.distance_function.sigmas = np.asarray(self.cif_meta.sigmas)

        self.frame_number += 1
        self.prune_active(self.frame_number)

        pose_start = time.perf_counter()
        pose_annotations = self.pose_generator(fields)
        LOG.debug('pose time = %.3fs', time.perf_counter() - pose_start)

        cost_start = time.perf_counter()
        cost = np.full((len(self.active) * 2, len(pose_annotations)), 1000.0)
        for track_i, track in enumerate(self.active):
            for pose_i, pose in enumerate(pose_annotations):
                cost[track_i, pose_i] = self.distance_function(
                    self.frame_number, pose, track, self.track_is_good(track, self.frame_number))

                # option to loose track (e.g. occlusion)
                cost[track_i + len(self.active), pose_i] = 10.0
        LOG.debug('cost time = %.3fs', time.perf_counter() - cost_start)

        track_indices, pose_indices = scipy.optimize.linear_sum_assignment(cost)
        matched_poses = set()
        for track_i, pose_i in zip(track_indices, pose_indices):
            # was track lost?
            if track_i >= len(self.active):
                continue

            pose = pose_annotations[pose_i]
            track = self.active[track_i]

            track.add(self.frame_number, pose)
            matched_poses.add(pose)

        for new_pose in pose_annotations:
            if new_pose in matched_poses:
                continue
            self.active.append(TrackAnnotation().add(self.frame_number, new_pose))

        # tag ignore regions TODO
        # if self.gt_anns:
        #     self.tag_ignore_region(self.frame_number, self.gt_anns)

        # pruning lost tracks
        self.active = [t for t in self.active
                       if self.track_is_viable(t, self.frame_number)]

        LOG.info('active tracks = %d, good = %d',
                 len(self.active),
                 len([t for t in self.active if self.track_is_good(t, self.frame_number)]))
        LOG.info('track ids = %s', [t.id_ for t in self.active])
        if self.track_visualizer:
            self.track_visualizer.predicted(
                self.frame_number,
                [t for t in self.active if self.track_is_good(t, self.frame_number)],
            )
        return self.annotations(self.frame_number)
