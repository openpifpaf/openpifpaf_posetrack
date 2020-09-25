import argparse

import numpy as np
import torch

import openpifpaf

from . import collate, datasets, encoder, headmeta, transforms
from .transforms import SingleImage as S


KEYPOINTS = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

SIGMAS = [
    0.026,  # 1 nose
    0.08,  # 2 head_bottom ==> changed versus COCO
    0.06,  # 3 head_top ==> changed versus COCO
    0.035,  # 4 ears ==> never annotated
    0.035,  # 5 ears ==> never annotated
    0.079,  # 6 shoulders
    0.079,  # 7 shoulders
    0.072,  # 8 elbows
    0.072,  # 9 elbows
    0.062,  # 10 wrists
    0.062,  # 11 wrists
    0.107,  # 12 hips
    0.107,  # 13 hips
    0.087,  # 14 knees
    0.087,  # 15 knees
    0.089,  # 16 ankles
    0.089,  # 17 ankles
]

UPRIGHT_POSE = np.array([
    [0.2, 9.3, 2.0],  # 'nose',            # 1
    [-0.05, 9.0, 2.0],  # 'head_bottom',        # 2
    [0.05, 10.0, 2.0],  # 'head_top',       # 3
    [-0.7, 9.5, 2.0],  # 'left_ear',        # 4
    [0.7, 9.5, 2.0],  # 'right_ear',       # 5
    [-1.4, 8.0, 2.0],  # 'left_shoulder',   # 6
    [1.4, 8.0, 2.0],  # 'right_shoulder',  # 7
    [-1.75, 6.0, 2.0],  # 'left_elbow',      # 8
    [1.75, 6.2, 2.0],  # 'right_elbow',     # 9
    [-1.75, 4.0, 2.0],  # 'left_wrist',      # 10
    [1.75, 4.2, 2.0],  # 'right_wrist',     # 11
    [-1.26, 4.0, 2.0],  # 'left_hip',        # 12
    [1.26, 4.0, 2.0],  # 'right_hip',       # 13
    [-1.4, 2.0, 2.0],  # 'left_knee',       # 14
    [1.4, 2.1, 2.0],  # 'right_knee',      # 15
    [-1.4, 0.0, 2.0],  # 'left_ankle',      # 16
    [1.4, 0.1, 2.0],  # 'right_ankle',     # 17
])

SKELETON = [
    [16, 14],
    [14, 12],
    [17, 15],
    [15, 13],
    [12, 13],
    [6, 12],
    [7, 13],
    [6, 8],
    [7, 9],
    [8, 10],
    [9, 11],
    [2, 6],
    [2, 7],
    [2, 3],
    [1, 2],
    [1, 3],
    [1, 4],  # 4 is never annotated
    [1, 5],  # 5 is never annotated
]

DENSER_CONNECTIONS = [
    [6, 7],  # shoulders
    [8, 9],  # elbows
    [10, 11],  # wrists
    [14, 15],  # knees
    [16, 17],  # ankles
    [6, 10],  # shoulder - wrist
    [7, 11],
    [10, 12],  # wrists - hips
    [11, 13],
    [2, 10],  # headbottom - wrists
    [2, 11],
    [12, 15],  # hip knee cross
    [13, 14],
    [14, 17],  # knee ancle cross
    [15, 16],
    [6, 13],  # shoulders hip cross
    [7, 12],
    [6, 3],  # shoulders head top
    [7, 3],
    [6, 1],  # shoulders head nose
    [7, 1],
    [8, 2],  # elbows head_bottom
    [9, 2],  # elbows head_bottom
]


class Posetrack2018(openpifpaf.datasets.DataModule):
    # cli configurable
    train_annotations = 'data-posetrack2018/annotations/train/*.json'
    val_annotations = 'data-posetrack2018/annotations/val/*.json'
    eval_annotations = val_annotations
    data_root = 'data-posetrack2018'

    square_edge = 513
    augmentation = True
    rescale_images = 1.0
    upsample_stride = 1
    min_kp_anns = 1

    eval_long_edge = 801
    eval_orientation_invariant = 0.0
    eval_extended_scale = False

    def __init__(self):
        super().__init__()

        cif = headmeta.TBaseCif('cif', 'posetrack2018',
                                keypoints=KEYPOINTS,
                                sigmas=SIGMAS,
                                pose=UPRIGHT_POSE,
                                draw_skeleton=SKELETON)
        caf = headmeta.TBaseCaf('caf', 'posetrack2018',
                                keypoints=KEYPOINTS,
                                sigmas=SIGMAS,
                                pose=UPRIGHT_POSE,
                                skeleton=SKELETON)
        dcaf = headmeta.TBaseCaf('dcaf', 'posetrack2018',
                                 keypoints=KEYPOINTS,
                                 sigmas=SIGMAS,
                                 pose=UPRIGHT_POSE,
                                 skeleton=DENSER_CONNECTIONS,
                                 sparse_skeleton=SKELETON,
                                 only_in_field_of_view=True)
        tcaf = headmeta.Tcaf('tcaf', 'posetrack2018',
                             keypoints=KEYPOINTS,
                             sigmas=SIGMAS,
                             pose=UPRIGHT_POSE,
                             draw_skeleton=SKELETON,
                             only_in_field_of_view=True)

        cif.upsample_stride = self.upsample_stride
        caf.upsample_stride = self.upsample_stride
        dcaf.upsample_stride = self.upsample_stride
        tcaf.upsample_stride = self.upsample_stride
        self.head_metas = [cif, caf, dcaf, tcaf]

    @classmethod
    def cli(cls, parser: argparse.ArgumentParser):
        group = parser.add_argument_group('data module Posetrack2018')

        group.add_argument('--posetrack2018-train-annotations',
                           default=cls.train_annotations)
        group.add_argument('--posetrack2018-val-annotations',
                           default=cls.val_annotations)
        group.add_argument('--posetrack2018-eval-annotations',
                           default=cls.eval_annotations)
        group.add_argument('--posetrack2018-data-root',
                           default=cls.data_root)

        group.add_argument('--posetrack-square-edge',
                           default=cls.square_edge, type=int,
                           help='square edge of input images')
        assert cls.augmentation
        group.add_argument('--posetrack-no-augmentation',
                           dest='posetrack_augmentation',
                           default=True, action='store_false',
                           help='do not apply data augmentation')
        group.add_argument('--posetrack-rescale-images',
                           default=cls.rescale_images, type=float,
                           help='overall rescale factor for images')
        group.add_argument('--posetrack-upsample',
                           default=cls.upsample_stride, type=int,
                           help='head upsample stride')
        group.add_argument('--posetrack-min-kp-anns',
                           default=cls.min_kp_anns, type=int,
                           help='filter images with fewer keypoint annotations')

        group.add_argument('--posetrack-eval-long-edge', default=cls.eval_long_edge, type=int)
        assert not cls.eval_extended_scale
        group.add_argument('--posetrack-eval-extended-scale', default=False, action='store_true')
        group.add_argument('--posetrack-eval-orientation-invariant',
                           default=cls.eval_orientation_invariant, type=float)

    @classmethod
    def configure(cls, args: argparse.Namespace):
        # extract global information
        cls.debug = args.debug
        cls.pin_memory = args.pin_memory

        # posetrack2018 specific
        cls.train_annotations = args.posetrack2018_train_annotations
        cls.val_annotations = args.posetrack2018_val_annotations
        cls.eval_annotations = args.posetrack2018_eval_annotations
        cls.data_root = args.posetrack2018_data_root

        cls.square_edge = args.posetrack_square_edge
        cls.augmentation = args.posetrack_augmentation
        cls.rescale_images = args.posetrack_rescale_images
        cls.upsample_stride = args.posetrack_upsample
        cls.min_kp_anns = args.posetrack_min_kp_anns

        # evaluation
        cls.eval_long_edge = args.posetrack_eval_long_edge
        cls.eval_orientation_invariant = args.posetrack_eval_orientation_invariant
        cls.eval_extended_scale = args.posetrack_eval_extended_scale

    def _preprocess(self):
        encoders = (
            encoder.SingleImage(openpifpaf.encoder.Cif(self.head_metas[0])),
            encoder.SingleImage(openpifpaf.encoder.Caf(self.head_metas[1])),
            encoder.SingleImage(openpifpaf.encoder.Caf(self.head_metas[2])),
            encoder.Tcaf(self.head_metas[3]),
        )

        if not self.augmentation:
            return openpifpaf.transforms.Compose([
                openpifpaf.transforms.NormalizeAnnotations(),
                openpifpaf.transforms.RescaleAbsolute(self.square_edge),
                openpifpaf.transforms.CenterPad(self.square_edge),
                openpifpaf.transforms.EVAL_TRANSFORM,
                transforms.Encoders(encoders),
            ])

        hflip_posetrack = openpifpaf.transforms.HFlip(
            KEYPOINTS,
            openpifpaf.datasets.constants.HFLIP)
        return openpifpaf.transforms.Compose([
            S(transforms.NormalizePosetrack()),
            openpifpaf.transforms.RandomApply(transforms.PreviousPast(), 0.2),
            S(openpifpaf.transforms.AnnotationJitter()),
            S(transforms.AddCrowdForIncompleteHead()),
            S(openpifpaf.transforms.RandomApply(hflip_posetrack, 0.5)),
            S(openpifpaf.transforms.RescaleRelative(
                (0.5, 2.0), power_law=True, absolute_reference=801, stretch_range=(0.75, 1.33))),
            transforms.Crop(self.square_edge, max_shift=30.0),
            S(openpifpaf.transforms.CenterPad(self.square_edge)),
            S(openpifpaf.transforms.RandomApply(openpifpaf.transforms.RotateBy90(), 0.5)),
            S(openpifpaf.transforms.TRAIN_TRANSFORM),
            transforms.Encoders(encoders),
        ])

    def train_loader(self):
        train_data = datasets.Posetrack2018(
            annotation_files=self.train_annotations,
            data_root=self.data_root,
            group=[(0, -12), (0, -8), (0, -4)],
            preprocess=self._preprocess(),
            only_annotated=True,
            max_per_sequence=10,
        )
        return torch.utils.data.DataLoader(
            train_data, batch_size=self.batch_size, shuffle=not self.debug,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate.collate_tracking_images_targets_meta)

    def val_loader(self):
        val_data = datasets.Posetrack2018(
            annotation_files=self.val_annotations,
            data_root=self.data_root,
            group=[(0, -12), (0, -8), (0, -4)],
            preprocess=self._preprocess(),
            only_annotated=True,
            max_per_sequence=10,
        )
        return torch.utils.data.DataLoader(
            val_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=True,
            collate_fn=collate.collate_tracking_images_targets_meta)

    @classmethod
    def common_eval_preprocess(cls):
        rescale_t = None
        if cls.eval_extended_scale:
            assert cls.eval_long_edge
            rescale_t = [
                openpifpaf.transforms.DeterministicEqualChoice([
                    openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge),
                    openpifpaf.transforms.RescaleAbsolute((cls.eval_long_edge - 1) // 2 + 1),
                ], salt=1)
            ]
        elif cls.eval_long_edge:
            rescale_t = openpifpaf.transforms.RescaleAbsolute(cls.eval_long_edge)

        if cls.batch_size == 1:
            padding_t = openpifpaf.transforms.CenterPadTight(16)
        else:
            assert cls.eval_long_edge
            padding_t = openpifpaf.transforms.CenterPad(cls.eval_long_edge)

        orientation_t = None
        if cls.eval_orientation_invariant:
            orientation_t = openpifpaf.transforms.DeterministicEqualChoice([
                None,
                openpifpaf.transforms.RotateBy90(fixed_angle=90),
                openpifpaf.transforms.RotateBy90(fixed_angle=180),
                openpifpaf.transforms.RotateBy90(fixed_angle=270),
            ], salt=3)

        return [
            transforms.Ungroup(),
            transforms.NormalizePosetrack(),
            rescale_t,
            padding_t,
            orientation_t,
        ]

    def _eval_preprocess(self):
        return openpifpaf.transforms.Compose([
            *self.common_eval_preprocess(),
            openpifpaf.transforms.ToAnnotations([
                openpifpaf.transforms.ToKpAnnotations(
                    ['person'],
                    keypoints_by_category={1: self.head_metas[0].keypoints},
                    skeleton_by_category={1: self.head_metas[1].skeleton},
                ),
                openpifpaf.transforms.ToCrowdAnnotations(['person']),
            ]),
            openpifpaf.transforms.EVAL_TRANSFORM,
        ])

    def eval_loader(self):
        eval_data = datasets.Posetrack2018(
            annotation_files=self.eval_annotations,
            data_root=self.data_root,
            preprocess=self._eval_preprocess(),
        )
        return torch.utils.data.DataLoader(
            eval_data, batch_size=self.batch_size, shuffle=False,
            pin_memory=self.pin_memory, num_workers=self.loader_workers, drop_last=False,
            collate_fn=openpifpaf.datasets.collate_images_anns_meta)

    def metrics(self):
        return []
