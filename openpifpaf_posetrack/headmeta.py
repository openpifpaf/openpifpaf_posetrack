from dataclasses import dataclass
from typing import Any, ClassVar, List, Tuple

import numpy as np

import openpifpaf


@dataclass
class TBaseCif(openpifpaf.headmeta.Cif):
    pass


@dataclass
class TBaseCaf(openpifpaf.headmeta.Caf):
    pass


@dataclass
class Tcaf(openpifpaf.headmeta.Base):
    keypoints_single_frame: List[str]
    sigmas_single_frame: List[float]
    pose_single_frame: Any
    draw_skeleton_single_frame: List[Tuple[int, int]] = None
    keypoints: List[str] = None
    sigmas: List[float] = None
    pose: Any = None
    draw_skeleton: List[Tuple[int, int]] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    vector_offsets = [True, True]

    def __post_init__(self):
        if self.keypoints is None:
            self.keypoints = np.concatenate((
                self.keypoints_single_frame,
                self.keypoints_single_frame,
            ), axis=0)
        if self.sigmas is None:
            self.sigmas = np.concatenate((
                self.sigmas_single_frame,
                self.sigmas_single_frame,
            ), axis=0)
        if self.pose is None:
            self.pose = np.concatenate((
                self.pose_single_frame,
                self.pose_single_frame,
            ), axis=0)
        if self.draw_skeleton is None:
            self.draw_skeleton = np.concatenate((
                self.draw_skeleton_single_frame,
                self.draw_skeleton_single_frame,
            ), axis=0)

    @property
    def skeleton(self):
        return [(i + 1, i + 1 + len(self.keypoints_single_frame))
                for i, _ in enumerate(self.keypoints_single_frame)]

    @property
    def n_fields(self):
        return len(self.keypoints_single_frame)
