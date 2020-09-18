from dataclasses import dataclass
from typing import Any, ClassVar, List, Tuple

import openpifpaf


@dataclass
class TBaseCif(openpifpaf.headmeta.Cif):
    pass


@dataclass
class TBaseCaf(openpifpaf.headmeta.Caf):
    pass


@dataclass
class Tcaf(openpifpaf.headmeta.Base):
    keypoints: List[str]
    sigmas: List[float]
    pose: Any
    draw_skeleton: List[Tuple[int, int]] = None
    only_in_field_of_view: bool = False

    n_confidences: ClassVar[int] = 1
    n_vectors: ClassVar[int] = 2
    n_scales: ClassVar[int] = 2

    vector_offsets = [True, True]

    @property
    def n_fields(self):
        return len(self.keypoints)
