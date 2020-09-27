import dataclasses
import logging
from typing import ClassVar

import numpy as np
import torch

import openpifpaf

from .. import headmeta
from .annrescaler import TrackingAnnRescaler

LOG = logging.getLogger(__name__)


@dataclasses.dataclass
class Tcaf:
    meta: headmeta.Tcaf
    rescaler: TrackingAnnRescaler = None
    v_threshold: int = 0
    bmin: float = 0.1
    visualizer: openpifpaf.visualizer.Caf = None

    min_size: ClassVar[int] = 3
    fixed_size: ClassVar[bool] = True
    aspect_ratio: ClassVar[float] = 0.0
    padding: ClassVar[int] = 10

    def __call__(self, images, all_anns, metas):
        return TcafGenerator(self)(images, all_anns, metas)


class TcafGenerator:
    def __init__(self, config: Tcaf):
        self.config = config

        self.rescaler = config.rescaler or TrackingAnnRescaler(
            config.meta.stride, len(config.meta.keypoints), config.meta.pose)
        self.visualizer = config.visualizer or openpifpaf.visualizer.Caf(config.meta)

        if self.config.fixed_size:
            assert self.config.aspect_ratio == 0.0

        self.intensities = None
        self.fields_reg1 = None
        self.fields_reg2 = None
        self.fields_bmin1 = None
        self.fields_bmin2 = None
        self.fields_scale1 = None
        self.fields_scale2 = None
        self.fields_reg_l = None

    def __call__(self, images, all_anns, metas):
        width_height_original = images[0].shape[2:0:-1]

        keypoint_sets = self.rescaler.keypoint_sets(all_anns[0], all_anns[1])
        bg_mask = self.rescaler.bg_mask(
            all_anns[0], all_anns[1], width_height_original)  # TODO add crowd margin
        valid_area = self.rescaler.valid_area(metas[0])
        LOG.debug('valid area: %s, kpsets = %d, tpaf min size = %d',
                  valid_area, len(keypoint_sets), self.config.min_size)

        n_fields = keypoint_sets.shape[-2]
        self.init_fields(n_fields, bg_mask)
        self.fill(keypoint_sets)
        fields = self.fields(valid_area)

        self.visualizer.processed_image(images[0])
        self.visualizer.targets(fields, annotation_dicts=all_anns[0])
        self.visualizer.processed_image(images[1])
        self.visualizer.targets(fields, annotation_dicts=all_anns[1])

        return fields

    def init_fields(self, n_fields, bg_mask):
        field_w = bg_mask.shape[1] + 2 * self.config.padding
        field_h = bg_mask.shape[0] + 2 * self.config.padding
        self.intensities = np.zeros((n_fields, field_h, field_w), dtype=np.float32)
        self.fields_reg1 = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg2 = np.full((n_fields, 2, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin1 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_bmin2 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale1 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_scale2 = np.full((n_fields, field_h, field_w), np.nan, dtype=np.float32)
        self.fields_reg_l = np.full((n_fields, field_h, field_w), np.inf, dtype=np.float32)

        # set background
        p = self.config.padding
        self.fields_reg_l[:, p:-p, p:-p][:, bg_mask == 0] = 1.0
        self.intensities[:, p:-p, p:-p][:, bg_mask == 0] = np.nan

    def fill(self, keypoint_sets):
        for keypoints in keypoint_sets:
            self.fill_keypoints(keypoints)

    def fill_keypoints(self, keypoints):
        visible = keypoints[0, :, 2] > 0
        if not np.any(visible):
            return

        scale = self.rescaler.scale(keypoints[0])

        for joint_i in range(keypoints.shape[-2]):
            joint1 = keypoints[0, joint_i]
            joint2 = keypoints[1, joint_i]
            if joint1[2] <= self.config.v_threshold or joint2[2] <= self.config.v_threshold:
                continue

            # if there is no continuous visual connection, endpoints outside
            # the field of view cannot be inferred
            # LOG.debug('fov check: j1 = %s, j2 = %s', joint1, joint2)
            out_field_of_view_1 = (
                joint1[0] < 0 or \
                joint1[1] < 0 or \
                joint1[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding or \
                joint1[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding
            )
            out_field_of_view_2 = (
                joint2[0] < 0 or \
                joint2[1] < 0 or \
                joint2[0] > self.intensities.shape[2] - 1 - 2 * self.config.padding or \
                joint2[1] > self.intensities.shape[1] - 1 - 2 * self.config.padding
            )
            if out_field_of_view_1 and out_field_of_view_2:
                continue
            if self.config.meta.only_in_field_of_view:
                if out_field_of_view_1 or out_field_of_view_2:
                    continue

            if self.config.meta.sigmas is not None:
                joint_scale = scale * self.config.meta.sigmas[joint_i]
            self.fill_association(joint_i, joint1, joint2, joint_scale)

    def fill_association(self, tcaf_i, joint1, joint2, joint_scale):
        # offset between joints
        offset = joint2[:2] - joint1[:2]
        offset_d = np.linalg.norm(offset)

        # dynamically create s
        s = max(self.config.min_size, int(offset_d * self.config.aspect_ratio))
        sink = openpifpaf.utils.create_sink(s)
        s_offset = (s - 1.0) / 2.0

        # pixel coordinates of top-left joint pixel
        joint1ij = np.round(joint1[:2] - s_offset)
        joint2ij = np.round(joint2[:2] - s_offset)
        offsetij = joint2ij - joint1ij

        # set fields
        num = max(2, int(np.ceil(offset_d)))
        fmargin = min(0.4, (s_offset + 1) / (offset_d + np.spacing(1)))
        # fmargin = 0.0
        frange = np.linspace(fmargin, 1.0-fmargin, num=num)
        if self.config.fixed_size:
            frange = [0.5]
        for f in frange:
            fij = np.round(joint1ij + f * offsetij) + self.config.padding
            fminx, fminy = int(fij[0]), int(fij[1])
            fmaxx, fmaxy = fminx + s, fminy + s
            if fminx < 0 or fmaxx > self.intensities.shape[2] or \
               fminy < 0 or fmaxy > self.intensities.shape[1]:
                continue
            fxy = (fij - self.config.padding) + s_offset

            # precise floating point offset of sinks
            joint1_offset = (joint1[:2] - fxy).reshape(2, 1, 1)
            joint2_offset = (joint2[:2] - fxy).reshape(2, 1, 1)
            sink1 = sink + joint1_offset
            sink2 = sink + joint2_offset

            # mask
            # perpendicular distance computation:
            # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
            # Coordinate systems for this computation is such that
            # joint1 is at (0, 0).
            sink_l = np.fabs(
                offset[1] * sink1[0]
                - offset[0] * sink1[1]
            ) / (offset_d + 0.01)
            mask = sink_l < self.fields_reg_l[tcaf_i, fminy:fmaxy, fminx:fmaxx]
            self.fields_reg_l[tcaf_i, fminy:fmaxy, fminx:fmaxx][mask] = sink_l[mask]

            # update intensity
            self.intensities[tcaf_i, fminy:fmaxy, fminx:fmaxx][mask] = 1.0

            # update regressions
            self.fields_reg1[tcaf_i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink1[:, mask]
            self.fields_reg2[tcaf_i, :, fminy:fmaxy, fminx:fmaxx][:, mask] = \
                sink2[:, mask]

            # update bmin
            self.fields_bmin1[tcaf_i, fminy:fmaxy, fminx:fmaxx][mask] = self.config.bmin
            self.fields_bmin2[tcaf_i, fminy:fmaxy, fminx:fmaxx][mask] = self.config.bmin

            # update scale
            self.fields_scale1[tcaf_i, fminy:fmaxy, fminx:fmaxx][mask] = joint_scale
            self.fields_scale2[tcaf_i, fminy:fmaxy, fminx:fmaxx][mask] = joint_scale

    def fields(self, valid_area):
        p = self.config.padding
        intensities = self.intensities[:, p:-p, p:-p]
        fields_reg1 = self.fields_reg1[:, :, p:-p, p:-p]
        fields_reg2 = self.fields_reg2[:, :, p:-p, p:-p]
        fields_bmin1 = self.fields_bmin1[:, p:-p, p:-p]
        fields_bmin2 = self.fields_bmin2[:, p:-p, p:-p]
        fields_scale1 = self.fields_scale1[:, p:-p, p:-p]
        fields_scale2 = self.fields_scale2[:, p:-p, p:-p]

        openpifpaf.utils.mask_valid_area(intensities, valid_area)
        openpifpaf.utils.mask_valid_area(fields_reg1[:, 0], valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_reg1[:, 1], valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_reg2[:, 0], valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_reg2[:, 1], valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_bmin1, valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_bmin2, valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_scale1, valid_area, fill_value=np.nan)
        openpifpaf.utils.mask_valid_area(fields_scale2, valid_area, fill_value=np.nan)

        return torch.from_numpy(np.concatenate([
            np.expand_dims(intensities, 1),
            fields_reg1,
            fields_reg2,
            np.expand_dims(fields_bmin1, 1),
            np.expand_dims(fields_bmin2, 1),
            np.expand_dims(fields_scale1, 1),
            np.expand_dims(fields_scale2, 1),
        ], axis=1))
