import numpy as np

import openpifpaf

from .posetrack2018 import KEYPOINTS, SKELETON, SIGMAS, UPRIGHT_POSE


def main():
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    keypoint_painter = openpifpaf.show.KeypointPainter(color_connections=True, linewidth=6)

    scale = np.sqrt(
        (np.max(UPRIGHT_POSE[:, 0]) - np.min(UPRIGHT_POSE[:, 0]))
        * (np.max(UPRIGHT_POSE[:, 1]) - np.min(UPRIGHT_POSE[:, 1]))
    )

    ann = openpifpaf.Annotation(KEYPOINTS, SKELETON)
    ann.set(UPRIGHT_POSE, np.array(SIGMAS) * scale)
    openpifpaf.plugins.coco.constants.draw_ann(
        ann, keypoint_painter=keypoint_painter, filename='docs/skeleton_posetrack.png')

    upright_pose_2tracking = np.concatenate([
        UPRIGHT_POSE,
        0.9 * UPRIGHT_POSE + np.array([-1.5, 1.5, 0.0]),
    ])
    sigmas_2tracking = np.concatenate([np.array(SIGMAS) * scale, 0.8 * np.array(SIGMAS) * scale])
    tracking2_skeleton = np.concatenate([
        np.array(SKELETON) + 17,
        np.array([(j, j + 17) for j in range(1, 18)]),
        np.array(SKELETON),
    ])
    ann = openpifpaf.Annotation(KEYPOINTS + KEYPOINTS, tracking2_skeleton)
    ann.set(
        upright_pose_2tracking,
        sigmas_2tracking,
    )
    openpifpaf.plugins.coco.constants.draw_ann(
        ann, keypoint_painter=keypoint_painter, filename='docs/skeleton_tracking2.png')


if __name__ == '__main__':
    main()
