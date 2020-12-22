import numpy as np

import openpifpaf

from .posetrack2018 import KEYPOINTS, SKELETON, SIGMAS, UPRIGHT_POSE


def main():
    openpifpaf.show.KeypointPainter.show_joint_scales = True
    openpifpaf.show.KeypointPainter.line_width = 6
    openpifpaf.show.KeypointPainter.monocolor_connections = False
    keypoint_painter = openpifpaf.show.KeypointPainter()

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

    tracking2_skeleton_forward = np.concatenate([
        np.array([(j, j + 17) for j in range(1, 18)]),
        np.array(SKELETON),
    ])
    ann = openpifpaf.Annotation(KEYPOINTS + KEYPOINTS, tracking2_skeleton_forward)
    ann.set(
        upright_pose_2tracking,
        sigmas_2tracking,
    )
    openpifpaf.plugins.coco.constants.draw_ann(
        ann, keypoint_painter=keypoint_painter, filename='docs/skeleton_tracking2_forward.png')

    # COCO
    coco_keypoints = openpifpaf.plugins.coco.constants.COCO_KEYPOINTS
    coco_skeleton = openpifpaf.plugins.coco.constants.COCO_PERSON_SKELETON
    coco_skeleton_forward = np.concatenate([
        np.array([(j, j + 17) for j in range(1, 18)]),
        np.array(coco_skeleton),
    ])
    coco_upright_pose_2tracking = np.concatenate([
        openpifpaf.plugins.coco.constants.COCO_UPRIGHT_POSE,
        0.9 * openpifpaf.plugins.coco.constants.COCO_UPRIGHT_POSE + np.array([-1.5, 1.5, 0.0]),
    ])
    coco_sigmas_2tracking = np.concatenate([
        np.array(openpifpaf.plugins.coco.constants.COCO_PERSON_SIGMAS) * scale,
        0.8 * np.array(openpifpaf.plugins.coco.constants.COCO_PERSON_SIGMAS) * scale,
    ])
    ann = openpifpaf.Annotation(coco_keypoints + coco_keypoints, coco_skeleton_forward)
    ann.set(
        coco_upright_pose_2tracking,
        coco_sigmas_2tracking,
    )
    openpifpaf.plugins.coco.constants.draw_ann(
        ann, keypoint_painter=keypoint_painter, filename='docs/coco_skeleton_forward.png')


if __name__ == '__main__':
    main()
