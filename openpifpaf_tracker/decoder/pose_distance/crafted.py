import numpy as np


class Crafted():
    """Compute hand crafted distance between a track and a new pose candidate.

    Ingredients:
    * compare to last pose in track and previous poses in case of temp corruption
    * require low distance for points that have high confidence in both poses (>=3 points)
    * "high confidence" is a dynamic measure dependent on the past track
    * penalize crappy tracks
    * penalize crappy poses
    """
    valid_keypoint_mask = None

    def __init__(self, *, invisible_penalty=50.0):
        self.invisible_penalty = invisible_penalty

    def __call__(self, frame_number, pose, track, track_is_good):
        return min((
            self.distance(frame_number, pose, track, track_is_good),
            self.distance(frame_number, pose, track, track_is_good, -4),
            self.distance(frame_number, pose, track, track_is_good, -8),
            self.distance(frame_number, pose, track, track_is_good, -12),
        ))

    # pylint: disable=too-many-return-statements,too-many-branches
    def distance(self, frame_number, pose, track, track_is_good, track_frame=None):
        last_track_frame = track.data[-1][0]
        skipped_frames = frame_number - last_track_frame - 1
        assert skipped_frames >= 0
        if skipped_frames > 12:
            return 1000.0

        # skipping frames cost
        skipped_frame_cost = 5.0 if track_frame else 0.0

        # correct track_frame with skipped_frames
        if track_frame is not None:
            track_frame += skipped_frames
        else:
            track_frame = -1

        if track_frame > -1:
            return 1000.0

        if len(track.data) < -1.0 * track_frame:
            return 1000.0

        pose1 = pose.data
        pose2 = track.data[track_frame][1].data
        # common valid (cv) keypoints
        cv = np.logical_and(pose1[:, 2] > 0.05, pose2[:, 2] > 0.05)
        if not np.any(cv):
            return 1000.0

        if np.min(pose2[cv, 0]) - np.max(pose1[cv, 0]) > self.invisible_penalty or \
           np.min(pose1[cv, 0]) - np.max(pose2[cv, 0]) > self.invisible_penalty or \
           np.min(pose2[cv, 1]) - np.max(pose1[cv, 1]) > self.invisible_penalty or \
           np.min(pose1[cv, 1]) - np.max(pose2[cv, 1]) > self.invisible_penalty:
            return 1000.0

        keypoint_scores = pose1[:, 2] * pose2[:, 2]
        kps_order = np.argsort(keypoint_scores)[::-1]
        if pose1[kps_order[2], 2] < 0.05 or pose2[kps_order[2], 2] < 0.05:
            return 1000.0
        pose1_center = np.mean(pose1[kps_order[:3], :2], axis=0)
        pose1_centered = np.copy(pose1)
        pose1_centered[:, :2] -= pose1_center
        pose2_center = np.mean(pose2[kps_order[:3], :2], axis=0)
        pose2_centered = np.copy(pose2)
        pose2_centered[:, :2] -= pose2_center
        center_distance = np.linalg.norm(pose2_center - pose1_center)

        kps_distances = np.linalg.norm(pose2_centered[:, :2] - pose1_centered[:, :2], axis=1)
        kps_distances[pose1[:, 2] < 0.05] = self.invisible_penalty
        kps_distances[pose2[:, 2] < 0.05] = self.invisible_penalty
        kps_distances[kps_order[3:]] = np.minimum(
            self.invisible_penalty,
            kps_distances[kps_order[3:]],
        )
        kps_distance = np.sum(kps_distances[self.valid_keypoint_mask]) / pose1.shape[0]

        crappy_track_penalty = 0.0
        if len(track.data) < 4:
            crappy_track_penalty = 5.0
        elif len(track.data) < 8:
            crappy_track_penalty = 1.0
        if not track_is_good:
            crappy_track_penalty = max(crappy_track_penalty, 1.0)

        crappy_pose_penalty = 0.0
        if pose.score() < 0.2:
            crappy_pose_penalty = 5.0
        elif pose.score() < 0.5:
            crappy_pose_penalty = 1.0

        return (
            center_distance / 10.0
            + kps_distance
            + crappy_track_penalty
            + crappy_pose_penalty
            + skipped_frame_cost
        )
