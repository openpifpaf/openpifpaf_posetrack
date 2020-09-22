import numpy as np


class Euclidean():
    """Compute Euclidean distance between a track and a new pose candidate.

    Ingredients:
    * compare to last pose in track and previous poses in case of temp corruption
    * require low distance for points that have high confidence in both poses (>=3 points)
    * "high confidence" is a dynamic measure dependent on the past track
    * penalize crappy tracks
    * penalize crappy poses
    """

    valid_mask = None

    def __init__(self, *, invisible_penalty=50.0, metric=1.0):
        self.invisible_penalty = invisible_penalty
        self.metric = metric

    def __call__(self, frame_number, pose, track, track_is_good):
        return min((
            self.distance(frame_number, pose, track, track_is_good),
            self.distance(frame_number, pose, track, track_is_good, -4),
            self.distance(frame_number, pose, track, track_is_good, -8),
            self.distance(frame_number, pose, track, track_is_good, -12),
        ))

    def distance(self, frame_number, pose, track, track_is_good, track_frame=None):
        # TODO: incorporate track_is_good similarly to how it is done in crafted
        last_track_frame = track.frame_pose[-1][0]
        skipped_frames = frame_number - last_track_frame - 1
        assert skipped_frames >= 0
        if skipped_frames > 12:
            return 1000.0

        # correct track_frame with skipped_frames
        if track_frame is not None:
            track_frame += skipped_frames
        else:
            track_frame = -1

        if track_frame > -1:
            return 1000.0

        if len(track.frame_pose) < -1.0 * track_frame:
            return 1000.0

        pose1 = pose.data[self.valid_mask]
        pose2 = track.frame_pose[track_frame][1].data[self.valid_mask]

        kps_distances = np.linalg.norm((pose2[:, :2] - pose1[:, :2]) / self.metric, axis=1)
        kps_distances = np.clip(kps_distances, 0.0, self.invisible_penalty)
        kps_distances[pose1[:, 2] < 0.05] = self.invisible_penalty
        kps_distances[pose2[:, 2] < 0.05] = self.invisible_penalty
        kps_distance = np.mean(kps_distances)

        return kps_distance
