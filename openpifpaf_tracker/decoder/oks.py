import numpy as np


class Oks:
    """Compute OKS distance between a track and a new pose candidate.

    From http://cocodataset.org/#keypoints-eval:
    OKS = Σi[exp(-di2/2s2κi2)δ(vi>0)] / Σi[δ(vi>0)]
    with κi=2σi, s is object scale and sigma is the keypoint sigma.

    Ingredients:
    * compare to last pose in track and previous poses in case of temp corruption
    * require low distance for points that have high confidence in both poses (>=3 points)
    * "high confidence" is a dynamic measure dependent on the past track
    * penalize crappy tracks
    * penalize crappy poses
    """
    valid_mask = None
    sigmas = None
    inflate = 1.0

    def __call__(self, frame_number, pose, track, track_is_good):
        return min((
            self.distance(frame_number, pose, track, track_is_good),
            self.distance(frame_number, pose, track, track_is_good, -4),
            self.distance(frame_number, pose, track, track_is_good, -8),
            self.distance(frame_number, pose, track, track_is_good, -12),
        ))

    @staticmethod
    def scale(pose):
        pose = pose[pose[:, 2] > 0.05]
        area = (pose[:, 0].max() - pose[:, 0].min()) * (pose[:, 1].max() - pose[:, 1].min())
        return np.sqrt(area)

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
        visible = np.logical_and(pose1[:, 2] > 0.05, pose2[:, 2] > 0.05)
        if not np.any(visible):
            return 1000.0
        scale = 0.5 * (self.scale(pose1) + self.scale(pose2))
        scale = max(0.1, scale)

        d = np.linalg.norm((pose2[:, :2] - pose1[:, :2]), axis=1)
        k = 2.0 * self.sigmas
        k *= self.inflate
        g = np.exp(-0.5 * d**2 / (scale**2 * k**2))
        oks = np.mean(g[visible])

        return 11.0 - 11.0*oks
