import openpifpaf
import torch


class TBaseSingleImage(openpifpaf.network.HeadNetwork):
    """Filter the feature map so that they can be used by single image loss.

    Training: only apply loss to image 0 of an image pair of image 0 and 1.
    Evaluation with forward tracking pose: only keep image 0.
    Evaluation with full tracking pose: keep all.
    """
    forward_tracking_pose = True
    tracking_pose_length = 2

    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        self.head = openpifpaf.network.heads.CompositeField3(meta, in_features)

    def forward(self, *args):
        x = args[0]

        if self.training:
            x = x[::2]
        elif self.forward_tracking_pose:
            x = x[::self.tracking_pose_length]

        return self.head(x)


class Tcaf(openpifpaf.network.HeadNetwork):
    """Filter the feature map so that they can be used by single image loss.

    Training: only apply loss to image 0 of an image pair of image 0 and 1.
    Evaluation with forward tracking pose: only keep image 0.
    Evaluation with full tracking pose: keep all.
    """
    tracking_pose_length = 2
    reduced_features = 512

    def __init__(self, meta, in_features):
        super().__init__(meta, in_features)
        self.feature_reduction = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_features,
                self.reduced_features,
                kernel_size=1, bias=True,
                # groups=feature_reduction[1],
            ),
            torch.nn.ReLU(),
        )
        self.head = openpifpaf.network.heads.CompositeField3(meta, self.reduced_features * 2)

    def forward(self, *args):
        x = args[0]

        # Batches that are not intended for tracking loss might have an
        # odd number of images (or only 1 image).
        # In that case, simply do not execute this head as the result should
        # never be used.
        if len(x) % 2 == 1:
            return None

        if self.feature_reduction:
            x = self.feature_reduction(x)

        group_length = 2 if self.training else self.tracking_pose_length
        primary = x[::group_length]
        others = [x[i::group_length] for i in range(1, group_length)]

        x = torch.stack([
            torch.cat([primary, other], dim=1)
            for other in others
        ], dim=1)
        x_shape = x.size()
        x = torch.reshape(x, [x_shape[0] * x_shape[1]] + list(x_shape[2:]))

        return self.head(x)
