import torch

from .signal import Signal


def collate_tracking_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([
        im for group in batch for im in group[0]])

    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]

    return images, targets, metas


class CollateImagesAnnsMetaWithReset:
    def __init__(self, key_to_monitor):
        self.key_to_monitor = key_to_monitor
        self.previous_value = None

    def __call__(self, batch):
        images = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
        anns = [b[1] for b in batch]
        metas = [b[2] for b in batch]

        value = metas[0][self.key_to_monitor]
        if len(metas) >= 2:
            assert all(m[self.key_to_monitor] == value for m in metas[1:])

        if value != self.previous_value:
            Signal.emit('eval_reset')
            self.previous_value = value

        return images, anns, metas
