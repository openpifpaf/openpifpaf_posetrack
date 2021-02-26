import torch


def collate_tracking_images_targets_meta(batch):
    images = torch.utils.data.dataloader.default_collate([
        im for group in batch for im in group[0]])

    targets = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    metas = [b[2] for b in batch]

    return images, targets, metas
