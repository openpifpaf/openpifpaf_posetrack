import logging

import numpy as np
import PIL
import scipy
import torch

import openpifpaf

LOG = logging.getLogger(__name__)


class HorizontalBlur(openpifpaf.transforms.Preprocess):
    def __init__(self, sigma=5.0):
        self.sigma = sigma

    def __call__(self, image, anns, meta):
        im_np = np.asarray(image)
        sigma = self.sigma * (0.8 + 0.4 * float(torch.rand(1).item()))
        LOG.debug('horizontal blur with %f', sigma)
        im_np = scipy.ndimage.filters.gaussian_filter1d(im_np, sigma=sigma, axis=1)
        return PIL.Image.fromarray(im_np), anns, meta
