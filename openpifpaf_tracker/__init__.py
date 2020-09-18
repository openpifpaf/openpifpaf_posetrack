import openpifpaf

from . import headmeta, heads
from .crowdpose import CrowdPose
from .posetrack2018 import Posetrack2018

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def register():
    openpifpaf.DATAMODULES['crowdpose'] = CrowdPose
    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018
    openpifpaf.HEAD_FACTORIES[headmeta.TBaseCif] = heads.TBaseSingleImage
    openpifpaf.HEAD_FACTORIES[headmeta.TBaseCaf] = heads.TBaseSingleImage
    openpifpaf.HEAD_FACTORIES[headmeta.Tcaf] = heads.Tcaf
