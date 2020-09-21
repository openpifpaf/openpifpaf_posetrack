import openpifpaf

from . import headmeta, heads
from .backbone import TBackbone
from .crowdpose import CrowdPose
from .posetrack2018 import Posetrack2018

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def register():
    openpifpaf.DATAMODULES['crowdpose'] = CrowdPose
    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018

    # TODO resolve conflicting names: TBase TBackbone
    openpifpaf.HEAD_FACTORIES[headmeta.TBaseCif] = heads.TBaseSingleImage
    openpifpaf.HEAD_FACTORIES[headmeta.TBaseCaf] = heads.TBaseSingleImage
    openpifpaf.HEAD_FACTORIES[headmeta.Tcaf] = heads.Tcaf

    openpifpaf.BASE_TYPES.add(TBackbone)
    openpifpaf.BASE_FACTORIES['tshufflenetv2k16'] = lambda: TBackbone(
        openpifpaf.network.factory(checkpoint='shufflenetv2k16')[0].base_net)
    openpifpaf.BASE_FACTORIES['tshufflenetv2k30'] = lambda: TBackbone(
        openpifpaf.network.factory(checkpoint='shufflenetv2k30')[0].base_net)
    openpifpaf.BASE_FACTORIES['tresnet50'] = lambda: TBackbone(
        openpifpaf.network.factory(checkpoint='resnet50')[0].base_net)
