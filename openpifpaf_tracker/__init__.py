import openpifpaf

from . import decoder, headmeta, heads
from .backbone import TBackbone
from .cocokpst import CocoKpSt
from .crowdpose import CrowdPose
from .posetrack2018 import Posetrack2018
from .signal import Signal

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions


def fix_feature_cache(model):
    for m in model.modules():
        if not isinstance(m, TBackbone):
            continue
        m.reset()


def subscribe_cache_reset(model):
    for m in model.modules():
        if not isinstance(m, TBackbone):
            continue
        Signal.subscribe('eval_reset', m.reset)


def register():
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k16'] = (
        'https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.12a5/'
        'tshufflenetv2k16-201112-085543-posetrack2018-cocokpst-o50-f8d3e7d5.pkl'
    )
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k30'] = (
        'https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.12a5/'
        'tshufflenetv2k30-201115-090131-posetrack2018-cocokpst-o50-934fdc1c.pkl'
    )

    openpifpaf.DATAMODULES['crowdpose'] = CrowdPose
    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018
    openpifpaf.DATAMODULES['cocokpst'] = CocoKpSt

    # TODO resolve conflicting names: TBase TBackbone
    openpifpaf.HEAD_FACTORIES[headmeta.TBaseCif] = heads.TBaseSingleImage
    openpifpaf.HEAD_FACTORIES[headmeta.TBaseCaf] = heads.TBaseSingleImage
    openpifpaf.HEAD_FACTORIES[headmeta.Tcaf] = heads.Tcaf

    openpifpaf.BASE_TYPES.add(TBackbone)
    openpifpaf.BASE_FACTORIES['tshufflenetv2k16'] = lambda: TBackbone(
        openpifpaf.BASE_FACTORIES['shufflenetv2k16']())
    openpifpaf.BASE_FACTORIES['tshufflenetv2k30'] = lambda: TBackbone(
        openpifpaf.BASE_FACTORIES['shufflenetv2k30']())
    openpifpaf.BASE_FACTORIES['tresnet50'] = lambda: TBackbone(
        openpifpaf.BASE_FACTORIES['resnet50']())

    openpifpaf.DECODERS.add(decoder.PoseSimilarity)
    openpifpaf.DECODERS.add(decoder.TrackingPose)

    openpifpaf.MODEL_MIGRATION.add(fix_feature_cache)
    openpifpaf.MODEL_MIGRATION.add(subscribe_cache_reset)
