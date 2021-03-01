import openpifpaf

from . import decoder, headmeta, heads
from .backbone import TBackbone
from .cocokpst import CocoKpSt
from .posetrack2018 import Posetrack2018
from .posetrack2017 import Posetrack2017
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


def tcaf_shared_preprocessing(model):
    for m in model.modules():
        if not isinstance(m, heads.Tcaf):
            continue

        # pylint: disable=protected-access
        heads.Tcaf._global_feature_reduction = m.feature_reduction
        heads.Tcaf._global_feature_compute = m.feature_compute
        return


def register():
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k16'] = (
        'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
        'v0.12.2/tshufflenetv2k16-210228-220045-posetrack2018-cocokpst-o10-856584da.pkl'
    )
    # openpifpaf.CHECKPOINT_URLS['tshufflenetv2k30'] = (
    #     'http://github.com/vita-epfl/openpifpaf-torchhub/releases/download/'
    #     'v0.12.2/tshufflenetv2k30-210222-112623-posetrack2018-cocokpst-o10-123ec670.pkl'
    # )

    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018
    openpifpaf.DATAMODULES['posetrack2017'] = Posetrack2017
    openpifpaf.DATAMODULES['cocokpst'] = CocoKpSt

    # TODO resolve conflicting names: TBase TBackbone
    openpifpaf.HEADS[headmeta.TBaseCif] = heads.TBaseSingleImage
    openpifpaf.HEADS[headmeta.TBaseCaf] = heads.TBaseSingleImage
    openpifpaf.HEADS[headmeta.Tcaf] = heads.Tcaf

    openpifpaf.LOSSES[headmeta.TBaseCif] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.TBaseCaf] = openpifpaf.network.losses.CompositeLoss
    openpifpaf.LOSSES[headmeta.Tcaf] = openpifpaf.network.losses.CompositeLoss

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
    openpifpaf.MODEL_MIGRATION.add(tcaf_shared_preprocessing)
