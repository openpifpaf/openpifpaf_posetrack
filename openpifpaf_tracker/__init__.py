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
        'https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.12a5/'
        'tshufflenetv2k16-201112-085543-posetrack2018-cocokpst-o50-f8d3e7d5.pkl'
    )
    openpifpaf.CHECKPOINT_URLS['tshufflenetv2k30'] = (
        'https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.12a5/'
        'tshufflenetv2k30-201115-090131-posetrack2018-cocokpst-o50-934fdc1c.pkl'
    )
    openpifpaf.CHECKPOINT_URLS['resnet50-crowdpose'] = (
        'https://github.com/vita-epfl/openpifpaf-torchhub/releases/download/v0.12a7/'
        'resnet50-201005-100758-crowdpose-d978a89f.pkl'
    )

    openpifpaf.DATAMODULES['crowdpose'] = CrowdPose
    openpifpaf.DATAMODULES['posetrack2018'] = Posetrack2018
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
