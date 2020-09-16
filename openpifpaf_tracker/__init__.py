import openpifpaf

from .crowdpose import CrowdPose


def register():
    openpifpaf.DATAMODULES['crowdpose'] = CrowdPose

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions
