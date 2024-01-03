from __future__ import absolute_import
from pymic.net_run.self_sup.self_genesis import SelfSupModelGenesis
from pymic.net_run.self_sup.self_patch_swapping import SelfSupPatchSwapping
from pymic.net_run.self_sup.self_volume_fusion import SelfSupVolumeFusion

SelfSupMethodDict = {
    'ModelGenesis': SelfSupModelGenesis,
    'PatchSwapping': SelfSupPatchSwapping,
    'VolumeFusion': SelfSupVolumeFusion
    }