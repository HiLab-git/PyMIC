from __future__ import absolute_import
from pymic.net_run.self_sup.self_genesis import SelfSupModelGenesis
from pymic.net_run.self_sup.self_patch_swapping import SelfSupPatchSwapping
# from pymic.net_run.self_sup.self_mim import SelfSupMIM
# from pymic.net_run.self_sup.self_dino import SelfSupDINO
from pymic.net_run.self_sup.self_vox2vec import SelfSupVox2Vec
from pymic.net_run.self_sup.self_volf import SelfSupVolumeFusion

SelfSupMethodDict = {
    # 'DINO': SelfSupDINO,
    'Vox2Vec': SelfSupVox2Vec,
    'ModelGenesis': SelfSupModelGenesis,
    'PatchSwapping': SelfSupPatchSwapping,
    'VolumeFusion': SelfSupVolumeFusion
    # 'MaskedImageModeling': SelfSupMIM
    }