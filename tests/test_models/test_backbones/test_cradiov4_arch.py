# Copyright (c) OpenMMLab. All rights reserved.
"""C-RADIOv4 trunk selection: ``arch='vit_h'`` (ViT-H) vs ``'vit_so400m'``."""

import importlib
import os.path as osp
import shutil
import sys
from pathlib import Path

import pytest
import torch
from mmengine.config import Config

from mmrotate.models.backbones import CRADIOv4, CRADIOv4CoMer
from mmrotate.models.backbones.cradiov4.checkpoint import load_radio_state_dict
from mmrotate.models.backbones.cradiov4.cradiov4 import default_out_indices, load_radio_weights
from mmrotate.models.backbones.cradiov4.radio_vit import RADIO_ARCHS, RADIOVisionTransformer
from mmrotate.models.backbones.cradiov4.vit_comer import split_stages

SO400M_DIR = '/mnt/zfs/models/nvidia/C-RADIOv4-SO400M'
SO400M_CFG = 'configs/cradio/cradiov4-so400m-comer_rtmdet-3x-dota_ms.yaml'
TINY_VIT = dict(embed_dim=32, depth=4, num_heads=2, max_img_size=256)
TINY_COMER = dict(
    vit_cfg=TINY_VIT,
    conv_inplane=8,
    deform_num_heads=2,
    norm_cfg=dict(type='BN'),
    msda_backend='pytorch',
)


@pytest.mark.parametrize(
    ('arch', 'dim', 'depth', 'hidden', 'num_params'),
    [('vit_h', 1280, 32, 5120, 651_645_440), ('vit_so400m', 1152, 27, 4304, 431_237_232)],
)
def test_arch_presets(arch: str, dim: int, depth: int, hidden: int, num_params: int) -> None:
    vit = RADIOVisionTransformer(arch=arch)
    assert (vit.arch, vit.embed_dim, vit.depth) == (arch, dim, depth)
    assert len(vit.blocks) == depth
    assert vit.blocks[0].mlp.fc1.out_features == hidden
    assert vit.blocks[0].attn.num_heads == 16
    assert vit.num_skip == 10
    assert sum(p.numel() for p in vit.parameters()) == num_params


def test_arch_overrides_and_errors() -> None:
    vit = RADIOVisionTransformer(arch='vit_so400m', **TINY_VIT)
    assert (vit.embed_dim, vit.depth) == (32, 4)
    assert vit.blocks[0].mlp.fc1.out_features == int(32 * 4304 / 1152)
    with pytest.raises(ValueError, match='Unknown arch'):
        RADIOVisionTransformer(arch='vit_g')
    assert set(RADIO_ARCHS) == {'vit_h', 'vit_so400m'}


def test_default_indices() -> None:
    assert split_stages(32) == [(0, 7), (8, 15), (16, 23), (24, 31)]
    assert split_stages(27) == [(0, 6), (7, 13), (14, 20), (21, 26)]
    assert split_stages(4) == [(0, 0), (1, 1), (2, 2), (3, 3)]
    assert default_out_indices(32, 3) == (15, 23, 31)
    assert default_out_indices(27, 3) == (12, 19, 26)
    assert default_out_indices(27, 1) == (26,)
    with pytest.raises(ValueError):
        default_out_indices(27, 5)


def test_top_level_arch_reaches_vit() -> None:
    # Top-level `arch` wins over vit_cfg; the tiny sizes still override the preset.
    vit_cfg = dict(TINY_VIT, arch='vit_h')
    plain = CRADIOv4(arch='vit_so400m', vit_cfg=vit_cfg)
    assert plain.vit.arch == 'vit_so400m'
    assert plain.out_indices == default_out_indices(4, 3)
    comer = CRADIOv4CoMer(arch='vit_so400m', **dict(TINY_COMER, vit_cfg=vit_cfg))
    assert comer.vit.arch == 'vit_so400m'
    assert comer.interaction_indexes == split_stages(4)
    outs = comer(torch.rand(1, 3, 64, 96))
    assert [o.shape[-2:] for o in outs] == [(8, 12), (4, 6), (2, 3)]


def test_comer_rejects_partial_interactions() -> None:
    # e.g. ViT-H stages [.., [24, 31]] left over on a 27 block SO400M.
    with pytest.raises(ValueError, match='end at block 2'):
        CRADIOv4CoMer(interaction_indexes=[[0, 0], [1, 1], [2, 2]], **TINY_COMER)


def test_arch_checkpoint_mismatch(tmp_path: Path) -> None:
    src = RADIOVisionTransformer(arch='vit_so400m', **dict(TINY_VIT, depth=3))
    torch.save(src.state_dict(), tmp_path / 'radio.pth')
    vit = RADIOVisionTransformer(arch='vit_so400m', **TINY_VIT)
    with pytest.raises(ValueError, match='depth=3.*Set the backbone `arch`'):
        load_radio_weights(vit, str(tmp_path / 'radio.pth'))


def test_so400m_config() -> None:
    cfg = Config.fromfile(SO400M_CFG)
    backbone = cfg.model.backbone
    assert backbone.type == 'CRADIOv4CoMer'
    assert backbone.arch == 'vit_so400m'
    assert backbone.checkpoint == SO400M_DIR
    assert [tuple(i) for i in backbone.interaction_indexes] == split_stages(27)
    assert cfg.model.neck.in_channels == [1152] * 3
    assert Config.fromfile(SO400M_CFG.replace('3x-dota_ms', 'proxy')).model == cfg.model


@pytest.mark.skipif(not osp.isdir(SO400M_DIR), reason='C-RADIOv4-SO400M weights not available')
def test_load_so400m_weights_strict() -> None:
    vit = RADIOVisionTransformer(arch='vit_so400m')
    vit.load_state_dict(load_radio_state_dict(SO400M_DIR), strict=True)
    with pytest.raises(ValueError, match='embed_dim=1152, depth=27'):
        load_radio_weights(RADIOVisionTransformer(arch='vit_h'), SO400M_DIR)


def _import_hf_reference(tmp_path: Path) -> object:
    # Own package name: the ViT-H parity test caches its copy as ``cradio_ref``.
    pkg = tmp_path / 'cradio_so400m_ref'
    pkg.mkdir()
    for src in Path(SO400M_DIR).glob('*.py'):
        shutil.copy(src, pkg / src.name)
    (pkg / '__init__.py').touch()
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module('cradio_so400m_ref.hf_model')
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.skipif(
    not (osp.isdir(SO400M_DIR) and torch.cuda.is_available()),
    reason='requires the C-RADIOv4-SO400M download and a GPU',
)
def test_so400m_parity_with_hf_reference(tmp_path: Path) -> None:
    """``arch='vit_so400m'`` must reproduce the official HF SO400M model."""
    pytest.importorskip('timm')
    pytest.importorskip('transformers')
    from safetensors.torch import load_file

    hf_model = _import_hf_reference(tmp_path)
    ref = hf_model.RADIOModel(hf_model.RADIOConfig.from_pretrained(SO400M_DIR))
    ref.load_state_dict(load_file(osp.join(SO400M_DIR, 'model.safetensors')), strict=True)
    ref = ref.cuda().eval()
    ours = RADIOVisionTransformer(arch='vit_so400m', normalize_input=True)
    ours.load_state_dict(load_radio_state_dict(SO400M_DIR), strict=True)
    ours = ours.cuda().eval()

    x = torch.rand(1, 3, 256, 384, device='cuda')
    with torch.no_grad():
        ref_summary, ref_feat = ref(x)
        summary, feat = ours(x)
    torch.testing.assert_close(summary, ref_summary, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(feat, ref_feat, atol=1e-4, rtol=1e-4)
