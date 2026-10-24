# Copyright (c) OpenMMLab. All rights reserved.
import importlib
import os.path as osp
import shutil
import sys
from pathlib import Path

import pytest
import torch

from mmrotate.models.backbones.cradiov4.checkpoint import (
    HF_PREFIX,
    load_radio_state_dict,
    resolve_checkpoint,
)
from mmrotate.models.backbones.cradiov4.comer_modules import (
    CNN,
    MRFP,
    ConvFFN,
    deform_inputs,
    get_reference_points,
)
from mmrotate.models.backbones.cradiov4.cradiov4 import resize_to_stride
from mmrotate.models.backbones.cradiov4.radio_vit import (
    RADIOVisionTransformer,
    ViTPatchGenerator,
)

HF_DIR = '/mnt/zfs/models/nvidia/C-RADIOv4-H'
TINY_VIT = dict(embed_dim=32, depth=2, num_heads=2, max_img_size=128)


def _tiny_state_dict() -> dict[str, torch.Tensor]:
    return RADIOVisionTransformer(**TINY_VIT).state_dict()


def test_load_radio_state_dict_safetensors(tmp_path: Path) -> None:
    from safetensors.torch import save_file

    state = _tiny_state_dict()
    save_file(
        {HF_PREFIX + k: v.contiguous() for k, v in state.items()}, tmp_path / 'model.safetensors'
    )
    # A directory resolves to its model.safetensors and the prefix is stripped.
    assert resolve_checkpoint(str(tmp_path)) == str(tmp_path / 'model.safetensors')
    loaded = load_radio_state_dict(str(tmp_path))
    assert loaded.keys() == state.keys()
    RADIOVisionTransformer(**TINY_VIT).load_state_dict(loaded, strict=True)


def test_load_radio_state_dict_torch(tmp_path: Path) -> None:
    state = _tiny_state_dict()
    torch.save({'state_dict': state}, tmp_path / 'radio.pth')
    loaded = load_radio_state_dict(str(tmp_path / 'radio.pth'))
    assert loaded.keys() == state.keys()


def test_resolve_checkpoint_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path))
    with pytest.raises(FileNotFoundError):
        resolve_checkpoint(str(tmp_path / 'missing.safetensors'))


def test_patch_generator_pos_embed() -> None:
    gen = ViTPatchGenerator(16, 8, 128, num_cls_tokens=3, num_registers=7)
    torch.nn.init.normal_(gen.pos_embed)
    # Full grid -> unchanged; smaller grids -> square resize then top-left crop.
    assert gen.get_pos_embed(8, 8) is gen.pos_embed
    assert gen.get_pos_embed(2, 4).shape == (1, 8, 8)
    tokens, h, w = gen(torch.rand(2, 3, 32, 64))
    assert (h, w) == (2, 4)
    assert tokens.shape == (2, gen.num_skip + 8, 8)
    assert gen.num_skip == 10


def test_patch_embedding_matches_conv() -> None:
    """Patch features are ordered (c, ky, kx), i.e. a flattened conv weight."""
    gen = ViTPatchGenerator(16, 8, 128, num_cls_tokens=1, num_registers=0)
    x = torch.rand(1, 3, 32, 48)
    patches, _, _ = gen.embed_patches(x)
    weight = gen.embedder.weight.view(8, 3, 16, 16)
    ref = torch.nn.functional.conv2d(x, weight, stride=16).flatten(2).transpose(1, 2)
    torch.testing.assert_close(patches, ref)


@pytest.mark.parametrize(
    'dst_stride, expected', [(4, (32, 48)), (8, (16, 24)), (16, (8, 12)), (32, (4, 6))]
)
def test_resize_to_stride(dst_stride: int, expected: tuple[int, int]) -> None:
    feat = torch.rand(1, 5, 8, 12)
    out = resize_to_stride(feat, 16, dst_stride)
    assert out.shape[-2:] == expected
    if dst_stride == 16:
        assert out is feat


def test_reference_points_and_deform_inputs() -> None:
    ref = get_reference_points([(2, 4)], torch.device('cpu'))
    assert ref.shape == (1, 8, 1, 2)
    assert ref.min() > 0 and ref.max() < 1
    (ref1, shapes1, starts1), (ref2, shapes2, starts2) = deform_inputs(torch.rand(1, 3, 128, 64))
    assert shapes1.tolist() == [[16, 8], [8, 4], [4, 2]]
    assert starts1.tolist() == [0, 128, 160]
    assert ref1.shape == (1, 8 * 4, 1, 2)
    assert shapes2.tolist() == [[8, 4]] and starts2.tolist() == [0]
    assert ref2.shape == (1, 128 + 32 + 8, 1, 2)


def test_spm_and_conv_ffn_shapes() -> None:
    spm = CNN(inplanes=8, embed_dim=16, norm_cfg=dict(type='BN'))
    c1, c2, c3, c4 = spm(torch.rand(2, 3, 64, 64))
    assert c1 is not None and c1.shape == (2, 16, 16, 16)
    assert (c2.shape, c3.shape, c4.shape) == ((2, 64, 16), (2, 16, 16), (2, 4, 16))
    assert CNN(8, 16, dict(type='BN'), with_stride4=False)(torch.rand(1, 3, 64, 64))[0] is None
    tokens = torch.cat([c2, c3, c4], dim=1)
    assert ConvFFN(16, 8)(tokens, 4, 4).shape == tokens.shape
    assert MRFP(16, 8)(tokens, 4, 4).shape == tokens.shape


def _import_hf_reference(tmp_path: Path) -> type:
    pkg = tmp_path / 'cradio_ref'
    pkg.mkdir()
    for src in Path(HF_DIR).glob('*.py'):
        shutil.copy(src, pkg / src.name)
    (pkg / '__init__.py').touch()
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module('cradio_ref.hf_model')
    finally:
        sys.path.remove(str(tmp_path))


@pytest.mark.skipif(
    not (osp.isdir(HF_DIR) and torch.cuda.is_available()),
    reason='requires the C-RADIOv4-H download and a GPU',
)
def test_parity_with_hf_reference(tmp_path: Path) -> None:
    """Our re-implementation must reproduce the official HF model exactly."""
    pytest.importorskip('timm')
    pytest.importorskip('transformers')
    from safetensors.torch import load_file

    hf_model = _import_hf_reference(tmp_path)
    ref = hf_model.RADIOModel(hf_model.RADIOConfig.from_pretrained(HF_DIR))
    ref.load_state_dict(load_file(osp.join(HF_DIR, 'model.safetensors')), strict=True)
    ref = ref.cuda().eval()
    ours = RADIOVisionTransformer(normalize_input=True)
    ours.load_state_dict(load_radio_state_dict(HF_DIR), strict=True)
    ours = ours.cuda().eval()

    x = torch.rand(1, 3, 256, 384, device='cuda')
    with torch.no_grad():
        ref_summary, ref_feat = ref(x)
        summary, feat = ours(x)
    torch.testing.assert_close(summary, ref_summary, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(feat, ref_feat, atol=1e-4, rtol=1e-4)
