# Copyright (c) OpenMMLab. All rights reserved.
"""Engines for other GPUs: hardware compatibility levels, plugin PTX target, rebuilds."""

import json
import subprocess
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from torch import nn
from trt_test_utils import convert_tiny

from mmrotate.tensorrt.builder import compat_covers
from mmrotate.tensorrt.plugins.msda import ptx_target

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')
REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ('level', 'built', 'running', 'covered'),
    [
        ('none', '12.0', '12.0', True),
        ('none', '12.0', '8.9', False),
        ('same-cc', '12.0', '12.0', True),
        ('same-cc', '12.0', '10.0', False),
        ('ampere+', '12.0', '8.6', True),
        ('ampere+', '12.0', '8.9', True),
        ('ampere+', '8.6', '12.0', True),
        ('ampere+', '12.0', '7.5', False),
        (None, None, '8.6', True),  # meta of an old deploy dir without build info
    ],
)
def test_compat_covers(level: str | None, built: str | None, running: str, covered: bool) -> None:
    assert compat_covers(level, built, running) is covered


def test_ptx_target() -> None:
    assert ptx_target('.version 8.8\n.target sm_80\n.address_size 64\n') == 'sm_80'
    assert ptx_target('//\n.version 9.0\n.target sm_120a\n') == 'sm_120a'
    with pytest.raises(ValueError):
        ptx_target('.version 9.0\n')
    from mmrotate.tensorrt.plugins.msda import ptx_version

    assert ptx_version('//\n.version 9.0\n.target sm_120a\n') == '9.0'


@needs_cuda
def test_plugin_ptx_arch_compiles_portable_ptx() -> None:
    pytest.importorskip('triton')
    from mmrotate.tensorrt.plugins.msda import compile_msda_kernel, plugin_ptx_arch

    dtypes = dict(value='fp16', shapes='i64', ref='fp32', off='fp16', logit='fp16', out='fp16')
    kwargs = dict(H=2, C=16, L=3, P=4, LR=1, block_n=2, num_warps=1)
    major, minor = torch.cuda.get_device_capability()
    _, native, _ = compile_msda_kernel(dtypes, **kwargs)
    assert ptx_target(native).startswith(f'sm_{major}{minor}')
    with plugin_ptx_arch(80):
        name, ptx, _ = compile_msda_kernel(dtypes, **kwargs)
    assert ptx_target(ptx) == 'sm_80'
    entry = ptx[ptx.index(f'.entry {name}(') : ptx.index('\n)', ptx.index(f'.entry {name}('))]
    assert entry.count('.param') == 9
    # The override is undone on exit.
    _, again, _ = compile_msda_kernel(dtypes, **kwargs)
    assert ptx_target(again) == ptx_target(native)


@needs_cuda
def test_plugin_ptx_version_is_loadable_by_the_driver() -> None:
    pytest.importorskip('triton')
    from mmrotate.tensorrt.plugins.msda import (
        PTX_VERSION_CAP,
        compile_msda_kernel,
        driver_ptx_version,
        plugin_ptx_arch,
        ptx_version,
    )

    dtypes = dict(value='fp16', shapes='i64', ref='fp32', off='fp16', logit='fp16', out='fp16')
    kwargs = dict(H=2, C=16, L=3, P=4, LR=1, block_n=2, num_warps=1)
    limit = min(PTX_VERSION_CAP, driver_ptx_version())
    for arch in (None, 80):
        with plugin_ptx_arch(arch):
            _, ptx, _ = compile_msda_kernel(dtypes, **kwargs)
        major, minor = map(int, ptx_version(ptx).split('.'))
        assert 10 * major + minor <= limit, (arch, ptx_version(ptx), limit)


def _run(model: nn.Module, deploy_dir: Path, x: torch.Tensor) -> list[torch.Tensor]:
    from mmrotate.tensorrt import init_trt_detector
    from mmrotate.tensorrt.export import flatten_outputs

    with torch.no_grad():
        outs = init_trt_detector(deploy_dir, 'cuda')._forward(x)
    return [t.float() for t in flatten_outputs(outs)[0]]


@needs_cuda
@pytest.mark.parametrize('plugins', ['aot', 'jit'])
def test_ampere_plus_engine_with_msda_plugin_matches_native(
    plugins: str,
    tiny_detector: Callable[..., nn.Module],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip('tensorrt')
    pytest.importorskip('triton')
    from mmrotate.tensorrt import builder

    model = tiny_detector('comer', 'cuda')
    shape = (2, 3, 128, 128)
    native_dir, compat_dir = tmp_path / 'native', tmp_path / 'compat'
    native_dir.mkdir()
    compat_dir.mkdir()
    convert_tiny(model, 'comer', native_dir, shape, precision='fp16', plugins=plugins)

    levels = []
    original = builder.build_engine

    def compat_build(*args: object, **kwargs: object) -> None:
        kwargs['hardware_compat'] = 'ampere+'
        levels.append('ampere+')
        original(*args, **kwargs)

    monkeypatch.setattr(builder, 'build_engine', compat_build)
    meta = convert_tiny(model, 'comer', compat_dir, shape, precision='fp16', plugins=plugins)
    assert levels == ['ampere+'] and meta['fusion']['plugins'] > 0

    x = torch.randn(shape, device='cuda')
    for a, b in zip(_run(model, native_dir, x), _run(model, compat_dir, x), strict=True):
        assert torch.isfinite(b).all()
        assert float((a - b).abs().max()) <= 0.02 * max(1.0, float(a.abs().max()))


@needs_cuda
def test_detector_warns_only_when_the_engine_cannot_cover_the_gpu(
    tiny_detector: Callable[..., nn.Module], tmp_path: Path
) -> None:
    pytest.importorskip('tensorrt')
    from mmrotate.tensorrt import init_trt_detector
    from mmrotate.tensorrt.deploy import write_meta

    model = tiny_detector('radio', 'cuda')
    meta = convert_tiny(model, 'radio', tmp_path, (1, 3, 64, 64))
    major, minor = torch.cuda.get_device_capability()
    other = '7.5' if major >= 8 else '8.0'

    def warnings_for(build: dict) -> list[str]:
        write_meta(tmp_path, dict(meta, build=build))
        detector = init_trt_detector(tmp_path, 'cuda')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            detector._check_environment(torch.device('cuda'))
        return [str(w.message) for w in caught if 'compute capability' in str(w.message)]

    assert not warnings_for(dict(compute_capability=f'{major}.{minor}'))
    assert warnings_for(dict(compute_capability=other, hardware_compat='none'))
    if major >= 8:
        assert not warnings_for(dict(compute_capability=other, hardware_compat='ampere+'))
    assert warnings_for(dict(compute_capability=other, hardware_compat='same-cc'))


@needs_cuda
def test_build_engine_tool_rebuilds_a_deploy_dir(
    tiny_detector: Callable[..., nn.Module], tmp_path: Path
) -> None:
    pytest.importorskip('tensorrt')
    pytest.importorskip('triton')
    from mmrotate.tensorrt.deploy import ONNX_FILE, write_meta

    model = tiny_detector('comer', 'cuda')
    shape = (1, 3, 128, 128)
    src, dst = tmp_path / 'src', tmp_path / 'dst'
    src.mkdir()
    meta = convert_tiny(model, 'comer', src, shape, precision='fp16', plugins='aot')
    (src / 'm.onnx').rename(src / ONNX_FILE)
    write_meta(src, dict(meta, build=dict(gpu='elsewhere', compute_capability='8.6')))
    for stale in ('test.log', 'profile.json', '.tested', 'timing.cache'):  # source results
        (src / stale).write_text('from the source GPU')

    cmd = [
        sys.executable, str(REPO / 'tools/tensorrt/build_engine.py'), str(src),
        '--out-dir', str(dst), '--hardware-compat', 'ampere+',
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=REPO)
    new = json.loads((dst / 'trt_meta.json').read_text())
    major, minor = torch.cuda.get_device_capability()
    assert new['build']['compute_capability'] == f'{major}.{minor}'
    assert new['build']['hardware_compat'] == 'ampere+'
    assert new['build']['rebuilt_from']['compute_capability'] == '8.6'
    assert new['fusion']['plugins'] == meta['fusion']['plugins'] > 0
    for stale in ('test.log', 'profile.json', '.tested'):
        assert not (dst / stale).exists(), stale
    assert (dst / 'timing.cache').read_bytes() != b'from the source GPU'
    x = torch.randn(shape, device='cuda')
    for a, b in zip(_run(model, src, x), _run(model, dst, x), strict=True):
        assert float((a - b).abs().max()) <= 0.02 * max(1.0, float(a.abs().max()))


def test_fusion_summary_counts_plugins_inside_myelin_graphs() -> None:
    from mmrotate.tensorrt import engine_report

    layers = [
        dict(Name='/backbone/attn/MSDAFused_myl0_49', LayerType='custom_layer'),
        dict(Name='x', LayerType='PluginV3'),
        dict(Name='y', LayerType='gemm'),
        dict(Name='__myl_Custom_myl0_3', LayerType='custom_layer'),  # not a plugin
    ]
    assert engine_report.fusion_summary(layers)['plugins'] == 2
