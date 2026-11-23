# Copyright (c) OpenMMLab. All rights reserved.
"""What a built engine contains and where its time goes.

- :func:`layer_info`: the engine's layers after TensorRT's fusions (engines are
  built with detailed profiling verbosity, see ``builder.py``).
- :func:`fusion_summary`: how many fused attention kernels (``mha`` layers)
  and plugin layers the engine has. ``convert.py`` compares the former with
  the number of attention calls of the model.
- :func:`profile_layers` / :func:`group_times`: per-layer times from TensorRT's
  profiler, summed per model component.
"""

import json
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence

import torch
from torch import Tensor

from .plugins.msda import PLUGIN_NAME
from .runtime import TRTEngine

# First match wins. TensorRT names fused layers after the ONNX nodes (module
# scopes) they contain, or gives them a synthetic name (``__myl_...``); the
# names of a layer's output tensors, then of its inputs, are matched as well.
DEFAULT_GROUPS: tuple[tuple[str, str], ...] = (
    ('msda', r'MSDAFused|/attn/(GridSample|value_proj|sampling_offsets|attention_weights)'),
    ('vit attention', r'blocks?[./]\d+/attn/'),
    ('vit mlp', r'blocks?[./]\d+/mlp/'),
    ('vit other', r'/backbone/(blocks?[./]\d+|patch_embed|model|vit|norm|rope)'),
    ('backbone other', r'/backbone/'),
    ('neck', r'/neck/'),
    ('head', r'/bbox_head/|/(cls|reg)_convs|/rtm_'),
)
MHA_GROUP = 'vit attention'


def layer_info(engine: TRTEngine) -> list[dict]:
    """Layers of ``engine`` as dicts (``Name``, ``LayerType``, I/O, ...)."""
    import tensorrt as trt

    inspector = engine.engine.create_engine_inspector()
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info.get('Layers', [])
    # Without detailed verbosity the layers are plain name strings.
    return [layer if isinstance(layer, dict) else {'Name': layer} for layer in layers]


def _is_mha(layer: dict) -> bool:
    text = ' '.join(str(layer.get(k, '')) for k in ('LayerType', 'ParameterType', 'TacticName'))
    return 'mha' in text.lower() or 'attention' in text.lower()


def _is_plugin(layer: dict) -> bool:
    layer_type = str(layer.get('LayerType', '')).lower()
    if 'plugin' in layer_type:
        return True
    # TensorRT 11.4 on Blackwell places plugins inside a Myelin graph, where they are listed as
    # ``custom_layer`` named after the ONNX node; other custom layers (e.g. in INT8 graphs) exist.
    return layer_type == 'custom_layer' and PLUGIN_NAME in str(layer.get('Name', ''))


def fusion_summary(layers: Sequence[dict]) -> dict:
    """Counts of the layer kinds this package cares about."""
    return dict(
        layers=len(layers),
        mha=sum(_is_mha(layer) for layer in layers),
        plugins=sum(_is_plugin(layer) for layer in layers),
    )


class _Profiler:
    """Accumulates TensorRT's per-layer times (ms)."""

    def __init__(self) -> None:
        self.times: dict[str, float] = defaultdict(float)

    def report_layer_time(self, layer_name: str, ms: float) -> None:
        self.times[layer_name] += ms


def _make_profiler() -> object:
    import tensorrt as trt

    class Profiler(trt.IProfiler):
        def __init__(self) -> None:
            trt.IProfiler.__init__(self)
            self.impl = _Profiler()

        def report_layer_time(self, layer_name: str, ms: float) -> None:
            self.impl.report_layer_time(layer_name, ms)

    return Profiler()


@torch.no_grad()
def profile_layers(
    engine: TRTEngine, batches: Iterable[Tensor], warmup: int = 2
) -> dict[str, float]:
    """Mean milliseconds per batch of every engine layer.

    TensorRT synchronises after each layer while profiling, so the total is a
    little above the un-profiled latency; the shares are what matters.
    """
    batches = list(batches)
    if not batches:
        raise ValueError('No batches to profile')
    for x in batches[:warmup]:
        engine(x)
    profiler = _make_profiler()
    engine.context.profiler = profiler
    try:
        for x in batches:
            engine(x)
        torch.cuda.synchronize(engine.device)
    finally:
        # A context keeps its profiler; the engine makes a fresh one on its next run.
        engine.reset_context()
    return {name: ms / len(batches) for name, ms in profiler.impl.times.items()}


def _tensor_names(layer: dict, key: str) -> str:
    return ' '.join(
        str(t.get('Name', '')) if isinstance(t, dict) else str(t) for t in layer.get(key, [])
    )


def group_times(
    times: dict[str, float],
    layers: Sequence[dict] = (),
    groups: Sequence[tuple[str, str]] = DEFAULT_GROUPS,
) -> dict[str, float]:
    """Sum layer times per component (``'other'`` for unmatched layers).

    Args:
        times: Output of :func:`profile_layers`.
        layers: Output of :func:`layer_info`; lets layers with synthetic names
            be attributed through their tensor names, and fused attention
            layers through their type.
        groups: ``(component, regex)`` pairs, first match wins.
    """
    compiled = [(name, re.compile(pattern)) for name, pattern in groups]
    info = {str(layer.get('Name')): layer for layer in layers}
    out: dict[str, float] = {name: 0.0 for name, _ in groups}
    out['other'] = 0.0
    for layer_name, ms in times.items():
        layer = info.get(layer_name, {})
        if layer and _is_mha(layer):
            out[MHA_GROUP] = out.get(MHA_GROUP, 0.0) + ms
            continue
        texts = (
            layer_name + ' ' + _tensor_names(layer, 'Outputs'),
            _tensor_names(layer, 'Inputs'),
        )
        group = next(
            (name for text in texts for name, pattern in compiled if pattern.search(text)),
            'other',
        )
        out[group] += ms
    return {name: ms for name, ms in out.items() if ms > 0}


def format_report(
    times: dict[str, float], layers: Sequence[dict], batch_size: int, top: int = 15
) -> str:
    """Markdown report: fusion counts, time per component, slowest layers."""
    total = sum(times.values())
    summary = fusion_summary(layers)
    types = {str(layer.get('Name')): str(layer.get('LayerType', '')) for layer in layers}
    lines = [
        f'{summary["layers"]} layers, {summary["mha"]} fused attention (mha), '
        f'{summary["plugins"]} plugin layers; {total:.1f} ms / batch of {batch_size} '
        f'({batch_size / total * 1000:.1f} img/s while profiling)',
        '',
        '| component | ms / batch | share |',
        '|---|---|---|',
    ]
    for name, ms in sorted(group_times(times, layers).items(), key=lambda kv: -kv[1]):
        lines.append(f'| {name} | {ms:.2f} | {ms / total:.1%} |')
    lines += ['', '| slowest layers | type | ms / batch |', '|---|---|---|']
    for name, ms in sorted(times.items(), key=lambda kv: -kv[1])[:top]:
        short = name if len(name) <= 90 else name[:43] + ' ... ' + name[-42:]
        lines.append(f'| `{short}` | {types.get(name, "")} | {ms:.2f} |')
    return '\n'.join(lines)
