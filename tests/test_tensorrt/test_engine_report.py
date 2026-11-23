# Copyright (c) OpenMMLab. All rights reserved.
"""Engine report helpers (no TensorRT needed: layer descriptions are plain dicts)."""

from mmrotate.tensorrt import engine_report as er

LAYERS = [
    dict(Name='_gemm_mha_v2_myl3_4', LayerType='kgen', TacticName='_gemm_mha_v2_0xabc'),
    dict(
        Name='/backbone/blocks.0/attn/qkv/MatMul_myl3_2',
        LayerType='gemm',
        Outputs=[dict(Name='/backbone/blocks.0/attn/qkv/MatMul_output_0')],
    ),
    dict(
        Name='__myl_AddMul_myl3_9',
        LayerType='kgen',
        Outputs=[dict(Name='/backbone/blocks.0/mlp/fc1/Add_output_0')],
    ),
    dict(Name='/backbone/interactions.0/cti_toc/cfinter/attn/MSDAFused', LayerType='PluginV3'),
    dict(
        Name='__myl_Cast_myl5_0',
        LayerType='kgen',
        Outputs=[dict(Name='__mye12')],
        Inputs=[dict(Name='/neck/upsample/Resize_output_0')],
    ),
    dict(Name='/rtm_cls.0/Conv', LayerType='CaskConvolution'),
    dict(Name='entry^bb^signal^1_myl0_0', LayerType='signal'),
]
TIMES = {layer['Name']: float(i + 1) for i, layer in enumerate(LAYERS)}


def test_fusion_summary_counts_attention_and_plugin_layers() -> None:
    assert er.fusion_summary(LAYERS) == dict(layers=7, mha=1, plugins=1)
    assert er.fusion_summary([dict(Name='only a name')]) == dict(layers=1, mha=0, plugins=0)


def test_group_times_uses_names_then_tensor_names() -> None:
    groups = er.group_times(TIMES, LAYERS)
    assert groups == {
        'vit attention': 1.0 + 2.0,  # the mha layer (by type) and the qkv projection
        'vit mlp': 3.0,  # synthetic name, attributed through its output tensor
        'msda': 4.0,
        'neck': 5.0,  # synthetic name and output, attributed through its input tensor
        'head': 6.0,
        'other': 7.0,
    }
    assert sum(groups.values()) == sum(TIMES.values())
    # Without layer descriptions only the names can be matched.
    by_name = er.group_times(TIMES)
    assert by_name['other'] == 1.0 + 3.0 + 5.0 + 7.0


def test_format_report_lists_components_and_slowest_layers() -> None:
    report = er.format_report(TIMES, LAYERS, batch_size=8, top=2)
    assert '1 fused attention (mha), 1 plugin layers' in report
    assert '| vit attention | 3.00 | 10.7% |' in report
    lines = report.splitlines()
    slowest = lines[lines.index('| slowest layers | type | ms / batch |') + 2 :]
    assert len(slowest) == 2 and 'entry^bb^signal' in slowest[0] and '/rtm_cls.0/Conv' in slowest[1]
