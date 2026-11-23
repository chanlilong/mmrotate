# Copyright (c) OpenMMLab. All rights reserved.
"""ONNX -> serialized TensorRT engine.

Networks are built strongly typed: the precision of every layer is the one of
the ONNX graph (fp32 / fp16 / bf16 tensors, Q/DQ nodes for int8 / fp8). This
is the only mode of TensorRT 11 (no FP16/INT8 builder flags or implicit
calibrators) and is supported by TensorRT 10.
"""

import logging
from contextlib import nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)

# --hardware-compat values -> TensorRT HardwareCompatibilityLevel names.
HARDWARE_COMPAT = {
    'none': 'NONE',
    'ampere+': 'AMPERE_PLUS',
    'same-cc': 'SAME_COMPUTE_CAPABILITY',
}
# Oldest compute capability an ``ampere+`` engine runs on; its plugin PTX targets it.
AMPERE_PLUS_ARCH = 80


def compat_covers(level: str | None, built_cc: str | None, running_cc: str) -> bool:
    """Whether an engine built for ``built_cc`` at ``level`` runs on ``running_cc``.

    Compute capabilities are ``'major.minor'`` strings (``'12.0'``).
    """
    if built_cc is None or built_cc == running_cc:
        return True
    if level == 'ampere+':
        return tuple(map(int, running_cc.split('.'))) >= (AMPERE_PLUS_ARCH // 10, 0)
    return False


def build_engine(
    onnx_file: str | Path,
    engine_file: str | Path,
    workspace_gb: float | None = None,
    opt_level: int = 3,
    timing_cache: str | Path | None = None,
    tf32: bool = True,
    verbose: bool = False,
    plugins: str | None = None,
    hardware_compat: str = 'none',
) -> None:
    """Parse ``onnx_file`` and serialize a static-shape engine to ``engine_file``.

    Args:
        onnx_file: ONNX model (external data next to it is picked up).
        engine_file: Output path.
        workspace_gb: Workspace memory pool limit; ``None`` leaves it to TensorRT.
        opt_level: Builder optimization level (0-5).
        timing_cache: Optional timing cache file, read if present and updated.
        tf32: Allow TF32 for fp32 convolutions / matmuls (TensorRT's default).
        verbose: Verbose TensorRT logging.
        plugins: Implementation of this package's Python plugins
            (:mod:`~mmrotate.tensorrt.plugins`) to build into the engine:
            ``'aot'`` (kernels embedded in the engine), ``'jit'`` (Python
            callbacks at run time) or ``None`` when the graph has no plugin
            nodes.
        hardware_compat: GPUs the engine may run on: ``'none'`` (the
            build GPU's compute capability), ``'ampere+'`` (every Ampere or
            newer GPU; plugin kernels are then compiled to sm_80 PTX) or
            ``'same-cc'`` (GPUs with the build GPU's compute capability).
    """
    import tensorrt as trt

    if plugins not in (None, 'aot', 'jit'):
        raise ValueError(f'Unknown plugin implementation {plugins!r}')
    if hardware_compat not in HARDWARE_COMPAT:
        raise ValueError(f'Unknown hardware compatibility {hardware_compat!r}')

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, '')
    builder = trt.Builder(trt_logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED'):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):  # TensorRT < 10
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if plugins is not None:
        from .plugins import register_plugins

        register_plugins()
        prefer = 'PREFER_AOT_PYTHON_PLUGINS' if plugins == 'aot' else 'PREFER_JIT_PYTHON_PLUGINS'
        flags |= 1 << int(getattr(trt.NetworkDefinitionCreationFlag, prefer))
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse {onnx_file}:\n{errors}')

    config = builder.create_builder_config()
    if workspace_gb is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = opt_level
    # Layer names and fusions for engine_report (no effect on the engine's speed).
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    config.hardware_compatibility_level = getattr(
        trt.HardwareCompatibilityLevel, HARDWARE_COMPAT[hardware_compat]
    )
    cache = None
    if timing_cache is not None:
        data = Path(timing_cache).read_bytes() if Path(timing_cache).exists() else b''
        cache = config.create_timing_cache(data)
        config.set_timing_cache(cache, ignore_mismatch=False)

    logger.info(
        'Building TensorRT engine from %s (hardware compatibility %s) ...',
        onnx_file,
        hardware_compat,
    )
    ptx_arch = nullcontext()
    if plugins == 'aot' and hardware_compat == 'ampere+':
        from .plugins.msda import plugin_ptx_arch

        ptx_arch = plugin_ptx_arch(AMPERE_PLUS_ARCH)
    with ptx_arch:
        serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT failed to build an engine from {onnx_file}')
    Path(engine_file).write_bytes(bytes(serialized))
    if cache is not None:
        Path(timing_cache).write_bytes(bytes(cache.serialize()))
