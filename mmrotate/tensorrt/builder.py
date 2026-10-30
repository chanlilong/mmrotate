# Copyright (c) OpenMMLab. All rights reserved.
"""ONNX -> serialized TensorRT engine.

Networks are built strongly typed: the precision of every layer is the one of
the ONNX graph (fp32 / fp16 / bf16 tensors, Q/DQ nodes for int8 / fp8). This
is the only mode of TensorRT 11 (no FP16/INT8 builder flags or implicit
calibrators) and is supported by TensorRT 10.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def build_engine(
    onnx_file: str | Path,
    engine_file: str | Path,
    workspace_gb: float | None = None,
    opt_level: int = 3,
    timing_cache: str | Path | None = None,
    tf32: bool = True,
    verbose: bool = False,
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
    """
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(trt_logger, '')
    builder = trt.Builder(trt_logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, 'STRONGLY_TYPED'):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    if hasattr(trt.NetworkDefinitionCreationFlag, 'EXPLICIT_BATCH'):  # TensorRT < 10
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse_from_file(str(onnx_file)):
        errors = '\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f'Failed to parse {onnx_file}:\n{errors}')

    config = builder.create_builder_config()
    if workspace_gb is not None:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.builder_optimization_level = opt_level
    if not tf32:
        config.clear_flag(trt.BuilderFlag.TF32)
    cache = None
    if timing_cache is not None:
        data = Path(timing_cache).read_bytes() if Path(timing_cache).exists() else b''
        cache = config.create_timing_cache(data)
        config.set_timing_cache(cache, ignore_mismatch=False)

    logger.info('Building TensorRT engine from %s ...', onnx_file)
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f'TensorRT failed to build an engine from {onnx_file}')
    Path(engine_file).write_bytes(bytes(serialized))
    if cache is not None:
        Path(timing_cache).write_bytes(bytes(cache.serialize()))
