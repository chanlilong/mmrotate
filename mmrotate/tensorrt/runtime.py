# Copyright (c) OpenMMLab. All rights reserved.
"""Run a serialized TensorRT engine on torch CUDA tensors."""

from pathlib import Path

import torch
from torch import Tensor


def _torch_dtype(trt_dtype: object) -> torch.dtype:
    import tensorrt as trt

    table = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.BF16: torch.bfloat16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT64: torch.int64,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    if trt_dtype not in table:
        raise TypeError(f'Unsupported engine I/O dtype {trt_dtype}')
    return table[trt_dtype]


class TRTEngine:
    """A deserialized engine with one execution context.

    Inputs and outputs are torch tensors on the engine's device, so no extra
    CUDA bindings are needed. The engine runs on its own CUDA stream, ordered
    after the caller's stream and before the caller's next operation
    (TensorRT adds blocking synchronisations when given the default stream).

    Args:
        engine_file: Serialized engine (``.engine`` / ``.plan``).
        device: CUDA device the engine is loaded on.
    """

    def __init__(self, engine_file: str | Path, device: torch.device | str = 'cuda') -> None:
        import tensorrt as trt

        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('TensorRT engines run on CUDA devices only')
        self.logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(self.logger, '')
        with torch.cuda.device(self.device):
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(engine_file).read_bytes())
            if self.engine is None:
                raise RuntimeError(
                    f'Could not deserialize {engine_file}: it may have been built with another '
                    'TensorRT version or for another GPU'
                )
            self.stream = torch.cuda.Stream(self.device)
        self._context = None
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        is_input = {n: self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT for n in names}
        self.input_names = [n for n in names if is_input[n]]
        self.output_names = [n for n in names if not is_input[n]]
        self.dtypes = {n: _torch_dtype(self.engine.get_tensor_dtype(n)) for n in names}

    @property
    def context(self) -> object:
        """The execution context, created on first use.

        Lazily, because TensorRT 11.4 corrupts the heap when a context of an
        engine with Python-callback (``jit``) plugins is destroyed without
        ever having run; inspecting an engine must not create one.
        """
        if self._context is None:
            with torch.cuda.device(self.device):
                self._context = self.engine.create_execution_context()
        return self._context

    def reset_context(self) -> None:
        """Drop the execution context (and a profiler attached to it)."""
        self._context = None

    def __del__(self) -> None:
        # Context before engine before runtime.
        for name in ('_context', 'engine', 'runtime'):
            self.__dict__.pop(name, None)

    def input_shape(self, name: str | None = None) -> tuple[int, ...]:
        """Static shape of an input (the first one by default)."""
        return tuple(self.engine.get_tensor_shape(name or self.input_names[0]))

    def __call__(self, *inputs: Tensor) -> list[Tensor]:
        """Run the engine; returns the outputs in ``output_names`` order."""
        if len(inputs) != len(self.input_names):
            raise ValueError(f'Expected {len(self.input_names)} inputs, got {len(inputs)}')
        with torch.cuda.device(self.device):
            # Converted inputs must outlive the enqueue; freeing them afterwards is
            # safe, as the caller's stream waits for the engine below before it can
            # reuse their memory.
            keep = []
            for name, x in zip(self.input_names, inputs, strict=True):
                x = x.to(self.device, self.dtypes[name]).contiguous()
                keep.append(x)
                if not self.context.set_input_shape(name, tuple(x.shape)):
                    raise ValueError(
                        f'Input {name!r} shape {tuple(x.shape)} does not fit the engine '
                        f'({self.input_shape(name)})'
                    )
                self.context.set_tensor_address(name, x.data_ptr())
            outputs = []
            for name in self.output_names:
                shape = tuple(self.context.get_tensor_shape(name))
                out = torch.empty(shape, dtype=self.dtypes[name], device=self.device)
                self.context.set_tensor_address(name, out.data_ptr())
                outputs.append(out)
            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError('TensorRT execution failed')
            caller.wait_stream(self.stream)
        del keep
        return outputs
