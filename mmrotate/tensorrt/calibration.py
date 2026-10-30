# Copyright (c) OpenMMLab. All rights reserved.
"""Network inputs drawn from a config's dataset, for calibration and checks."""

import copy
from collections.abc import Iterator

import torch
import torch.nn.functional as F
from mmengine.config import Config
from mmengine.runner import Runner
from torch import Tensor, nn

# Pipeline steps that need annotations; calibration only needs images.
_ANNOTATION_STEPS = ('LoadAnnotations', 'ConvertBoxType')


def build_input_loader(
    cfg: Config, split: str = 'val', shuffle: bool = True, num_workers: int = 4, seed: int = 0
) -> torch.utils.data.DataLoader:
    """Dataloader over ``cfg.<split>_dataloader``'s dataset with the test pipeline.

    The test pipeline (without annotation loading) is used for every split,
    so the train split gives un-augmented images at the deployed resolution.
    """
    key = f'{split}_dataloader'
    if key not in cfg:
        raise KeyError(f'Config has no {key}')
    dataset = copy.deepcopy(cfg[key].dataset)
    pipeline = copy.deepcopy(cfg.test_dataloader.dataset.pipeline)
    dataset.pipeline = [
        step for step in pipeline if not str(step['type']).endswith(_ANNOTATION_STEPS)
    ]
    loader_cfg = dict(
        batch_size=1,
        num_workers=num_workers,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=shuffle, seed=seed),
        dataset=dataset,
    )
    return Runner.build_dataloader(loader_cfg, seed=seed)


def pad_inputs(inputs: Tensor, hw: tuple[int, int]) -> Tensor:
    """Zero-pad ``(B, C, h, w)`` bottom/right to ``hw`` (as the preprocessor pads)."""
    h, w = inputs.shape[-2:]
    if h > hw[0] or w > hw[1]:
        raise ValueError(f'Input {(h, w)} is larger than the engine input {hw}')
    return F.pad(inputs, (0, hw[1] - w, 0, hw[0] - h))


@torch.no_grad()
def iter_batches(
    loader: torch.utils.data.DataLoader,
    data_preprocessor: nn.Module,
    input_shape: tuple[int, int, int, int],
    num_images: int,
) -> Iterator[tuple[Tensor, list]]:
    """Yield ``(inputs, data_samples)`` batches of the engine input shape.

    ``inputs`` are preprocessed (normalised, padded) exactly as for the torch
    model. The last batch may be smaller than the engine batch size.
    """
    batch_size, _, h, w = input_shape
    images: list[Tensor] = []
    samples: list = []
    seen = 0
    for data in loader:
        data = data_preprocessor(data, False)
        images.append(pad_inputs(data['inputs'], (h, w)))
        samples += data['data_samples']
        seen += data['inputs'].shape[0]
        if sum(x.shape[0] for x in images) >= batch_size or seen >= num_images:
            yield torch.cat(images), samples
            images, samples = [], []
        if seen >= num_images:
            return
    if images:
        yield torch.cat(images), samples
