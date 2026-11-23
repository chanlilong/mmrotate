# Copyright (c) OpenMMLab. All rights reserved.
"""The detector service (``/detector/health``, ``/detector/detect``).

CPU: the contract on a random-init rotated_rtmdet_tiny (1024 input). GPU: real SODA-A
checkpoints from work_dirs (ViTDet SepBN head and ConvNeXt DCFL head).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import torch
from api_test_utils import (
    DCFL,
    VITDET,
    check_detector_health,
    needs_cuda,
    needs_run,
    random_tile,
)
from fastapi.testclient import TestClient
from mmcv.ops import box_iou_rotated

from api.app import create_app
from api.codec import encode_png
from api.detector import DetectorService
from api.model import LoadedModel
from api.settings import Settings
from mmrotate.structures.bbox import qbox2rbox


def _settings(lm: LoadedModel, **kwargs: object) -> Settings:
    return Settings(config=lm.config_path, weights=lm.weights_path, **kwargs)  # type: ignore[arg-type]


@pytest.fixture(scope='module')
def tiny_client(tiny_lm: LoadedModel) -> TestClient:
    app = create_app(_settings(tiny_lm, gsd_m=0.5, max_batch=2), lm=tiny_lm)
    return TestClient(app)


def test_health_contract(tiny_client: TestClient, tiny_lm: LoadedModel) -> None:
    h = tiny_client.get('/detector/health').json()
    check_detector_health(h)
    assert h['classes'] == tiny_lm.classes and len(h['classes']) == 15
    assert h['gsd_m'] == 0.5
    assert h['native_size'] == 1024 and h['input_size'] == [1024, 1024]
    index = tiny_client.get('/').json()
    assert [s['path'] for s in index['services']] == ['/detector', '/similarity']
    assert index['input_size_source'].startswith('test pipeline')


def test_detect_one_result_per_tile(tiny_client: TestClient) -> None:
    images = [encode_png(random_tile(1024, 1024)), encode_png(random_tile(300, 1024, 1))]
    r = tiny_client.post('/detector/detect', json={'images': images, 'score_thr': 0.0})
    assert r.status_code == 200, r.text
    results = r.json()['results']
    assert len(results) == 2
    for res in results:
        for inst in res['instances']:
            assert len(inst['quad']) == 4 and 0 <= inst['score'] <= 1


def test_detect_errors(tiny_client: TestClient) -> None:
    tile = encode_png(random_tile(64, 64))
    r = tiny_client.post('/detector/detect', json={'images': [tile] * 3})
    assert r.status_code == 400 and 'max_batch' in r.json()['detail']
    r = tiny_client.post('/detector/detect', json={'images': ['not base64!!']})
    assert r.status_code == 400
    r = tiny_client.post('/detector/detect', json={'images': ['aGVsbG8=']})  # 'hello'
    assert r.status_code == 400 and 'PNG' in r.json()['detail']
    big = encode_png(random_tile(64, 2049))
    r = tiny_client.post('/detector/detect', json={'images': [big]})
    assert r.status_code == 400 and 'max_side_px' in r.json()['detail']
    r = tiny_client.post('/detector/detect', json={'images': [tile], 'score_thr': 2})
    assert r.status_code == 422


def test_nms_override_is_restored(tiny_lm: LoadedModel) -> None:
    before = tiny_lm.nms_cfg()
    with tiny_lm.nms_override(0.77, True):
        assert tiny_lm.model.bbox_head.test_cfg.nms.iou_threshold == 0.77
        assert tiny_lm.model.bbox_head.test_cfg.nms.class_agnostic is True
    assert tiny_lm.nms_cfg() == before


def test_single_service_also_at_root(tiny_lm: LoadedModel) -> None:
    app = create_app(_settings(tiny_lm, services=('detector',), self_test=False), lm=tiny_lm)
    client = TestClient(app)
    assert client.get('/health').json()['model_name'] == tiny_lm.model_name
    assert client.get('/detector/health').status_code == 200
    assert client.get('/similarity/health').status_code == 404


def test_large_tile_runs_as_windows(tiny_lm: LoadedModel) -> None:
    svc = DetectorService(tiny_lm, tile_size=1600, max_batch=4)
    assert svc.health().native_size == 1600 and svc.max_side_px == 3200
    out = svc.detect([random_tile(1600, 1500)], score_thr=0.0)
    assert len(out) == 1


# ----------------------------------------------------------------- real models (GPU)
def _rboxes(quads: np.ndarray) -> torch.Tensor:
    return qbox2rbox(torch.as_tensor(quads, dtype=torch.float32))


def _recall(instances: list, gt_quads: np.ndarray, iou: float = 0.5) -> float:
    if not instances:
        return 0.0
    det = _rboxes(np.array([np.ravel(i.quad) for i in instances]))
    ious = box_iou_rotated(_rboxes(gt_quads), det)
    return float((ious.max(1).values >= iou).float().mean())


@needs_cuda
@needs_run(VITDET)
def test_vitdet_detect_matches_model_and_finds_objects(
    load_run: Callable[..., LoadedModel], soda_patches: list
) -> None:
    lm = load_run(VITDET)
    assert lm.input_size == (800, 800) and len(lm.classes) == 9
    svc = DetectorService(lm, gsd_m=0.5)
    img, gt, _ = soda_patches[0]
    got = svc.detect([img], score_thr=0.3)[0]
    # the same tile straight through the model
    with torch.inference_mode():
        ref = lm.model.test_step(lm.collate([img]))[0].pred_instances
    ref = ref[ref.scores >= 0.3]
    assert len(got) == len(ref)
    recalls = [_recall(svc.detect([im], 0.3)[0], q) for im, q, _ in soda_patches]
    assert np.mean(recalls) > 0.4, recalls


@needs_cuda
@needs_run(VITDET)
def test_vitdet_http_roundtrip(load_run: Callable[..., LoadedModel], soda_patches: list) -> None:
    lm = load_run(VITDET)
    client = TestClient(create_app(_settings(lm, gsd_m=0.5, services=('detector',)), lm=lm))
    check_detector_health(client.get('/health').json())
    img, gt, names = soda_patches[0]
    pad = np.zeros((800, 800, 3), np.uint8)  # the client pads edge tiles bottom-right
    pad[:600, :700] = img[:600, :700]
    r = client.post('/detect', json={'images': [encode_png(img), encode_png(pad)]})
    assert r.status_code == 200
    full, cut = r.json()['results']
    assert full['instances'] and {i['label'] for i in full['instances']} <= set(lm.classes)
    for inst in cut['instances']:
        cx, cy = np.mean(inst['quad'], 0)
        assert cx < 720 and cy < 620  # nothing invented in the black padding


@needs_cuda
@needs_run(VITDET)
def test_vitdet_nms_overrides(load_run: Callable[..., LoadedModel], soda_patches: list) -> None:
    lm = load_run(VITDET)
    svc = DetectorService(lm)
    img = soda_patches[1][0]
    strict = svc.detect([img], 0.05, nms_iou=0.05)[0]
    loose = svc.detect([img], 0.05, nms_iou=0.9)[0]
    assert len(loose) >= len(strict)
    agnostic = svc.detect([img], 0.05, class_agnostic=True)[0]
    per_class = svc.detect([img], 0.05, class_agnostic=False)[0]
    assert len(agnostic) <= len(per_class)
    assert lm.config_nms_iou() == pytest.approx(0.1)


@needs_cuda
@needs_run(VITDET)
def test_vitdet_windows_merge_seams(
    load_run: Callable[..., LoadedModel], soda_patches: list
) -> None:
    """A 1600 px tile (2x2 SODA patches) as 800 px windows: no duplicate across seams."""
    lm = load_run(VITDET)
    mosaic = np.zeros((1600, 1600, 3), np.uint8)
    for k, (y, x) in enumerate([(0, 0), (0, 800), (800, 0), (800, 800)]):
        mosaic[y : y + 800, x : x + 800] = soda_patches[k % len(soda_patches)][0]
    big = DetectorService(lm, tile_size=1600)
    small = DetectorService(lm)
    merged = big.detect([mosaic], 0.3)[0]
    separate = sum(len(small.detect([p[0]], 0.3)[0]) for p in soda_patches[:4])
    assert abs(len(merged) - separate) <= 0.25 * max(separate, 1) + 5
    boxes = _rboxes(np.array([np.ravel(i.quad) for i in merged]))
    labels = np.array([i.label for i in merged])
    ious = box_iou_rotated(boxes, boxes)
    ious.fill_diagonal_(0)
    same = torch.as_tensor(labels[:, None] == labels[None, :])
    assert float((ious * same).max()) <= lm.config_nms_iou() + 1e-3


@needs_cuda
@needs_run(VITDET)
def test_vitdet_other_input_size(load_run: Callable[..., LoadedModel], soda_patches: list) -> None:
    lm = load_run(VITDET, input_size=(1024, 1024))
    svc = DetectorService(lm)
    assert svc.health().native_size == 1024
    tile = np.zeros((1024, 1024, 3), np.uint8)
    tile[:800, :800] = soda_patches[0][0]
    assert svc.detect([tile], 0.3)[0]


@needs_cuda
@needs_run(DCFL)
def test_dcfl_detect(load_run: Callable[..., LoadedModel], soda_patches: list) -> None:
    lm = load_run(DCFL)
    svc = DetectorService(lm)
    recalls = [_recall(svc.detect([im], 0.3)[0], q) for im, q, _ in soda_patches]
    assert np.mean(recalls) > 0.4, recalls
