# Copyright (c) OpenMMLab. All rights reserved.
"""The FastAPI app: one model, a detector and/or a similarity service on it."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import detector, similarity
from .docs import install_offline_docs
from .features import FeatureSpec, default_spec
from .model import LoadedModel, load
from .profile import load_profile
from .settings import Settings

log = logging.getLogger('api')

DESCRIPTION = """
Serves an mmrotate single-stage detector over the qgis_annotations REST contracts.

* **Detector** (`/detector/health`, `/detector/detect`): register `http://<host>:<port>/detector`
  in the admin under *Setup → AI services* with kind **fixed classes**.
* **Similarity** (`/similarity/health`, `/similarity/embed`): register
  `http://<host>:<port>/similarity` with kind **Similarity features**.

When only one service is enabled it is also served at the root (`/health`, `/detect` or `/embed`).
`GET /` lists what this instance serves.
"""


class ServiceInfo(BaseModel):
    kind: str
    path: str = Field(description='append to http://<host>:<port> to register it')
    health: dict


class ServiceIndex(BaseModel):
    model_name: str
    config: str
    weights: str
    classes: list[str]
    input_size: list[int]
    input_size_source: str
    services: list[ServiceInfo]


def build_services(
    settings: Settings, lm: LoadedModel
) -> tuple[detector.DetectorService | None, similarity.SimilarityService | None]:
    det = sim = None
    if 'detector' in settings.services:
        det = detector.DetectorService(
            lm,
            gsd_m=settings.gsd_m,
            tile_size=settings.detector_tile_size,
            max_batch=settings.max_batch,
            max_side_px=settings.max_side_px,
            patch_overlap=settings.patch_overlap,
            model_version=settings.model_version,
        )
    if 'similarity' in settings.services:
        overrides = settings.sim.as_dict()
        projector, meta = None, {}
        if settings.similarity_profile:
            spec, projector, meta = load_profile(settings.similarity_profile)
            log.info('similarity profile %s: %s', settings.similarity_profile, spec.tag())
        elif overrides:
            spec = FeatureSpec()  # --sim-* flags build on the base spec (neck, mean, all levels)
        else:
            spec = default_spec(lm.model)
            log.info('similarity: no profile, serving the default %s', spec.tag())
        spec = spec.with_overrides(overrides)
        sim = similarity.SimilarityService(
            lm,
            spec,
            projector,
            gsd_m=settings.gsd_m,
            imagery_gsd_m=settings.imagery_gsd_m,
            tile_size=settings.similarity_tile_size,
            patch_overlap=settings.patch_overlap,
            max_batch=settings.max_batch,
            model_version=settings.model_version if det is None else None,
            feature_space=settings.feature_space,
            profile_meta=meta,
        )
    return det, sim


def create_app(settings: Settings, lm: LoadedModel | None = None) -> FastAPI:
    """Load the model (unless given), build the services, self-test them, return the app.

    Everything heavy happens here, before uvicorn starts listening: a config and checkpoint
    that do not fit fail at start-up, not on an annotator's click.
    """
    if lm is None:
        lm = load(
            settings.config,
            settings.weights,
            device=settings.device,
            precision=settings.precision,
            model_name=settings.model_name,
            input_size=settings.input_size,
            resize_mode=settings.resize_mode,
            pad_mode=settings.pad_mode,
        )
    det, sim = build_services(settings, lm)
    if settings.self_test:
        if det is not None:
            det.self_test()
        if sim is not None:
            sim.self_test()

    app = FastAPI(
        title=f'mmrotate API: {lm.model_name}',
        description=DESCRIPTION,
        version='1.0',
        docs_url=None,
        redoc_url=None,
    )
    install_offline_docs(app)
    app.state.model = lm
    app.state.detector = det
    app.state.similarity = sim

    single = len([s for s in (det, sim) if s is not None]) == 1
    if det is not None:
        router = detector.make_router(lambda: det)
        app.include_router(router, prefix='/detector')
        if single:
            app.include_router(router)
    if sim is not None:
        router = similarity.make_router(lambda: sim)
        app.include_router(router, prefix='/similarity')
        if single:
            app.include_router(router)

    @app.get('/', response_model=ServiceIndex, tags=['index'], summary='What this instance serves')
    def index() -> ServiceIndex:
        services = []
        if det is not None:
            services.append(
                ServiceInfo(
                    kind='detector (fixed classes)',
                    path='/detector',
                    health=det.health().model_dump(exclude_none=True),
                )
            )
        if sim is not None:
            services.append(
                ServiceInfo(
                    kind='similarity features',
                    path='/similarity',
                    health=sim.health().model_dump(exclude_none=True),
                )
            )
        return ServiceIndex(
            model_name=lm.model_name,
            config=lm.config_path,
            weights=lm.weights_path,
            classes=lm.classes,
            input_size=list(lm.input_size),
            input_size_source=lm.input_size_source,
            services=services,
        )

    return app
