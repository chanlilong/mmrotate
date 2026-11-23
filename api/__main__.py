# Copyright (c) OpenMMLab. All rights reserved.
"""``python -m api --config cfg.yaml --weights w.pth --gsd 0.5`` (see ``--help``)."""

from __future__ import annotations

import logging

import uvicorn

from .app import create_app
from .settings import settings_from_args


def main() -> None:
    settings = settings_from_args()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )
    app = create_app(settings)
    log = logging.getLogger('api')
    base = f'http://<host>:{settings.port}'
    for svc in settings.services:
        log.info('register %s at %s/%s', svc, base, svc)
    log.info('Swagger UI at %s/docs (served locally, works offline)', base)
    # one worker: the model lives in this process and the GPU is serialised by a lock
    uvicorn.run(
        app, host=settings.host, port=settings.port, log_level=settings.log_level, workers=1
    )


if __name__ == '__main__':
    main()
