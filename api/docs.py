# Copyright (c) OpenMMLab. All rights reserved.
"""Swagger UI and ReDoc served from ``api/static`` (no CDN: works airgapped)."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.openapi.docs import (
    get_redoc_html,
    get_swagger_ui_html,
    get_swagger_ui_oauth2_redirect_html,
)
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parent / 'static'


def install_offline_docs(app: FastAPI) -> None:
    """Mount ``/static`` and serve ``/docs``, ``/docs/oauth2-redirect`` and ``/redoc`` from it.

    The app must be created with ``docs_url=None, redoc_url=None``. Every URL honours the
    ASGI ``root_path``, so the pages also work behind a reverse proxy prefix.
    """
    app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')
    oauth_path = '/docs/oauth2-redirect'

    @app.get('/docs', include_in_schema=False)
    def swagger_ui(request: Request) -> HTMLResponse:
        root = request.scope.get('root_path', '').rstrip('/')
        return get_swagger_ui_html(
            openapi_url=f'{root}{app.openapi_url}',
            title=f'{app.title} - Swagger UI',
            swagger_js_url=f'{root}/static/swagger-ui/swagger-ui-bundle.js',
            swagger_css_url=f'{root}/static/swagger-ui/swagger-ui.css',
            swagger_favicon_url=f'{root}/static/swagger-ui/favicon-32x32.png',
            oauth2_redirect_url=f'{root}{oauth_path}',
            # no call-home to validator.swagger.io
            swagger_ui_parameters={'validatorUrl': None},
        )

    @app.get(oauth_path, include_in_schema=False)
    def swagger_ui_redirect() -> HTMLResponse:
        return get_swagger_ui_oauth2_redirect_html()

    @app.get('/redoc', include_in_schema=False)
    def redoc(request: Request) -> HTMLResponse:
        root = request.scope.get('root_path', '').rstrip('/')
        return get_redoc_html(
            openapi_url=f'{root}{app.openapi_url}',
            title=f'{app.title} - ReDoc',
            redoc_js_url=f'{root}/static/redoc/redoc.standalone.js',
            redoc_favicon_url=f'{root}/static/swagger-ui/favicon-32x32.png',
            with_google_fonts=False,
        )
