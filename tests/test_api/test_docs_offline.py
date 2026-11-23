# Copyright (c) OpenMMLab. All rights reserved.
"""Swagger UI / ReDoc work without internet: every asset is served from api/static."""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.model import LoadedModel
from api.settings import Settings

EXTERNAL = re.compile(r'(src|href)="(https?:)?//', re.I)


@pytest.fixture(scope='module')
def app(tiny_lm: LoadedModel) -> object:
    settings = Settings(config=tiny_lm.config_path, weights='', self_test=False)
    return create_app(settings, lm=tiny_lm)


@pytest.mark.parametrize('page', ['/docs', '/redoc'])
def test_pages_use_local_assets(app: object, page: str) -> None:
    client = TestClient(app)
    html = client.get(page).text
    assert not EXTERNAL.search(html), EXTERNAL.findall(html)
    for host in ('cdn.jsdelivr', 'unpkg', 'googleapis', 'fastapi.tiangolo'):
        assert host not in html
    assets = re.findall(r'(?:src|href)=["\'](/static/[^"\']+)', html)
    assert assets
    for url in assets:
        assert client.get(url).status_code == 200, url
    assert client.get('/openapi.json').status_code == 200


def test_swagger_parameters(app: object) -> None:
    html = TestClient(app).get('/docs').text
    assert '"validatorUrl": null' in html  # no call-home to validator.swagger.io
    assert TestClient(app).get('/docs/oauth2-redirect').status_code == 200


def test_root_path_prefix(app: object) -> None:
    """Behind a reverse proxy at /mmrotate every asset URL keeps the prefix."""
    html = TestClient(app, root_path='/mmrotate').get('/docs').text
    assert '/mmrotate/static/swagger-ui/swagger-ui-bundle.js' in html
    assert '/mmrotate/openapi.json' in html


def test_openapi_documents_both_services(app: object) -> None:
    spec = TestClient(app).get('/openapi.json').json()
    for path in ('/detector/health', '/detector/detect', '/similarity/health', '/similarity/embed'):
        assert path in spec['paths'], path
    schemas = spec['components']['schemas']
    for name in ('DetectRequest', 'DetectResponse', 'EmbedRequest', 'EmbedResponse'):
        assert name in schemas
    detect = spec['paths']['/detector/detect']['post']
    assert '200' in detect['responses'] and 'content' in detect['responses']['200']
