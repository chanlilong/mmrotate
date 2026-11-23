# Copyright (c) OpenMMLab. All rights reserved.
"""PNG tiles on the wire <-> numpy arrays."""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np


class BadImageError(ValueError):
    """A tile that is not base64 or not a decodable image (answered with HTTP 400)."""


def decode_png(data: str) -> np.ndarray:
    """Base64 PNG -> ``[H, W, 3]`` uint8 **BGR** (what mmdet's pipeline and preprocessor
    expect: ``LoadImageFromFile`` reads BGR, and ``bgr_to_rgb`` flips it for the network).

    The qgis server writes true RGB colours into the PNG, so ``cv2.imdecode`` returning BGR
    is exactly right. Grey and RGBA PNGs are converted to 3 channels.
    """
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as e:
        raise BadImageError(f'image is not valid base64: {e}') from e
    buf = np.frombuffer(raw, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR) if buf.size else None
    if img is None:
        raise BadImageError('image is not a decodable PNG')
    return img  # IMREAD_COLOR: always 3-channel 8-bit


def encode_png(bgr: np.ndarray) -> str:
    """``[H, W, 3]`` uint8 BGR -> base64 PNG (true RGB colours in the file)."""
    ok, buf = cv2.imencode('.png', bgr)
    if not ok:
        raise ValueError('PNG encoding failed')
    return base64.b64encode(buf.tobytes()).decode()
