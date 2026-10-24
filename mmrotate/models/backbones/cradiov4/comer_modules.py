# Copyright (c) OpenMMLab. All rights reserved.
"""ViT-CoMer building blocks, ported to mmcv 2.x / mmengine.

Ported from https://github.com/Traffic-X/ViT-CoMer (Apache-2.0),
``detection/mmdet_custom/models/backbones/comer_modules.py``. The math and the
parameter names are unchanged; the deformable attention comes from
:mod:`.ms_deform_attn` and the normalisation layers are configurable.
"""

from collections.abc import Callable, Sequence
from functools import partial

import torch
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks import DropPath
from torch import Tensor, nn

from .ms_deform_attn import MSDeformAttn

NormLayer = Callable[[int], nn.Module]
DeformInputs = tuple[Tensor, Tensor, Tensor]
default_norm_layer: NormLayer = partial(nn.LayerNorm, eps=1e-6)


def get_reference_points(spatial_shapes: Sequence[tuple[int, int]], device: torch.device) -> Tensor:
    reference_points_list = []
    for H_, W_ in spatial_shapes:
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            indexing='ij',
        )
        ref_y = ref_y.reshape(-1)[None] / H_
        ref_x = ref_x.reshape(-1)[None] / W_
        reference_points_list.append(torch.stack((ref_x, ref_y), -1))
    reference_points = torch.cat(reference_points_list, 1)
    return reference_points[:, :, None]


def _shapes_and_starts(
    shapes: list[tuple[int, int]], device: torch.device
) -> tuple[Tensor, Tensor]:
    spatial_shapes = torch.as_tensor(shapes, dtype=torch.long, device=device)
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    return spatial_shapes, level_start_index


def deform_inputs(x: Tensor) -> tuple[DeformInputs, DeformInputs]:
    """Deformable-attention inputs for an image ``x`` of shape (B, C, h, w)."""
    _, _, h, w = x.shape
    pyramid = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
    shapes, starts = _shapes_and_starts(pyramid, x.device)
    deform_inputs1 = (get_reference_points([(h // 16, w // 16)], x.device), shapes, starts)
    shapes, starts = _shapes_and_starts([(h // 16, w // 16)], x.device)
    deform_inputs2 = (get_reference_points(pyramid, x.device), shapes, starts)
    return deform_inputs1, deform_inputs2


def deform_inputs_only_one(x: Tensor, h: int, w: int) -> DeformInputs:
    pyramid = [(h // 8, w // 8), (h // 16, w // 16), (h // 32, w // 32)]
    shapes, starts = _shapes_and_starts(pyramid, x.device)
    return get_reference_points(pyramid, x.device), shapes, starts


def _split_levels(x: Tensor, H: int, W: int) -> tuple[Tensor, Tensor, Tensor]:
    """Split concatenated stride 8/16/32 tokens into NCHW maps."""
    B, N, C = x.shape
    n = N // 21
    x1 = x[:, 0 : 16 * n, :].transpose(1, 2).reshape(B, C, H * 2, W * 2)
    x2 = x[:, 16 * n : 20 * n, :].transpose(1, 2).reshape(B, C, H, W)
    x3 = x[:, 20 * n :, :].transpose(1, 2).reshape(B, C, H // 2, W // 2)
    return x1, x2, x3


class DWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        xs = _split_levels(x, H, W)
        return torch.cat([self.dwconv(xi).flatten(2).transpose(1, 2) for xi in xs], dim=1)


class MultiDWConv(nn.Module):
    def __init__(self, dim: int = 768) -> None:
        super().__init__()
        dim1 = dim
        dim = dim // 2
        self.dwconv1 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv3 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv4 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.dwconv5 = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.dwconv6 = nn.Conv2d(dim, dim, 5, 1, 2, bias=True, groups=dim)
        self.act1 = nn.GELU()
        self.bn1 = nn.BatchNorm2d(dim1)
        self.act2 = nn.GELU()
        self.bn2 = nn.BatchNorm2d(dim1)
        self.act3 = nn.GELU()
        self.bn3 = nn.BatchNorm2d(dim1)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        C = x.shape[-1]
        x1, x2, x3 = _split_levels(x, H, W)
        convs = (
            (self.dwconv1, self.dwconv2, self.bn1, self.act1),
            (self.dwconv3, self.dwconv4, self.bn2, self.act2),
            (self.dwconv5, self.dwconv6, self.bn3, self.act3),
        )
        outs = []
        for xi, (conv_a, conv_b, bn, act) in zip((x1, x2, x3), convs, strict=True):
            xi = torch.cat([conv_a(xi[:, : C // 2]), conv_b(xi[:, C // 2 :])], dim=1)
            outs.append(act(bn(xi)).flatten(2).transpose(1, 2))
        return torch.cat(outs, dim=1)


class ConvFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor, H: int, W: int) -> Tensor:
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return self.drop(x)


class MRFP(ConvFFN):
    """Multi-receptive-field feature pyramid (ConvFFN with MultiDWConv)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int | None = None,
        out_features: int | None = None,
        act_layer: type[nn.Module] = nn.GELU,
        drop: float = 0.0,
    ) -> None:
        super().__init__(in_features, hidden_features, out_features, act_layer, drop)
        self.dwconv = MultiDWConv(hidden_features or in_features)


class MultiscaleExtractor(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: NormLayer = default_norm_layer,
        with_cp: bool = False,
        msda_backend: str = 'auto',
    ) -> None:
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
            backend=msda_backend,
        )
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        feat: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        def _inner_forward(query: Tensor, feat: Tensor) -> Tensor:
            attn = self.attn(
                self.query_norm(query),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            query = query + attn
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return query

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class CTI_toC(nn.Module):  # noqa: N801 - keep upstream name
    """CNN-Transformer interaction: ViT tokens -> CNN pyramid."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: NormLayer = default_norm_layer,
        with_cp: bool = False,
        cnn_feature_interaction: bool = True,
        msda_backend: str = 'auto',
    ) -> None:
        super().__init__()
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.with_cffn = with_cffn
        self.with_cp = with_cp
        self.cnn_feature_interaction = cnn_feature_interaction
        if cnn_feature_interaction:
            self.cfinter = MultiscaleExtractor(
                dim=dim,
                n_levels=3,
                num_heads=num_heads,
                n_points=n_points,
                norm_layer=norm_layer,
                deform_ratio=deform_ratio,
                with_cffn=with_cffn,
                cffn_ratio=cffn_ratio,
                drop=drop,
                drop_path=drop_path,
                with_cp=with_cp,
                msda_backend=msda_backend,
            )

    def _add_vit(self, query: Tensor, feat: Tensor) -> Tensor:
        n = query.shape[1] // 21
        x1 = query[:, 0 : 16 * n, :]
        x2 = query[:, 16 * n : 20 * n, :] + feat
        x3 = query[:, 20 * n :, :]
        return torch.cat([x1, x2, x3], dim=1)

    def _interact(self, query: Tensor, H: int, W: int) -> Tensor:
        if not self.cnn_feature_interaction:
            return query
        ref, shapes, starts = deform_inputs_only_one(query, H * 16, W * 16)
        return self.cfinter(
            query=self.query_norm(query),
            reference_points=ref,
            feat=self.feat_norm(query),
            spatial_shapes=shapes,
            level_start_index=starts,
            H=H,
            W=W,
        )

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        feat: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        def _inner_forward(query: Tensor, feat: Tensor) -> Tensor:
            return self._interact(self._add_vit(query, feat), H, W)

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class Extractor_CTI(CTI_toC):  # noqa: N801 - keep upstream name
    """``CTI_toC`` with an extra ConvFFN after injecting the ViT tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        drop: float = 0.0,
        drop_path: float = 0.0,
        norm_layer: NormLayer = default_norm_layer,
        with_cp: bool = False,
        cnn_feature_interaction: bool = True,
        msda_backend: str = 'auto',
    ) -> None:
        super().__init__(
            dim,
            num_heads,
            n_points,
            n_levels,
            deform_ratio,
            with_cffn,
            cffn_ratio,
            drop,
            drop_path,
            norm_layer,
            with_cp,
            cnn_feature_interaction,
            msda_backend,
        )
        if with_cffn:
            self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
            self.ffn_norm = norm_layer(dim)
            self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        feat: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        def _inner_forward(query: Tensor, feat: Tensor) -> Tensor:
            query = self._add_vit(query, feat)
            if self.with_cffn:
                query = query + self.drop_path(self.ffn(self.ffn_norm(query), H, W))
            return self._interact(query, H, W)

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class CTI_toV(nn.Module):  # noqa: N801 - keep upstream name
    """CNN-Transformer interaction: CNN pyramid -> ViT tokens."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        n_levels: int = 1,
        deform_ratio: float = 1.0,
        norm_layer: NormLayer = default_norm_layer,
        init_values: float = 0.0,
        with_cp: bool = False,
        drop: float = 0.0,
        drop_path: float = 0.0,
        cffn_ratio: float = 0.25,
        msda_backend: str = 'auto',
    ) -> None:
        super().__init__()
        self.with_cp = with_cp
        self.query_norm = norm_layer(dim)
        self.feat_norm = norm_layer(dim)
        self.attn = MSDeformAttn(
            d_model=dim,
            n_levels=n_levels,
            n_heads=num_heads,
            n_points=n_points,
            ratio=deform_ratio,
            backend=msda_backend,
        )
        self.gamma = nn.Parameter(init_values * torch.ones(dim), requires_grad=True)
        self.ffn = ConvFFN(in_features=dim, hidden_features=int(dim * cffn_ratio), drop=drop)
        self.ffn_norm = norm_layer(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(
        self,
        query: Tensor,
        reference_points: Tensor,
        feat: Tensor,
        spatial_shapes: Tensor,
        level_start_index: Tensor,
        H: int,
        W: int,
    ) -> Tensor:
        def _inner_forward(query: Tensor, feat: Tensor) -> Tensor:
            B, _, C = feat.shape
            # Upstream uses the CNN tokens themselves as queries.
            c1 = self.attn(
                self.query_norm(feat),
                reference_points,
                self.feat_norm(feat),
                spatial_shapes,
                level_start_index,
                None,
            )
            c1 = c1 + self.drop_path(self.ffn(self.ffn_norm(c1), H, W))
            s1 = c1[:, : H * W * 4, :]
            s2 = c1[:, H * W * 4 : H * W * 5, :]
            s3 = c1[:, H * W * 5 :, :]
            s1 = s1.permute(0, 2, 1).reshape(B, C, H * 2, W * 2)
            s1 = F.interpolate(s1, scale_factor=0.5, mode='bilinear', align_corners=False)
            s3 = s3.permute(0, 2, 1).reshape(B, C, H // 2, W // 2)
            s3 = F.interpolate(s3, scale_factor=2, mode='bilinear', align_corners=False)
            s1 = s1.flatten(2).permute(0, 2, 1)
            s3 = s3.flatten(2).permute(0, 2, 1)
            return query + self.gamma * (s1 + s2 + s3)

        if self.with_cp and query.requires_grad:
            return cp.checkpoint(_inner_forward, query, feat, use_reentrant=False)
        return _inner_forward(query, feat)


class CTIBlock(nn.Module):
    """One CoMer interaction stage around a chunk of ViT blocks."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 6,
        n_points: int = 4,
        norm_layer: NormLayer = default_norm_layer,
        drop: float = 0.0,
        drop_path: float = 0.0,
        with_cffn: bool = True,
        cffn_ratio: float = 0.25,
        init_values: float = 0.0,
        deform_ratio: float = 1.0,
        extra_CTI: bool = False,
        with_cp: bool = False,
        use_CTI_toV: bool = True,
        use_CTI_toC: bool = True,
        cnn_feature_interaction: bool = False,
        extra_num: int = 4,
        msda_backend: str = 'auto',
    ) -> None:
        super().__init__()
        if use_CTI_toV:
            self.cti_tov = CTI_toV(
                dim=dim,
                n_levels=3,
                num_heads=num_heads,
                init_values=init_values,
                n_points=n_points,
                norm_layer=norm_layer,
                deform_ratio=deform_ratio,
                with_cp=with_cp,
                drop=drop,
                drop_path=drop_path,
                cffn_ratio=cffn_ratio,
                msda_backend=msda_backend,
            )
        cti_kwargs = dict(
            dim=dim,
            n_levels=1,
            num_heads=num_heads,
            n_points=n_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            with_cffn=with_cffn,
            cffn_ratio=cffn_ratio,
            drop=drop,
            drop_path=drop_path,
            with_cp=with_cp,
            cnn_feature_interaction=cnn_feature_interaction,
            msda_backend=msda_backend,
        )
        if use_CTI_toC:
            self.cti_toc = CTI_toC(**cti_kwargs)
        self.extra_CTIs: nn.Sequential | None = (
            nn.Sequential(*[Extractor_CTI(**cti_kwargs) for _ in range(extra_num)])
            if extra_CTI
            else None
        )
        self.use_CTI_toV = use_CTI_toV
        self.use_CTI_toC = use_CTI_toC
        if use_CTI_toV:
            self.mrfp = MRFP(dim, hidden_features=int(dim * 0.5))

    def forward(
        self,
        x: Tensor,
        c: Tensor,
        run_blocks: Callable[[Tensor], Tensor],
        deform_inputs1: DeformInputs,
        deform_inputs2: DeformInputs,
        H: int,
        W: int,
    ) -> tuple[Tensor, Tensor]:
        """``run_blocks`` maps spatial ViT tokens through this stage's blocks."""
        if self.use_CTI_toV:
            ref, shapes, starts = deform_inputs_only_one(x, H * 16, W * 16)
            c = self.mrfp(c, H, W)
            c1 = c[:, : H * W * 4, :]
            c2 = c[:, H * W * 4 : H * W * 5, :]
            c3 = c[:, H * W * 5 :, :]
            c = torch.cat([c1, c2 + x, c3], dim=1)
            x = self.cti_tov(
                query=x,
                reference_points=ref,
                feat=c,
                spatial_shapes=shapes,
                level_start_index=starts,
                H=H,
                W=W,
            )
        x = run_blocks(x)
        ref, shapes, starts = deform_inputs2
        if self.use_CTI_toC:
            c = self.cti_toc(
                query=c,
                reference_points=ref,
                feat=x,
                spatial_shapes=shapes,
                level_start_index=starts,
                H=H,
                W=W,
            )
        if self.extra_CTIs is not None:
            for cti in self.extra_CTIs:
                c = cti(
                    query=c,
                    reference_points=ref,
                    feat=x,
                    spatial_shapes=shapes,
                    level_start_index=starts,
                    H=H,
                    W=W,
                )
        return x, c


class CNN(nn.Module):
    """Spatial prior module: a small CNN producing a stride 4-32 pyramid.

    ``with_stride4=False`` drops the stride 4 projection (``fc1``) when that
    level is not consumed, so no parameter is left without gradients.
    """

    def __init__(
        self,
        inplanes: int = 64,
        embed_dim: int = 384,
        norm_cfg: dict | None = None,
        with_stride4: bool = True,
    ) -> None:
        super().__init__()
        norm_cfg = norm_cfg or dict(type='SyncBN', requires_grad=True)

        def conv_bn_relu(cin: int, cout: int, stride: int) -> list[nn.Module]:
            return [
                nn.Conv2d(cin, cout, kernel_size=3, stride=stride, padding=1, bias=False),
                build_norm_layer(norm_cfg, cout)[1],
                nn.ReLU(inplace=True),
            ]

        self.stem = nn.Sequential(
            *conv_bn_relu(3, inplanes, 2),
            *conv_bn_relu(inplanes, inplanes, 1),
            *conv_bn_relu(inplanes, inplanes, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.conv2 = nn.Sequential(*conv_bn_relu(inplanes, 2 * inplanes, 2))
        self.conv3 = nn.Sequential(*conv_bn_relu(2 * inplanes, 4 * inplanes, 2))
        self.conv4 = nn.Sequential(*conv_bn_relu(4 * inplanes, 4 * inplanes, 2))
        self.fc1 = (
            nn.Conv2d(inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
            if with_stride4
            else None
        )
        self.fc2 = nn.Conv2d(2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc3 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.fc4 = nn.Conv2d(4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x: Tensor) -> tuple[Tensor | None, Tensor, Tensor, Tensor]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c1 = self.fc1(c1) if self.fc1 is not None else None
        c2 = self.fc2(c2).flatten(2).transpose(1, 2)  # 8s
        c3 = self.fc3(c3).flatten(2).transpose(1, 2)  # 16s
        c4 = self.fc4(c4).flatten(2).transpose(1, 2)  # 32s
        return c1, c2, c3, c4
