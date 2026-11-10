from __future__ import annotations

from typing import Optional

import torch
from torch import nn


class V1ConvNextCore(nn.Module):
    """
    Effective released macaque V1 core.

    Historical nnvision built the full HF classifier and then inclusively
    replaced `convnextv2.encoder.stages.1.layers.0` and everything after it
    with Identity. The released checkpoint confirms that only embeddings,
    stage 0, and stage-1 downsampling have learned backbone parameters.

    This clean implementation instantiates only the effective retained path.
    """

    def __init__(
        self,
        model_id: str = "facebook/convnextv2-atto-1k-224",
        revision: Optional[str] = None,
        batchnorm_momentum: float = 0.9,
        stage0_blocks: Optional[int] = None,
        stage1_blocks: int = 0,
        include_stage1_downsampling: bool = True,
        stage2_blocks: int = 0,
        stage3_blocks: int = 0,
        final_norm: str = "batchnorm",
        final_nonlinearity: str = "gelu",
        input_hw=(93, 93),
    ) -> None:
        super().__init__()
        try:
            from transformers import ConvNextV2ForImageClassification
        except ImportError as exc:
            raise ImportError(
                "transformers is required to construct V1ConvNextCore"
            ) from exc

        kwargs = {}
        if revision:
            kwargs["revision"] = revision

        full = ConvNextV2ForImageClassification.from_pretrained(
            model_id,
            **kwargs
        )
        backbone = full.convnextv2

        self.embeddings = backbone.embeddings
        self.stage0 = backbone.encoder.stages[0]
        available0 = len(self.stage0.layers)
        n0 = available0 if stage0_blocks is None else int(stage0_blocks)
        n1 = int(stage1_blocks)
        n2, n3 = int(stage2_blocks), int(stage3_blocks)
        available1 = len(backbone.encoder.stages[1].layers)
        available2 = len(backbone.encoder.stages[2].layers)
        available3 = len(backbone.encoder.stages[3].layers)
        if not (0 <= n0 <= available0 and 0 <= n1 <= available1
                and 0 <= n2 <= available2 and 0 <= n3 <= available3):
            raise ValueError("Requested block count exceeds ConvNeXt stage depth")
        if n1 and (n0 != available0 or not include_stage1_downsampling):
            raise ValueError("Stage 1 blocks require full stage 0 and stage 1 downsampling")
        if n2 and n1 != available1 or n3 and n2 != available2:
            raise ValueError("Later stages require all blocks in preceding stages")
        if n0 != available0:
            self.stage0.layers = nn.ModuleList(list(self.stage0.layers[:n0]))
        self.stage1_downsampling = (
            backbone.encoder.stages[1].downsampling_layer
            if include_stage1_downsampling else nn.Identity()
        )
        self.stage1_layers = nn.ModuleList(list(backbone.encoder.stages[1].layers[:n1])) if n1 else None
        self.stage2 = backbone.encoder.stages[2] if n2 else None
        self.stage3 = backbone.encoder.stages[3] if n3 else None
        if self.stage2 is not None:
            self.stage2.layers = nn.ModuleList(list(self.stage2.layers[:n2]))
        if self.stage3 is not None:
            self.stage3.layers = nn.ModuleList(list(self.stage3.layers[:n3]))

        # Use the requested input resolution to determine the readout's shape.
        was_training = self.training
        self.eval()
        with torch.no_grad():
            features = self._features(torch.zeros(1, 3, *input_hw))
        self.train(was_training)
        self.output_shape = tuple(features.shape[1:])

        if final_norm == "batchnorm":
            self.out_batchnorm = nn.BatchNorm2d(self.output_shape[0], momentum=batchnorm_momentum)
        elif final_norm == "identity":
            self.out_batchnorm = nn.Identity()
        elif final_norm == "groupnorm":
            self.out_batchnorm = nn.GroupNorm(1, self.output_shape[0])
        else:
            raise ValueError("Unsupported core.final_norm %r" % final_norm)
        activations = {"gelu": nn.GELU, "elu": nn.ELU, "identity": nn.Identity}
        if final_nonlinearity not in activations:
            raise ValueError("Unsupported core.final_nonlinearity %r" % final_nonlinearity)
        self.out_nonlinearity = activations[final_nonlinearity]()

        # Drop references to the unused classifier and later stages.
        del full
        del backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x[:, None, :, :]
        if x.ndim != 4:
            raise ValueError("Expected input [B,C,H,W] or [B,H,W]")

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError("Expected 1 or 3 input channels")

        x = self._features(x)
        x = self.out_batchnorm(x)
        x = self.out_nonlinearity(x)

        if x.shape[1:] != self.output_shape:
            raise RuntimeError(
                "Unexpected core output shape %r; expected %r"
                % (tuple(x.shape[1:]), self.output_shape)
            )
        return x

    def _features(self, x):
        x = self.embeddings(x)
        x = self.stage0(x)
        if isinstance(self.stage1_downsampling, (nn.ModuleList, nn.Sequential)):
            for layer in self.stage1_downsampling:
                x = layer(x)
        else:
            x = self.stage1_downsampling(x)
        if self.stage1_layers is not None:
            for layer in self.stage1_layers:
                x = layer(x)
        if self.stage2 is not None:
            x = self.stage2(x)
        if self.stage3 is not None:
            x = self.stage3(x)
        return x

    def set_trainable(self, trainable: bool) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = bool(trainable)

    def regularizer(self) -> torch.Tensor:
        # Historical ConvNextCore.regularizer() returned zero.
        parameter = next(self.parameters())
        return parameter.new_zeros(())


# ---------------------------------------------------------------------------

from typing import Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F


class FullGaussian2d(nn.Module):
    """
    Exact-use-case clean port of the historical isotropic FullGaussian2d path.

    Unsupported historical generic features (shared grids/features, predicted
    grids, shifts) are intentionally omitted because the released V1 model did
    not use them.
    """

    def __init__(
        self,
        in_shape: Sequence[int] = (80, 11, 11),
        outdims: int = 458,
        bias: bool = True,
        init_mu_range: float = 0.4,
        init_sigma: float = 0.6,
        batch_sample: bool = True,
        align_corners: bool = True,
        gauss_type: str = "isotropic",
    ) -> None:
        super().__init__()
        if gauss_type != "isotropic":
            raise ValueError("This clean V1 port supports only gauss_type='isotropic'")
        if init_mu_range <= 0 or init_mu_range > 1:
            raise ValueError("init_mu_range must be in (0, 1]")
        if init_sigma <= 0:
            raise ValueError("init_sigma must be positive")

        self.in_shape = tuple(int(v) for v in in_shape)
        self.outdims = int(outdims)
        self.batch_sample = bool(batch_sample)
        self.align_corners = bool(align_corners)
        self.gauss_type = gauss_type
        self.init_mu_range = float(init_mu_range)
        self.init_sigma = float(init_sigma)

        channels = self.in_shape[0]

        self._mu = nn.Parameter(torch.empty(1, self.outdims, 1, 2))
        self.sigma = nn.Parameter(torch.empty(1, self.outdims, 1, 1))
        self._features = nn.Parameter(
            torch.empty(1, channels, 1, self.outdims)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.outdims))
        else:
            self.register_parameter("bias", None)

        self.initialize()

    @property
    def mu(self) -> torch.Tensor:
        return self._mu

    @property
    def features(self) -> torch.Tensor:
        return self._features

    @property
    def grid(self) -> torch.Tensor:
        return self.sample_grid(batch_size=1, sample=False)

    def initialize(self) -> None:
        self._mu.data.uniform_(-self.init_mu_range, self.init_mu_range)
        self.sigma.data.fill_(self.init_sigma)
        self._features.data.fill_(1.0 / self.in_shape[0])
        if self.bias is not None:
            self.bias.data.zero_()

    def feature_l1(self, average: bool = True) -> torch.Tensor:
        if average:
            return self._features.abs().mean()
        return self._features.abs().sum()

    def sample_grid(
        self,
        batch_size: int,
        sample: Optional[bool] = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            self._mu.clamp_(min=-1.0, max=1.0)
            self.sigma.clamp_(min=0.0)

        do_sample = self.training if sample is None else bool(sample)
        grid_batch = int(batch_size) if self.batch_sample else 1

        if do_sample:
            norm = self._mu.new_empty(
                grid_batch, self.outdims, 1, 2
            ).normal_()
        else:
            norm = self._mu.new_zeros(
                grid_batch, self.outdims, 1, 2
            )

        grid = torch.clamp(
            norm * self.sigma + self._mu,
            min=-1.0,
            max=1.0,
        )

        if grid_batch == 1 and batch_size != 1:
            grid = grid.expand(batch_size, -1, -1, -1)
        return grid

    def forward(
        self,
        x: torch.Tensor,
        sample: Optional[bool] = None,
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Expected core tensor [B,C,H,W]")
        n, c, w, h = x.shape
        if (c, w, h) != self.in_shape:
            raise ValueError(
                "Expected core shape %r, got %r"
                % (self.in_shape, (c, w, h))
            )

        grid = self.sample_grid(batch_size=n, sample=sample)
        y = F.grid_sample(
            x,
            grid,
            align_corners=self.align_corners,
        )
        # [N,C,outdims,1] -> [N,C,outdims]
        y = y.squeeze(-1)
        features = self._features.view(1, c, self.outdims)
        y = (y * features).sum(dim=1).view(n, self.outdims)
        if self.bias is not None:
            y = y + self.bias
        return y


class PointPooledReadout(nn.Module):
    """Learn one spatial position per neuron and pool features at several scales."""

    def __init__(self, in_shape, outdims, bias=True, init_mu_range=0.4,
                 pool_steps=2, pool_kernel=2, align_corners=True):
        super().__init__()
        self.in_shape = tuple(int(v) for v in in_shape)
        self.outdims = int(outdims)
        self.pool_steps = int(pool_steps)
        self.pool_kernel = int(pool_kernel)
        self.align_corners = bool(align_corners)
        if self.pool_steps < 0 or self.pool_kernel < 2:
            raise ValueError("pool_steps must be >= 0 and pool_kernel >= 2")
        if not 0 < float(init_mu_range) <= 1:
            raise ValueError("init_mu_range must be in (0, 1]")
        if min(self.in_shape[1:]) < self.pool_kernel ** self.pool_steps:
            raise ValueError("Requested pooling steps exceed core spatial resolution")
        self.grid = nn.Parameter(torch.empty(1, self.outdims, 1, 2))
        self._features = nn.Parameter(torch.empty(
            1, self.in_shape[0] * (self.pool_steps + 1), 1, self.outdims
        ))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.outdims))
        else:
            self.register_parameter("bias", None)
        nn.init.uniform_(self.grid, -float(init_mu_range), float(init_mu_range))
        nn.init.constant_(self._features, 1.0 / self._features.shape[1])

    @property
    def features(self):
        return self._features

    def feature_l1(self, average=True):
        return self._features.abs().mean() if average else self._features.abs().sum()

    def forward(self, x, sample=None):
        if tuple(x.shape[1:]) != self.in_shape:
            raise ValueError("Expected core shape %r, got %r" %
                             (self.in_shape, tuple(x.shape[1:])))
        grid = self.grid.clamp(-1, 1).expand(x.shape[0], -1, -1, -1)
        scales = []
        for level in range(self.pool_steps + 1):
            if level:
                x = F.avg_pool2d(x, self.pool_kernel, stride=self.pool_kernel)
            scales.append(F.grid_sample(x, grid, align_corners=self.align_corners).squeeze(-1))
        sampled = torch.cat(scales, dim=1)
        y = (sampled * self._features.view(1, -1, self.outdims)).sum(dim=1)
        return y if self.bias is None else y + self.bias


# ---------------------------------------------------------------------------

from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F



class SinzV1Model(nn.Module):
    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        model_cfg = cfg["model"]
        backbone_cfg = model_cfg["backbone"]
        core_cfg = model_cfg["core"]
        readout_cfg = model_cfg["readout"]

        self.core = V1ConvNextCore(
            model_id=backbone_cfg["model_id"],
            revision=backbone_cfg.get("revision"),
            batchnorm_momentum=float(core_cfg["batchnorm_momentum"]),
            stage0_blocks=core_cfg.get("stage0_blocks"),
            stage1_blocks=core_cfg.get("stage1_blocks", 0),
            include_stage1_downsampling=core_cfg.get("include_stage1_downsampling", True),
            stage2_blocks=core_cfg.get("stage2_blocks", 0),
            stage3_blocks=core_cfg.get("stage3_blocks", 0),
            final_norm=core_cfg.get("final_norm", "batchnorm"),
            final_nonlinearity=core_cfg.get("final_nonlinearity", "gelu"),
            input_hw=tuple(cfg["data"]["preprocessing"]["expected_output_hw"]),
        )
        expected_shape = core_cfg.get("output_channels"), core_cfg.get("output_hw")
        if expected_shape[0] is not None and int(expected_shape[0]) != self.core.output_shape[0]:
            raise ValueError("core.output_channels does not match selected cutoff")
        if expected_shape[1] is not None and tuple(expected_shape[1]) != self.core.output_shape[1:]:
            raise ValueError("core.output_hw does not match selected cutoff")
        readout_type = readout_cfg.get("type", "gaussian")
        common = dict(in_shape=self.core.output_shape,
                      outdims=int(readout_cfg["outdims"]),
                      bias=bool(readout_cfg["bias"]),
                      init_mu_range=float(readout_cfg["init_mu_range"]),
                      align_corners=bool(readout_cfg["align_corners"]))
        if readout_type == "gaussian":
            self.readout = FullGaussian2d(
                **common, init_sigma=float(readout_cfg["init_sigma"]),
                batch_sample=bool(readout_cfg["batch_sample"]),
                gauss_type=str(readout_cfg["gauss_type"]),
            )
        elif readout_type == "point_pooled":
            self.readout = PointPooledReadout(
                **common, pool_steps=int(readout_cfg.get("pool_steps", 2)),
                pool_kernel=int(readout_cfg.get("pool_kernel", 2)),
            )
        else:
            raise ValueError("Unknown model.readout.type %r" % readout_type)
        self.elu_offset = float(model_cfg["output"]["elu_offset"])

    def forward(
        self,
        x: torch.Tensor,
        sample=None,
    ) -> torch.Tensor:
        x = self.core(x)
        x = self.readout(x, sample=sample)
        return F.elu(x + self.elu_offset) + 1.0

    def regularizer(self, gamma_readout: float) -> torch.Tensor:
        return (
            self.core.regularizer()
            + float(gamma_readout) * self.readout.feature_l1(average=False)
        )

    def set_core_trainable(self, trainable: bool) -> None:
        self.core.set_trainable(trainable)


def load_best_model(checkpoint_path, device="cpu"):
    """Load a standalone best.pt using its own recorded architecture/config."""
    payload = torch.load(checkpoint_path, map_location="cpu")
    if "config" not in payload or "model_state_dict" not in payload:
        raise ValueError("Checkpoint lacks config or model weights")
    cfg = payload["config"]
    model = TVSDV1Model(cfg) if cfg.get("data", {}).get("kind") == "tvsd" else SinzV1Model(cfg)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, payload["config"]


class TVSDV1Model(SinzV1Model):
    """Original Sinz core + ONE Gaussian readout over [F | N] neuron columns.

    Neuron parameters remain independent. F and N have disjoint row masks,
    but share the same feature representation and readout implementation.
    """
    def __init__(self, cfg):
        rc = cfg['model']['readout']
        sizes = rc['outdims_by_animal']
        n_f, n_n = int(sizes['F']), int(sizes['N'])
        if min(n_f, n_n) <= 0 or n_f + n_n != int(rc['outdims']):
            raise ValueError('Combined Gaussian outdims must equal F + N readout indices')
        if rc.get('type', 'gaussian') != 'gaussian':
            raise ValueError('TVSD baseline requires Gaussian readout')
        super().__init__(cfg)
