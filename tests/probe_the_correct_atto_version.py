# import torch
# import transformers
# from transformers import ConvNextV2ForImageClassification


# MODEL_NAME = "facebook/convnextv2-atto-1k-224"


# # Match historical backend state
# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = False
# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = True


# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# print("torch:", torch.__version__)
# print("transformers:", transformers.__version__)
# print("device:", device)


# # Historical constructor
# model = ConvNextV2ForImageClassification.from_pretrained(
#     MODEL_NAME
# ).to(device)

# model.eval()

# # Dummy grayscale model input
# x = torch.randn(4, 1, 93, 93, device=device)

# # Historical ConvNextCore grayscale handling
# x = x.repeat(1, 3, 1, 1)
# print("after grayscale repeat:", tuple(x.shape))


# with torch.no_grad():

#     # ConvNeXtV2 patch embedding
#     x = model.convnextv2.embeddings(x)
#     print("after embeddings:", tuple(x.shape))

#     # Entire stage 0 = exactly its two residual blocks
#     x = model.convnextv2.encoder.stages[0](x)
#     print("after stage 0:", tuple(x.shape))

#     # Historical clipping keeps the stage-1 downsampling,
#     # but removes stage1.layers.0 and everything afterwards.
#     x = model.convnextv2.encoder.stages[1].downsampling_layer(x)
#     print("after stage 1 downsampling:", tuple(x.shape))


# assert x.shape == (4, 80, 11, 11), x.shape

# # Historical ConvNextCore output transform
# bn = torch.nn.BatchNorm2d(80, momentum=0.9).to(device)
# gelu = torch.nn.GELU().to(device)

# with torch.no_grad():
#     y = gelu(bn(x))

# print("after BatchNorm + GELU:", tuple(y.shape))

# assert y.shape == (4, 80, 11, 11)

# print("\nEXACT CORE SMOKE TEST PASS")


import torch
from torch import nn
from transformers import ConvNextV2ForImageClassification


class V1ConvNextCore(nn.Module):
    """
    Clean implementation of the ConvNeXtV2 core used by the released
    macaque V1 digital twin.

    Historical nnvision equivalent:
        model_name = "facebook/convnextv2-atto-1k-224"
        cutoff_layer = "convnextv2.encoder.stages.1.layers.0"
        patch_embedding_stride = None
        final_norm = "BatchNorm"
        momentum = 0.9
        final_nonlinearity = "GELU"

    Effective active backbone:
        patch embedding
        -> stage 0 block 0
        -> stage 0 block 1
        -> stage 1 downsampling
        -> BatchNorm2d(80, momentum=0.9)
        -> GELU

    Input:
        [B, 1, 93, 93] grayscale

    Output:
        [B, 80, 11, 11]
    """

    MODEL_NAME = "facebook/convnextv2-atto-1k-224"

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        pretrained: bool = True,
    ):
        super().__init__()

        if not pretrained:
            raise NotImplementedError(
                "Historical model construction starts from the pretrained "
                "Hugging Face ConvNeXtV2 checkpoint."
            )

        full_model = ConvNextV2ForImageClassification.from_pretrained(
            model_name
        )

        convnext = full_model.convnextv2

        # These are exactly the modules that survive the historical
        # inclusive clipping at:
        # convnextv2.encoder.stages.1.layers.0
        self.embeddings = convnext.embeddings
        self.stage0 = convnext.encoder.stages[0]
        self.stage1_downsampling = (
            convnext.encoder.stages[1].downsampling_layer
        )

        # Historical ConvNextCore additions.
        self.out_batchnorm = nn.BatchNorm2d(
            80,
            momentum=0.9,
        )
        self.out_nonlinearity = nn.GELU()

        # Release references are no longer needed.
        del full_model
        del convnext

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if x.ndim != 4:
            raise ValueError(
                f"Expected [B,H,W] or [B,C,H,W], got {tuple(x.shape)}"
            )

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError(
                f"Expected 1 or 3 input channels, got {x.shape[1]}"
            )

        x = self.embeddings(x)
        x = self.stage0(x)
        x = self.stage1_downsampling(x)

        x = self.out_batchnorm(x)
        x = self.out_nonlinearity(x)

        return x

    def set_trainable(self, trainable: bool) -> None:
        """
        Used for the two-stage training protocol:
            phase 1: False
            phase 2: True
        """
        for parameter in self.parameters():
            parameter.requires_grad = trainable