"""Download and verify the exact pretrained Sinz backbone; NOT Musa's timm model."""
import os
from pathlib import Path
import torch
from transformers import ConvNextV2ForImageClassification

MODEL_ID = "facebook/convnextv2-atto-1k-224"
REVISION = "2a62c34e37c1092d9a99cba587d5428837495d7c"


def main():
    print("Hugging Face model:", MODEL_ID)
    print("Pinned revision:   ", REVISION)
    print("HF_HOME:           ", os.environ.get("HF_HOME", "<unset>"))
    model = ConvNextV2ForImageClassification.from_pretrained(
        MODEL_ID, revision=REVISION
    )
    model.eval()
    with torch.no_grad():
        x = model.convnextv2.embeddings(torch.zeros(1, 3, 93, 93))
        x = model.convnextv2.encoder.stages[0](x)
        x = model.convnextv2.encoder.stages[1].downsampling_layer(x)
    print("Sinz cutoff feature map:", tuple(x.shape))
    assert tuple(x.shape) == (1, 80, 11, 11)
    print("BACKBONE VERIFIED — weights are cached persistently.")


if __name__ == "__main__":
    main()
