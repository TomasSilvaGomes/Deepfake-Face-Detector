import os
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import transforms
from transformers import (
    CLIPVisionConfig,
    CLIPVisionModel,
    Swinv2Config,
    Swinv2ForImageClassification,
)

def get_swinv2_transform():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_clip_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


class SwinV2Classifier(nn.Module):
    HF_MODEL_ID = "microsoft/swinv2-small-patch4-window16-256"

    def __init__(self, ckpt_path: str, num_labels: int = 2):
        super().__init__()
        self.num_labels = num_labels
        config = Swinv2Config.from_pretrained(
            self.HF_MODEL_ID, num_labels=num_labels, ignore_mismatched_sizes=True
        )
        self.model = Swinv2ForImageClassification(config)
        self._load_weights(ckpt_path)

    def _load_weights(self, ckpt_path: str):
        path = Path(ckpt_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint não encontrado: {ckpt_path}")

        if path.suffix == ".safetensors":
            from safetensors.torch import load_file
            state_dict = load_file(str(path), device="cpu")
        else:
            state_dict = torch.load(str(path), map_location="cpu")

        self.model.load_state_dict(state_dict, strict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(pixel_values=x).logits

    def predict_prob(self, x: torch.Tensor) -> float:
        with torch.no_grad():
            logits = self.forward(x)
            return torch.softmax(logits, dim=1)[0, 1].item()


class DF40CLIPModel(nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14")
        self.backbone = CLIPVisionModel(config)
        self.head = nn.Linear(config.hidden_size, num_labels)
        

    def forward(self, pixel_values):
        outputs = self.backbone(pixel_values=pixel_values)
        return self.head(outputs.pooler_output)