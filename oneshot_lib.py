"""Foundation-model backbone, prototype head and test-time adaptation for one-shot classification."""
import re
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

# ----------------------------------------------------------------------------
# Text prompts
# ----------------------------------------------------------------------------
PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a photo of the {}.",
    "a picture of a {}.",
    "a low resolution photo of a {}.",
    "a {} scene.",
    "an image showing a {}.",
]

# Human-readable phrasing for class folder names. Edit to match your class list.
DEFAULT_DESCRIPTIONS = {
    "Bedroom": "bedroom with 2 nightstands and a lamp.",
    "Coast": "coast with ocean and beach with the sun passing the treeline",
    "Flower": "4 flowers",
    "Forest": "forest with a single tree in the foreground",
    "Highway": "highway road with few cars and above street lights",
    "Industrial": "industrial building site in the distacne releasing steam",
    "InsideCity": "downtown city street of a large apartment",
    "Kitchen": "kitchen with a stove sink and fridge",
    "LivingRoom": "sperad out living room with a couch and two lovfe seats",
    "Mountain": "snowy mountain top landscape vieing the peak from a distance",
    "Office": "office interior with a computer a window and fax machine",
    "OpenCountry": "open countryside field over rolling fields",
    "Store": "store interior with shelves and a worker",
    "Street": "busy city street with moving cars and pedastrians",
    "Suburb": "suburban neighborhood with houses and foliage",
    "TallBuilding": "several skyscrapers",
}


def class_to_text(name, overrides=None):
    overrides = DEFAULT_DESCRIPTIONS if overrides is None else overrides
    if name in overrides:
        return overrides[name]
    return re.sub(r"(?<!^)(?=[A-Z])", " ", name).replace("_", " ").lower()


def _load(x):
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    return x.convert("RGB") if x.mode != "RGB" else x


def _as_tensor(x):
    if torch.is_tensor(x):
        return x
    for attr in ("pooler_output", "image_embeds", "text_embeds"):
        v = getattr(x, attr, None)
        if v is not None:
            return v
    return x[0]


# ----------------------------------------------------------------------------
# Frozen backbone
# ----------------------------------------------------------------------------
class FoundationModel:
    """Frozen backbone that returns L2-normalized image (and text) embeddings."""

    def __init__(self, kind, hf_name, device):
        from transformers import AutoModel, AutoProcessor, AutoTokenizer
        self.kind, self.hf_name, self.device = kind, hf_name, device
        model = AutoModel.from_pretrained(hf_name)
        if device.type == "cuda":
            model = model.half()
        self.model = model.to(device).eval()
        self.dtype = next(self.model.parameters()).dtype
        self.processor = AutoProcessor.from_pretrained(hf_name)
        self.tokenizer = None if kind == "dinov2" else AutoTokenizer.from_pretrained(hf_name)

    @property
    def has_text(self):
        return self.tokenizer is not None

    @torch.no_grad()
    def encode_images(self, images, batch_size=64):
        feats = []
        for i in range(0, len(images), batch_size):
            batch = [_load(x) for x in images[i:i + batch_size]]
            px = self.processor(images=batch, return_tensors="pt")["pixel_values"]
            px = px.to(self.device, dtype=self.dtype)
            if self.kind == "dinov2":
                h = self.model(pixel_values=px).last_hidden_state
                f = torch.cat([h[:, 0], h[:, 1:].mean(dim=1)], dim=-1)   # CLS + mean patch token
            else:
                f = _as_tensor(self.model.get_image_features(pixel_values=px))
            feats.append(F.normalize(f.float(), dim=-1).cpu())
        return torch.cat(feats)

    @torch.no_grad()
    def encode_texts(self, texts, batch_size=64):
        assert self.has_text, f"{self.kind} has no text encoder."
        feats = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            if self.kind == "siglip":
                tok = self.tokenizer(chunk, padding="max_length", max_length=64,
                                     truncation=True, return_tensors="pt")
                f = self.model.get_text_features(input_ids=tok["input_ids"].to(self.device))
            else:
                tok = self.tokenizer(chunk, padding=True, max_length=77,
                                     truncation=True, return_tensors="pt")
                f = self.model.get_text_features(
                    input_ids=tok["input_ids"].to(self.device),
                    attention_mask=tok["attention_mask"].to(self.device))
            feats.append(F.normalize(_as_tensor(f).float(), dim=-1).cpu())
        return torch.cat(feats)


def text_prototypes(fm, class_names, templates=PROMPT_TEMPLATES, overrides=None):
    """Zero-shot classifier: mean text embedding over prompt templates, one row per class."""
    protos = []
    for name in class_names:
        phrase = class_to_text(name, overrides)
        feats = fm.encode_texts([t.format(phrase) for t in templates])
        protos.append(F.normalize(feats.mean(0), dim=0))
    return torch.stack(protos)


# ----------------------------------------------------------------------------
# Trainable prototype head
# ----------------------------------------------------------------------------
class PrototypeHead(nn.Module):
    """Cosine-similarity classifier on frozen features. Rows of W are class prototypes."""

    def __init__(self, init_W, scale=100.0):
        super().__init__()
        self.W = nn.Parameter(init_W.clone())
        self.register_buffer("W0", init_W.clone())     # starting point used as an anchor
        self.scale = scale

    def forward(self, feats):
        return self.scale * feats @ F.normalize(self.W, dim=-1).T

    def prototypes(self):
        return F.normalize(self.W.detach(), dim=-1).cpu()


def prototypes_from_views(view_feats):
    """[num_classes, n_views, D] -> [num_classes, D] mean prototype."""
    return F.normalize(view_feats.mean(dim=1), dim=-1)


def fuse(text_proto, image_proto, alpha=0.5):
    """Blend the zero-shot text prototype with the one-shot image prototype."""
    return F.normalize(alpha * text_proto + (1 - alpha) * image_proto, dim=-1)


# ----------------------------------------------------------------------------
# Test-time augmentation and adaptation (use UNLABELED test images only)
# ----------------------------------------------------------------------------
def tta_views(image):
    """Original, flip, and two center crops (each also flipped)."""
    w, h = image.size
    views = [image, T.functional.hflip(image)]
    for frac in (0.9, 0.75):
        cw, ch = int(w * frac), int(h * frac)
        left, top = (w - cw) // 2, (h - ch) // 2
        crop = image.crop((left, top, left + cw, top + ch))
        views += [crop, T.functional.hflip(crop)]
    return views


def encode_images_tta(fm, images, views_fn=tta_views, chunk=32):
    """Average the embeddings of several views of every image."""
    out = []
    for i in range(0, len(images), chunk):
        all_views, counts = [], []
        for x in images[i:i + chunk]:
            v = views_fn(_load(x))
            all_views += v
            counts.append(len(v))
        feats = fm.encode_images(all_views)
        j = 0
        for c in counts:
            out.append(F.normalize(feats[j:j + c].mean(0), dim=0))
            j += c
    return torch.stack(out)


@torch.no_grad()
def sinkhorn(logits, n_iter=3):
    """Soft assignments with equal total mass per class (assumes a roughly balanced pool)."""
    logits = logits.double()
    Q = torch.exp(logits - logits.max())
    for _ in range(n_iter):
        Q = Q / Q.sum(0, keepdim=True)
        Q = Q / Q.sum(1, keepdim=True)
    return Q.float()


@torch.no_grad()
def transductive_refine(X, W0, iters=10, temp=0.03, gamma=0.5, balance=True):
    """Soft k-means in embedding space, anchored to the one-shot prototypes W0."""
    W = W0.clone()
    for _ in range(iters):
        logits = X @ W.T / temp
        Q = sinkhorn(logits) if balance else torch.softmax(logits, dim=1)
        centers = F.normalize(Q.T @ X, dim=-1)
        W = F.normalize((1 - gamma) * W0 + gamma * centers, dim=-1)
    return W


def accuracy(logits, labels):
    return (logits.argmax(dim=1) == labels).float().mean().item() * 100
