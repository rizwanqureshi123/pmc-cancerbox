"""
Run MCP-MedSAM on PMC relevance-labeled JSON files.

Workflow:
1. Parse labeled JSON and keep only relevant images.
2. Extract bounding boxes from region annotations when present.
3. Infer modality prompts from captions/metadata.
4. Run MCP-MedSAM segmentation for each (image, bbox, modality) pair.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from transformers import CLIPModel, CLIPTokenizer

from models import MaskDecoder_F4, PromptEncoder, TinyViT, TwoWayTransformer
from pmc_labeled_dataset import load_relevant_scans


def clip_feature_tensor(features):
    if isinstance(features, torch.Tensor):
        return features
    if hasattr(features, "pooler_output") and features.pooler_output is not None:
        return features.pooler_output
    if hasattr(features, "image_embeds") and features.image_embeds is not None:
        return features.image_embeds
    if hasattr(features, "text_embeds") and features.text_embeds is not None:
        return features.text_embeds
    raise TypeError(f"Unsupported CLIP feature type: {type(features)}")


class MedSAM_Lite(torch.nn.Module):
    def __init__(self, image_encoder, mask_decoder, prompt_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

    def postprocess_masks(self, masks, new_size, original_size):
        masks = masks[..., : new_size[0], : new_size[1]]
        return F.interpolate(
            masks,
            size=(original_size[0], original_size[1]),
            mode="bilinear",
            align_corners=False,
        )


def resize_longest_side(image, target_length=256):
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
    return cv2.resize(image, (neww, newh), interpolation=cv2.INTER_AREA)


def pad_image(image, target_size=256):
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3:
        return np.pad(image, ((0, padh), (0, padw), (0, 0)))
    return np.pad(image, ((0, padh), (0, padw)))


def normalize_bbox(bbox: list[int], height: int, width: int, min_size: int = 2) -> list[int] | None:
    x0, y0, x1, y1 = [int(v) for v in bbox]
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width))
    y1 = max(0, min(y1, height))
    if x1 <= x0:
        x1 = min(width, x0 + min_size)
    if y1 <= y0:
        y1 = min(height, y0 + min_size)
    if x1 - x0 < min_size or y1 - y0 < min_size:
        return None
    return [x0, y0, x1, y1]


def resize_box_to_256(box, original_size):
    new_box = np.zeros_like(box)
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)
    return new_box, ratio


def m2_pre_img(image_data, image_size=224):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(
                [image_size, image_size],
                interpolation=transforms.InterpolationMode.BILINEAR,
                antialias=True,
            ),
        ]
    )
    return transform(image_data)


def get_contents(img, box, clip_model):
    height, width = img.shape[:2]
    normalized = normalize_bbox(box, height, width)
    if normalized is None:
        raise ValueError(f"Degenerate bbox {box} for image size {(height, width)}")
    x_mino, y_mino, x_maxo, y_maxo = normalized
    crops = img[y_mino:y_maxo, x_mino:x_maxo, :]
    if crops.size == 0 or crops.shape[0] == 0 or crops.shape[1] == 0:
        raise ValueError(f"Degenerate crop for bbox {box}")
    crops_64 = m2_pre_img(crops, image_size=64)
    crops_224 = m2_pre_img(crops).unsqueeze(0)
    with torch.no_grad():
        image_features = clip_feature_tensor(clip_model.get_image_features(crops_224))
    return crops_64, image_features


def get_text_features(modality_text, clip_model, tokenizer):
    text_token = tokenizer(
        modality_text,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.no_grad():
        return clip_feature_tensor(clip_model.get_text_features(text_token))


@torch.no_grad()
def medsam_inference(
    medsam_model,
    img_embed,
    box_256,
    features,
    crops,
    text_features,
    category_idx,
    new_size,
    original_size,
    device,
):
    box_torch = torch.as_tensor(box_256[None, None, ...], dtype=torch.float, device=device)
    if features.dim() == 2:
        features = features.unsqueeze(0)
    features = features.to(device)
    crops = crops.unsqueeze(0).to(device)
    category_idx_tensor = torch.tensor([category_idx]).to(device)
    text_features = text_features.to(device)
    if text_features.dim() == 2:
        text_features = text_features.unsqueeze(0)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
        features=features,
        crops=crops,
        text_features=text_features,
        category_idx=category_idx_tensor,
    )
    low_res_logits, iou, _, _, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred).squeeze().cpu().numpy()
    return (low_res_pred > 0.5).astype(np.uint8), iou


def show_mask(mask, ax, alpha=0.5):
    color = np.array([251 / 255, 252 / 255, 30 / 255, alpha])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_box(box, ax, edgecolor="blue"):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor=edgecolor, facecolor=(0, 0, 0, 0), lw=2)
    )


def load_image_rgb(source: str | None, url: str | None, cache_dir: Path) -> np.ndarray:
    if source is not None:
        path = Path(source)
        if path.exists():
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Failed to read image: {path}")
            if image.ndim == 2:
                image = np.repeat(image[:, :, None], 3, axis=-1)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image.astype(np.uint8)

    if url is None:
        raise ValueError("No local image path or URL available for this entry.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).name or hashlib.sha1(url.encode("utf-8")).hexdigest() + ".jpg"
    cached_path = cache_dir / filename
    if not cached_path.exists():
        with urlopen(url, timeout=60) as response:
            cached_path.write_bytes(response.read())

    image = cv2.imread(str(cached_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Failed to read downloaded image: {cached_path}")
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=-1)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.uint8)


def build_model(checkpoint_path: str, device: torch.device) -> MedSAM_Lite:
    image_encoder = TinyViT(
        img_size=256,
        in_chans=3,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.0,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=256,
        image_embedding_size=(64, 64),
        input_image_size=(256, 256),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder_F4(
        num_multimask_outputs=3,
        transformer=TwoWayTransformer(
            depth=2,
            embedding_dim=256,
            mlp_dim=2048,
            num_heads=8,
        ),
        modality=True,
        contents=True,
        transformer_dim=256,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )
    model = MedSAM_Lite(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=prompt_encoder,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()
    return model


def run_single_inference(
    model: MedSAM_Lite,
    clip_model: CLIPModel,
    tokenizer: CLIPTokenizer,
    image_rgb: np.ndarray,
    bbox: list[int],
    modality_text: str,
    category_idx: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    height, width = image_rgb.shape[:2]
    normalized = normalize_bbox(bbox, height, width)
    if normalized is None:
        raise ValueError(f"Invalid bbox {bbox} for image size {(height, width)}")
    bbox = normalized

    text_features = get_text_features(modality_text, clip_model, tokenizer)

    img_256 = resize_longest_side(image_rgb, 256)
    new_h, new_w = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = (
        torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )

    with torch.no_grad():
        image_embedding = model.image_encoder(img_256_tensor)

    crops, features = get_contents(image_rgb, bbox, clip_model)
    box_256, _ = resize_box_to_256(np.array(bbox), original_size=(height, width))
    mask, iou = medsam_inference(
        model,
        image_embedding,
        box_256,
        features,
        crops,
        text_features,
        category_idx,
        (new_h, new_w),
        (height, width),
        device,
    )
    return mask, float(iou.item())


def save_overlay(image_rgb: np.ndarray, mask: np.ndarray, bbox: list[int], output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_rgb)
    axes[0].set_title("Image")
    axes[0].axis("off")

    axes[1].imshow(image_rgb)
    show_box(np.array(bbox), axes[1])
    show_mask(mask.astype(np.uint8), axes[1])
    axes[1].set_title("MCP-MedSAM Segmentation")
    axes[1].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MCP-MedSAM on PMC labeled JSON files.")
    parser.add_argument("--json", nargs="+", required=True, help="One or more labeled JSON files.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Directory containing downloaded PMC images.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="pmc_image_cache",
        help="Directory used when images must be downloaded from URL.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for masks/overlays.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to MCP-MedSAM checkpoint.")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device.")
    parser.add_argument(
        "--allow-full-image-bbox",
        action="store_true",
        help="Use a centered full-image bbox when regions are missing.",
    )
    parser.add_argument("--save-overlay", action="store_true", help="Save side-by-side overlay PNGs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on processed scans.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    cache_dir = Path(args.cache_dir)
    image_dir = Path(args.image_dir) if args.image_dir else None
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)

    clip_model = CLIPModel.from_pretrained(
        "flaviagiammarino/pubmed-clip-vit-base-patch32",
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        "openai/clip-vit-base-patch16",
    )
    clip_model.requires_grad_(False)
    clip_model.eval()

    model = build_model(args.checkpoint, device)

    scans: list[dict] = []
    for json_path in args.json:
        scans.extend(
            load_relevant_scans(
                json_path,
                image_dir=image_dir,
                require_bbox=not args.allow_full_image_bbox,
                allow_full_image_bbox=args.allow_full_image_bbox,
            )
        )

    if args.limit > 0:
        scans = scans[: args.limit]

    if not scans:
        print(
            "No relevant scans with bounding boxes were found. "
            "Add `regions`/`bbox` annotations to relevant images, "
            "or rerun with --allow-full-image-bbox for exploratory inference."
        )
        return

    skipped = 0
    processed = 0
    for index, scan in enumerate(tqdm(scans, desc="MCP-MedSAM PMC inference")):
        bbox = scan.get("bbox")
        original_bbox_was_none = bbox is None
        if bbox is None and not args.allow_full_image_bbox:
            skipped += 1
            continue

        stem = f"{index:06d}_{scan.get('pmcid', 'unknown')}"
        if (mask_dir / f"{stem}.npz").exists():
            processed += 1
            continue

        try:
            image_rgb = load_image_rgb(scan.get("image_path"), scan.get("url"), cache_dir)
        except ValueError:
            skipped += 1
            continue

        if bbox is None:
            height, width = image_rgb.shape[:2]
            margin = max(1, min(height, width) // 20)
            bbox = [margin, margin, width - margin, height - margin]
        else:
            height, width = image_rgb.shape[:2]
            normalized = normalize_bbox(bbox, height, width)
            if normalized is None:
                print(
                    f"Skipping invalid bbox {bbox} for {scan.get('url', scan.get('pmcid', index))}"
                )
                skipped += 1
                continue
            bbox = normalized

        try:
            mask, iou = run_single_inference(
                model=model,
                clip_model=clip_model,
                tokenizer=tokenizer,
                image_rgb=image_rgb,
                bbox=bbox,
                modality_text=scan["modality_text"],
                category_idx=scan["category_idx"],
                device=device,
            )
        except (ValueError, RuntimeError) as exc:
            print(
                f"Skipping inference error for {scan.get('url', scan.get('pmcid', index))}: {exc}"
            )
            skipped += 1
            continue

        if original_bbox_was_none:
            bbox_source = "full_image_fallback"
        else:
            bbox_source = scan.get("bbox_source") or "manual_or_regions"

        np.savez_compressed(
            mask_dir / f"{stem}.npz",
            mask=mask.astype(np.uint8),
            bbox=np.array(bbox),
            iou=iou,
            modality=scan["modality"],
            caption=scan.get("caption", ""),
            url=scan.get("url", ""),
            image_path=scan.get("image_path", ""),
            bbox_source=bbox_source,
        )
        processed += 1

        if args.save_overlay:
            save_overlay(image_rgb, mask, bbox, overlay_dir / f"{stem}.png")

    print(f"Processed {processed} scans. Skipped {skipped} entries.")


if __name__ == "__main__":
    main()