from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw
import random
import numpy as np


def load_rgb(path):
    return Image.open(path).convert("RGB")


def apply_corruption(img, corruption="clean", severity=1, seed=42):
    rng = random.Random(seed)

    if corruption == "clean":
        return img

    if corruption == "blur":
        radius = {1: 1.0, 2: 2.0, 3: 3.0}.get(severity, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "brightness":
        factor = {1: 0.75, 2: 0.55, 3: 0.35}.get(severity, 0.55)
        return ImageEnhance.Brightness(img).enhance(factor)

    if corruption == "noise":
        arr = np.asarray(img).astype(np.float32)
        sigma = {1: 10, 2: 25, 3: 40}.get(severity, 25)
        noise = np.random.default_rng(seed).normal(0, sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    if corruption == "occlusion":
        out = img.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size
        ratio = {1: 0.18, 2: 0.28, 3: 0.38}.get(severity, 0.28)
        bw, bh = int(w * ratio), int(h * ratio)
        x0 = rng.randint(0, max(0, w - bw))
        y0 = rng.randint(0, max(0, h - bh))
        draw.rectangle([x0, y0, x0 + bw, y0 + bh], fill=(0, 0, 0))
        return out

    raise ValueError(f"Unknown corruption: {corruption}")


def corrupt_option_images(sample, project_root, corruption="clean", severity=1, only_positive=False, seed=42):
    if corruption == "clean":
        new_sample = dict(sample)
        new_sample["corruption"] = "clean"
        new_sample["severity"] = 0
        new_sample["corruption_scope"] = "none"
        return new_sample

    project_root = Path(project_root)
    new_sample = dict(sample)
    new_options = []
    scope = "positive_only" if only_positive else "all_images"

    for opt in sample["options"]:
        new_opt = dict(opt)
        src = project_root / opt["path"]

        if (not only_positive) or opt.get("is_correct", False):
            img = load_rgb(src)
            corrupted = apply_corruption(
                img,
                corruption=corruption,
                severity=severity,
                seed=seed + int(opt["image_id"]),
            )

            rel_out = Path("data/corrupted") / f"{corruption}_s{severity}_{scope}" / opt["file_name"]
            abs_out = project_root / rel_out
            abs_out.parent.mkdir(parents=True, exist_ok=True)

            if not abs_out.exists():
                corrupted.save(abs_out, quality=95)

            new_opt["path"] = str(rel_out)
        else:
            new_opt["path"] = opt["path"]

        new_options.append(new_opt)

    new_sample["options"] = new_options
    new_sample["corruption"] = corruption
    new_sample["severity"] = severity
    new_sample["corruption_scope"] = scope
    return new_sample
