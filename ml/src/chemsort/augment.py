"""Data augmentation presets shared by every ChemSort pipeline.

One :class:`AugmentConfig` describes a policy; the same preset drives

* :func:`augment_image` - PIL/NumPy implementation used for offline
  augmentation (``tools.augment_dataset``) and previews;
* :func:`keras_augmentation` - on-the-fly Keras preprocessing layers
  (``pipelines.train_keras``);
* :func:`yolo_hyperparameters` - matching Ultralytics classification
  arguments (``pipelines.train_yolo``);
* :func:`mix_batch` - MixUp / CutMix on one-hot Keras batches.

Geometric transforms never introduce empty borders: after rotation the image
is cropped to the largest axis-aligned rectangle inside the rotated content,
so the model never learns "black corner" artefacts that the camera never
produces. Augmentation is applied to training images only.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


@dataclass(frozen=True)
class AugmentConfig:
    name: str
    hflip: float = 0.5  # probability
    vflip: float = 0.0  # probability
    rotate_degrees: float = 0.0  # uniform in [-d, d]
    zoom: float = 0.0  # crop-in up to this fraction (0.2 -> up to 1.2x)
    translate: float = 0.0  # max shift as a fraction of the side
    brightness: float = 0.0  # factor uniform in [1-b, 1+b]
    contrast: float = 0.0
    saturation: float = 0.0
    hue: float = 0.0  # shift as a fraction of the hue circle
    blur: float = 0.0  # probability of a light Gaussian blur
    noise: float = 0.0  # max Gaussian noise std in 0..255 units
    erasing: float = 0.0  # probability of one random-erasing box

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, AugmentConfig] = {
    "none": AugmentConfig("none", hflip=0.0),
    "light": AugmentConfig(
        "light", hflip=0.5, rotate_degrees=10, zoom=0.10, translate=0.05,
        brightness=0.10, contrast=0.10, saturation=0.10, hue=0.01,
    ),
    "medium": AugmentConfig(
        "medium", hflip=0.5, rotate_degrees=20, zoom=0.15, translate=0.08,
        brightness=0.20, contrast=0.20, saturation=0.30, hue=0.015,
        blur=0.10, noise=4.0,
    ),
    "strong": AugmentConfig(
        "strong", hflip=0.5, vflip=0.2, rotate_degrees=30, zoom=0.25,
        translate=0.12, brightness=0.30, contrast=0.30, saturation=0.50,
        hue=0.03, blur=0.20, noise=8.0, erasing=0.40,
    ),
}


def get_preset(name: str | AugmentConfig) -> AugmentConfig:
    if isinstance(name, AugmentConfig):
        return name
    if name not in PRESETS:
        raise ValueError(f"Unknown augmentation preset {name!r}; expected one of {sorted(PRESETS)}")
    return PRESETS[name]


# --------------------------------------------------------------------------
# Offline (PIL / NumPy)
# --------------------------------------------------------------------------


def inscribed_scale(width: int, height: int, degrees: float) -> float:
    """Largest k so that a centred (k*w, k*h) box fits in the rotated image."""
    theta = math.radians(abs(degrees))
    cos, sin = math.cos(theta), math.sin(theta)
    return min(width / (width * cos + height * sin), height / (width * sin + height * cos))


def _rotate_zoom_translate(image: Image.Image, config: AugmentConfig, rng: np.random.Generator) -> Image.Image:
    width, height = image.size
    angle = float(rng.uniform(-config.rotate_degrees, config.rotate_degrees)) if config.rotate_degrees else 0.0
    zoom = float(rng.uniform(0.0, config.zoom)) if config.zoom else 0.0
    if angle == 0.0 and zoom == 0.0 and config.translate == 0.0:
        return image
    rotated = image.rotate(angle, resample=Image.BICUBIC, expand=False) if angle else image
    valid = inscribed_scale(width, height, angle)
    scale = valid / (1.0 + zoom)
    crop_w, crop_h = width * scale, height * scale
    # Shift only inside the inscribed box, so no rotation border is visible.
    slack_x = (valid - scale) * width / 2
    slack_y = (valid - scale) * height / 2
    dx = float(rng.uniform(-1, 1)) * min(slack_x, config.translate * width)
    dy = float(rng.uniform(-1, 1)) * min(slack_y, config.translate * height)
    left = (width - crop_w) / 2 + dx
    top = (height - crop_h) / 2 + dy
    box = (left, top, left + crop_w, top + crop_h)
    return rotated.resize((width, height), resample=Image.BICUBIC, box=box)


def _shift_hue(image: Image.Image, shift: float) -> Image.Image:
    hsv = np.asarray(image.convert("HSV"), dtype=np.int16).copy()
    hsv[..., 0] = (hsv[..., 0] + int(round(shift * 255))) % 256
    return Image.fromarray(hsv.astype(np.uint8), mode="HSV").convert("RGB")


def _random_erase(array: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = array.shape[:2]
    for _ in range(10):
        area = rng.uniform(0.02, 0.15) * height * width
        ratio = math.exp(rng.uniform(math.log(0.3), math.log(3.3)))
        box_h = int(round(math.sqrt(area * ratio)))
        box_w = int(round(math.sqrt(area / ratio)))
        if 0 < box_h < height and 0 < box_w < width:
            top = int(rng.integers(0, height - box_h + 1))
            left = int(rng.integers(0, width - box_w + 1))
            array[top:top + box_h, left:left + box_w] = rng.integers(0, 256, size=(box_h, box_w, 3))
            break
    return array


def augment_image(
    image: Image.Image,
    config: AugmentConfig | str,
    rng: np.random.Generator,
) -> Image.Image:
    """Return a randomly augmented RGB copy of ``image`` (same size)."""
    config = get_preset(config)
    image = image.convert("RGB")
    if config.hflip and rng.random() < config.hflip:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    if config.vflip and rng.random() < config.vflip:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    image = _rotate_zoom_translate(image, config, rng)

    # Colour jitter in random order, like torchvision's ColorJitter.
    operations = []
    if config.brightness:
        operations.append(lambda im: ImageEnhance.Brightness(im).enhance(rng.uniform(1 - config.brightness, 1 + config.brightness)))
    if config.contrast:
        operations.append(lambda im: ImageEnhance.Contrast(im).enhance(rng.uniform(1 - config.contrast, 1 + config.contrast)))
    if config.saturation:
        operations.append(lambda im: ImageEnhance.Color(im).enhance(rng.uniform(1 - config.saturation, 1 + config.saturation)))
    if config.hue:
        operations.append(lambda im: _shift_hue(im, rng.uniform(-config.hue, config.hue)))
    for index in rng.permutation(len(operations)):
        image = operations[index](image)

    if config.blur and rng.random() < config.blur:
        image = image.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(0.3, 1.2))))

    if config.noise or config.erasing:
        array = np.asarray(image, dtype=np.float32).copy()
        if config.noise:
            array += rng.normal(0.0, rng.uniform(0.0, config.noise), size=array.shape)
        array = np.clip(array, 0, 255)
        if config.erasing and rng.random() < config.erasing:
            array = _random_erase(array, rng)
        image = Image.fromarray(array.round().astype(np.uint8))
    return image


def limit_size(image: Image.Image, max_side: int | None) -> Image.Image:
    """Downscale so the longer side is at most ``max_side`` (never upscales)."""
    if not max_side or max(image.size) <= max_side:
        return image
    scale = max_side / max(image.size)
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, resample=Image.BOX)


# --------------------------------------------------------------------------
# Keras (on the fly)
# --------------------------------------------------------------------------


def keras_augmentation(tf, config: AugmentConfig | str, value_range=(0.0, 255.0)):
    """Keras ``Sequential`` of preprocessing layers implementing ``config``.

    Blur and pixel noise are offline-only (``augment_image``): Keras 3 has no
    layer that matches them on a 0..255 input.

    Layers missing from the installed Keras (legacy ``tf_keras`` used by
    EfficientNet-Lite0 has no hue/saturation/erasing layers) are skipped and
    listed in the model's ``skipped_layers`` attribute.
    """
    config = get_preset(config)
    layers = tf.keras.layers
    stack, skipped = [], []

    def add(name: str, *args, **kwargs) -> None:
        layer = getattr(layers, name, None)
        if layer is None:
            skipped.append(name)
            return
        try:
            stack.append(layer(*args, **kwargs))
        except TypeError:
            # Older signatures lack value_range.
            kwargs.pop("value_range", None)
            stack.append(layer(*args, **kwargs))

    if config.hflip and config.vflip:
        add("RandomFlip", "horizontal_and_vertical")
    elif config.hflip:
        add("RandomFlip", "horizontal")
    elif config.vflip:
        add("RandomFlip", "vertical")
    if config.rotate_degrees:
        add("RandomRotation", config.rotate_degrees / 360.0, fill_mode="reflect")
    if config.zoom:
        add("RandomZoom", (-config.zoom, 0.0), fill_mode="reflect")
    if config.translate:
        add("RandomTranslation", config.translate, config.translate, fill_mode="reflect")
    if config.brightness:
        add("RandomBrightness", config.brightness, value_range=value_range)
    if config.contrast:
        add("RandomContrast", config.contrast)
    if config.saturation:
        add("RandomSaturation", (max(0.0, 0.5 - config.saturation / 2), min(1.0, 0.5 + config.saturation / 2)), value_range=value_range)
    if config.hue:
        add("RandomHue", config.hue * 2, value_range=value_range)
    if config.erasing:
        add("RandomErasing", factor=config.erasing, scale=(0.02, 0.15), value_range=value_range)

    model = tf.keras.Sequential(stack, name=f"augment_{config.name}")
    model.skipped_layers = skipped
    return model


def sample_beta(tf, alpha: float):
    first = tf.random.gamma([], alpha)
    second = tf.random.gamma([], alpha)
    return first / (first + second)


def mixup(tf, images, labels, alpha: float):
    lam = sample_beta(tf, alpha)
    index = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    images = lam * images + (1.0 - lam) * tf.gather(images, index)
    labels = lam * labels + (1.0 - lam) * tf.gather(labels, index)
    return images, labels


def cutmix(tf, images, labels, alpha: float):
    lam = sample_beta(tf, alpha)
    index = tf.random.shuffle(tf.range(tf.shape(images)[0]))
    height = tf.shape(images)[1]
    width = tf.shape(images)[2]
    cut = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(height, tf.float32) * cut, tf.int32)
    cut_w = tf.cast(tf.cast(width, tf.float32) * cut, tf.int32)
    cy = tf.random.uniform([], 0, height, dtype=tf.int32)
    cx = tf.random.uniform([], 0, width, dtype=tf.int32)
    top, bottom = tf.maximum(cy - cut_h // 2, 0), tf.minimum(cy + cut_h // 2, height)
    left, right = tf.maximum(cx - cut_w // 2, 0), tf.minimum(cx + cut_w // 2, width)
    rows = tf.range(height)[:, None]
    cols = tf.range(width)[None, :]
    box = (rows >= top) & (rows < bottom) & (cols >= left) & (cols < right)
    mask = tf.cast(box, images.dtype)[None, :, :, None]
    images = images * (1.0 - mask) + tf.gather(images, index) * mask
    area = tf.cast((bottom - top) * (right - left), tf.float32) / tf.cast(height * width, tf.float32)
    labels = (1.0 - area) * labels + area * tf.gather(labels, index)
    return images, labels


def mix_batch(tf, images, labels, mixup_alpha: float = 0.0, cutmix_alpha: float = 0.0):
    """Apply MixUp or CutMix (50/50 when both are enabled) to one-hot labels."""
    if mixup_alpha > 0 and cutmix_alpha > 0:
        return tf.cond(
            tf.random.uniform([]) < 0.5,
            lambda: mixup(tf, images, labels, mixup_alpha),
            lambda: cutmix(tf, images, labels, cutmix_alpha),
        )
    if mixup_alpha > 0:
        return mixup(tf, images, labels, mixup_alpha)
    if cutmix_alpha > 0:
        return cutmix(tf, images, labels, cutmix_alpha)
    return images, labels


# --------------------------------------------------------------------------
# Ultralytics YOLO-cls
# --------------------------------------------------------------------------


# Ultralytics classification transforms use flips, HSV jitter, auto_augment
# and random erasing; rotation/translation are detection-only options. The
# "medium" row equals the Ultralytics defaults that trained the app's model.
YOLO_PRESETS: dict[str, dict] = {
    "none": {"fliplr": 0.0, "flipud": 0.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0, "erasing": 0.0, "auto_augment": None},
    "light": {"fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.01, "hsv_s": 0.3, "hsv_v": 0.2, "erasing": 0.0, "auto_augment": None},
    "medium": {"fliplr": 0.5, "flipud": 0.0, "hsv_h": 0.015, "hsv_s": 0.7, "hsv_v": 0.4, "erasing": 0.4, "auto_augment": "randaugment"},
    "strong": {"fliplr": 0.5, "flipud": 0.2, "hsv_h": 0.03, "hsv_s": 0.8, "hsv_v": 0.5, "erasing": 0.6, "auto_augment": "randaugment"},
}


def yolo_hyperparameters(preset: str) -> dict:
    """Ultralytics ``train()`` augmentation arguments for a ``-cls`` run."""
    if preset not in YOLO_PRESETS:
        raise ValueError(f"Unknown augmentation preset {preset!r}; expected one of {sorted(YOLO_PRESETS)}")
    return dict(YOLO_PRESETS[preset])


def with_overrides(config: AugmentConfig | str, **overrides) -> AugmentConfig:
    config = get_preset(config)
    values = {key: value for key, value in overrides.items() if value is not None}
    return replace(config, **values) if values else config
