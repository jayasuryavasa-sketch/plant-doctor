from pathlib import Path
from PIL import Image, ImageStat, UnidentifiedImageError


class ImageValidationError(ValueError):
    pass


def validate_image(path: Path, max_bytes: int) -> None:
    if path.stat().st_size > max_bytes:
        raise ImageValidationError("That image is too large. Please choose one under 8 MB.")
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image = image.convert("RGB")
            width, height = image.size
            if width < 96 or height < 96:
                raise ImageValidationError("This image is too small. Please use a clear, close photo of the leaf.")
            thumb = image.resize((64, 64))
            brightness = sum(ImageStat.Stat(thumb).mean) / 3
            if brightness < 12:
                raise ImageValidationError("This photo is very dark. Please take it in brighter, even light.")
    except UnidentifiedImageError as exc:
        raise ImageValidationError("We couldn't read this image. Please upload a valid photo file.") from exc
