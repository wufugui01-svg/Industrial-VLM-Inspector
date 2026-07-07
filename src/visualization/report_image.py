"""Create a visual inspection report without pixel-level localization."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from src.agent.schema import InspectionResult

PANEL_WIDTH = 480
PADDING = 28
MAX_IMAGE_SIDE = 1200


def _load_font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_width(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
) -> float:
    left, _, right, _ = draw.textbbox((0, 0), text, font=font)
    return right - left


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_width: int,
) -> list[str]:
    """Wrap English or unspaced text according to rendered pixel width."""

    wrapped: list[str] = []
    for paragraph in (text or "").splitlines() or [""]:
        use_spaces = bool(paragraph.split()) and " " in paragraph
        tokens = paragraph.split() if use_spaces else list(paragraph)
        separator = " " if use_spaces else ""
        current = ""
        for token in tokens:
            candidate = token if not current else f"{current}{separator}{token}"
            if current and _text_width(draw, candidate, font) > max_width:
                wrapped.append(current)
                current = token
            else:
                current = candidate
        wrapped.append(current)
    return wrapped or [""]


def _prepare_image(image_path: Path) -> Image.Image:
    with Image.open(image_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return image


def create_report_image(
    image_path: str,
    result: InspectionResult,
    output_path: str,
) -> str:
    """Save an original-image-plus-summary report and return its absolute path."""

    source_path = Path(image_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Inspection image does not exist: {source_path}")

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = _prepare_image(source_path)

    title_font = _load_font(28)
    status_font = _load_font(24)
    body_font = _load_font(20)
    small_font = _load_font(15)
    measuring_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    reason_lines = _wrap_text(
        measuring_draw,
        result.reason,
        body_font,
        PANEL_WIDTH - 2 * PADDING,
    )

    line_height = 30
    content_height = (
        PADDING
        + 42
        + 42
        + 20
        + 4 * line_height
        + 18
        + len(reason_lines) * line_height
        + 70
    )
    canvas_height = max(image.height, content_height)
    canvas = Image.new(
        "RGB",
        (image.width + PANEL_WIDTH, canvas_height),
        color=(247, 248, 250),
    )
    canvas.paste(image, (0, (canvas_height - image.height) // 2))

    panel_x = image.width
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (panel_x, 0, panel_x + PANEL_WIDTH, canvas_height),
        fill=(247, 248, 250),
    )
    accent = (190, 45, 45) if result.is_anomaly else (34, 139, 94)
    draw.rectangle((panel_x, 0, panel_x + 8, canvas_height), fill=accent)

    x = panel_x + PADDING
    y = PADDING
    draw.text((x, y), "Inspection Report", font=title_font, fill=(30, 35, 42))
    y += 46
    status = "ANOMALY DETECTED" if result.is_anomaly else "NO ANOMALY"
    draw.text((x, y), status, font=status_font, fill=accent)
    y += 46
    draw.line(
        (x, y, panel_x + PANEL_WIDTH - PADDING, y),
        fill=(205, 208, 214),
        width=2,
    )
    y += 18

    fields = (
        ("is_anomaly", str(result.is_anomaly).lower()),
        ("defect_type", result.defect_type),
        ("severity", result.severity),
        ("confidence", f"{result.confidence:.3f}"),
    )
    for label, value in fields:
        draw.text(
            (x, y),
            f"{label}:",
            font=body_font,
            fill=(75, 80, 88),
        )
        draw.text(
            (x + 150, y),
            value,
            font=body_font,
            fill=(25, 30, 36),
        )
        y += line_height

    y += 10
    draw.text((x, y), "reason:", font=body_font, fill=(75, 80, 88))
    y += line_height
    for line in reason_lines:
        draw.text((x, y), line, font=body_font, fill=(25, 30, 36))
        y += line_height

    footer_y = canvas_height - PADDING - 38
    draw.text(
        (x, footer_y),
        "Classification summary only.\nNo pixel-level localization is shown.",
        font=small_font,
        fill=(110, 115, 124),
        spacing=4,
    )

    save_format = "PNG" if not destination.suffix else None
    canvas.save(destination, format=save_format)
    return str(destination)

