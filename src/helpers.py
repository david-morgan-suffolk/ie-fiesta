import torch
from PIL import ImageDraw, ImageFont
from PIL.Image import Image
from transformers import (
    BatchEncoding,
    LayoutLMv2Tokenizer,
)

# Types, lol
Encoding = dict[str, torch.Tensor]


def plot_results(
    pil_img: Image,
    encoding: BatchEncoding,
    original_size: tuple[int, int],
    tokenizer: LayoutLMv2Tokenizer | None = None,
) -> Image:
    """
    Draws bounding boxes from the model's encoding onto the original image.

    Args:
        pil_img: The original PIL Image object.
        encoding: The dictionary of tensors returned by the LayoutLMv2Processor.
        original_size: A tuple containing the (width, height) of the original image.

    Returns:
        A new PIL Image object with the bounding boxes and tokens drawn on it.
    """
    original_width, original_height = original_size
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    # The processor normalizes the bbox to a 1000x1000 space
    boxes: list[list[int]] = encoding["bbox"].squeeze().tolist()  # pyright: ignore[reportAttributeAccessIssue]
    token_ids: list[int] = encoding["input_ids"].squeeze().tolist()  # pyright: ignore[reportAttributeAccessIssue]

    # Get the tokenizer to decode token IDs back to text
    if tokenizer is None:
        tokenizer = LayoutLMv2Tokenizer.from_pretrained(
            "microsoft/layoutlmv2-base-uncased"
        )

    if tokenizer is None:
        raise ValueError("No tokenizer found")

    for box, token_id in zip(boxes, token_ids, strict=False):
        # Skip special tokens like [CLS], [SEP], [PAD]
        if token_id in [
            tokenizer.cls_token_id,
            tokenizer.sep_token_id,
            tokenizer.pad_token_id,
        ]:
            continue

        # Denormalize the bounding box coordinates
        x0, y0, x1, y1 = box
        scaled_box: list[float] = [
            (x0 / 1000) * original_width,
            (y0 / 1000) * original_height,
            (x1 / 1000) * original_width,
            (y1 / 1000) * original_height,
        ]

        # Draw the rectangle and the decoded token
        draw.rectangle(scaled_box, outline="red", width=2)
        token_text = tokenizer.decode([token_id])

        # Add a small background for the text for better visibility
        text_bbox = draw.textbbox((scaled_box[0], scaled_box[1]), token_text, font=font)
        draw.rectangle(text_bbox, fill="red")
        draw.text((scaled_box[0], scaled_box[1]), token_text, fill="white", font=font)

    return pil_img
