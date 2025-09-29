import pathlib
import re

import pymupdf
from loguru import logger

from utils import get_files, get_sample_path


def to_page(limit: int = 5):
    files = get_files(get_sample_path(), ".pdf")

    def strip(value: str) -> str:
        return re.sub(r"\\", "", value)

    for file_path, file_name in files:
        logger.info(f"Extracting {file_name}")
        file_name = strip(file_name)
        output_path = pathlib.Path(pathlib.Path(file_path).parent, file_name)

        if pathlib.Path.exists(output_path):
            logger.info("Skipping")
            continue
        pathlib.Path.mkdir(output_path)

        doc = pymupdf.open(file_path)
        for page in doc:
            label = page.get_label()
            name = strip(f"{page.number}-{label if label else 'page'}")
            logger.info(f"Saving {name}")
            pix = page.get_pixmap()
            pix.save(output_path / f"{name}.png")

    logger.info("Extraction complete")


if __name__ == "__main__":
    to_page()
