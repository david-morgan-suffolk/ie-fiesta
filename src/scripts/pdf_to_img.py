import pathlib
import re

from utils import get_samples_path

Test_FileName = "AMAN MB - Fire Alarm.pdf"


def to_page(file: str = Test_FileName):
    import pymupdf

    location = get_samples_path() / "test"
    if pathlib.Path.exists(location) is False:
        pathlib.Path.mkdir(location)

    doc = pymupdf.open(location / file)
    for page in doc:
        pix = page.get_pixmap()
        pix.save(re.sub(r"\\", "", f"{location}/{page.number}-{page.get_label()}.png"))


if __name__ == "__main__":
    to_page()
