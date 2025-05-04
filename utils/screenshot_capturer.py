# utils/screenshot_capturer.py
import mss, mss.tools, pathlib
from datetime import datetime
import PIL.Image as Image


CAPTURE_DIR = pathlib.Path("captures")
CAPTURE_DIR.mkdir(exist_ok=True)

def capture() -> str:
    """
    Grab the primary screen, save as PNG, and return the file path.
    Loss-less PNG keeps text crisp for OCR if the profile uses it.
    """
    fname = CAPTURE_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    with mss.mss() as sct:
        img = sct.grab(sct.monitors[1])
        mss.tools.to_png(img.rgb, img.size, output=str(fname))
        im = Image.open(fname)
        im.thumbnail((1920, 1920))
        im.save(fname)
    return str(fname)
