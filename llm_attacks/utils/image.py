from .typing import _pair, _size_2_t

CM_PER_INCH = 2.54


def compute_image_shape(
    inch: _size_2_t, dpi: _size_2_t, pixel_ori: _size_2_t = None, is_cm=False
):
    """Compute image shape with physical size and dpi.

    Args:
        inch (tuple[float, float] | float):
            Size in inch. At most one edge could be None.
        dpi (tuple[int, int] | int):
            Resolution per inch.
        pixel_ori (tuple[int, int] | int | None):
            Original pixel size to provide ratio. Default: None.
        is_cm (bool):
            True if the `inch` parameter accepts a input in centimeter.

    Returns:
        tuple[int, int]: pixel size of new image, which has the expected physical
        size with the input dpi.
    """
    h_ori, w_ori = _pair(pixel_ori)
    h_inch, w_inch = _pair(inch)
    dpi = _pair(dpi)
    if w_inch is None:
        w_inch = w_ori / h_ori * h_inch
    if h_inch is None:
        h_inch = h_ori / w_ori * w_inch
    if is_cm:
        w_inch = w_inch / CM_PER_INCH
        h_inch = h_inch / CM_PER_INCH
    return int(h_inch * dpi[0]), int(w_inch * dpi[1])
