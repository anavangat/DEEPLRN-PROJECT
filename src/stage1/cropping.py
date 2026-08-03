def yolo_to_xyxy(xc, yc, bw, bh, img_w, img_h):
    """Normalized YOLO cx,cy,w,h -> absolute pixel x1,y1,x2,y2 (floats)."""
    x1 = (xc - bw / 2.0) * img_w
    y1 = (yc - bh / 2.0) * img_h
    x2 = (xc + bw / 2.0) * img_w
    y2 = (yc + bh / 2.0) * img_h
    return x1, y1, x2, y2


def pad_and_clip(x1, y1, x2, y2, img_w, img_h, pad=0.10):
    """Expand a box by `pad` on EACH side (fraction of that box's own w/h),
    then clip to the image. Returns integer pixel coords with x2>x1, y2>y1."""
    bw = x2 - x1
    bh = y2 - y1
    x1 -= bw * pad
    x2 += bw * pad
    y1 -= bh * pad
    y2 += bh * pad

    x1 = max(0, int(round(x1)))
    y1 = max(0, int(round(y1)))
    x2 = min(img_w, int(round(x2)))
    y2 = min(img_h, int(round(y2)))

    # Degenerate boxes (tiny or fully off-image) would crash PIL.crop downstream.
    if x2 <= x1:
        x1, x2 = max(0, min(x1, img_w - 1)), min(img_w, max(x1 + 1, 1))
    if y2 <= y1:
        y1, y2 = max(0, min(y1, img_h - 1)), min(img_h, max(y1 + 1, 1))
    return x1, y1, x2, y2


def iou_xyxy(a, b):
    """Intersection over union of two absolute boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0
