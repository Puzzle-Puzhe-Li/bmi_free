"""
video.py
========

Utilities for pulling individual frames out of session/camera videos and
plotting them in a grid, with optional pose-keypoint / skeleton overlays.

Main entry point: `plot_frame_grid`.

--------------------------------------------------------------------------
Keypoint dataframe format
--------------------------------------------------------------------------
`kpt_df` is expected to have:
  - a MultiIndex on rows: (session, sn, frame_id)
  - a MultiIndex on columns: (joint, 'x' / 'y')

Its coordinates are assumed to have been tracked on the RAW frame after it
was downsampled so that its width is 720px (aspect ratio preserved), i.e.
a 1280x720 raw frame was tracked at ~720x405. This module rescales those
coordinates back to full raw-frame resolution before rotating / cropping.

--------------------------------------------------------------------------
Example usage
--------------------------------------------------------------------------
    from video import plot_frame_grid

    plot_frame_grid(
        session="A0",
        rows=[100, 200, 300],
        cols=["BO", "LO", "RO"],          # locations -> resolved to sn automatically
        df=df,
        scatter_kwargs=dict(s=15, c="cyan", edgecolors="k", linewidths=0.3),
        line_kwargs=dict(color="lime", linewidth=1.2),
        share_lim=True,
    )
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Sequence, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bmi_free.io import path, import_camera_info, import_skeleton_config

# Coordinates are assumed to have been tracked on frames downsampled to
# this width (aspect ratio preserved).
TRACKED_WIDTH = 720


# ===========================================================================
# Video reading
# ===========================================================================
class _VideoCache:
    """Keeps cv2.VideoCapture handles open across a grid of (session, sn)
    reads so we don't reopen the same video file for every frame."""

    def __init__(self):
        self._caps: dict[str, cv2.VideoCapture] = {}

    def _get_cap(self, video_path: Union[str, Path]) -> cv2.VideoCapture:
        key = str(video_path)
        cap = self._caps.get(key)
        if cap is None:
            cap = cv2.VideoCapture(key)
            if not cap.isOpened():
                raise FileNotFoundError(f"Could not open video: {video_path}")
            self._caps[key] = cap
        return cap

    def read_frame(self, video_path: Union[str, Path], frame_id: int) -> np.ndarray:
        cap = self._get_cap(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Could not read frame {frame_id} from {video_path}")
        return frame  # BGR, shape (H, W, 3)

    def close(self):
        for cap in self._caps.values():
            cap.release()
        self._caps.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _video_path(raw_root: Union[str, Path], session, sn) -> Path:
    return Path(raw_root) / str(session) / f"{sn}.mp4"


# ===========================================================================
# Image processing: brighten / rotate
# ===========================================================================
def _brighten(img_bgr: np.ndarray, brighten: Optional[Union[float, tuple]]) -> np.ndarray:
    """Brighten a BGR image. `brighten` can be:
      - None: no-op
      - float: multiplicative gain (alpha), e.g. 1.3
      - (alpha, beta): multiplicative gain + additive offset
    """
    if brighten is None:
        return img_bgr
    if isinstance(brighten, (tuple, list)):
        alpha, beta = brighten
    else:
        alpha, beta = float(brighten), 0.0
    return cv2.convertScaleAbs(img_bgr, alpha=alpha, beta=beta)


_ROTATE_CODE_BY_ANGLE = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def _rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    """Rotate image clockwise by `angle` degrees (0/90/180/270 use fast
    cv2.rotate; any other angle falls back to a general affine warp)."""
    angle = int(angle) % 360
    code = _ROTATE_CODE_BY_ANGLE.get(angle, "not_multiple_of_90")
    if angle == 0:
        return img
    if code != "not_multiple_of_90":
        return cv2.rotate(img, code)
    # General fallback for non-multiple-of-90 angles.
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)  # cv2 angle is CCW-positive
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(img, M, (new_w, new_h))


def _rotate_point(x: float, y: float, w: int, h: int, angle: int) -> tuple[float, float]:
    """Map a point (x, y) in a WxH image to its location after the image is
    rotated clockwise by `angle` degrees (0/90/180/270)."""
    angle = int(angle) % 360
    if angle == 0:
        return x, y
    if angle == 90:
        return h - 1 - y, x
    if angle == 180:
        return w - 1 - x, h - 1 - y
    if angle == 270:
        return y, w - 1 - x
    # General fallback, matching _rotate_image's warpAffine above.
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    new_w = h * sin + w * cos
    new_h = h * cos + w * sin
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    pt = M @ np.array([x, y, 1.0])
    return float(pt[0]), float(pt[1])


# ===========================================================================
# Axis (rows/cols) classification: sn vs. location vs. frame_id
# ===========================================================================
def _classify_and_resolve(values: Sequence, df_camera: pd.DataFrame) -> tuple[str, list]:
    """Given a list of row/col values, work out whether they're camera
    serial numbers, camera locations (resolved to sn), or frame ids.

    Returns (kind, resolved_values) where kind is 'sn' or 'frame_id'.
    """
    values = list(values)
    sn_set = set(df_camera["sn"].astype(str))
    loc_lookup = {
        str(loc).upper(): sn
        for loc, sn in zip(df_camera["location"], df_camera["sn"])
    }

    resolved = []
    kinds = set()
    for v in values:
        vs = str(v)
        if vs in sn_set:
            resolved.append(vs)
            kinds.add("sn")
        elif vs.upper() in loc_lookup:
            resolved.append(loc_lookup[vs.upper()])
            kinds.add("sn")
        elif isinstance(v, (int, np.integer)) or (
            isinstance(v, str) and v.lstrip("-").isdigit()
        ):
            resolved.append(int(v))
            kinds.add("frame_id")
        else:
            raise ValueError(
                f"Could not classify axis value {v!r} as a camera sn, a camera "
                f"location, or a frame_id. Known sn: {sorted(sn_set)}. "
                f"Known locations: {sorted(loc_lookup)}."
            )

    if len(kinds) > 1:
        raise ValueError(
            f"Axis values are a mix of {kinds} — each of `rows`/`cols` must be "
            "entirely camera sn/location, or entirely frame_id, not a mix."
        )
    return kinds.pop(), resolved


def _resolve_axes(rows, cols, df_camera):
    """Figure out which of rows/cols is the sn/location axis and which is
    the frame_id axis, and resolve locations -> sn along the way."""
    row_kind, row_vals = _classify_and_resolve(rows, df_camera)
    col_kind, col_vals = _classify_and_resolve(cols, df_camera)

    if row_kind == col_kind:
        raise ValueError(
            f"Both `rows` and `cols` were classified as '{row_kind}'. One of "
            "them must be camera sn/location and the other must be frame_id."
        )
    return (row_kind, row_vals), (col_kind, col_vals)

def _resolve_axes_matrix(rows, cols, df_camera):
    """Matrix mode: `rows` is a 2D array of shape (n_rows, len(cols)),
    giving a per-cell frame_id for each (row, col). `cols` must be
    entirely sn/location (frame_id varies per cell, not per row alone).
    Returns (row_kind, row_vals), (col_kind, col_vals) in the same shape
    as _resolve_axes, except row_vals is the raw 2D frame_id array.
    """
    col_kind, col_vals = _classify_and_resolve(cols, df_camera)
    if col_kind != "sn":
        raise ValueError(
            "In matrix mode (rows is a 2D array), `cols` must be camera "
            f"sn/location, got '{col_kind}'."
        )
    rows = np.asarray(rows)
    if rows.ndim != 2 or rows.shape[1] != len(col_vals):
        raise ValueError(
            f"`rows` matrix must have shape (n_rows, len(cols)="
            f"{len(col_vals)}), got {rows.shape}."
        )
    return ("frame_id", rows), (col_kind, col_vals)

# ===========================================================================
# Keypoints
# ===========================================================================
def _get_raw_keypoints(kpt_df: Optional[pd.DataFrame], session, sn, frame_id, threshold: float = 0.7) -> Optional[dict]:
    """Pull out {joint_name: (x, y)} for one (session, sn, frame_id) row,
    in the ORIGINAL tracked (downsampled-to-720-wide) coordinate space."""
    if kpt_df is None:
        return None
    key = (session, sn, frame_id)
    try:
        row = kpt_df.loc[key]
    except KeyError:
        return None
    joints = row.index.get_level_values(0).unique()
    xy = {}
    for j in joints:
        try:
            x = float(row[(j, "x")])
            y = float(row[(j, "y")])
        except KeyError:
            continue
        if not (np.isnan(x) or np.isnan(y)):
            if (j, 'likelihood') in row.index:
                if row[(j, "likelihood")] > threshold:
                    xy[j] = (x, y)
            else:
                xy[j] = (x, y)
    return xy if xy else None


def _transform_keypoints(xy: dict, scale: float, raw_w: int, raw_h: int, angle: int) -> dict:
    """Rescale tracked-space keypoints to raw-frame pixels, then rotate them
    to match the rotated image."""
    out = {}
    for joint, (x, y) in xy.items():
        rx, ry = x * scale, y * scale
        rx, ry = _rotate_point(rx, ry, raw_w, raw_h, angle)
        out[joint] = (rx, ry)
    return out


# ===========================================================================
# Cropping
# ===========================================================================
def _bbox_from_keypoints(xy_list: Sequence[dict], margin_ratio: float = 0.3):
    """Union bounding box (x, y, w, h) over one or more {joint: (x,y)} dicts,
    expanded by `margin_ratio` and made square (side = max(w, h))."""
    xs, ys = [], []
    for xy in xy_list:
        if not xy:
            continue
        for x, y in xy.values():
            xs.append(x)
            ys.append(y)
    if not xs:
        return None
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    w, h = x1 - x0, y1 - y0
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(w, h) * (1 + margin_ratio)
    side = max(side, 1.0)
    return (cx - side / 2, cy - side / 2, side, side)


def _normalize_crop_box(crop) -> tuple[float, float, float, float]:
    """Accept (x, y, w) [h=w] or (x, y, w, h) and return (x, y, w, h)."""
    if len(crop) == 3:
        x, y, w = crop
        return (x, y, w, w)
    if len(crop) == 4:
        return tuple(crop)
    raise ValueError(f"`crop` must have 3 or 4 elements, got {crop!r}")

def _clamp_box_to_image(box, img_w, img_h):
    """Shift (and if necessary shrink) a crop box so it lies entirely
    within an img_w x img_h image, instead of letting it hang off the
    edge (which _apply_crop would otherwise pad with black)."""
    x, y, w, h = box
    w = min(w, img_w)
    h = min(h, img_h)
    x = min(max(x, 0), img_w - w)
    y = min(max(y, 0), img_h - h)
    return (x, y, w, h)

def _apply_crop(img: np.ndarray, xy: Optional[dict], box: tuple[float, float, float, float]):
    """Crop `img` to `box` = (x, y, w, h) in pixel coords, padding with
    black if the box extends beyond the image, and shift keypoints to be
    relative to the crop. Returns (cropped_img, shifted_xy)."""
    x, y, w, h = box
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    out_w = max(1, round(w))
    out_h = max(1, round(h))
    x1 = x0 + out_w
    y1 = y0 + out_h
    img_h, img_w = img.shape[:2]

    src_x0, src_y0 = max(x0, 0), max(y0, 0)
    src_x1, src_y1 = min(x1, img_w), min(y1, img_h)

    canvas = np.zeros((out_h, out_w, img.shape[2]), dtype=img.dtype)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0, dst_y0 = src_x0 - x0, src_y0 - y0
        patch = img[src_y0:src_y1, src_x0:src_x1]
        canvas[dst_y0:dst_y0 + patch.shape[0], dst_x0:dst_x0 + patch.shape[1]] = patch

    shifted = None
    if xy is not None:
        shifted = {j: (px - x0, py - y0) for j, (px, py) in xy.items()}
    return canvas, shifted


# ===========================================================================
# Main entry point
# ===========================================================================
def plot_frame_grid(
    session,
    rows: Sequence,
    cols: Sequence,
    df: Optional[pd.DataFrame] = None,
    edges: Optional[Sequence[tuple]] = None,
    threshold: float = 0.7,
    df_camera: Optional[pd.DataFrame] = None,
    raw_root: Optional[Union[str, Path]] = None,
    crop: Optional[Union[tuple, None]] = None,
    share_lim: bool = False,
    crop_margin_ratio: float = 0.3,
    brighten: Optional[Union[float, tuple]] = 1.3,
    scatter_kwargs: Optional[dict] = None,
    line_kwargs: Optional[dict] = None,
    subplot_size: float = 0.5,
    titles: bool = False,
    dpi: int = 300,
    overlay: Optional[Union[slice, Sequence, np.ndarray]] = None,

):
    """Plot a grid of video frames, optionally overlaid with pose keypoints
    and skeleton edges.

    Parameters
    ----------
    session : the session id shared by every cell in the grid.
    rows, cols : sequences. Exactly one of them must be camera identifiers
        (sn strings, e.g. "213622077617", OR locations, e.g. "BO" — mixing
        sn and location within one axis is fine since both resolve to sn,
        but you can't mix sn/location with frame_id on the same axis).
        The other must be a sequence of frame_id ints.
    df : optional DataFrame, index (session, sn, frame_id), columns
        MultiIndex (joint, 'x'/'y'), in tracked (720-wide-downsampled)
        coordinates. If None, no keypoints/edges are drawn and cropping
        (unless `crop` is given explicitly) is disabled.
    edges : optional list of (joint_a, joint_b) tuples to draw as skeleton
        lines. Defaults to `import_skeleton_config()` if not given.
    df_camera : optional DataFrame with columns sn/location/rotate.
        Defaults to `import_camera_info()` if not given.
    raw_root : folder containing `{session}/{sn}.mp4`. Defaults to
        `path['raw']` if not given.
    crop : None, (x, y, w), or (x, y, w, h) in RAW (post-rotation) pixel
        coordinates.
        - None + no keypoints -> no cropping, full rotated frame is shown.
        - None + keypoints present -> auto-crop to a square box around
          that cell's keypoints (with `crop_margin_ratio` padding).
        - explicit tuple -> that exact box is used for every cell.
    share_lim : if True and `crop` is None, compute ONE auto-crop box from
        the union of keypoints across the *entire* grid and use it,
        identically, for every cell (instead of per-cell auto-crop).
    crop_margin_ratio : float
        当自动裁剪时，关键点边界框向外扩展的比例，默认 0.3。
    brighten : float or tuple, optional
        cv2.convertScaleAbs. None to disable.
    scatter_kwargs : kwargs forwarded to ax.scatter() for joints.
    line_kwargs : kwargs forwarded to ax.plot() for skeleton edges.
    subplot_size : inches, per-cell size used to compute the figure size.
    titles : if True, label each cell with its (location/sn, frame_id).
    dpi : figure dpi.
    overlay : controls which subplots show keypoint/skeleton overlays.
        - None: all subplots show overlays (default).
        - slice: e.g., slice(0, 3) applies to rows (if rows is the frame axis)
          or cols (if cols is the frame axis).
        - Sequence of ints/bools: applied to the frame axis (either rows or cols).
        - 2D np.ndarray with shape (n_rows, n_cols): boolean mask.
        - 1D np.ndarray with shape (n,) where n is the length of the frame axis.
        For 1D indexing, the index is applied to the axis that contains frame_ids.
        For 2D boolean arrays, shape must match (len(rows), len(cols)).
    Returns
    -------
    (fig, axes)
    """
    if df_camera is None:
        df_camera = import_camera_info()
    if edges is None and df is not None:
        edges = import_skeleton_config()
    if raw_root is None:
        if path is None:
            raise ValueError(
                "raw_root was not given and no project-level `path` object is "
                "available in video.py. Pass raw_root=... explicitly."
            )
        raw_root = path["raw"]

    scatter_kwargs = dict(s=18, c="deepskyblue", edgecolors="black", linewidths=0.3, zorder=3) | (scatter_kwargs or {})
    line_kwargs = dict(color="lime", linewidth=1.2, zorder=2) | (line_kwargs or {})

    if isinstance(rows, np.ndarray) and rows.ndim == 2:
        (row_kind, row_vals), (col_kind, col_vals) = _resolve_axes_matrix(rows, cols, df_camera)
    else:
        (row_kind, row_vals), (col_kind, col_vals) = _resolve_axes(rows, cols, df_camera)
    sn_axis, sn_vals = (row_kind, row_vals) if row_kind == "sn" else (col_kind, col_vals)

    # frame_axis, frame_vals = (row_kind, row_vals) if row_kind == "frame_id" else (col_kind, col_vals)
    # sn_axis, sn_vals = (row_kind, row_vals) if row_kind == "sn" else (col_kind, col_vals)
    # frame_axis, frame_vals = (row_kind, row_vals) if row_kind == "frame_id" else (col_kind, col_vals)

    # Build the (row_val, col_val) -> (sn, frame_id) grid, preserving the
    # user-specified rows/cols order.
    matrix_mode = isinstance(row_vals, np.ndarray) and row_vals.ndim == 2

    def cell_sn_frame(r_idx, c_idx, row_val, col_val):
        if matrix_mode:
            # cols is always the sn axis in matrix mode
            return col_val, row_vals[r_idx, c_idx]
        return (row_val, col_val) if row_kind == "sn" else (col_val, row_val)

    rotate_by_sn = dict(zip(df_camera["sn"].astype(str), df_camera["rotate"]))
    loc_by_sn = dict(zip(df_camera["sn"].astype(str), df_camera["location"]))

    n_rows = row_vals.shape[0] if matrix_mode else len(row_vals)
    n_cols = len(col_vals)

    # ------------------------------------------------------------------
    # Pass 1: read + rotate + brighten every frame, compute raw-space
    # (rotated) keypoints for every cell. We need this up-front so that
    # share_lim can pool keypoints across the whole grid before cropping.
    # ------------------------------------------------------------------
    cells = {}  # (r_idx, c_idx) -> dict(img=..., xy=..., sn=..., frame_id=..., angle=...)
    with _VideoCache() as cache:
        for r_idx in range(n_rows):
            row_val = row_vals[r_idx] if not matrix_mode else None
            for c_idx, col_val in enumerate(col_vals):
                sn, frame_id = cell_sn_frame(r_idx, c_idx, row_val, col_val)
                angle = int(rotate_by_sn.get(sn, 0))
                video_path = _video_path(raw_root, session, sn)
 
                raw_img = cache.read_frame(video_path, frame_id)
                raw_h, raw_w = raw_img.shape[:2]
 
                img = _brighten(raw_img, brighten)
                img = _rotate_image(img, angle)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
                xy_tracked = _get_raw_keypoints(df, session, sn, frame_id, threshold)
                xy_rot = None
                if xy_tracked is not None:
                    scale = raw_w / TRACKED_WIDTH
                    xy_rot = _transform_keypoints(xy_tracked, scale, raw_w, raw_h, angle)
 
                cells[(r_idx, c_idx)] = dict(
                    img=img, xy=xy_rot, sn=sn, frame_id=frame_id, angle=angle,
                )
 
    # ------------------------------------------------------------------
    # Work out the crop box(es).
    #
    # share_lim groups cells by sn (camera) and shares one box across the
    # different frame_ids of that sn -- NOT across different sn. Different
    # cameras generally have different resolutions/fields of view after
    # rotation, so sharing one box across sn tends to produce a bad crop
    # for at least some of them.
    # ------------------------------------------------------------------
    fixed_box = None
    box_by_sn: dict = {}
    if crop is not None:
        fixed_box = _normalize_crop_box(crop)
    elif share_lim:
        xy_by_sn: dict = {}
        for c in cells.values():
            if c["xy"] is not None:
                xy_by_sn.setdefault(c["sn"], []).append(c["xy"])
        for sn, xy_list in xy_by_sn.items():
            box_by_sn[sn] = _bbox_from_keypoints(xy_list, margin_ratio=crop_margin_ratio)
        if not box_by_sn:
            warnings.warn("share_lim=True but no keypoints were found anywhere in "
                           "the grid; showing full frames instead.")
 
    # ------------------------------------------------------------------
    # Pass 2: crop (per-cell auto-crop if needed) and plot.
    # ------------------------------------------------------------------
    fig_w = subplot_size * n_cols
    fig_h = subplot_size * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), 
                             dpi=dpi, squeeze=False, gridspec_kw=dict(hspace=0.05, wspace=0.05))
 
    for r_idx in range(n_rows):
        row_val = row_vals[r_idx] if not matrix_mode else None
        for c_idx, col_val in enumerate(col_vals):
            ax = axes[r_idx][c_idx]
            cell = cells[(r_idx, c_idx)]
            img, xy = cell["img"], cell["xy"]
 
            if fixed_box is not None:
                box = fixed_box
            elif share_lim:
                box = box_by_sn.get(cell["sn"]) 
            else:
                box = None
            if box is None and xy is not None:
                box = _bbox_from_keypoints([xy], margin_ratio=crop_margin_ratio)
 
            if box is not None:
                box = _clamp_box_to_image(box, img.shape[1], img.shape[0])
                img, xy = _apply_crop(img, xy, box)
 
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
 
            if xy:
                if edges:
                    for j1, j2 in edges:
                        if j1 in xy and j2 in xy:
                            (x1, y1), (x2, y2) = xy[j1], xy[j2]
                            ax.plot([x1, x2], [y1, y2], **line_kwargs)
                xs = [p[0] for p in xy.values()]
                ys = [p[1] for p in xy.values()]
                ax.scatter(xs, ys, **scatter_kwargs)
 
            if titles:
                sn = cell["sn"]
                loc = loc_by_sn.get(sn, sn)
                ax.set_title(f"{loc} | frame {cell['frame_id']}", fontsize=9)
 
    fig.tight_layout()
    return fig, axes