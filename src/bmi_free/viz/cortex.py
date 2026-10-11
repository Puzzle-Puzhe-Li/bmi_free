"""
cortex.py — Monkey cortical topomap visualization utilities
===========================================================
Three PDFs (cortex.pdf, electrodes.pdf, range.pdf) share one 800x1000-point
canvas:
  cortex.pdf     cortical outline
  electrodes.pdf electrode positions
  range.pdf      interpolation / masking region

Layering
--------
plot_cortical_frame : outline (+ optional electrodes, + optional labels)
plot_heatmap        : interpolated heatmap clipped to range.pdf
plot_topomap        : plot_cortical_frame + plot_heatmap (+ colorbar)

Public API
----------
plot_cortical_frame, plot_heatmap, plot_topomap, plot_topomap_grid,
cortical_subplots, build_montage, color_elec
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import fitz
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MplPath
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler

from scipy.stats import rankdata

from bmi_free.io import path as _data_path

_HERE = Path(__file__).parent
_ASSET_DIR = _HERE / "cortex"
_DEFAULT_CORTEX = _ASSET_DIR / "cortex.pdf"
_DEFAULT_ELECTRODES = _ASSET_DIR / "electrodes.pdf"
_DEFAULT_RANGE = _ASSET_DIR / "range.pdf"
_ELEC_LOC_PATH = _data_path["config"]

_SKIP_CHANNELS = {"L11", "R11"}
_PROJ_PITCH_DEG = -15.0

ColorSpec = Union[str, tuple, dict, list, np.ndarray]

def _ball_points(n, inner=0.0, seed=0):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    u = np.random.default_rng(seed).random(n)
    r = np.cbrt(inner ** 3 + (1 - inner ** 3) * u)
    return r[:, None] * np.column_stack([
        np.cos(theta) * np.sin(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(phi),
    ])


# ===========================================================================
# 1. Montage
# ===========================================================================

def build_montage(
    elec_xlsx: Union[str, Path, None] = None,
    elec_pdf: Union[str, Path, None] = None,
    pitch_deg: float = _PROJ_PITCH_DEG,
    skip_channels: set[str] = _SKIP_CHANNELS,
    version: str = "64_20250606",
) -> mne.channels.DigMontage:
    """Build an electrode montage aligned to the shared PDF canvas.

    Positions are pixel coordinates on the canvas (z = 0).  The canvas size
    and mean electrode radius are stored in the ``nasion`` slot as
    ``[W, H, elec_radius]``.  The result is cached as
    ``cortex/montage_{version}.fif``.
    """
    cache_path = _ASSET_DIR / f"montage_{version}.fif"
    if cache_path.exists():
        return mne.channels.read_dig_fif(str(cache_path))

    if elec_xlsx is None:
        elec_xlsx = _ELEC_LOC_PATH / f"elec_loc_{version}.xlsx"
    if elec_pdf is None:
        elec_pdf = _DEFAULT_ELECTRODES

    df_loc = pd.read_excel(str(elec_xlsx), index_col="label")

    loc_3d = df_loc[["r", "a", "s"]].values.astype(np.float64)
    theta = np.radians(pitch_deg)
    proj_r = loc_3d[:, 0]
    proj_y = loc_3d[:, 1] * np.cos(theta) - loc_3d[:, 2] * np.sin(theta)
    loc_2d = np.column_stack((proj_r, proj_y))

    page = fitz.open(str(elec_pdf))[0]
    H, W = page.rect.height, page.rect.width

    centers, radii = [], []
    for d in page.get_drawings():
        r = d["rect"]
        centers.append([(r.x0 + r.x1) / 2, (H - r.y0 + H - r.y1) / 2])
        radii.append(((r.x1 - r.x0) + (r.y1 - r.y0)) / 4)
    centers = np.array(centers)
    elec_radius = float(np.mean(radii))

    loc_norm = MinMaxScaler().fit_transform(loc_2d)
    centers_norm = MinMaxScaler().fit_transform(centers)
    dist = cdist(loc_norm, centers_norm, metric="euclidean")
    matched_idx = np.argmin(dist, axis=1)
    max_err = np.max(np.min(dist, axis=1))
    if max_err > 0.05:
        warnings.warn(
            f"build_montage: largest normalised matching error = {max_err:.4f}. "
            "Check that elec_xlsx and elec_pdf are consistent.",
            stacklevel=2,
        )
    reordered = centers[matched_idx]

    ch_pos = {
        name: np.array([*reordered[i], 0.0])
        for i, name in enumerate(df_loc.index.tolist())
        if name not in skip_channels
    }
    montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos,
        nasion=np.array([W, H, elec_radius]),
        lpa=np.zeros(3),
        rpa=np.zeros(3),
        coord_frame="unknown",
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(str(cache_path), overwrite=True)
    return montage


def _montage_info(montage, version):
    """Return (montage, W, H, R, {ch_name: xy})."""
    if montage is None:
        montage = build_montage(version=version)
    pos = montage.get_positions()
    W, H, R = (float(v) for v in pos["nasion"])
    ch_xy = {k: np.asarray(v, dtype=float)[:2] for k, v in pos["ch_pos"].items()}
    return montage, W, H, R, ch_xy


# ===========================================================================
# 2. Outline (+ electrodes, labels)
# ===========================================================================

def _resolve_colors(elec_color: ColorSpec, names: list[str]) -> np.ndarray:
    """Turn a colour spec into an (N, 4) or (1, 4) RGBA array."""
    if isinstance(elec_color, dict):
        return mcolors.to_rgba_array([elec_color.get(n, "k") for n in names])
    if mcolors.is_color_like(elec_color):
        return mcolors.to_rgba_array([elec_color])
    arr = mcolors.to_rgba_array(elec_color)
    if len(arr) != len(names):
        raise ValueError(
            f"elec_color has {len(arr)} entries but {len(names)} electrodes are drawn."
        )
    return arr


def _default_elec_size(ax: plt.Axes, W: float, R: float) -> float:
    """Marker area (pt^2) whose diameter matches the electrode radius on canvas."""
    bbox = ax.get_window_extent().transformed(ax.get_figure().dpi_scale_trans.inverted())
    data_per_point = (W / bbox.width) / 72.0
    return (R / data_per_point) ** 2


def _draw_electrodes(
    ax, montage, version, ch_names, show_electrodes, show_labels,
    elec_color, elec_size, label_kwargs, label_offset,
):
    montage, W, H, R, ch_xy = _montage_info(montage, version)

    names = list(ch_xy) if ch_names is None else list(ch_names)
    missing = [n for n in names if n not in ch_xy]
    if missing:
        warnings.warn(f"channels not in montage (skipped): {missing}")
    names = [n for n in names if n in ch_xy]
    xy = np.array([ch_xy[n] for n in names])

    if show_electrodes:
        if elec_size is None:
            elec_size = _default_elec_size(ax, W, R)
        ax.scatter(
            xy[:, 0], xy[:, 1],
            s=elec_size,
            c=_resolve_colors(elec_color, names),
            marker="o",
            zorder=3,
        )

    if show_labels:
        kw = dict(fontsize=4, color="k", ha="center", va="bottom", zorder=4)
        kw.update(label_kwargs or {})
        for name, (x, y) in zip(names, xy):
            ax.annotate(
                name, (x, y),
                xytext=label_offset, textcoords="offset points",
                **kw,
            )


def plot_cortical_frame(
    fname: Union[str, Path, None] = None,
    ax: Optional[plt.Axes] = None,
    edgecolor: str = "k",
    alpha: float = 0.8,
    lw: float = 1,
    # --- electrodes / labels ---
    show_electrodes: bool = False,
    show_labels: bool = False,
    montage: Optional[mne.channels.DigMontage] = None,
    version: str = "64_20250606",
    ch_names: Optional[list[str]] = None,
    elec_color: ColorSpec = "k",
    elec_size: Optional[float] = None,
    label_kwargs: Optional[dict] = None,
    label_offset: tuple[float, float] = (0, 4),
) -> plt.Axes:
    """Draw the cortical outline, optionally with electrodes and labels.

    Parameters
    ----------
    fname : path-like, optional
        Override path to cortex.pdf.
    ax : Axes, optional
        Target axes (``plt.gca()`` if omitted).
    edgecolor, alpha, lw :
        Outline style.
    show_electrodes : bool
        Draw one marker per electrode.
    show_labels : bool
        Annotate each electrode with its channel name.
    montage, version :
        Montage used for electrode positions (built from *version* if None).
    ch_names : list of str, optional
        Electrodes to draw, in this order.  Defaults to all montage channels.
    elec_color : colour, dict ``{ch_name: colour}``, or sequence of colours
        A sequence must match *ch_names* (or the montage order) in length.
        Channels missing from a dict are drawn black.
    elec_size : float, optional
        Marker area in pt^2.  Defaults to the electrode size on the canvas.
    label_kwargs : dict, optional
        Forwarded to ``ax.annotate`` (e.g. ``fontsize``, ``color``).
    label_offset : (dx, dy)
        Label offset from the electrode centre, in points.
    """
    if fname is None:
        fname = _DEFAULT_CORTEX
    if ax is None:
        ax = plt.gca()

    page = fitz.open(str(fname))[0]
    H, W = page.rect.height, page.rect.width

    for drawing in page.get_drawings():
        verts: list[tuple[float, float]] = []
        codes: list[int] = []
        for item in drawing["items"]:
            if item[0] == "c":
                p1, p2, p3, _p4 = item[1:]
                if not verts:
                    verts.append((p1.x, H - p1.y))
                    codes.append(MplPath.MOVETO)
                verts.extend([(p1.x, H - p1.y), (p2.x, H - p2.y), (p3.x, H - p3.y)])
                codes.extend([MplPath.CURVE4] * 3)
        if verts:
            ax.add_patch(PathPatch(
                MplPath(verts, codes),
                facecolor="none", edgecolor=edgecolor, alpha=alpha, lw=lw,
            ))

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    if show_electrodes or show_labels:
        _draw_electrodes(
            ax, montage, version, ch_names, show_electrodes, show_labels,
            elec_color, elec_size, label_kwargs, label_offset,
        )
    return ax


# ===========================================================================
# 3. Heatmap
# ===========================================================================

def _align_data(data, ch_names, montage_ch_xy):
    """Match data values to montage channels -> (names, xy (N,2), values (N,))."""
    if isinstance(data, dict):
        ch_names = list(data.keys())
        vals = np.array(list(data.values()), dtype=float)
    else:
        vals = np.asarray(data, dtype=float).ravel()
        if ch_names is None:
            if len(vals) != len(montage_ch_xy):
                raise ValueError(
                    f"data has {len(vals)} values but montage has "
                    f"{len(montage_ch_xy)} channels.  Pass ch_names explicitly."
                )
            ch_names = list(montage_ch_xy)
        elif len(vals) != len(ch_names):
            raise ValueError(
                f"data length ({len(vals)}) != ch_names length ({len(ch_names)})."
            )

    kept, xy, out = [], [], []
    missing = []
    for name, v in zip(ch_names, vals):
        if name in montage_ch_xy:
            kept.append(name)
            xy.append(montage_ch_xy[name])
            out.append(v)
        else:
            missing.append(name)
    if missing:
        warnings.warn(f"channels not in montage (skipped): {missing}")
    return kept, np.array(xy), np.array(out)


def _draw_heatmap(
    ax, xy_pts, values, W, H,
    range_pdf, mask_outside, cmap, vmin, vmax, symmetric_cmap, interp_resolution,
):
    nx = int(interp_resolution * W / min(W, H))
    ny = int(interp_resolution * H / min(W, H))
    grid_x, grid_y = np.meshgrid(np.linspace(0, W, nx), np.linspace(0, H, ny))

    grid_z = CloughTocher2DInterpolator(xy_pts, values)(grid_x, grid_y)

    mask = None
    if mask_outside:
        range_path = range_pdf if range_pdf is not None else _DEFAULT_RANGE
        mask = _build_range_mask(str(range_path), grid_x, grid_y, H)
        nan_in_mask = mask & ~np.isfinite(grid_z)
        if nan_in_mask.any():
            rbf = RBFInterpolator(xy_pts, values, kernel="cubic")
            query = np.column_stack([grid_x[nan_in_mask], grid_y[nan_in_mask]])
            grid_z[nan_in_mask] = np.clip(rbf(query), values.min(), values.max())
        grid_z[~mask] = np.nan

    finite = grid_z[np.isfinite(grid_z)]
    if vmin is None and vmax is None:
        if symmetric_cmap and len(finite):
            lim = np.nanmax(np.abs(finite))
            vmin, vmax = -lim, lim
        else:
            vmin = float(np.nanmin(finite)) if len(finite) else 0.0
            vmax = float(np.nanmax(finite)) if len(finite) else 1.0
    elif vmin is None:
        vmin = float(np.nanmin(finite)) if len(finite) else 0.0
    elif vmax is None:
        vmax = float(np.nanmax(finite)) if len(finite) else 1.0

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    rgba = plt.get_cmap(cmap)(norm(np.where(np.isfinite(grid_z), grid_z, vmin)))
    rgba[..., 3] = mask.astype(float) if mask is not None else np.isfinite(grid_z).astype(float)

    im = ax.imshow(
        rgba,
        extent=[0, W, 0, H],
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
        zorder=-10,
    )
    im.set_cmap(cmap)
    im.set_clim(vmin, vmax)
    return im


def plot_heatmap(
    data: Union[np.ndarray, dict],
    ax: Optional[plt.Axes] = None,
    montage: Optional[mne.channels.DigMontage] = None,
    ch_names: Optional[list[str]] = None,
    version: str = "64_20250606",
    range_pdf: Union[str, Path, None] = None,
    mask_outside: bool = True,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    interp_resolution: int = 1000,
):
    """Draw only the interpolated heatmap on *ax*.  Returns the AxesImage."""
    if ax is None:
        ax = plt.gca()
    montage, W, H, _R, ch_xy = _montage_info(montage, version)
    _, xy_pts, values = _align_data(data, ch_names, ch_xy)
    return _draw_heatmap(
        ax, xy_pts, values, W, H,
        range_pdf, mask_outside, cmap, vmin, vmax, symmetric_cmap, interp_resolution,
    )


# ===========================================================================
# 4. plot_topomap = frame (+ electrodes) -> heatmap -> colorbar
# ===========================================================================

def plot_topomap(
    data: Union[np.ndarray, dict],
    montage: Optional[mne.channels.DigMontage] = None,
    ch_names: Optional[list[str]] = None,
    version: str = "64_20250606",
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (2, 2.5),
    # --- frame ---
    cortex_pdf: Union[str, Path, None] = None,
    frame_color: str = "k",
    frame_alpha: float = 0.8,
    frame_lw: float = 1,
    # --- electrodes / labels ---
    show_electrodes: bool = True,
    show_labels: bool = False,
    elec_color: ColorSpec = "k",
    elec_size: Optional[float] = None,
    label_kwargs: Optional[dict] = None,
    label_offset: tuple[float, float] = (0, 4),
    # --- heatmap ---
    range_pdf: Union[str, Path, None] = None,
    mask_outside: bool = True,
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    interp_resolution: int = 1000,
    # --- colorbar ---
    show_colorbar: bool = True,
    cbar_label: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a topomap: cortical frame (with electrodes) first, then heatmap.

    *data* is an array of per-channel values (with optional *ch_names*) or a
    ``{ch_name: value}`` dict.  Only channels with data are drawn as electrodes.
    See :func:`plot_cortical_frame` for the frame / electrode / label options
    and :func:`plot_heatmap` for the heatmap options.
    """
    montage, W, H, _R, ch_xy = _montage_info(montage, version)
    kept, xy_pts, values = _align_data(data, ch_names, ch_xy)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    plot_cortical_frame(
        fname=cortex_pdf, ax=ax,
        edgecolor=frame_color, alpha=frame_alpha, lw=frame_lw,
        show_electrodes=show_electrodes, show_labels=show_labels,
        montage=montage, version=version, ch_names=kept,
        elec_color=elec_color, elec_size=elec_size,
        label_kwargs=label_kwargs, label_offset=label_offset,
    )

    im = _draw_heatmap(
        ax, xy_pts, values, W, H,
        range_pdf, mask_outside, cmap, vmin, vmax, symmetric_cmap, interp_resolution,
    )

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)

    return fig, ax


# ===========================================================================
# 5. Per-electrode subplots
# ===========================================================================

def cortical_subplots(
    montage: Optional[mne.channels.DigMontage] = None,
    version: str = "64_20250606",
    figsize: Optional[tuple[float, float]] = None,
    ax_size: Optional[tuple[float, float]] = None,
    ax_size_scale: float = 1.0,
    aspect_ratio: float = 0.6,
    cortex_pdf: Union[str, Path, None] = None,
    frame_color: str = "k",
    frame_alpha: float = 0.05,
    frame_lw: float = 1,
    share_x: bool = False,
    share_y: bool = False,
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure with one small transparent axes centred on each electrode.

    Returns ``(fig, axes)``; ``axes`` follows the montage channel order and each
    axes is labelled with its channel name (``ax.get_label()``).
    """
    montage, W, H, elec_radius, ch_xy = _montage_info(montage, version)
    ch_names = list(ch_xy)

    if figsize is None:
        figsize = (5.0 * W / H, 5.0)
    fig_w, fig_h = float(figsize[0]), float(figsize[1])

    if ax_size is None:
        ax_width = max(elec_radius / W * fig_w * ax_size_scale, 0.15)
        ax_size = (ax_width, ax_width * aspect_ratio)
    ax_w_frac = ax_size[0] / fig_w
    ax_h_frac = ax_size[1] / fig_h

    fig = plt.figure(figsize=figsize)
    bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    plot_cortical_frame(
        fname=cortex_pdf, ax=bg_ax,
        edgecolor=frame_color, alpha=frame_alpha, lw=frame_lw,
    )
    bg_ax.set_aspect("auto")
    bg_ax.patch.set_alpha(0)

    axes = np.empty(len(ch_names), dtype=object)
    ref_ax = None
    for i, name in enumerate(ch_names):
        cx, cy = ch_xy[name]
        left = cx / W - ax_w_frac / 2.0
        bottom = cy / H - ax_h_frac / 2.0

        kwargs = {}
        if share_x and ref_ax is not None:
            kwargs["sharex"] = ref_ax
        if share_y and ref_ax is not None:
            kwargs["sharey"] = ref_ax

        ax = fig.add_axes([left, bottom, ax_w_frac, ax_h_frac], **kwargs)
        ax.set_facecolor("none")
        ax.set_label(name)
        ax.set_xticks([])
        ax.set_yticks([])
        if ref_ax is None:
            ref_ax = ax
        axes[i] = ax

    return fig, axes


# ===========================================================================
# 6. Internal helpers
# ===========================================================================

def _pdf_paths_to_mpl(fname: str, H: float) -> list[MplPath]:
    page = fitz.open(fname)[0]
    mpl_paths = []
    for drawing in page.get_drawings():
        verts: list[tuple[float, float]] = []
        codes: list[int] = []
        for item in drawing["items"]:
            if item[0] == "c":
                p1, p2, p3, _p4 = item[1:]
                if not verts:
                    verts.append((p1.x, H - p1.y))
                    codes.append(MplPath.MOVETO)
                verts.extend([(p1.x, H - p1.y), (p2.x, H - p2.y), (p3.x, H - p3.y)])
                codes.extend([MplPath.CURVE4] * 3)
        if verts:
            verts.append(verts[0])
            codes.append(MplPath.CLOSEPOLY)
            mpl_paths.append(MplPath(verts, codes))
    return mpl_paths


def _build_range_mask(range_fname: str, grid_x, grid_y, H: float) -> np.ndarray:
    pts = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    mask = np.zeros(pts.shape[0], dtype=bool)
    for p in _pdf_paths_to_mpl(range_fname, H):
        mask |= p.contains_points(pts)
    return mask.reshape(grid_x.shape)


def _compute_vlim(data_flat, symmetric_cmap):
    finite = data_flat[np.isfinite(data_flat)]
    if len(finite) == 0:
        return (-1.0, 1.0) if symmetric_cmap else (0.0, 1.0)
    if symmetric_cmap:
        lim = np.nanmax(np.abs(finite))
        return -lim, lim
    return float(np.nanmin(finite)), float(np.nanmax(finite))


# ===========================================================================
# 7. plot_topomap_grid
# ===========================================================================

def plot_topomap_grid(
    data: np.ndarray,
    montage: Optional[mne.channels.DigMontage] = None,
    ch_names: Optional[list[str]] = None,
    version: str = "64_20250606",
    panel_size: tuple[float, float] = (0.8, 1),
    share_lim: str = "all",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    cbar_label: str = "",
    row_labels: Optional[list[str]] = None,
    col_labels: Optional[list[str]] = None,
    row_title: Optional[str] = None,
    col_title: Optional[str] = None,
    **topomap_kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Grid of topomaps for data of shape (62,), (col, 62) or (row, col, 62).

    ``share_lim`` is one of "all", "row", "col", "none" and controls how the
    colour limits (and colorbars) are shared.  Extra keyword arguments are
    forwarded to :func:`plot_topomap` (e.g. ``elec_color``, ``show_labels``).
    """
    if share_lim not in ("all", "row", "col", "none"):
        raise ValueError('share_lim must be one of "all", "row", "col", "none"')

    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, np.newaxis, :]
    elif data.ndim == 2:
        data = data[np.newaxis, :, :]
    elif data.ndim != 3:
        raise ValueError(f"data must have 1, 2, or 3 dimensions; got shape {data.shape}")

    n_row, n_col, n_ch = data.shape

    if montage is None:
        montage = build_montage(version=version)

    frame_alpha = topomap_kwargs.pop("frame_alpha", 0.8)
    frame_lw = topomap_kwargs.pop("frame_lw", 0.5)
    interp_resolution = topomap_kwargs.pop("interp_resolution", 300)

    vlims = np.empty((n_row, n_col, 2), dtype=float)

    def _lim(arr):
        lo, hi = _compute_vlim(arr.ravel(), symmetric_cmap)
        return (vmin if vmin is not None else lo, vmax if vmax is not None else hi)

    if vmin is not None and vmax is not None:
        vlims[:, :, 0], vlims[:, :, 1] = vmin, vmax
    elif share_lim == "all":
        vlims[:, :, :] = _lim(data)
    elif share_lim == "row":
        for i in range(n_row):
            vlims[i, :, :] = _lim(data[i])
    elif share_lim == "col":
        for j in range(n_col):
            vlims[:, j, :] = _lim(data[:, j, :])
    else:
        for i in range(n_row):
            for j in range(n_col):
                vlims[i, j, :] = _lim(data[i, j])

    panel_w, panel_h = panel_size
    has_row_labels = row_labels is not None
    has_col_labels = col_labels is not None
    has_row_title = row_title is not None
    has_col_title = col_title is not None

    row_title_w = 0.35 if has_row_title else 0.0
    row_label_w = 0.6 if has_row_labels else 0.0
    col_title_h = 0.35 if has_col_title else 0.0
    col_label_h = 0.4 if has_col_labels else 0.0
    cbar_margin_w = 0.35 if (show_colorbar and share_lim in ("all", "row")) else 0.0
    cbar_margin_h = 0.35 if (show_colorbar and share_lim == "col") else 0.0

    n_grid_cols = n_col + int(has_row_title) + int(has_row_labels)
    n_grid_rows = n_row + int(has_col_title) + int(has_col_labels)

    fig = plt.figure(figsize=(
        n_col * panel_w + row_title_w + row_label_w + cbar_margin_w,
        n_row * panel_h + col_title_h + col_label_h + cbar_margin_h,
    ))

    width_ratios = (
        ([row_title_w] if has_row_title else [])
        + ([row_label_w] if has_row_labels else [])
        + [panel_w] * n_col
    )
    height_ratios = (
        ([col_title_h] if has_col_title else [])
        + ([col_label_h] if has_col_labels else [])
        + [panel_h] * n_row
    )

    none_cbar = show_colorbar and share_lim == "none"
    gs = fig.add_gridspec(
        n_grid_rows, n_grid_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        wspace=0.5 if none_cbar else 0.05,
        hspace=0.01,
    )

    col0 = int(has_row_title) + int(has_row_labels)
    row0 = int(has_col_title) + int(has_col_labels)

    axes = np.empty((n_row, n_col), dtype=object)
    images = np.empty((n_row, n_col), dtype=object)

    for i in range(n_row):
        for j in range(n_col):
            ax = fig.add_subplot(gs[row0 + i, col0 + j])
            pmin, pmax = vlims[i, j]
            _, ax = plot_topomap(
                data[i, j],
                montage=montage,
                ch_names=ch_names,
                version=version,
                ax=ax,
                cmap=cmap,
                vmin=pmin,
                vmax=pmax,
                symmetric_cmap=symmetric_cmap,
                show_colorbar=show_colorbar and share_lim == "none",
                cbar_label=cbar_label,
                frame_alpha=frame_alpha,
                frame_lw=frame_lw,
                interp_resolution=interp_resolution,
                **topomap_kwargs,
            )
            axes[i, j] = ax
            for im in ax.get_images():
                images[i, j] = im

    if show_colorbar and share_lim == "all":
        host_ax = axes[0, n_col - 1]
        cb = fig.colorbar(images[0, n_col - 1], cax=host_ax.inset_axes([1.02, 0.2, 0.05, 0.6]))
        if cbar_label:
            cb.set_label(cbar_label)
    elif show_colorbar and share_lim == "row":
        for i in range(n_row):
            host_ax = axes[i, n_col - 1]
            cb = fig.colorbar(images[i, n_col - 1], cax=host_ax.inset_axes([1.02, 0.2, 0.05, 0.6]))
            if cbar_label:
                cb.set_label(cbar_label)
    elif show_colorbar and share_lim == "col":
        for j in range(n_col):
            host_ax = axes[n_row - 1, j]
            cb = fig.colorbar(
                images[n_row - 1, j],
                cax=host_ax.inset_axes([0.0, -0.05, 1.0, 0.05]),
                orientation="horizontal",
            )
            if cbar_label:
                cb.set_label(cbar_label)

    if has_col_labels:
        if len(col_labels) != n_col:
            raise ValueError(f"col_labels has {len(col_labels)} entries but data has {n_col} columns.")
        for j, label in enumerate(col_labels):
            lax = fig.add_subplot(gs[row0 - 1, col0 + j])
            lax.axis("off")
            lax.text(0.5, 0.1, label, ha="center", va="bottom", fontsize=11, transform=lax.transAxes)

    if has_row_labels:
        if len(row_labels) != n_row:
            raise ValueError(f"row_labels has {len(row_labels)} entries but data has {n_row} rows.")
        for i, label in enumerate(row_labels):
            lax = fig.add_subplot(gs[row0 + i, col0 - 1])
            lax.axis("off")
            lax.text(0.9, 0.5, label, ha="right", va="center", fontsize=11,
                     rotation=90, transform=lax.transAxes)

    if has_col_title:
        tax = fig.add_subplot(gs[0, col0:col0 + n_col])
        tax.axis("off")
        tax.text(0.5, 0.1, col_title, ha="center", va="bottom", fontsize=13,
                 fontweight="bold", transform=tax.transAxes)

    if has_row_title:
        tax = fig.add_subplot(gs[row0:row0 + n_row, 0])
        tax.axis("off")
        tax.text(0.9, 0.5, row_title, ha="right", va="center", fontsize=13,
                 fontweight="bold", rotation=90, transform=tax.transAxes)

    return fig, axes


# ===========================================================================
# 8. Electrode colouring
# ===========================================================================

def color_elec(df, shrink=0.5, elev=30, wrap=330, equalize=0.9):
    """
    给电极分配 RGB 颜色：电极在 2D 展开面上绕质心一圈，映射为绕 RGB 立方体
    黑白对角线（灰轴）一圈。

    - 绕质心的角度  -> 绕灰轴的角度（色相）
    - 距质心的距离  -> 离灰轴的距离（饱和度，rho=1 恰好落在立方体表面）
    - 第三主成分    -> 沿灰轴的位置（明度，幅度由 lift 控制）
    - 同一 (半球, 脑区) 内的电极先向组质心收缩，再计算颜色
    - 仅用左半球计算，右半球按顺序镜像复用

    参数
    ----
    df : DataFrame，含 ['r', 'a', 's', 'region']，label 为索引或一列
    shrink : 脑区内向质心收缩的比例，越大同区颜色越接近
    rho_min : 最小饱和度比例，避免质心附近电极全是灰色
    lift : 明度随第三主成分变化的幅度（0 表示固定中灰）
    wrap : 色相覆盖的总角度（度），小于 360 可避免缺口两侧电极同色

    返回
    ----
    dict {label: (R, G, B)}，取值 [0, 1]
    """
    df = df.copy().dropna()

    if df.index.name == 'label' or 'label' not in df.columns:
        df = df.reset_index()
        if 'label' not in df.columns and 'index' in df.columns:
            df = df.rename(columns={'index': 'label'})

    df['hemi'] = df['label'].astype(str).str[0]
    coords = df[['r', 'a', 's']].to_numpy(dtype=float)

    centroids = (
        df.groupby(['hemi', 'region'], dropna=False)[['r', 'a', 's']]
        .transform('mean')
        .to_numpy(dtype=float)
    )
    shrunk = centroids + (1 - shrink) * (coords - centroids)

    left_mask = (df['hemi'] == 'L').to_numpy()
    right_mask = (df['hemi'] == 'R').to_numpy()

    left = shrunk[left_mask]
    left = left - left.mean(axis=0)
    _, _, Vt = np.linalg.svd(left, full_matrices=False)
    scores = left @ Vt.T

    if np.corrcoef(scores[:, 0], df.loc[left_mask, 'a'])[0, 1] > 0:
        scores[:, 0] *= -1
    if np.corrcoef(scores[:, 1], df.loc[left_mask, 'r'])[0, 1] < 0:
        scores[:, 1] *= -1

    ang = np.arctan2(scores[:, 1], scores[:, 0])
    rad = np.hypot(scores[:, 0], scores[:, 1])
    n = len(ang)

    srt = np.sort(ang)
    gaps = np.diff(np.append(srt, srt[0] + 2 * np.pi))
    start = srt[(np.argmax(gaps) + 1) % n]
    phi = (ang - start) % (2 * np.pi)

    def spread(x):
        lin = (x - x.min()) / (x.max() - x.min())
        rk = (rankdata(x) - 1) / (len(x) - 1)
        return (1 - equalize) * lin + equalize * rk

    hue = spread(phi) * np.deg2rad(wrap)
    lat = (2 * spread(rad) - 1) * np.deg2rad(elev)

    e1 = np.array([2, -1, -1]) / np.sqrt(6)
    e2 = np.array([0, 1, -1]) / np.sqrt(2)
    e3 = np.ones(3) / np.sqrt(3)
    u = (np.cos(lat) * np.cos(hue))[:, None] * e1 \
        + (np.cos(lat) * np.sin(hue))[:, None] * e2 \
        + np.sin(lat)[:, None] * e3

    left_norm = 0.5 + 0.5 * u / np.abs(u).max(axis=1, keepdims=True)

    right_norm = left_norm

    labels_left = df.loc[left_mask, 'label'].to_numpy()
    labels_right = df.loc[right_mask, 'label'].to_numpy()

    order_left = np.argsort(-scores[:, 0])
    order_right = np.argsort(scores[:, 0])

    color_dict = {}
    for i in order_left:
        color_dict[labels_left[i]] = left_norm[i]
    for i in order_right:
        color_dict[labels_right[i]] = right_norm[i]

    return color_dict
