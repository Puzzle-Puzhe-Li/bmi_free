"""
viz.py — Monkey cortical topomap visualization utilities
=========================================================
Analogous to MNE's plot_topomap, but for macaque multi-channel
cranial-nail (颅钉) arrays.

Three PDFs (cortex.pdf, electrodes.pdf, range.pdf) share the same
800×1000-point coordinate system and are used as:
  cortex.pdf    — cortical outline (background frame)
  electrodes.pdf — electrode positions in the shared coordinate system
  range.pdf     — interpolation / masking region

Public API
----------
plot_cortical_frame(fname, ax)          -> ax
build_montage(elec_xlsx, elec_pdf)      -> montage dict
plot_topomap(data, montage, ...)        -> (fig, ax)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional, Union

import fitz                               # PyMuPDF
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.interpolate import CloughTocher2DInterpolator, RBFInterpolator
from scipy.spatial.distance import cdist
from sklearn.preprocessing import MinMaxScaler
from bmi_free.io import path
import mne

# ---------------------------------------------------------------------------
# Default asset paths (can be overridden in every function call)
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
_DEFAULT_CORTEX    = _HERE / "cortex" / "cortex.pdf"
_DEFAULT_ELECTRODES = _HERE / "cortex" / "electrodes.pdf"
_DEFAULT_RANGE     = _HERE / "cortex" / "range.pdf"
_ELEC_LOC_PATH = path['config']

# Electrodes that are GND / REF and should never be plotted
_SKIP_CHANNELS = {"L11", "R11"}

# Pitch (degrees) used when projecting 3-D RAS coords onto the 2-D frame
_PROJ_PITCH_DEG = -15.0


# ===========================================================================
# 1. Background cortical frame
# ===========================================================================

def plot_cortical_frame(
    fname: Union[str, Path, None] = None,
    ax: Optional[plt.Axes] = None,
    edgecolor: str = "k",
    alpha: float = 0.8,
    lw: float = 1,
) -> plt.Axes:
    """Draw the cortical outline from *cortex.pdf* onto *ax*.

    Parameters
    ----------
    fname : path-like, optional
        Path to cortex.pdf.  Defaults to ``viz/cortex.pdf`` next to this file.
    ax : matplotlib Axes, optional
        Target axes.  Created with ``plt.gca()`` if not supplied.
    edgecolor, alpha, lw :
        Style parameters forwarded to :class:`matplotlib.patches.PathPatch`.

    Returns
    -------
    ax : matplotlib Axes
        The axes that were drawn on.
    """
    if fname is None:
        fname = _DEFAULT_CORTEX
    if ax is None:
        ax = plt.gca()

    pdf  = fitz.open(str(fname))
    page = pdf[0]
    H    = page.rect.height
    W    = page.rect.width

    for drawing in page.get_drawings():
        verts: list[tuple[float, float]] = []
        codes: list[int] = []

        for item in drawing["items"]:
            if item[0] == "c":          # cubic Bézier
                p1, p2, p3, _p4 = item[1:]

                if not verts:           # first point → MOVETO
                    verts.append((p1.x, H - p1.y))
                    codes.append(MplPath.MOVETO)

                verts.extend([
                    (p1.x, H - p1.y),
                    (p2.x, H - p2.y),
                    (p3.x, H - p3.y),
                ])
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

        if verts:
            path  = MplPath(verts, codes)
            patch = PathPatch(
                path,
                facecolor="none",
                edgecolor=edgecolor,
                alpha=alpha,
                lw=lw,
            )
            ax.add_patch(patch)

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")
    return ax


# ===========================================================================
# 2. Build montage  (3-D RAS → pixel positions in the shared PDF space)
# ===========================================================================

def build_montage(
    elec_xlsx: Union[str, Path, None] = None,
    elec_pdf:  Union[str, Path, None] = None,
    pitch_deg: float = _PROJ_PITCH_DEG,
    skip_channels: set[str] = _SKIP_CHANNELS,
    version: str = "64_20250606",
) -> mne.channels.DigMontage:
    """Build an electrode-position montage aligned to the shared PDF canvas.

    Checks for a cached ``viz/montage_{version}.fif`` before recomputing.
    If absent, builds the montage and saves the cache.

    Returns
    -------
    montage : dict
        ``{ch_name: np.ndarray([x, y])}``  — pixel positions on the canvas.
        Also stores ``_canvas_wh`` = (W, H) and ``_elec_radius`` (mean
        electrode radius in pixels, extracted from elec_pdf).
    """
    # ------------------------------------------------------------------
    # Cache: viz/montage_{version}.fif
    # ------------------------------------------------------------------
    cache_path = _HERE / "viz" / f"montage_{version}.fif"

    if cache_path.exists():
        dig_montage = mne.channels.read_dig_fif(str(cache_path))
        fids = dig_montage.get_positions()
        first_pos = next(iter(fids["ch_pos"].values()))
        H, W, elec_radius = fids['nasion']
        dig_montage._canvas_wh    = (W, H)
        dig_montage._elec_radius  = elec_radius
        return dig_montage

    # ------------------------------------------------------------------
    # 2a. Load 3-D RAS coordinates
    # ------------------------------------------------------------------
    if elec_xlsx is None:
        elec_xlsx = _ELEC_LOC_PATH / f'elec_loc_{version}.xlsx'
    if elec_pdf is None:
        elec_pdf = _DEFAULT_ELECTRODES

    df_loc = pd.read_excel(str(elec_xlsx), index_col="label")

    # ------------------------------------------------------------------
    # 2b. Project 3-D → 2-D  (pitch rotation around R-axis)
    # ------------------------------------------------------------------
    loc_3d   = df_loc[["r", "a", "s"]].values.astype(np.float64)
    theta    = np.radians(pitch_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    proj_r = loc_3d[:, 0]
    proj_y = loc_3d[:, 1] * cos_t - loc_3d[:, 2] * sin_t

    loc_2d = np.column_stack((proj_r, proj_y))

    # ------------------------------------------------------------------
    # 2c. Extract electrode dot centres and radii from the PDF
    # ------------------------------------------------------------------
    pdf  = fitz.open(str(elec_pdf))
    page = pdf[0]
    H    = page.rect.height
    W    = page.rect.width

    centers = []
    radii   = []
    for d in page.get_drawings():
        r = d["rect"]
        cx = (r.x0 + r.x1) / 2
        cy = (H - r.y0 + H - r.y1) / 2
        centers.append([cx, cy])
        # Use mean of half-widths in x and y as radius
        radii.append(((r.x1 - r.x0) + (r.y1 - r.y0)) / 4)

    centers     = np.array(centers)
    elec_radius = float(np.mean(radii))

    # ------------------------------------------------------------------
    # 2d. Nearest-neighbour matching in normalised [0,1]² space
    # ------------------------------------------------------------------
    scaler_loc     = MinMaxScaler()
    scaler_centers = MinMaxScaler()

    loc_norm     = scaler_loc.fit_transform(loc_2d)
    centers_norm = scaler_centers.fit_transform(centers)

    dist_matrix = cdist(loc_norm, centers_norm, metric="euclidean")
    matched_idx = np.argmin(dist_matrix, axis=1)
    max_err     = np.max(np.min(dist_matrix, axis=1))

    if max_err > 0.05:
        warnings.warn(
            f"build_montage: largest normalised matching error = {max_err:.4f}. "
            "Check that elec_xlsx and elec_pdf are consistent.",
            stacklevel=2,
        )

    reordered = centers[matched_idx]

    # ------------------------------------------------------------------
    # 2e. Build output dict and MNE DigMontage
    # ------------------------------------------------------------------
    ch_names = df_loc.index.tolist()
    ch_pos_dict = {}

    for i, name in enumerate(ch_names):
        if name in skip_channels:
            continue
        x, y = reordered[i]
        # z encodes W (constant per channel) so we can recover it on load
        ch_pos_dict[name] = np.array([x, y, 0])

    dig_montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos_dict,
        nasion=np.array([W, H, elec_radius]),
        lpa   =np.array([0.0, 0.0, 0.0]),
        rpa   =np.array([0.0, 0.0, 0.0]),
        coord_frame='unknown',
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dig_montage.save(str(cache_path), overwrite=True)

    # dig_montage._canvas_wh   = (W, H)
    # dig_montage._elec_radius = elec_radius
    return dig_montage


# ===========================================================================
# 3. plot_topomap  (the main entry point)
# ===========================================================================

def plot_topomap(
    data: Union[np.ndarray, dict],
    montage: Optional[dict] = None,
    ch_names: Optional[list[str]] = None,
    version: str = "64_20250606",          # ← 新增
    # --- layout --
    ax: Optional[plt.Axes] = None,
    figsize: tuple[float, float] = (2, 2.5),
    # --- cortical frame ---
    cortex_pdf:    Union[str, Path, None] = None,
    frame_color:   str   = "k",
    frame_alpha:   float = 0.8,
    frame_lw:      float = 1,
    # --- interpolation mask ---
    range_pdf: Union[str, Path, None] = None,
    mask_outside: bool = True,
    # --- colormap ---
    cmap: str = "RdBu_r",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    # --- electrode markers ---
    show_electrodes: bool = True,
    elec_color: str = "k",
    elec_size:  Optional[float] = None,    # ← 改为 None，从 montage 读取
    # --- colorbar ---
    show_colorbar: bool = True,
    cbar_label: str = "",
    # --- resolution ---
    interp_resolution: int = 1000,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot a topomap on the monkey cortical frame.

    Parameters
    ----------
    data : array-like of shape (62,) or dict {ch_name: value}
        Data to visualise.  When an array is provided, *ch_names* must be
        supplied or the montage channel order is used.  The 62 active
        channels are the 64-channel array **minus** L11 (GND) and R11 (REF).
    montage : dict, optional
        Output of :func:`build_montage`.  Built automatically (with default
        paths) if not supplied.
    ch_names : list of str, optional
        Channel names corresponding to *data* when *data* is an array.
        If omitted, the keys of *montage* (excluding ``_canvas_wh``) are
        used in insertion order.
    ax : matplotlib Axes, optional
        Axes to draw on.  A new figure/axes is created if not supplied.
    figsize : (width, height)
        Figure size in inches (used only when *ax* is ``None``).
    cortex_pdf : path-like, optional
        Override path to cortex.pdf.
    frame_color, frame_alpha, frame_lw :
        Style for the cortical outline.
    range_pdf : path-like, optional
        Override path to range.pdf.  The two closed paths in this file
        define the left- and right-hemisphere interpolation regions.
    mask_outside : bool
        If ``True`` (default), only render the heatmap inside the region
        defined by range.pdf.
    cmap : str
        Matplotlib colormap name.
    vmin, vmax : float, optional
        Colormap limits.  When both are ``None`` and *symmetric_cmap* is
        ``True``, limits are set to ``±max(|data|)``.
    symmetric_cmap : bool
        Force the colormap to be symmetric around zero (ignored if both
        *vmin* and *vmax* are provided explicitly).
    show_electrodes : bool
        Overlay electrode positions as scatter dots.
    elec_color, elec_size :
        Style for electrode markers.
    show_colorbar : bool
        Attach a colorbar to the figure.
    cbar_label : str
        Label for the colorbar axis.
    interp_resolution : int
        Number of pixels along the shorter canvas dimension for the
        interpolation grid.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """

    # ------------------------------------------------------------------
    # 3a. Build / load montage
    # ------------------------------------------------------------------
    if montage is None:
        cache_path = _HERE / "cortex" / f"montage_{version}.fif"
        if cache_path.exists():
            montage = build_montage(version=version)
        else:
            montage = build_montage(version=version)   # will build & cache

    W, H, R = montage.get_positions()["nasion"]
    ch_pos = montage.get_positions()["ch_pos"]
    active_chs = list(ch_pos.keys())

    # if elec_size is None:
    #     raw_radius = R
    #     elec_size = float(raw_radius ** 2) if raw_radius is not None else 4.0


    # ------------------------------------------------------------------
    # 3b. Validate / align data
    # ------------------------------------------------------------------
    if isinstance(data, dict):
        ch_names  = list(data.keys())
        data_vals = np.array(list(data.values()), dtype=float)
    else:
        data_vals = np.asarray(data, dtype=float).ravel()
        if ch_names is None:
            if len(data_vals) != len(active_chs):
                raise ValueError(
                    f"data has {len(data_vals)} values but montage has "
                    f"{len(active_chs)} active channels.  Pass ch_names explicitly."
                )
            ch_names = active_chs
        else:
            if len(data_vals) != len(ch_names):
                raise ValueError(
                    f"data length ({len(data_vals)}) != ch_names length ({len(ch_names)})."
                )

    # Keep only channels present in montage
    xy_list, val_list = [], []
    missing = []
    for name, val in zip(ch_names, data_vals):
        if name in montage.ch_names:
            xy_list.append(montage.get_positions()['ch_pos'][name][:2])
            val_list.append(val)
        else:
            missing.append(name)
    if missing:
        warnings.warn(f"plot_topomap: channels not in montage (skipped): {missing}")

    xy_pts  = np.array(xy_list)    # (N, 2)  pixel coords
    values  = np.array(val_list)   # (N,)

    # ------------------------------------------------------------------
    # 3c. Build interpolation grid
    # ------------------------------------------------------------------
    nx = int(interp_resolution * W / min(W, H))
    ny = int(interp_resolution * H / min(W, H))

    grid_x, grid_y = np.meshgrid(
        np.linspace(0, W, nx),
        np.linspace(0, H, ny),
    )

    interp  = CloughTocher2DInterpolator(xy_pts, values)
    grid_z  = interp(grid_x, grid_y)   # NaN outside convex hull

    # ------------------------------------------------------------------
    # 3d. Build mask from range.pdf
    # ------------------------------------------------------------------
    # if mask_outside:
    #     range_path = range_pdf if range_pdf is not None else _DEFAULT_RANGE
    #     mask = _build_range_mask(str(range_path), grid_x, grid_y, H)
    #     grid_z[~mask] = np.nan
    # ------------------------------------------------------------------
    # 3d. Build mask from range.pdf, then extrapolate to fill it
    # ------------------------------------------------------------------
    if mask_outside:
        range_path = range_pdf if range_pdf is not None else _DEFAULT_RANGE
        mask = _build_range_mask(str(range_path), grid_x, grid_y, H)

        # Fill points that are inside the mask but outside the CT
        # interpolator's convex hull (currently NaN) via nearest-neighbour
        # extrapolation, so the heatmap covers the full mask region.
        # nan_in_mask = mask & ~np.isfinite(grid_z)
        # if nan_in_mask.any():
        #     nearest = RBFInterpolator(xy_pts, values)
        #     grid_z[nan_in_mask] = nearest(grid_x[nan_in_mask], grid_y[nan_in_mask])
        nan_in_mask = mask & ~np.isfinite(grid_z)
        if nan_in_mask.any():
            rbf = RBFInterpolator(xy_pts, values, kernel="cubic")
            query_pts = np.column_stack([grid_x[nan_in_mask], grid_y[nan_in_mask]])
            grid_z[nan_in_mask] = rbf(query_pts)
            grid_z[nan_in_mask] = np.clip(rbf(query_pts), values.min(), values.max())            
        grid_z[~mask] = np.nan    

    # ------------------------------------------------------------------
    # 3e. Colormap limits
    # ------------------------------------------------------------------
    finite = grid_z[np.isfinite(grid_z)]
    if vmin is None and vmax is None:
        if symmetric_cmap and len(finite):
            lim  = np.nanmax(np.abs(finite))
            vmin, vmax = -lim, lim
        else:
            vmin = float(np.nanmin(finite)) if len(finite) else 0.0
            vmax = float(np.nanmax(finite)) if len(finite) else 1.0
    elif vmin is None:
        vmin = float(np.nanmin(finite)) if len(finite) else 0.0
    elif vmax is None:
        vmax = float(np.nanmax(finite)) if len(finite) else 1.0

    # ------------------------------------------------------------------
    # 3f. Draw
    # ------------------------------------------------------------------
    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Cortical frame first
    plot_cortical_frame(
        fname=cortex_pdf,
        ax=ax,
        edgecolor=frame_color,
        alpha=frame_alpha,
        lw=frame_lw,
    )

    # Heatmap
    # im = ax.imshow(
    #     grid_z,
    #     extent=[0, W, 0, H],
    #     origin="lower",
    #     cmap=cmap,
    #     vmin=vmin,
    #     vmax=vmax,
    #     aspect="equal",
    #     interpolation="bilinear",
    #     zorder=-10,
    # )
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    grid_z_filled = np.where(np.isfinite(grid_z), grid_z, vmin)  # placeholder; alpha hides it
    rgba = plt.get_cmap(cmap)(norm(grid_z_filled))
    rgba[..., 3] = mask.astype(float) if mask_outside else np.isfinite(grid_z).astype(float)

    im = ax.imshow(
        rgba,
        extent=[0, W, 0, H],
        origin="lower",
        aspect="equal",
        interpolation="bilinear",
        zorder=-10,
    )
    # for colorbar reference (rgba image carries no scalar mappable info)
    im.set_cmap(cmap)
    im.set_clim(vmin, vmax)    

    if elec_size is None:
        raw_radius = R
        if raw_radius is not None:
            # data-units-per-point, derived from actual axes extent vs size
            bbox = ax.get_window_extent().transformed(
                fig.dpi_scale_trans.inverted()
            )
            ax_width_inches = bbox.width
            data_per_point = (W / ax_width_inches) / 72.0  # data units per point

            radius_points = raw_radius / data_per_point
            diameter_points = radius_points
            elec_size = diameter_points ** 2
        else:
            elec_size = 4.0    

    # Electrode markers
    if show_electrodes:
        ax.scatter(
            xy_pts[:, 0], xy_pts[:, 1],
            s=elec_size,
            c=elec_color,
            marker="o",
            zorder=3,
        )

    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.axis("off")

    # Colorbar
    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        if cbar_label:
            cbar.set_label(cbar_label)

    return fig, ax

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
    share_x: bool = False,   # ← 新增
    share_y: bool = False,   # ← 新增
) -> tuple[plt.Figure, np.ndarray]:
    """Create a figure whose axes are anchored at electrode positions.
 
    A cortical outline (from *cortex.pdf*) is drawn on a full-figure
    background axes.  A small, transparent :class:`~matplotlib.axes.Axes`
    is then placed *centred* at every electrode, in channel order.
 
    Parameters
    ----------
    montage : mne.channels.DigMontage, optional
        Output of :func:`build_montage`.  Built automatically (with
        default paths) if not supplied.
    version : str
        Electrode layout version forwarded to :func:`build_montage`.
    figsize : (width, height), optional
        Figure size in inches.  Defaults to a height of 5 in with the
        width set to preserve the W:H aspect ratio of the PDF canvas,
        so that canvas coordinates map linearly to figure fractions.
        If you supply a custom *figsize* whose aspect ratio differs from
        W:H, the cortex outline and the per-channel axes will still be
        self-consistent, but the cortex may not fill the figure edge-to-edge.
    ax_size : (width, height), optional
        Size of each per-channel axes **in inches**.  Defaults to
        ``1 × elec_radius`` (extracted from *electrodes.pdf*), which
        gives axes roughly three electrode-diameters wide — a sensible
        starting point for small bar charts or time series.
    ax_size_scale: float = 1.0,
        Scale factor for the axes size.  Defaults to 1.0.
    aspect_ratio : float, optional
        Aspect ratio of each per-channel axes.  Defaults to 1.5.
    cortex_pdf : path-like, optional
        Override path to ``cortex.pdf``.
    frame_color, frame_alpha, frame_lw :
        Style forwarded to :func:`plot_cortical_frame`.
 
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of matplotlib.axes.Axes, shape ``(n_channels,)``
        One axes per channel, ordered by ``montage.ch_names``.
        Every axes is centred at the corresponding electrode position.
        Axes labels (``ax.get_label()``) are set to the channel name so
        callers can look up an axes by name if needed.
 
    Notes
    -----
    The per-channel axes are created with a transparent face (``"none"``)
    so the cortical outline remains visible beneath them.  Callers are
    responsible for drawing into the axes (e.g. ``axes[i].plot(...)``).
 
    Example
    -------
    >>> fig, axes = cortical_subplots()
    >>> for i, ax in enumerate(axes):
    ...     ax.bar([0], [data[i]], color="steelblue")
    ...     ax.set_ylim(-1, 1)
    ...     ax.axis("off")
    >>> fig.savefig("per_channel.png", dpi=150)
    """
    # ------------------------------------------------------------------
    # Load montage
    # ------------------------------------------------------------------
    if montage is None:
        montage = build_montage(version=version)
 
    W, H, elec_radius = montage.get_positions()["nasion"]
    ch_pos   = montage.get_positions()["ch_pos"]
    ch_names = list(ch_pos.keys())   # canonical order
    n_ch     = len(ch_names)
 
    # ------------------------------------------------------------------
    # Figure size — default preserves canvas aspect ratio so that
    # canvas_x / W  ==  figure_fraction_x  (and same for y)
    # ------------------------------------------------------------------
    if figsize is None:
        fig_h = 5.0
        fig_w = fig_h * (W / H)
        figsize = (fig_w, fig_h)
    fig_w, fig_h = float(figsize[0]), float(figsize[1])
 
    # ------------------------------------------------------------------
    # Per-channel axes size — default: ~3 electrode diameters wide
    # ------------------------------------------------------------------
    if ax_size is None:
        r_in  = elec_radius / W * fig_w   # elec_radius in inches
        ax_width  = max(r_in * ax_size_scale, 0.15)     # at least 0.15 in
        ax_hight = ax_width * aspect_ratio
        ax_size = (ax_width, ax_hight)
    ax_w_frac = ax_size[0] / fig_w
    ax_h_frac = ax_size[1] / fig_h
 
    # ------------------------------------------------------------------
    # Figure + background (cortex) axes
    # ------------------------------------------------------------------
    fig   = plt.figure(figsize=figsize)
    bg_ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
 
    # plot_cortical_frame sets xlim/ylim to (0,W)/(0,H) and aspect='equal'.
    # We immediately override aspect to 'auto' so that the figsize (already
    # matched to W:H) drives the display geometry — no internal padding.
    plot_cortical_frame(
        fname=cortex_pdf,
        ax=bg_ax,
        edgecolor=frame_color,
        alpha=frame_alpha,
        lw=frame_lw,
    )
    bg_ax.set_aspect("auto")   # undo 'equal'; figsize already encodes the ratio
    bg_ax.patch.set_alpha(0)   # transparent axes background
 
    # ------------------------------------------------------------------
    # Per-channel axes — centred at each electrode
    # ------------------------------------------------------------------
    axes = np.empty(n_ch, dtype=object)
    ref_ax = None   # first axes, used as sharex/sharey target

    for i, name in enumerate(ch_names):
        cx, cy, _ = ch_pos[name]
        fig_cx = cx / W
        fig_cy = cy / H
        left   = fig_cx - ax_w_frac / 2.0
        bottom = fig_cy - ax_h_frac / 2.0

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
            ref_ax = ax   # anchor for subsequent shares

        axes[i] = ax

    return fig, axes

# ===========================================================================
# Internal helpers
# ===========================================================================

def _pdf_paths_to_mpl(fname: str, H: float) -> list[MplPath]:
    """Extract all closed Bézier paths from a PDF page as Matplotlib Path objects.

    The Y-axis is flipped so origin is at the bottom-left (matching
    imshow ``origin='lower'``).
    """
    pdf  = fitz.open(fname)
    page = pdf[0]
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
                verts.extend([
                    (p1.x, H - p1.y),
                    (p2.x, H - p2.y),
                    (p3.x, H - p3.y),
                ])
                codes.extend([MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])

        if verts:
            # Close the path
            verts.append(verts[0])
            codes.append(MplPath.CLOSEPOLY)
            mpl_paths.append(MplPath(verts, codes))

    return mpl_paths


def _build_range_mask(
    range_fname: str,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    H: float,
) -> np.ndarray:
    """Return a boolean mask (shape = grid_x.shape) that is True inside
    the combined regions defined by range.pdf.
    """
    paths = _pdf_paths_to_mpl(range_fname, H)
    pts   = np.column_stack([grid_x.ravel(), grid_y.ravel()])

    mask = np.zeros(pts.shape[0], dtype=bool)
    for p in paths:
        mask |= p.contains_points(pts)

    return mask.reshape(grid_x.shape)


# ===========================================================================
# 4. plot_topomap_grid  (multi-panel wrapper around plot_topomap)
# ===========================================================================

def _compute_vlim(data_flat, symmetric_cmap):
    """Compute (vmin, vmax) over a flat array of values, NaNs ignored."""
    finite = data_flat[np.isfinite(data_flat)]
    if len(finite) == 0:
        return (-1.0, 1.0) if symmetric_cmap else (0.0, 1.0)
    if symmetric_cmap:
        lim = np.nanmax(np.abs(finite))
        return -lim, lim
    return float(np.nanmin(finite)), float(np.nanmax(finite))


def plot_topomap_grid(
    data: np.ndarray,
    montage: Optional[mne.channels.DigMontage] = None,
    ch_names: Optional[list[str]] = None,
    version: str = "64_20250606",
    # --- layout ---
    panel_size: tuple[float, float] = (0.8, 1),
    # --- shared limits / colorbar ---
    share_lim: str = "all",     # "all" | "row" | "col" | "none"
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    symmetric_cmap: bool = True,
    cmap: str = "RdBu_r",
    show_colorbar: bool = True,
    cbar_label: str = "",
    # --- labels ---
    row_labels: Optional[list[str]] = None,
    col_labels: Optional[list[str]] = None,
    row_title: Optional[str] = None,
    col_title: Optional[str] = None,
    **topomap_kwargs,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot a grid of topomaps for 1-D, 2-D, or 3-D data.

    Parameters
    ----------
    data : array-like
        - shape ``(62,)``       → single topomap.
        - shape ``(col, 62)``   → ``1 x col`` row of topomaps.
        - shape ``(row, col, 62)`` → ``row x col`` grid of topomaps.
    montage, ch_names, version :
        Forwarded to :func:`plot_topomap` for every panel.  The montage is
        built once and reused across all panels.
    panel_size : (width, height)
        Size in inches of a *single* topomap panel.  The overall figure
        size is ``panel_size`` scaled by the grid shape, plus extra space
        for labels/colorbar.
    share_lim : {"all", "row", "col", "none"}
        How ``vmin``/``vmax`` (and therefore the colormap) are shared:
          - ``"all"``  : one global vmin/vmax for the whole grid. One
            colorbar, placed at the top-right corner.
          - ``"row"``  : one vmin/vmax per row. One colorbar per row,
            placed to the right of the rightmost panel in that row.
          - ``"col"``  : one vmin/vmax per column. One colorbar per
            column, placed below the bottom panel in that column.
          - ``"none"`` : every panel gets its own vmin/vmax (computed from
            its own data) and its own colorbar.
    vmin, vmax : float, optional
        Explicit color limits. If given, override the automatic
        per-group computation (still subject to *share_lim* grouping).
    symmetric_cmap : bool
        Force color limits to be symmetric around 0 (``±max(|data|)``)
        when *vmin*/*vmax* are not given.
    cmap : str
        Colormap name, forwarded to every panel.
    show_colorbar : bool
        Whether to draw colorbar(s) at all.
    cbar_label : str
        Label applied to every colorbar.
    row_labels, col_labels : list of str, optional
        Labels placed to the left of each row / above each column.
    row_title, col_title : str, optional
        An overall title placed further left of *row_labels* (rotated
        90°) / further above *col_labels*.
    **topomap_kwargs :
        Any other keyword arguments are forwarded verbatim to every call
        of :func:`plot_topomap` (e.g. ``frame_color``, ``elec_size``,
        ``range_pdf``, ``mask_outside``, ``interp_resolution``, ...).

    Returns
    -------
    fig : matplotlib Figure
    axes : np.ndarray of matplotlib Axes, shape ``(row, col)``
        Always 2-D, even for 1-D/single input (shape ``(1, 1)`` or
        ``(1, col)``).

    Examples
    --------
    See module-level docstring / usage examples at the bottom of this file.
    """
    if share_lim not in ("all", "row", "col", "none"):
        raise ValueError('share_lim must be one of "all", "row", "col", "none"')

    # ------------------------------------------------------------------
    # 4a. Normalise data to shape (row, col, 62)
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data[np.newaxis, np.newaxis, :]          # (1, 1, 62)
    elif data.ndim == 2:
        data = data[np.newaxis, :, :]                    # (1, col, 62)
    elif data.ndim != 3:
        raise ValueError(
            f"data must have 1, 2, or 3 dimensions; got shape {data.shape}"
        )

    n_row, n_col, n_ch = data.shape

    # ------------------------------------------------------------------
    # 4b. Build montage once, reuse for every panel
    # ------------------------------------------------------------------
    if montage is None:
        montage = build_montage(version=version)

    frame_alpha = topomap_kwargs.pop("frame_alpha", 0.8)
    frame_lw = topomap_kwargs.pop("frame_lw", .5)
    interp_resolution = topomap_kwargs.pop("interp_resolution", 300)

    # ------------------------------------------------------------------
    # 4c. Determine color limits per group, according to share_lim
    # ------------------------------------------------------------------
    # vlims[i, j] = (vmin, vmax) for panel (i, j)
    vlims = np.empty((n_row, n_col, 2), dtype=float)

    if vmin is not None and vmax is not None:
        vlims[:, :, 0] = vmin
        vlims[:, :, 1] = vmax
    elif share_lim == "all":
        gmin, gmax = _compute_vlim(data.ravel(), symmetric_cmap)
        if vmin is not None:
            gmin = vmin
        if vmax is not None:
            gmax = vmax
        vlims[:, :, 0] = gmin
        vlims[:, :, 1] = gmax
    elif share_lim == "row":
        for i in range(n_row):
            rmin, rmax = _compute_vlim(data[i].ravel(), symmetric_cmap)
            if vmin is not None:
                rmin = vmin
            if vmax is not None:
                rmax = vmax
            vlims[i, :, 0] = rmin
            vlims[i, :, 1] = rmax
    elif share_lim == "col":
        for j in range(n_col):
            cmin, cmax = _compute_vlim(data[:, j, :].ravel(), symmetric_cmap)
            if vmin is not None:
                cmin = vmin
            if vmax is not None:
                cmax = vmax
            vlims[:, j, 0] = cmin
            vlims[:, j, 1] = cmax
    else:  # "none"
        for i in range(n_row):
            for j in range(n_col):
                pmin, pmax = _compute_vlim(data[i, j].ravel(), symmetric_cmap)
                if vmin is not None:
                    pmin = vmin
                if vmax is not None:
                    pmax = vmax
                vlims[i, j, 0] = pmin
                vlims[i, j, 1] = pmax

    # ------------------------------------------------------------------
    # 4d. Figure / GridSpec layout
    #
    # Extra columns/rows are reserved for: row_title, row_labels (left);
    # col_title, col_labels (top); colorbar (right or bottom, depending
    # on share_lim).
    # ------------------------------------------------------------------
    panel_w, panel_h = panel_size

    has_row_labels = row_labels is not None
    has_col_labels = col_labels is not None
    has_row_title  = row_title is not None
    has_col_title  = col_title is not None

    # Sizes (inches) for the extra label columns/rows
    row_title_w  = 0.35 if has_row_title  else 0.0
    row_label_w  = 0.6  if has_row_labels else 0.0
    col_title_h  = 0.35 if has_col_title  else 0.0
    col_label_h  = 0.4  if has_col_labels else 0.0

    # Colorbar space
    cbar_margin_w = 0.35 if (show_colorbar and share_lim in ("all", "row")) else 0.0
    cbar_margin_h = 0.35 if (show_colorbar and share_lim == "col") else 0.0

    n_grid_cols = n_col + (1 if has_row_title else 0) + (1 if has_row_labels else 0)
    n_grid_rows = n_row + (1 if has_col_title else 0) + (1 if has_col_labels else 0)

    fig_w = n_col * panel_w + row_title_w + row_label_w + cbar_margin_w
    fig_h = n_row * panel_h + col_title_h + col_label_h + cbar_margin_h

    fig = plt.figure(figsize=(fig_w, fig_h))

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

    # Index offsets into the gridspec for the actual panel grid
    col0 = (1 if has_row_title else 0) + (1 if has_row_labels else 0)
    row0 = (1 if has_col_title else 0) + (1 if has_col_labels else 0)

    # ------------------------------------------------------------------
    # 4e. Draw panels
    # ------------------------------------------------------------------
    axes = np.empty((n_row, n_col), dtype=object)
    images = np.empty((n_row, n_col), dtype=object)  # for colorbars

    for i in range(n_row):
        for j in range(n_col):
            ax = fig.add_subplot(gs[row0 + i, col0 + j])
            pmin, pmax = vlims[i, j]

            # For "none" mode, let plot_topomap draw its own colorbar.
            panel_show_cbar = show_colorbar and (share_lim == "none")

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
                show_colorbar=panel_show_cbar,
                cbar_label=cbar_label,
                frame_alpha=frame_alpha,
                frame_lw=frame_lw,
                interp_resolution=interp_resolution,
                **topomap_kwargs,
            )
            axes[i, j] = ax

            # Keep a reference image (the last imshow on this ax) for
            # shared colorbars
            for im in ax.get_images():
                images[i, j] = im

    # ------------------------------------------------------------------
    # 4f. Shared colorbars
    # ------------------------------------------------------------------
    if show_colorbar and share_lim == "all":
        # Single colorbar, sized like plot_topomap's default
        # (fraction=0.03, pad=0.02 of the reference axes).
        host_ax = axes[0, n_col - 1]
        cax = host_ax.inset_axes([1.02, 0.2, 0.05, 0.6])
        ref_im = images[0, n_col - 1]
        cb = fig.colorbar(ref_im, cax=cax)
        if cbar_label:
            cb.set_label(cbar_label)

    elif show_colorbar and share_lim == "row":
        # One colorbar per row, to the right of the last column.
        for i in range(n_row):
            host_ax = axes[i, n_col - 1]
            cax = host_ax.inset_axes([1.02, 0.2, 0.05, 0.6])
            ref_im = images[i, n_col - 1]
            cb = fig.colorbar(ref_im, cax=cax)
            if cbar_label:
                cb.set_label(cbar_label)

    elif show_colorbar and share_lim == "col":
        # One colorbar per column, below the last row, horizontal.
        for j in range(n_col):
            host_ax = axes[n_row - 1, j]
            cax = host_ax.inset_axes([0.0, -0.05, 1.0, 0.05])
            ref_im = images[n_row - 1, j]
            cb = fig.colorbar(ref_im, cax=cax, orientation="horizontal")
            if cbar_label:
                cb.set_label(cbar_label)
                cb.set_label(cbar_label)

    # ------------------------------------------------------------------
    # 4g. Row / column labels and titles
    # ------------------------------------------------------------------
    if has_col_labels:
        if len(col_labels) != n_col:
            raise ValueError(
                f"col_labels has {len(col_labels)} entries but data has "
                f"{n_col} columns."
            )
        for j, label in enumerate(col_labels):
            lax = fig.add_subplot(gs[row0 - 1, col0 + j])
            lax.axis("off")
            lax.text(0.5, 0.1, label, ha="center", va="bottom",
                     fontsize=11, transform=lax.transAxes)

    if has_row_labels:
        if len(row_labels) != n_row:
            raise ValueError(
                f"row_labels has {len(row_labels)} entries but data has "
                f"{n_row} rows."
            )
        rlabel_col = col0 - 1
        for i, label in enumerate(row_labels):
            lax = fig.add_subplot(gs[row0 + i, rlabel_col])
            lax.axis("off")
            lax.text(0.9, 0.5, label, ha="right", va="center",
                     fontsize=11, rotation=90, transform=lax.transAxes)

    if has_col_title:
        title_row = 0
        tax = fig.add_subplot(gs[title_row, col0:col0 + n_col])
        tax.axis("off")
        tax.text(0.5, 0.1, col_title, ha="center", va="bottom",
                 fontsize=13, fontweight="bold", transform=tax.transAxes)

    if has_row_title:
        title_col = 0
        tax = fig.add_subplot(gs[row0:row0 + n_row, title_col])
        tax.axis("off")
        tax.text(0.9, 0.5, row_title, ha="right", va="center",
                 fontsize=13, fontweight="bold", rotation=90,
                 transform=tax.transAxes)

    return fig, axes

def color_elec(df, shrink=0.6):
    """
    根据电极的解剖坐标 (r, a, s) 给每个电极分配一个 RGB 颜色，
    使得同一半球、同一脑区内的电极颜色更接近，且左右半球的着色
    完全镜像对称（L12 与 R12 会用同一套 PCA 坐标系统投影，
    只是先把右半球的 r 坐标取负号，相当于把它"翻折"到左半球空间里）。
 
    参数
    ----
    df : pandas.DataFrame
        需要包含列 ['r', 'a', 's', 'region']，电极名 (label, 如 'L12',
        'R12') 既可以是索引 (index.name == 'label')，也可以是一列
        'label'。label 首字母 'L'/'R' 用来判断半球。
    shrink : float, 默认 0.6
        取值范围建议在 [0, 1)。表示每个 (半球, 脑区) 分组内的电极
        向该组质心收缩的比例：shrink=0 表示不收缩（用原始坐标算颜色），
        shrink 越接近 1，同一脑区的电极坐标越接近质心（颜色也就越接近）。
 
    返回
    ----
    dict
        {label: (R, G, B)}，R/G/B 均已归一化到 [0, 1]。
        字典的插入顺序：先左半球（PC0 从正到负，即从后往前），
        再右半球（镜像后的 PC0 从负到正，即从前往后）。
    """
    df = df.copy().dropna()
 
    # ---- 0. 确保 label 是一列 ----
    if df.index.name == 'label' or 'label' not in df.columns:
        df = df.reset_index()
        if 'label' not in df.columns and 'index' in df.columns:
            df = df.rename(columns={'index': 'label'})
 
    df['hemi'] = df['label'].astype(str).str[0]  # 'L' or 'R'
 
    coords = df[['r', 'a', 's']].to_numpy(dtype=float)
 
    # ---- 1. 每个 (半球, 脑区) 分组内，向组质心收缩 ----
    # dropna=False: 让 region 为 NaN 的电极也自成一组（互相收缩），
    # 而不会被 groupby 直接丢弃
    centroids = (
        df.groupby(['hemi', 'region'], dropna=False)[['r', 'a', 's']]
        .transform('mean')
        .to_numpy(dtype=float)
    )
    shrunk = centroids + (1 - shrink) * (coords - centroids)
 
    left_mask = (df['hemi'] == 'L').to_numpy()
    right_mask = (df['hemi'] == 'R').to_numpy()
 
    # ---- 2. 仅用左半球（收缩后）的坐标拟合 PCA ----
    left_coords = shrunk[left_mask]
    mean_left = left_coords.mean(axis=0)
    centered_left = left_coords - mean_left
 
    # SVD 求主成分，Vt 每一行是一个主轴（已按方差从大到小排序）
    _, _, Vt = np.linalg.svd(centered_left, full_matrices=False)
    components = Vt.copy()  # shape (3, 3)
 
    left_scores = centered_left @ components.T  # (n_left, 3) -> PC0, PC1, PC2
 
    # ---- 3. 保证 PC0 的正方向朝后（posterior） ----
    # a 坐标越大越靠前(anterior)，越小越靠后(posterior)。
    # 我们希望 PC0 得分越大 <-> a 越小（负相关）。若目前是正相关，则翻转符号。
    a_left = df.loc[left_mask, 'a'].to_numpy(dtype=float)
    corr0 = np.corrcoef(left_scores[:, 0], a_left)[0, 1]
    if corr0 > 0:
        components[0] *= -1
        left_scores[:, 0] *= -1
 
    # ---- 4. 用左半球的分数范围做 min-max 归一化 ----
    pc_min = left_scores.min(axis=0)
    pc_max = left_scores.max(axis=0)
    pc_range = np.where(pc_max - pc_min == 0, 1.0, pc_max - pc_min)
 
    def normalize(scores):
        return (scores - pc_min) / pc_range
 
    left_norm = np.clip(normalize(left_scores), 0, 1)
    # left_norm = left_scores
    right_norm = left_norm
 
    # ---- 6. 排序并组装结果字典 ----
    labels_left = df.loc[left_mask, 'label'].to_numpy()
    labels_right = df.loc[right_mask, 'label'].to_numpy()
 
    # 左半球：PC0 从正到负 -> 从后往前
    order_left = np.argsort(-left_scores[:, 0])
    # 右半球：镜像后的 PC0 从负到正 -> 从前往后
    order_right = np.argsort(left_scores[:, 0])
 
    color_dict = {}
    for i in order_left:
        color_dict[labels_left[i]] = left_norm[i]
    for i in order_right:
        color_dict[labels_right[i]] = right_norm[i]
 
    return color_dict