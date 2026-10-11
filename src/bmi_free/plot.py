import os
import time
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import mne

from bmi_free.io import import_joints


palette = {'monkey':
           {'A': '#6B4F3F', 'B': "#BCA07F"},
           'joint':
           dict(zip(import_joints(),
                    [
               (0.5, 0.5, 0.1),
               (0.5, 0.5, 0.3),
               (0.5, 0.5, 0.5),
               (0.5, 0.65, 0.5),
               (0.5, 0.8, 0.5),
               (0.5, 0.95, 0.5),
               (0.35, 0.5, 0.5),
               (0.2, 0.5, 0.5),
               (0.05, 0.5, 0.5),
               (0.5, 0.5, 0.7),
               (0.5, 0.5, 0.9),
               (0.65, 0.5, 0.5),
               (0.8, 0.5, 0.5),
               (0.95, 0.5, 0.5),
               (0.5, 0.35, 0.5),
               (0.5, 0.2, 0.5),
               (0.5, 0.05, 0.5),
           ]))
           }
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from itertools import product, combinations


def plot_cube_faces(data, cmap='viridis', vmin=None, vmax=None,
                     elev=25, azim=-45, figsize=(8, 8),
                     title=None, ax=None):
    """
    输入一个三维数组 data[nx, ny, nz]，在立方体的三个可见外表面上
    画出 pcolormesh 风格的图（顶面 z=nz-1, 正面 y=0, 侧面 x=nx-1）。

    Parameters
    ----------
    data : numpy.ndarray, shape (nx, ny, nz)
        三维数据。
    cmap : str or matplotlib.colors.Colormap
        颜色映射，默认 'viridis'。
    vmin : float, optional
        颜色范围下界，默认用 data 的全局 min。
    vmax : float, optional
        颜色范围上界，默认用 data 的全局 max。
    elev : float
        3D 视角仰角，默认 25。
    azim : float
        3D 视角方位角，默认 -45。
        默认 (25, -45) 对应取 x=nx-1, y=0, z=nz-1 三个面朝向相机。
        如果你改了视角，请确认对应的暴露面是否还正确。
    figsize : tuple of (float, float)
        图形大小，默认 (8, 8)。
    title : str, optional
        图形标题。
    ax : matplotlib.axes.Axes, optional
        传入已有的 3D 坐标轴则画在上面，否则新建一个 figure。

    Returns
    -------
    fig : matplotlib.figure.Figure
        图形对象。
    ax : matplotlib.axes.Axes
        3D 坐标轴。
    mappable : matplotlib.cm.ScalarMappable
        可用来添加 colorbar。
    """
    data = np.asarray(data)
    if data.ndim != 3:
        raise ValueError(f"data 必须是三维数组，当前 shape={data.shape}")
    nx, ny, nz = data.shape

    if vmin is None:
        vmin = np.nanmin(data)
    if vmax is None:
        vmax = np.nanmax(data)
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    def draw(X, Y, Z, values):
        facecolors = cmap(norm(values))
        ax.plot_surface(X, Y, Z, facecolors=facecolors,
                         rstride=1, cstride=1, shade=False,
                         linewidth=0, antialiased=False)

    # 网格坐标用整数下标的边界（0..n），使每个体素格子对应一个颜色格
    xg = np.arange(nx + 1)
    yg = np.arange(ny + 1)
    zg = np.arange(nz + 1)

    # ---- 顶面 z = nz (取 data[:, :, -1]) ----
    X, Y = np.meshgrid(xg, yg, indexing='ij')
    Z = np.full_like(X, nz, dtype=float)
    draw(X, Y, Z, data[:, :, -1])

    # ---- 正面 y = 0 (取 data[:, 0, :]) ----
    X, Z = np.meshgrid(xg, zg, indexing='ij')
    Y = np.zeros_like(X, dtype=float)
    draw(X, Y, Z, data[:, 0, :])

    # ---- 侧面 x = nx (取 data[-1, :, :]) ----
    Y, Z = np.meshgrid(yg, zg, indexing='ij')
    X = np.full_like(Y, nx, dtype=float)
    draw(X, Y, Z, data[-1, :, :])

    # 立方体边框线（可选，增强立体感）
    r_x, r_y, r_z = [0, nx], [0, ny], [0, nz]
    corners = np.array(list(product(r_x, r_y, r_z)))
    for s, e in combinations(corners, 2):
        diffs = np.abs(s - e)
        if np.sum(diffs > 0) == 1:  # 只连同一条棱上的顶点
            ax.plot3D(*zip(s, e), color="k", linewidth=0.8, alpha=0.4)

    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_zlim(0, nz)
    ax.set_box_aspect([nx, ny, nz])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()

    if title:
        ax.set_title(title)

    mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    return fig, ax, mappable

class NMTECoGMapper:
    """
    Python 封装类，用于调用 MATLAB 函数 nmt_ecog_map，
    并支持按样本 × 分组的批量可视化布局。
    """

    # ======================================================================
    # 初始化与基础配置
    # ======================================================================

    def __init__(self, maptmp='ecog64_20250606', matlab_engine=None, **kwargs):
        """
        初始化映射器并启动 MATLAB 引擎。

        Parameters
        ----------
        maptmp : str
            电极布局模板名称
        matlab_engine : matlab.engine.MatlabEngine | None
            已存在的 MATLAB 引擎实例
        """
        import matlab.engine

        if matlab_engine is None:
            self.eng = matlab.engine.start_matlab()
        else:
            self.eng = matlab_engine

        self.set_maptmp(maptmp)

        # 默认 ft_cfg
        self._ft_cfg = None
        self.set(**kwargs)

        # 启动 FieldTrip
        while not self.eng:
            time.sleep(0.5)
        self.eng.ft_defaults(nargout=0)

    def set_maptmp(self, maptmp):
        """设置电极布局模板"""
        self.maptmp = maptmp

    # ======================================================================
    # ft_sourceplot 参数配置
    # ======================================================================

    def set(self,
            method='surface', anaparameter=None, funparameter='powspctrm',
            maskparameter=None, maskstyle='opacity', downsample=1, atlas=None,
            visible='off', figure='yes', position=[1, 1, 3.6, 4.5], renderer='opengl',
            funcolormap='jet', funcolorlim=None, colorbar='yes',
            colorbartext='', latency=None, avgovertime='no', frequency=None,
            avgoverfreq='no', opacitymap='auto', opacitylim='auto', roi=None,
            location=None, locationcoordinates='head', crosshair='yes',
            axis='on', queryrange=1, clim=None, nslices=1, slicerange='auto',
            slicedim=3, title=None, figurename=None, surffile=None,
            surfinflated=None, surfdownsample=1, interpmethod='sphere_weighteddistance',
            projvec=None, projcomb='mean', projweight=None, projthresh=None,
            sphereradius=5, distmat=None, camlight='no', facecolor=None,
            vertexcolor='curv', edgecolor='none', cloudtype='cloud',
            radius=4, colorgrad='white', slice='none', ori='y',
            slicepos='auto', nslices_cloud=1, minspace=1, flip=None,
            interactive='yes', voxelratio='data', axisratio='data',
            inputfile=None
            ):
        """
        设置 FieldTrip ft_sourceplot 的配置参数。

        Parameters
        ----------
        method : str
            可视化方法，如 'surface', 'ortho', 'slice', 'glassbrain' 等。
        anaparameter : str, optional
            解剖数据字段名。
        funparameter : str
            功能参数字段名，默认 'powspctrm'。
        maskparameter : str, optional
            透明度掩码字段名。
        maskstyle : str
            掩码样式，默认 'opacity'。
        downsample : int
            降采样因子，默认 1。
        atlas : str, optional
            atlas 文件名。
        visible : str
            图形是否可见，默认 'off'。
        figure : str
            是否打开新图形，默认 'yes'。
        position : list
            图形位置和大小 [left, bottom, width, height]。
        renderer : str
            渲染器，默认 'opengl'。
        funcolormap : str or matplotlib Colormap
            功能数据颜色映射，默认 'jet'。
        funcolorlim : list, optional
            功能数据颜色范围 [min, max]。
        colorbar : str
            是否显示 colorbar，'yes' 或 'no'。
        colorbartext : str
            colorbar 旁文本。
        latency : float or str, optional
            时间范围（秒）。
        avgovertime : str
            是否对时间平均，默认 'no'。
        frequency : float or str, optional
            频率范围（Hz）。
        avgoverfreq : str
            是否对频率平均，默认 'no'。
        opacitymap : str
            透明度映射，默认 'auto'。
        opacitylim : str
            透明度范围，默认 'auto'。
        roi : str or list of str, optional
            感兴趣区域。
        location : optional
            切割位置。
        locationcoordinates : str
            位置坐标系，默认 'head'。
        crosshair : str
            是否显示十字线，默认 'yes'。
        axis : str
            是否显示坐标轴，默认 'on'。
        queryrange : int
            atlas 查询范围，默认 1。
        clim : list, optional
            解剖 MRI 颜色范围。
        nslices : int
            切片数，默认 1。
        slicerange : str
            切片范围，默认 'auto'。
        slicedim : int
            切片维度 (1=x, 2=y, 3=z)，默认 3。
        title : str, optional
            图形标题。
        figurename : str, optional
            图形窗口标题。
        surffile : str, optional
            表面文件路径。
        surfinflated : str, optional
            膨胀表面文件路径。
        surfdownsample : int
            表面降采样因子，默认 1。
        interpmethod : str
            投影插值方法，默认 'sphere_weighteddistance'。
        projvec : optional
            投影向量 (mm)。
        projcomb : str
            投影组合方法，默认 'mean'。
        projweight : optional
            投影权重向量。
        projthresh : optional
            表面级阈值。
        sphereradius : float
            球体投影最大距离 (mm)，默认 5。
        distmat : optional
            预计算距离矩阵。
        camlight : str
            是否使用 camlight，默认 'no'。
        facecolor : optional
            面颜色。
        vertexcolor : str
            顶点颜色，默认 'curv'。
        edgecolor : str
            边颜色，默认 'none'。
        cloudtype : str
            云类型，默认 'cloud'。
        radius : float
            云最大半径，默认 4。
        colorgrad : str
            云颜色梯度，默认 'white'。
        slice : str
            切片模式，默认 'none'。
        ori : str
            切片方向，默认 'y'。
        slicepos : str
            切片位置，默认 'auto'。
        nslices_cloud : int
            云切片数，默认 1。
        minspace : int
            最小切片间距，默认 1。
        flip : str, optional
            是否翻转。
        interactive : str
            是否交互，默认 'yes'。
        voxelratio : str
            体素显示比例，默认 'data'。
        axisratio : str
            坐标轴比例，默认 'data'。
        inputfile : str, optional
            输入文件路径。
        """

        # check cmap
        funcolormap = self._check_cmap(funcolormap)

        # 处理colorbar参数，保证只在plt里画图用到
        colorbar = self._check_colorbar(colorbar)

        # 构建ft_cfg结构体
        frame = inspect.currentframe()
        args_info = inspect.getargvalues(frame)

        ft_cfg = {arg: locals()[arg]
                  for arg in args_info.args if arg != 'self'}  # 过滤掉None值

        # 处理需要转换为MATLAB数组并reshape的参数
        params = ['funcolorlim', 'position']

        for param in params:
            # 检查参数是否存在且不为None
            if param in ft_cfg.keys():
                if ft_cfg[param] is not None:
                    # 将参数转换为MATLAB数组并reshape
                    ft_cfg.update({param: self._convert_to_matlab_array(
                        ft_cfg[param]).reshape(1, -1)})

        # 过滤None值
        ft_cfg = {k: v for k, v in ft_cfg.items() if v is not None}
        ft_cfg = {k: np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v
                    for k, v in ft_cfg.items()}

        # 识别是否首次调用，构建默认ft_cfg结构体
        if self._ft_cfg is None:
            self._ft_cfg = ft_cfg
            self.ft_cfg = ft_cfg
        else:
            # 检查和默认结构体中不一样的部分
            ft_cfg = {k: v for k, v in ft_cfg.items(
            ) if k not in self._ft_cfg or v is not self._ft_cfg[k]}
            self.ft_cfg.update(ft_cfg)

        

    def _convert_to_matlab_array(self, array):
        """
        将输入数组转换为 MATLAB 兼容的 double 数组。

        Parameters
        ----------
        array : array-like
            输入数组。

        Returns
        -------
        numpy.ndarray
            dtype 为 double 的数组。
        """
        return np.array(array).astype('double')

    # ======================================================================
    # values / picks 规范化（核心输入逻辑）
    # ======================================================================

    def _check_values(self, values, picks):
        """
        将不同类型的 values 统一整理为 (channels, samples, groups) 的三维 numpy 数组。

        Parameters
        ----------
        values : numpy.ndarray, pandas.Series, pandas.DataFrame, or array-like
            输入数据，支持 1–3 维。
        picks : list of str or int, optional
            通道名称或索引列表。

        Returns
        -------
        arr : numpy.ndarray, shape (n_channels, n_samples, n_groups)
            规范化后的三维数组。
        picks : list
            通道名称列表。
        """

        def _is_str_axis(obj):
            return all(isinstance(x, str) for x in obj)

        def _ensure_3d(arr):
            if arr.ndim == 1:
                return arr[:, None, None]
            elif arr.ndim == 2:
                return arr[:, :, None]
            elif arr.ndim == 3:
                return arr
            else:
                raise ValueError("values 只允许 1–3 维")

        # 规范 picks
        if picks is not None:
            if isinstance(picks, (str, int)):
                picks = [picks]
            else:
                picks = list(picks)

        # ---------- numpy array ----------
        if isinstance(values, np.ndarray):
            arr = values

            if picks is None:
                ch_axis = 0
            else:
                axes = [i for i, s in enumerate(arr.shape) if s == len(picks)]
                ch_axis = axes[0] if axes else 0

            if ch_axis != 0:
                arr = np.moveaxis(arr, ch_axis, 0)

            return _ensure_3d(arr), picks or []

        # ---------- pandas Series ----------
        if isinstance(values, pd.Series):
            s = values

            if picks is None and _is_str_axis(s.index):
                picks = list(s.index)

            elif picks is not None and _is_str_axis(s.index):
                if not set(picks).issubset(s.index):
                    raise ValueError("picks 不是 Series index 的子集")
                s = s.loc[picks]

            elif picks is not None and len(s) != len(picks):
                raise ValueError("Series 长度与 picks 不一致")

            return _ensure_3d(s.to_numpy()), picks or []

        # ---------- pandas DataFrame ----------
        if isinstance(values, pd.DataFrame):
            df = values
            idx_str = _is_str_axis(df.index)
            col_str = _is_str_axis(df.columns)

            if picks is None:
                if idx_str ^ col_str:
                    ch_axis = 0 if idx_str else 1
                elif idx_str and col_str:
                    ch_axis = 0
                    picks = list(df.index)
                else:
                    ch_axis = 0
            else:
                if len(df.index) == len(picks):
                    ch_axis = 0
                elif len(df.columns) == len(picks):
                    ch_axis = 1
                elif idx_str and set(picks).issubset(df.index):
                    df = df.loc[picks]
                    ch_axis = 0
                elif col_str and set(picks).issubset(df.columns):
                    df = df[picks]
                    ch_axis = 1
                else:
                    raise ValueError("无法根据 picks 确定通道轴")

            arr = df.to_numpy()
            if ch_axis == 1:
                arr = arr.T

            return _ensure_3d(arr), picks or []

        # ---------- 其他 ----------
        arr = np.asarray(values)
        return _ensure_3d(arr), picks or []

    # ======================================================================
    # compute / render / layout 三层
    # ======================================================================

    def _compute_maps(self, values, picks, savepath):
        """
        逐样本 × 分组调用 MATLAB nmt_ecog_map，每次生成一张临时 png。

        Parameters
        ----------
        values : numpy.ndarray, shape (n_channels, n_samples, n_groups)
            输入数据。
        picks : list
            通道名称列表。
        savepath : str or Path, optional
            图片保存路径；None 则自动生成临时文件名。

        Returns
        -------
        dict
            键为 (sample_idx, group_idx)，值为 MATLAB 返回的输出字典。
        """

        auto_save = True if savepath is None else False
        n_ch, n_samp, n_grp = values.shape
        outputs = {}

        for i in range(n_samp):
            for j in range(n_grp):
                if auto_save:
                    savepath = f"tmp_{i}_{j}.png"
                opts = ["picks", picks, "savepath", savepath]

                out = self.eng.nmt_ecog_map(
                    np.ascontiguousarray(values[:, i, j], dtype='double'),
                    self.maptmp,
                    self.ft_cfg,
                    *opts
                )
                self.ft_cfg["funcolorlim"] = out['clim']
                outputs[(i, j)] = out

        return outputs

    def _render_single(self, ax, png_path):
        """
        将单张 png 渲染到 matplotlib Axes。

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            目标坐标轴。
        png_path : str
            png 文件路径。
        """
        ax.imshow(plt.imread(png_path))
        ax.axis("off")

    def _layout_maps(self, outputs, n_samples, n_groups, share_x, share_y, row_labels, col_labels):
        """
        拼接所有子图，并根据 share_x / share_y 控制 colorbar 显示。

        Parameters
        ----------
        outputs : dict
            _compute_maps 的返回值。
        n_samples : int
            样本数（列数）。
        n_groups : int
            分组数（行数）。
        share_x : bool
            是否在列间共享颜色范围。
        share_y : bool
            是否在行间共享颜色范围。
        row_labels : list of str, optional
            行标签。
        col_labels : list of str, optional
            列标签。

        Returns
        -------
        matplotlib.figure.Figure
            拼接后的图形。
        """
        margin = 0.75

        base_w, base_h = self.ft_cfg["position"][0][2:]
        fig = plt.figure(figsize=(base_w * n_samples * 1.2, base_h * n_groups * 1.1))

        for (i, j), out in outputs.items():
            x0 = i / n_samples + 0.1
            y0 = 1 - (j + 1) / n_groups - 0.1
            ax = fig.add_axes([x0, y0, margin/n_samples, margin/n_groups])
            
            self._render_single(ax, out["savepath"])
            if 'tmp' in out["savepath"]:
                os.remove(out["savepath"])

            if self.plot_colorbar:
                show_cb = (
                    (share_x and share_y and i == n_samples - 1 and j == n_groups - 1) or
                    (share_x and not share_y and j == n_groups - 1) or
                    (share_y and not share_x and i == n_samples - 1) or
                    (not share_x and not share_y)
                )
                if show_cb:
                    self.colorbar(fig, ax)

                # if (not share_x) and (not share_y):
                #     self.ft_cfg["funcolorlim"] = 'auto'
    
            if row_labels is not None and (i == 0 or not share_x):
                fig.text(x0-0.05, y0 + 0.5 * margin/n_groups, row_labels[j], rotation=90, va="center", fontsize='large')
            
            if col_labels is not None and (j == 0 or not share_y):
                fig.text(x0 + 0.5 * margin/n_samples, y0 + 0.05 + margin/n_groups, col_labels[i], ha="center", fontsize='large')



        return fig

    # ======================================================================
    # map 主接口
    # ======================================================================

    def map(self, values, picks=None, plot=True, share_x=False, share_y=False, savepath=None, row_labels=None, col_labels=None, **kwargs):
        """
        高层接口：完成 values 规范化 → MATLAB 计算 → 图像布局。

        Parameters
        ----------
        values : numpy.ndarray, pandas.Series, pandas.DataFrame, or array-like
            输入数据。
        picks : list, optional
            通道名称列表。
        plot : bool
            若 True，返回 fig；否则返回 None。
        share_x : bool
            列间共享颜色范围。
        share_y : bool
            行间共享颜色范围。
        savepath : str or Path, optional
            图片保存路径。
        row_labels : list of str, optional
            行标签。
        col_labels : list of str, optional
            列标签。
        **kwargs
            转发给 set() 的额外配置参数。

        Returns
        -------
        matplotlib.figure.Figure or None
            若 plot=True 返回图形，否则返回 None。
        """
        values, picks = self._check_values(values, picks)
        n_ch, n_samp, n_grp = values.shape

        self.set(**kwargs)

        # funcolorlim 自动推断
        if self.ft_cfg.get("funcolorlim", None) is None:
            if share_x and share_y:
                lo, hi = np.percentile(values, [5, 95])
            elif share_x:
                lo, hi = np.percentile(values, [5, 95], axis=1)
                lo, hi = lo.min(), hi.max()
            elif share_y:
                lo, hi = np.percentile(values, [5, 95], axis=2)
                lo, hi = lo.min(), hi.max()
            else:
                self.ft_cfg["funcolorlim"] = 'auto'
                lo = hi = None

            if lo is not None:
                self.ft_cfg["funcolorlim"] = np.array([[lo, hi]])


        outputs = self._compute_maps(values, picks, savepath)
        fig = self._layout_maps(outputs, n_samp, n_grp, share_x, share_y, row_labels, col_labels)

        return fig if plot else None

    # ======================================================================
    # 工具函数
    # ======================================================================

    def _check_cmap(self, cmap):
        """
        检查并转换 colormap 为 MATLAB 兼容的 (256, 3) RGB 数组。

        Parameters
        ----------
        cmap : str or matplotlib.colors.Colormap
            colormap 名称或对象。

        Returns
        -------
        numpy.ndarray, shape (256, 3)
            RGB 颜色表。
        """
        if isinstance(cmap, str):
            if cmap not in plt.colormaps():
                raise ValueError(f"{cmap} 不是 matplotlib 内置 colormap")
            self.cmap = cmap
            return plt.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
        elif isinstance(cmap, Colormap):
            self.cmap = cmap
            return cmap(np.linspace(0, 1, 256))[:, :3]
        return cmap

    def _check_colorbar(self, colorbar):
        """
        检查 colorbar 参数并记录是否需要绘制 colorbar。

        Parameters
        ----------
        colorbar : str
            'yes' 或 'no'。

        Returns
        -------
        str
            始终返回 'no'（MATLAB 侧不画 colorbar，由 Python 侧控制）。
        """
        self.plot_colorbar = (colorbar == "yes")
        return "no"
    

    def colorbar(self, fig, ax):
        """
        在指定坐标轴的右侧添加 colorbar。

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            目标图形。
        ax : matplotlib.axes.Axes
            目标坐标轴。
        """
        norm = plt.Normalize(
            self.ft_cfg["funcolorlim"][0][0],
            self.ft_cfg["funcolorlim"][0][1]
        )
        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=self.cmap),
            # cax=ax.inset_axes([0.72, 0.2, 0.05, 0.6]),
            cax=ax.inset_axes([1.05, 0.2, 0.05, 0.6]),
            label=self.ft_cfg.get("colorbartext", "")
        )

    def close(self):
        """
        关闭 MATLAB 引擎实例。
        """
        if hasattr(self, "eng"):
            self.eng.quit()

# def show_features(X):
#     # 检查特征是否有伪迹
#     plt.figure(figsize=(10, 3))
#     plt.plot(X.mean(axis=(1, 2)))
#     plt.axis('off')
#     plt.show()

def plot_ica_properties_grid(ica, inst, **kwargs):
    """
    以网格布局绘制 ICA 各分量的属性图（拓扑、ERP 图像、ERP、PSD、方差）。

    Parameters
    ----------
    ica : mne.preprocessing.ICA
        拟合好的 ICA 对象。
    inst : mne.io.Raw
        原始数据对象。
    **kwargs
        转发给 ica.plot_properties 的额外参数。
        支持特殊参数：
        - picks : 范围或列表，选择要绘制的 IC 索引
        - psd_args : dict，传给 PSD 绘制的参数
        - width_ratios : 长度为 4 的序列，四列（topo / image+ERP / PSD / variance）
          的相对宽度，默认 (1, 2, 2, 5)                  

    Returns
    -------
    matplotlib.figure.Figure
        绘制好的图形。
    """
    picks = kwargs.pop('picks', range(ica.n_components))
    psd_args = kwargs.pop('psd_args', {'fmin': 1, 'fmax': 150})
    width_ratios = kwargs.pop('width_ratios', (1, 2, 2, 5))
    var_hist_ratio = kwargs.pop('var_hist_ratio', width_ratios[3]*2)   
    n_components = len(picks)

    fig = plt.figure(figsize=(10, n_components * 1.5))

    outer = gridspec.GridSpec(
        n_components, 4,
        figure=fig,
        hspace=0.15,
        wspace=0.35,
        width_ratios=list(width_ratios), 
    )

    ref_x = {'topo': None, 'spectrum': None, 'variance': None}
    ref_y = {'erp': None, 'spectrum': None, 'variance': None}

    def add(spec, xkey=None, ykey=None):
        ax = fig.add_subplot(
            spec,
            sharex=ref_x.get(xkey),
            sharey=ref_y.get(ykey),
        )
        if xkey and ref_x.get(xkey) is None:
            ref_x[xkey] = ax
        if ykey and ref_y.get(ykey) is None:
            ref_y[ykey] = ax
        return ax
    
    known_axes = set()  
    hist_axes = []  

    for row, ic_idx in enumerate(picks):
        ax_topo     = add(outer[row, 0], xkey='topo')
        ax_spectrum = add(outer[row, 2], xkey='spectrum', ykey='spectrum')
        ax_variance = add(outer[row, 3], xkey='variance')

        inner = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[row, 1], hspace=0, height_ratios=[2, 1]
        )
        ax_image = fig.add_subplot(inner[0])
        ax_erp   = add(inner[1], ykey='erp')

        ica.plot_properties(
            mne.make_fixed_length_epochs(inst, duration=1.0, preload=True, verbose="Critical"),
            picks=ic_idx,
            axes=[ax_topo, ax_image, ax_erp, ax_spectrum, ax_variance],
            show=False,
            psd_args=psd_args,
            verbose="Critical",
            **kwargs
        )

        # ---- PSD: x 轴 log，次级刻度每 10 Hz 一格 ----
        ax_spectrum.set_xscale('log')
        ax_spectrum.set_xlim(psd_args.get('fmin', 1), psd_args.get('fmax', 150))
        ax_spectrum.xaxis.set_major_locator(ticker.FixedLocator([1, 10, 100]))
        ax_spectrum.xaxis.set_major_formatter(ticker.ScalarFormatter())
        ax_spectrum.xaxis.set_minor_locator(ticker.MultipleLocator(10))
        ax_spectrum.xaxis.set_minor_formatter(ticker.NullFormatter())

        ax_topo.set_ylabel(f"IC {ic_idx:03d}", fontsize='large')
        for ax in [ax_topo, ax_image, ax_spectrum, ax_variance]:
            ax.set_title("")
        ax_spectrum.set_ylabel("PSD (dB/Hz)")

        # 缩小散点
        for collection in ax_variance.collections:
            collection.set_sizes([4])

        # 对 variance 轴进行自定义刻度
        ax_variance.set_xlim(0, 30 * 60)
        ax_variance.xaxis.set_major_locator(ticker.MultipleLocator(5 * 60))
        ax_variance.xaxis.set_minor_locator(ticker.MultipleLocator(1 * 60))
        ax_variance.xaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: f"{int(x/60)}")
        )        

        is_last = (row == n_components - 1)

        # x轴：erp 始终保留，其余非最后一行隐藏
        if not is_last:
            for ax in [ax_topo, ax_image, ax_spectrum, ax_variance, ax_erp]:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.set_xlabel("")
        else:
            ax_variance.set_xlabel("Time (min)")

        # erp 始终强制显示 x 轴（MNE 可能会隐藏它）
        ax_erp.xaxis.set_tick_params(labelbottom=True)

        # 对ax_extra进行处理，确保只处理本行的额外轴
        extra_axes = [ax for ax in fig.get_axes() if ax not in known_axes]
        # extra_axes 里前5个是本行的，剩下的是 MNE 自动添加的
        extra_axes = [ax for ax in extra_axes 
                      if ax not in {ax_topo, ax_image, ax_erp, ax_spectrum, ax_variance}]
        
        for i, ax_extra in enumerate(extra_axes):
            if row == 0:
                ref_x['variance_twin'] = ax_extra
            else:
                # 新版 matplotlib 的正确共享方式
                ax_extra.sharex(ref_x['variance_twin'])
            if not is_last:
                plt.setp(ax_extra.get_xticklabels(), visible=False)
                ax_extra.set_xlabel("")
            else:
                ax_extra.set_xticklabels([''] + ax_extra.get_xticklabels()[1:])

        hist_axes.append((ax_variance, list(extra_axes)))
        known_axes.update(fig.get_axes())  # 记录本行结束后所有已知轴            


    plt.tight_layout()
    fig.align_labels()

    # ---- 调整方差散点图 : 直方图 = var_hist_ratio : 1 ----
    fig.canvas.draw()   # 让 axes_grid1 的 locator 先算出实际位置
    for ax_var, extras in hist_axes:
        if not extras:
            continue
        ax_hist = max(extras, key=lambda a: a.get_position().x0)   # 最靠右的就是直方图
        pv, ph = ax_var.get_position(), ax_hist.get_position()

        left, right = pv.x0, ph.x1
        pad = max(ph.x0 - pv.x1, 0)
        usable = (right - left) - pad
        w_scatter = usable * var_hist_ratio / (var_hist_ratio + 1)
        w_hist = usable / (var_hist_ratio + 1)

        # 去掉 locator，否则 set_position 会被 axes_grid1 覆盖
        ax_var.set_axes_locator(None)
        ax_hist.set_axes_locator(None)
        ax_var.set_position([left, pv.y0, w_scatter, pv.height])
        ax_hist.set_position([left + w_scatter + pad, ph.y0, w_hist, ph.height])

    return fig

if __name__ == "__main__":
    """
    示例：
    - 构造 channels × samples × groups 的模拟数据
    - 使用 share_x / share_y 进行统一颜色映射
    - 展示多样本 × 多分组的布局效果
    """

    # -------------------------------
    # 创建 Mapper
    # -------------------------------
    mapper = NMTECoGMapper(
        maptmp="ecog64_20250606"
    )

    # -------------------------------
    # 构造示例数据
    # -------------------------------
    # 假设：
    #   - 64 个通道
    #   - 5 个样本（例如时间窗 / trial）
    #   - 3 个分组（例如条件）
    n_channels = 64
    n_samples = 3
    n_groups = 2

    values = np.random.rand(
        n_channels, n_samples, n_groups
    )

    # -------------------------------
    # 调用 map
    # -------------------------------
    axes = mapper.map(
        values,
        share_x=True,
        share_y=True,
        funcolormap="RdBu_r",
        colorbar="yes",
        colorbartext="Power (a.u.)",
    )

    # -------------------------------
    # 显示结果
    # -------------------------------
    plt.show()

    # -------------------------------
    # 清理资源
    # -------------------------------
    mapper.close()