function varargout = plot_skeleton(mode, varargin)
%PLOT_SKELETON 骨架可视化的 MATLAB 入口（供 Python MATLAB Engine 调用）
%
% MATLAB 的函数文件只有"文件里第一个函数"能被外部（包括 Python MATLAB
% Engine）直接调用，其余函数只能在同一文件内部相互调用（local function）。
% 这里用 mode 参数做分发，Python 侧统一调用本函数，通过第一个参数 mode
% 决定具体执行 'grid'（单帧也走这里，n_rows=n_cols=1）还是 'animation'。
%
% 架构说明（重构要点）:
%   所有"画笼子环境"（坐标轴范围 / 地面网格 / 爬杆 / 墙壁 / 视角 / 光照）
%   的逻辑都收敛到 local function plot_cage() 中；所有"画骨架本体"
%   （点 / 连线 / 三角面）的逻辑都收敛到 local function
%   plot_skeleton_core() 中（因本文件已叫 plot_skeleton，local function
%   不能与之同名，故加 _core 后缀）。
%     - plot_skeleton_grid   : 每个子图各调用一次 plot_cage + plot_skeleton_core。
%     - plot_skeleton_animation : 只调用一次 plot_cage + plot_skeleton_core
%       建场景，之后在循环里直接更新 plot_skeleton_core 返回的图形句柄的
%       XData/YData/ZData/Vertices，而不是每帧重新建图（更快）。
%   两个功能函数都用 arguments 块声明参数（必选位置参数 + name-value
%   可选参数），而不是用 nargin 手工判断——以后增删参数只需改 arguments
%   块，不用逐个调整 nargin 判断和默认值链。
%
% 用法 (通常由 skeleton.py 自动调用，不需要手动调用):
%
%   fig = plot_skeleton('grid', V_all, edges, faces, bounds_all, titles, ...
%                        n_rows, n_cols, Name, Value, ...)
%
%   plot_skeleton('animation', V_all, edges, faces, bounds, fps, filename, ...
%                 Name, Value, ...)   (无返回值)
%
% 两个 mode 通用的 Name-Value 参数见 plot_cage() 的说明:
%   'show_floor', 'show_poles', 'show_walls', 'show_axis', 'view_az_el'
%   ('grid' 额外有 'save_as', 'visible'；'animation' 额外有 'visible')

switch mode
    case 'grid'
        varargout{1} = plot_skeleton_grid(varargin{:});
    case 'animation'
        plot_skeleton_animation(varargin{:});
    otherwise
        error('plot_skeleton:UnknownMode', ...
            '未知 mode: %s，可选 ''grid'' / ''animation''', mode);
end

end


%% ============================================================
%  两个顶层功能: grid（含单帧）/ animation
%% ============================================================

function fig = plot_skeleton_grid(V_all, edges, faces, bounds_all, titles, n_rows, n_cols, opts)
%PLOT_SKELETON_GRID 单帧 / 多帧网格绘图（tiledlayout 子图）
%
% 单帧场景：V_all 传 [1 x N x 3]，n_rows=n_cols=1 即可；内部与多帧网格
% 走同一套逻辑（同一个 for 循环），只是恰好只有一个子图、不建 tiledlayout。
%
% 输入 (索引均为 1-based，Python 侧已从 0-based 转换好):
%   V_all      : T x N x 3 double，逐帧关节坐标（T=1 即单帧）
%   edges      : M x 2 double，连线顶点索引；无连线则传 []
%   faces      : F x 3 double，三角面顶点索引（由 edges 自动检测得到）；无则传 []
%   bounds_all : T x 6 double，每帧各自的 [xmin xmax ymin ymax zmin zmax]；
%                [] 则每帧都用 plot_cage 的默认笼子范围
%   titles     : 1 x T cell，每个子图标题；{} 则不加标题
%   n_rows, n_cols : 网格行列数（T=1 时通常为 1, 1；tiledlayout 只会用到
%                    的 nexttile 是前 T 个格子，多余格子自动留白）
%
% Name-Value 参数:
%   save_as     : char，保存为 .fig 文件的完整路径；'' 则不保存（默认）
%   visible     : logical，是否显示图窗，默认 true
%   show_floor  : logical，是否画地面网格，默认 false
%   show_poles  : logical，是否画笼子爬杆，默认 false
%   show_walls  : logical，是否画墙壁网格线 + F/L 标签，默认 false
%   show_axis   : logical，是否显示坐标轴刻度，默认 false
%   view_az_el  : 1 x 2 [az, el]；[] 用 plot_cage 默认视角
%
% 输出:
%   fig : 图窗句柄，供 Python 侧继续调用 savefig / exportgraphics 等

arguments
    V_all (:,:,:) double
    edges (:,2) double = zeros(0,2)
    faces (:,3) double = zeros(0,3)
    bounds_all (:,6) double = zeros(0,6)
    titles (1,:) cell = {}
    n_rows (1,1) double {mustBePositive, mustBeInteger} = 1
    n_cols (1,1) double {mustBePositive, mustBeInteger} = 1
    opts struct = struct
    % opts.save_as (1,:) char = ''
    % opts.visible (1,1) logical = true
    % opts.show_floor (1,1) logical = false
    % opts.show_poles (1,1) logical = false
    % opts.show_walls (1,1) logical = false
    % opts.show_axis (1,1) logical = false
    % opts.view_az_el double = []
end

n = size(V_all, 1);
fig = figure('Color', 'w', 'Visible', get(opts, 'visible', true));

if n_rows == 1 && n_cols == 1
    ax_list = {axes(fig)};
else
    tl = tiledlayout(fig, n_rows, n_cols, 'TileSpacing', 'compact', 'Padding', 'compact');
    ax_list = arrayfun(@(~) nexttile(tl), 1:n, 'UniformOutput', false);
end

for k = 1:n
    ax = ax_list{k};
    V = squeeze(V_all(k, :, :));
    if isvector(V)
        V = reshape(V, [], 3);  % 只有 1 个关节点时 squeeze 可能压成行/列向量，纠正形状
    end

    if ~isempty(bounds_all)
        b = bounds_all(k, :);
    else
        b = [];
    end
    opts.bound = b;

    plot_cage(ax, opts);
    plot_skeleton_core(ax, V, edges, faces, opts);

    if ~isempty(titles) && k <= numel(titles) && ~isempty(titles{k})
        title(ax, titles{k}, 'Interpreter', 'none');
    end
end

if get(opts, 'save_as', false)
    savefig(fig, opts.save_as);
end

end


function plot_skeleton_animation(V_all, edges, faces, bounds, fps, filename, opts)
%PLOT_SKELETON_ANIMATION 生成骨架动画视频
%
% 只建一次场景（一次 plot_cage + 一次 plot_skeleton_core 拿到图形句柄），
% 之后在循环里更新这些句柄的 XData/YData/ZData/Vertices（而不是每帧重新
% 画图），再写入视频，整段动画只需一次 MATLAB Engine 调用即可完成。
%
% 输入 (索引均为 1-based，Python 侧已从 0-based 转换好):
%   V_all      : T x N x 3 double，逐帧关节坐标
%   edges      : M x 2 double，连线索引；无则传 []
%   faces      : F x 3 double，三角面索引；无则传 []
%   bounds     : 1 x 6 double，坐标轴范围（动画期间固定不变）；[] 用默认值
%   fps        : 帧率，默认 30
%   filename   : 输出 mp4 文件路径，默认 'animation.mp4'
%
% Name-Value 参数（同 plot_skeleton_grid，除 save_as 外）:
%   visible, show_floor, show_poles, show_walls, show_axis, view_az_el

arguments
    V_all (:,:,:) double
    edges (:,2) double = zeros(0,2)
    faces (:,3) double = zeros(0,3)
    bounds (1,:) double = []
    fps (1,1) double {mustBePositive} = 30
    filename (1,:) char = 'animation.mp4'
    opts.visible (1,1) logical = true
    opts.show_floor (1,1) logical = false
    opts.show_poles (1,1) logical = false
    opts.show_walls (1,1) logical = false
    opts.show_axis (1,1) logical = false
    opts.view double = []
end

numFrames = size(V_all, 1);

v = VideoWriter(filename, 'MPEG-4');
v.FrameRate = fps;
open(v);

fig = figure('Color', 'w', 'Visible', opts.visible);
ax = axes(fig);
plot_cage(ax, 'bounds', bounds, 'show_floor', opts.show_floor, 'show_poles', opts.show_poles, ...
    'show_walls', opts.show_walls, 'show_axis', opts.show_axis, 'view', opts.view);

V0 = squeeze(V_all(1, :, :));
[hVerts, hEdges, hFaces] = plot_skeleton_core(ax, V0, edges, faces);

for frame = 1:numFrames
    V = squeeze(V_all(frame, :, :));

    hVerts.XData = V(:,1);
    hVerts.YData = V(:,2);
    hVerts.ZData = V(:,3);

    for i = 1:size(edges,1)
        pts = V(edges(i,:), :);
        hEdges(i).XData = pts(:,1);
        hEdges(i).YData = pts(:,2);
        hEdges(i).ZData = pts(:,3);
    end

    if ~isempty(faces)
        hFaces.Vertices = V;
    end

    drawnow;
    frameImage = getframe(fig);
    writeVideo(v, frameImage);
end

close(v);
fprintf('视频已保存为 %s\n', filename);

end


%% ============================================================
%  共用绘图工具: plot_cage / plot_skeleton_core
%% ============================================================

function ax = plot_cage(ax, opts)
%PLOT_CAGE 给指定坐标区画出统一的"训练笼"风格环境
% （坐标轴范围 / 地面网格 / 爬杆 / 墙壁网格 / 视角 / 光照）
%
%   ax = plot_cage()                  在新图窗上新建 axes 并画笼子
%   ax = plot_cage(ax)                在已有 axes 上画笼子（默认外观）
%   ax = plot_cage(ax, 'bounds', b, 'show_floor', true, ...)
%
% 被 plot_skeleton_grid（每个子图各调用一次）和 plot_skeleton_animation
% （只调用一次建场景）共用，保证所有场景下笼子的外观完全一致。
%
% 输入:
%   ax : 目标坐标区句柄；[] 或不传则新建一个 figure + axes
%
% Name-Value 参数:
%   bounds      : 1x6 [xmin xmax ymin ymax zmin zmax]；[] 用默认笼子范围
%   show_floor  : logical，是否画地面网格，默认 false
%   show_poles  : logical，是否画笼子爬杆，默认 false
%   show_walls  : logical，是否画墙壁网格线 + F/L 标签，默认 false
%   show_axis   : logical，是否显示坐标轴刻度，默认 false
%   view_az_el  : 1x2 [az, el]（同 MATLAB 内置 view(az,el) 的定义）；
%                 [] 则用默认 [150, 10]

arguments
    ax = []
    opts struct = struct
    % opts.bounds double = []
    % opts.show_floor (1,1) logical = false
    % opts.show_poles (1,1) logical = false
    % opts.show_walls (1,1) logical = false
    % opts.show_axis (1,1) logical = false
    % opts.view_az_el double = []
end

if isempty(ax)
    ax = axes(figure('Color', 'w'));
    rotate3d on;
end

bounds = get(opts, 'bound', [-5, 75, -5, 105, -5, 105]);

axes(ax); hold on; axis equal;
set(ax, 'Color', [.97 .97 .97]);

xlabel(''); xticks(bounds(1:2)); xticklabels([]); xlim(bounds(1:2));
ylabel(''); yticks(bounds(3:4)); yticklabels([]); ylim(bounds(3:4));
zlabel(''); zticks(bounds(5:6)); zticklabels([]); zlim(bounds(5:6));


view(get(opts, 'view', [150, 10]));

lighting gouraud;       % 光照插值
material dull;          % 材质
camlight headlight;     % 从视角方向打光
camlight left;          % 辅助光

if get(opts, 'show_poles', false)
    plot_cage_poles();
end
if get(opts, 'show_floor', false)
    plot_cage_floor();
end
if get(opts, 'show_walls', false)
    text(45, 135, 0, 'F');
    text(95, 60, 0, 'L');
    grid on;
end
if ~get(opts, 'show_axis', true)
    axis off;
end

end


function plot_cage_poles()
%PLOT_CAGE_POLES 画训练笼的爬杆（画在当前坐标区 gca 上）
%
% 坐标取自 viz_kinetics.m 中被注释掉的"笼子爬杆"代码块，
% 是特定实验装置的固定几何尺寸，不随数据/bounds 变化。

barstyles = {'LineWidth', 1, 'Color', [.8 .8 .8], 'Marker', 'o', 'MarkerSize', 2};
plot3([0 70], [10, 10], [15 15], barstyles{:});
plot3([0 70], [12, 12], [15 15], barstyles{:});
plot3([0 70], [10, 10], [70 70], barstyles{:});
plot3([0 70], [50, 50], [90 90], barstyles{:});
plot3([0 70], [90, 90], [70 70], barstyles{:});

end


function plot_cage_floor(xlim_range, ylim_range, nx, ny)
%PLOT_CAGE_FLOOR 画地面网格（画在当前坐标区 gca 上）
%
%   plot_cage_floor()                                默认值取自 viz_kinetics.m 注释代码
%   plot_cage_floor(xlim_range, ylim_range, nx, ny)   自定义范围和网格密度
%
% xlim_range, ylim_range : 1x2，地面在 x/y 方向的范围，默认 [0 70] / [0 100]
% nx, ny                 : 网格线密度（linspace 的点数），默认 8 / 11

arguments
    xlim_range (1,2) double = [0, 70]
    ylim_range (1,2) double = [0, 100]
    nx (1,1) double {mustBePositive, mustBeInteger} = 8
    ny (1,1) double {mustBePositive, mustBeInteger} = 11
end

x = linspace(xlim_range(1), xlim_range(2), nx);
y = linspace(ylim_range(1), ylim_range(2), ny);
[X, Y] = meshgrid(x, y);
Z = zeros(size(X));
surf(X, Y, Z, 'FaceColor', 'none', 'EdgeColor', [.8, .8, .8]);

end


function [hVerts, hEdges, hFaces] = plot_skeleton_core(ax, V, edges, faces, opts)
%PLOT_SKELETON_CORE 在指定坐标区上画骨架本体（点 + 连线 + 三角面）
%
%   [hVerts, hEdges, hFaces] = plot_skeleton_core(ax, V, edges, faces)
%
% 被 plot_skeleton_grid（每个子图各调用一次）和 plot_skeleton_animation
% （只调用一次，之后在循环里更新返回的句柄）共用。
%
% 输入:
%   ax    : 目标坐标区句柄
%   V     : N x 3 顶点坐标矩阵
%   edges : M x 2 线段顶点索引矩阵；无则传 []
%   faces : F x 3 三角面顶点索引矩阵（由 edges 自动检测得到）；无则传 []
%
% 输出（图形对象句柄，供动画场景循环里更新 XData/YData/ZData/Vertices 用）:
%   hVerts : 顶点 scatter3 句柄
%   hEdges : 连线 plot3 句柄数组（无 edges 时为空的 gobjects 数组）
%   hFaces : 三角面 patch 句柄（无 faces 时为 []）

arguments
    ax
    V (:,3) double
    edges (:,2) double = zeros(0,2)
    faces (:,3) double = zeros(0,3)
    opts struct = struct
    % opts.line_width = 1
    % opts.face_color = 'w'
    % opts.face_alpha = 0.7
    % opts.edge_color = 'k'
end

axes(ax); hold(ax, 'on');
c = get(opts, 'vert_color', zeros(size(V, 1), 3));

if get(opts, 'marker_joint', false)

    % 中间
    idx = [1:3, 10:11];
    hVerts1 = scatter3(ax, V(idx,1), V(idx,2), V(idx,3), get(opts, 'vert_size', 10), c(idx, :), 'filled', 'd');
    % 左
    idx = [4:6, 12:14 ];
    hVerts2 = scatter3(ax, V(idx,1), V(idx,2), V(idx,3), get(opts, 'vert_size', 30), c(idx, :), 'filled', '<');
    % 右
    idx = [7:9, 15:17];
    hVerts3 = scatter3(ax, V(idx,1), V(idx,2), V(idx,3), get(opts, 'vert_size', 30), c(idx, :), 'filled', '>');
    hVerts = {hVerts1, hVerts2, hVerts3};
else
    hVerts = scatter3(ax, V(:,1), V(:,2), V(:,3), get(opts, 'vert_size', 20), c(:, :), 'filled', 'o');
end

% idx1 = [1:3, 10:11];
% idx2 = [4:6, 12:14];
% idx3 = [7:9, 15:17]; 
% 
% markers = [ ...
%     repmat({'d'}, 1, numel(idx1)), ... % 中间：菱形
%     repmat({'<'}, 1, numel(idx2)), ... % 左：左三角
%     repmat({'>'}, 1, numel(idx3))  ... % 右：右三角
% ];
% 
% hVerts = scatter3(ax, V(:,1), V(:,2), V(:,3), ...
%     get(opts, 'vert_size', 10), 'k', 'filled');
% 
% hVerts.Marker = markers; 



hEdges = gobjects(size(edges,1), 1);
for i = 1:size(edges,1)
    edge_color = 'k';
    if get(opts, 'color_limb', false)
        if any(i == [7, 8, 20, 21])
            edge_color =  [31, 119, 180] / 255;
        elseif any(i == [9, 10, 22, 23])
            edge_color =  [214, 39, 40] / 255;
        end

    end

    pts = V(edges(i,:), :);
    hEdges(i) = plot3(ax, pts(:,1), pts(:,2), pts(:,3), 'Color' ,edge_color, 'LineWidth', get(opts, 'line_width', 1));
end

hFaces = [];
if ~isempty(faces)
    hFaces = patch(ax, 'Vertices', V, 'Faces', faces, ...
        'FaceColor', get(opts, 'face_color', 'w'), ...
        'FaceAlpha', get(opts, 'face_alpha', 0.7), ...
        'EdgeColor', get(opts, 'edge_color', 'k'));
end

camtarget(V(10, :));

end

function v = get(opts, k, default)
if isfield(opts, k), v = opts.(k); else, v=default; end
end
