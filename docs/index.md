# bmi_free

ECoG-based brain-machine interface: decoding, visualization, and analysis.

## 功能概览

| 模块 | 说明 |
|------|------|
| `io` | 数据导入：BDF/ECoG、运动捕捉坐标、骨架配置、标注等 |
| `features` | TFR 计算与特征切片（支持 MATLAB CWT 和 MNE 内置方法） |
| `kinematics` | 关节角度与角速度计算、EDF 导出 |
| `pls` | PLS 回归模型训练、滑动窗口 r 值、权重可视化 |
| `prep` | 预处理：双侧平均重参考等 |
| `plot` | 皮层拓扑图可视化（MATLAB NMT-ECoG-Mapper 封装） |
| `utils` | 工具函数：shuffle、epoch 筛选、Tag 管理等 |
| `video` | 视频帧网格可视化 |
| `viz.cortex` | 皮层拓扑图（纯 Python/MNE 实现） |
| `viz.skeleton` | 3D 骨架可视化与动画（MATLAB 封装） |

## 快速开始

```bash
pip install -e .
```

详细 API 文档请参阅左侧 **API 参考** 导航栏。