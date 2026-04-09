# 🎨 PyQtGraph 内置功能说明

## 📊 概述

HDF5 Viewer 使用 **pyqtgraph** 来显示数据的可视化。pyqtgraph 自带了强大的交互功能，包括右键菜单中的导出和归一化选项。

---

## 🖼️ ImageView 右键菜单 (2D/3D 图像)

### 可用功能

当您查看 **2D 数组**（热图）或 **3D RGB 图像**时，右键点击图像区域会显示菜单：

#### 1. **Export** 📤
- **功能**: 导出当前显示的图像
- **格式**:
  - SVG (矢量图，适合出版)
  - PNG (位图)
  - CSV (数据导出)
  - 等等

**使用方法**:
```
1. 显示图像数据 (Array2D 或 ImageRGB)
2. 右键点击图像
3. Export → 选择格式
4. 保存文件
```

#### 2. **Normalization** 🎨
- **功能**: 调整图像的亮度和对比度
- **选项**:
  - **Off**: 不归一化，使用原始数据范围
  - **Subtract Min**: 减去最小值
  - **Divide by Max**: 除以最大值
  - **By Color**: 按颜色通道归一化

**效果**:
```
原始数据: [0, 100, 200, 300]

Off:           显示 0-300 范围
Subtract Min:  显示 0-300 范围（减去最小值）
Divide by Max: 归一化到 0-1 范围
```

#### 3. **Histogram/Gradient Editor** 📈
- **功能**: 手动调整图像的颜色映射
- **可以调整**:
  - 最小值/最大值范围
  - 颜色查找表 (LUT)
  - Gamma 校正

**使用方法**:
```
1. 右键图像 → Histogram
2. 拖动直方图边界调整范围
3. 实时预览效果
```

#### 4. **View All** 🔍
- **功能**: 重置视图，显示完整图像
- **快捷键**: `Ctrl+0`

#### 5. **Mouse Mode** 🖱️
- **功能**: 切换鼠标交互模式
- **模式**:
  - **1 button**: 单键平移/缩放
  - **3 button**: 三键模式（左键平移，右键缩放）

---

## 📉 PlotWidget 右键菜单 (1D 数据)

### 可用功能

当您查看 **1D 数组**（线图）时，右键点击绘图区域：

#### 1. **Export** 📤
- SVG (矢量图)
  - 无损缩放
  - 适合论文/出版
- PNG
- CSV (导出数据点)
- MATLAB (导出为 .mat 文件)

#### 2. **View All** 🔍
- 重置视图到默认范围
- 显示所有数据点

#### 3. **X-Axis / Y-Axis** 📏
- **Auto Range**: 自动调整坐标轴范围
- **Mouse Enabled**: 启用/禁用鼠标交互
- **Log Mode**: 对数坐标轴

#### 4. **Grid** 📊
- 显示/隐藏网格线
- 调整网格透明度

#### 5. **FFT** 🌊
- 快速傅里叶变换（某些配置下可用）

---

## 🎯 实际使用示例

### 示例 1: 导出高质量图像用于论文

**场景**: 需要导出检测器数据的 SVG 矢量图

```
步骤:
  1. 在 HDF5 Viewer 中打开文件
  2. 选择检测器数据 (2D 数组)
  3. 数据自动显示为热图 (ImageView)
  4. 右键点击图像
  5. Export → SVG
  6. 保存为 detector_data.svg

优势:
  ✅ 矢量图，无损缩放
  ✅ 适合出版物
  ✅ 文件小
```

### 示例 2: 调整图像对比度

**场景**: 图像太暗，需要增强对比度

```
方法 1: 使用归一化
  1. 右键图像
  2. Normalization → Divide by Max
  3. 图像自动归一化到 0-1 范围

方法 2: 手动调整
  1. 右键图像
  2. Histogram
  3. 拖动直方图边界
  4. 调整到满意的对比度
```

### 示例 3: 导出数据点

**场景**: 需要将曲线数据导出到 Excel

```
步骤:
  1. 显示 1D 数组 (线图)
  2. 右键绘图区域
  3. Export → CSV
  4. 保存文件
  5. 在 Excel 中打开

注意:
  - pyqtgraph 导出的 CSV 格式可能略有不同
  - 建议使用我们的导出功能 (Ctrl+E) 获得更好的格式
```

---

## 🔧 与我们的导出功能对比

### pyqtgraph 内置导出

**优点**:
- ✅ 快速访问（右键菜单）
- ✅ 支持 SVG 矢量图
- ✅ 所见即所得（导出显示的内容）

**缺点**:
- ❌ 只导出当前视图
- ❌ 图像格式选项少
- ❌ CSV 格式不够友好

### HDF5 Viewer 导出功能 (Ctrl+E)

**优点**:
- ✅ 导出原始数据（完整分辨率）
- ✅ 更多格式选项（PNG, JPEG, TIFF, CSV）
- ✅ 友好的 CSV 格式（带列名）
- ✅ 自动归一化处理

**缺点**:
- ❌ 不支持 SVG

### 推荐使用场景

| 需求 | 推荐方法 |
|------|----------|
| **论文矢量图** | pyqtgraph → Export → SVG |
| **高质量位图** | HDF5 Viewer → Ctrl+E → PNG/TIFF |
| **数据分析** | HDF5 Viewer → Ctrl+E → CSV |
| **快速预览** | pyqtgraph → Export → PNG |
| **当前视图** | pyqtgraph → Export |
| **原始数据** | HDF5 Viewer → Ctrl+E |

---

## 🎨 高级功能

### 1. **颜色映射 (Colormap)**

**当前设置**:
```python
# 在 main_window.py 中
new_widget.setColorMap(pg.colormap.get("inferno"))
```

**可用颜色映射**:
- `inferno` - 默认，科学可视化标准
- `viridis` - 另一个流行选择
- `plasma`
- `cividis` - 色盲友好
- `gray` - 灰度
- `hot`
- `jet` - 经典彩虹色（不推荐）

**自定义颜色映射**:
```python
# 可以在代码中修改
new_widget.setColorMap(pg.colormap.get("viridis"))
```

### 2. **直方图均衡化**

pyqtgraph 的 ImageView 自带直方图工具：

```
1. 右键图像
2. 会看到图像旁边的直方图面板
3. 调整直方图边界
4. 实时预览效果
```

### 3. **ROI (Region of Interest) 选择**

某些配置下，ImageView 支持 ROI 选择：

```
功能:
  - 在图像上绘制矩形区域
  - 查看 ROI 内的统计信息
  - 导出 ROI 区域
```

---

## 📋 快捷键参考

### ImageView (图像)

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+0` | 重置视图（View All） |
| 鼠标滚轮 | 缩放 |
| 左键拖动 | 平移 |
| 右键 | 打开菜单 |

### PlotWidget (曲线)

| 快捷键 | 功能 |
|--------|------|
| `Ctrl+0` | 重置视图 |
| 鼠标滚轮 | 缩放 |
| 左键拖动 | 平移 |
| 右键拖动 | 矩形缩放（zoom） |
| 中键拖动 | 平移 |

---

## 💡 最佳实践

### 1. 导出工作流程

```
快速预览:
  右键 → Export → PNG

高质量论文图:
  右键 → Export → SVG → 在 Inkscape 中编辑

数据分析:
  Ctrl+E → CSV → Excel/MATLAB 分析

高分辨率存档:
  Ctrl+E → TIFF → 长期保存
```

### 2. 图像增强

```
暗图像:
  右键 → Normalization → Divide by Max

调整对比度:
  右键 → Histogram → 拖动边界

自定义颜色:
  修改代码中的 colormap
```

### 3. 交互式探索

```
1. 用鼠标缩放到感兴趣区域
2. 右键 → Export → 导出当前视图
3. 或者 Ctrl+E → 导出完整数据
```

---

## 🔍 技术细节

### pyqtgraph 版本

```python
# 查看版本
import pyqtgraph as pg
print(pg.__version__)

# requirements.txt 中
pyqtgraph>=0.13.7
```

### 代码位置

在 `main_window.py` 中创建图像视图：

```python
# Array2D (热图)
new_widget = pg.ImageView()
new_widget.setImage(data)
new_widget.setColorMap(pg.colormap.get("inferno"))

# Array1D (线图)
new_widget = pg.PlotWidget()
new_widget.plot(data)
```

---

## 🎯 总结

### PyQtGraph 内置功能

| 功能 | 位置 | 说明 |
|------|------|------|
| Export | 右键菜单 | 支持 SVG/PNG/CSV |
| Normalization | 右键菜单 | 调整亮度/对比度 |
| Histogram | 右键菜单 | 手动调整范围 |
| View All | 右键菜单 | 重置视图 |
| Mouse Mode | 右键菜单 | 切换交互模式 |

### HDF5 Viewer 增强功能

| 功能 | 快捷键 | 说明 |
|------|--------|------|
| Export Dataset | Ctrl+E | 导出原始数据 |
| 多格式支持 | - | PNG/JPEG/TIFF/CSV |
| 结构化数组 | - | 自动提取列名 |
| 特殊值处理 | - | NaN/Inf 友好显示 |

### 推荐组合

```
日常使用:
  - 快速查看: pyqtgraph 右键导出
  - 详细分析: Ctrl+E 导出原始数据

科研发表:
  - 矢量图: pyqtgraph → SVG
  - 高质量位图: Ctrl+E → PNG/TIFF

数据处理:
  - 数据表: Ctrl+E → CSV
  - 图像数据: Ctrl+E → TIFF
```

---

**这两套导出系统互为补充，为您提供最大的灵活性！** 🎉

## 📚 参考资料

- [PyQtGraph 官方文档](http://www.pyqtgraph.org/)
- [ImageView 文档](http://www.pyqtgraph.org/documentation/widgets/imageview.html)
- [PlotWidget 文档](http://www.pyqtgraph.org/documentation/graphicsItems/plotitem.html)

---

**文档版本**: v1.0
**更新日期**: 2026-02-11
