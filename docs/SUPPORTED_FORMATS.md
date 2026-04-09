# HDF5 Viewer 支持的文件格式

## 🎯 快速参考

HDF5 Viewer 现在支持**智能检测**和**自动识别** HDF5 文件，无论其扩展名是什么。

---

## 📁 支持的文件扩展名

### 标准 HDF5 格式
- **`.h5`** - HDF5 标准格式
- **`.hdf5`** - HDF5 标准格式
- **`.hdf`** - HDF4/HDF5 通用格式

### 科学仪器格式
- **`.nxs`** ⭐ **NeXus 数据格式**
  - 中子散射实验数据
  - 同步辐射光源数据
  - X 射线衍射数据
  - 常见于大型科研设施（如 ISIS, Diamond, ESRF 等）

- **`.nx5`** - NeXus HDF5 变体

### 专业领域格式
- **`.he5`** - HDF-EOS（地球观测系统）
  - 卫星遥感数据
  - 气象观测数据
  - NASA 地球科学数据

- **`.cxi`** - Coherent X-ray Imaging
  - 相干 X 射线成像数据
  - 自由电子激光实验数据

- **`.mat`** - MATLAB v7.3+
  - 现代 MATLAB 数据文件（v7.3 及以上版本使用 HDF5 格式）

---

## 🔍 智能检测功能

### 工作原理

1. **扩展名识别**（快速检查）
   ```
   文件: experiment.nxs
   → 识别为已知 HDF5 扩展名
   → 尝试打开
   ```

2. **内容验证**（深度检查）
   ```
   文件: data_file (无扩展名)
   → 尝试用 h5py 打开
   → 成功 → 识别为 HDF5 文件
   → 失败 → 跳过
   ```

3. **容错机制**
   ```
   文件: mydata.dat (实际是 HDF5)
   → 扩展名未知，但内容是 HDF5
   → 仍然可以打开！✅
   ```

---

## 📖 使用方法

### 方法 1: 文件菜单打开

```
File → Open File... → 选择任意支持的格式
```

**文件过滤器显示**:
```
HDF5 Files (*.cxi *.h5 *.hdf *.hdf5 *.he5 *.mat *.nx5 *.nxs);;All Files (*.*)
```

### 方法 2: 拖放打开

直接将文件拖放到窗口中：

```
✅ 支持的格式自动打开
❌ 不支持的格式自动跳过（日志中会显示）
```

### 方法 3: 批量打开文件夹

```
File → Open Folder... → 选择文件夹
```

程序会自动扫描并打开文件夹中所有有效的 HDF5 文件。

---

## 🎨 示例：NeXus (.nxs) 文件

### NeXus 文件结构示例

```
experiment.nxs/
├── entry1/
│   ├── instrument/
│   │   ├── detector/
│   │   │   ├── data [Dataset]
│   │   │   └── distance [Dataset]
│   │   └── source/
│   │       ├── name [Dataset]
│   │       └── type [Dataset]
│   ├── sample/
│   │   ├── name [Dataset]
│   │   └── temperature [Dataset]
│   └── data/
│       └── counts [Dataset]
└── entry2/
    └── ...
```

### 在 HDF5 Viewer 中的显示

- **树形视图**: 清晰展示 entry → instrument → detector 的层级结构
- **属性面板**: 显示每个节点的元数据
- **数据可视化**:
  - 数值数据 → 绘制曲线
  - 2D 数据 → 热图
  - 字符串 → 文本显示

---

## ❓ 常见问题

### Q1: 我的文件扩展名是 .dat，能打开吗？

**A**: 可以尝试！如果文件内容是有效的 HDF5 格式，程序可以识别并打开。

**操作方法**:
1. 在文件打开对话框中选择 "All Files (*.*)"
2. 或者直接拖放文件到窗口
3. 程序会自动检测文件格式

### Q2: 为什么有些 .mat 文件无法打开？

**A**: MATLAB 的 .mat 文件有多个版本：
- **v7.3 及以上**: 使用 HDF5 格式 ✅ **可以打开**
- **v7.2 及以下**: 使用专有格式 ❌ **无法打开**

**检查方法**:
```matlab
% 在 MATLAB 中查看版本
whos -file yourfile.mat
```

### Q3: 文件拖放被拒绝怎么办？

**可能原因**:
1. 文件不是有效的 HDF5 格式
2. 文件已损坏
3. 文件权限问题

**解决方法**:
1. 检查日志输出（会显示具体错误信息）
2. 尝试用 h5dump 或其他工具验证文件完整性
3. 确认文件读取权限

---

## 🛠️ 技术细节

### 文件检测流程图

```
文件输入
    ↓
检查扩展名
    ↓
  已知扩展名？
    ├─ 是 → 尝试打开
    └─ 否 → 尝试作为 HDF5 打开
        ↓
    成功打开？
    ├─ 是 → 加载文件结构
    └─ 否 → 显示错误/跳过
```

### 性能考虑

- **快速路径**: 已知扩展名的文件直接尝试打开（< 100ms）
- **安全检查**: 所有文件都会验证是否为有效 HDF5
- **错误处理**: 无效文件不会导致程序崩溃，只是记录警告

---

## 📚 参考资料

### HDF5 格式标准
- [HDF Group 官网](https://www.hdfgroup.org/)
- [HDF5 文件格式规范](https://support.hdfgroup.org/HDF5/doc/H5.format.html)

### NeXus 格式
- [NeXus 官方文档](https://www.nexusformat.org/)
- [NeXus 文件结构](https://manual.nexusformat.org/classes/base_classes/)

### HDF-EOS
- [NASA HDF-EOS 文档](https://www.earthdata.nasa.gov/esdis/eso/standards-and-references/hdf-eos)

---

## 🎓 最佳实践

### 1. 命名规范

推荐使用标准扩展名以便识别：

```
✅ 推荐
experiment_2024_01_15.nxs
detector_calibration.h5
satellite_data.he5

⚠️ 可用但不推荐
my_data.dat (虽然能打开，但不直观)
test_file (无扩展名)
```

### 2. 文件组织

```
project/
├── raw_data/
│   ├── scan001.nxs
│   ├── scan002.nxs
│   └── ...
├── processed/
│   ├── analysis.h5
│   └── results.h5
└── calibration/
    └── detector_config.h5
```

使用 "Open Folder" 功能可以一次性加载整个目录。

### 3. 数据验证

在处理重要数据前，建议：

1. 先在 HDF5 Viewer 中预览数据结构
2. 检查关键数据集的形状和类型
3. 验证元数据的完整性

---

**最后更新**: 2026-02-11
