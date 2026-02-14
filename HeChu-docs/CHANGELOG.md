# 变更列表

记录所有需求的变更历史。

---

## REQ-022: 增强图片输入体验 - 拖入提示与剪贴板粘贴功能

- [2026-02-14] 创建需求
- [2026-02-14] 完成开发：在谜题主体区域添加拖入提示（未加载时显示），在工具栏添加"粘贴截图"按钮（始终可用），支持 Ctrl+V 快捷键；修复 WinUI3 剪贴板位图编码问题（使用 BitmapDecoder/Encoder 正确编码为 PNG），修复 Path 命名冲突（使用 System.IO.Path）；编译通过 0 错误 0 警告

---

## REQ-021: Avalonia UI 功能完善 - 对齐 WinUI3 版本界面功能

- [2026-02-14] 创建需求
- [2026-02-14] 完成开发：修复颜色组显示（实际颜色 + HSV 值）、元件列表（编号、格数、形状可视化）、图标（文件夹图标）、状态栏样式（InfoBar 风格），编译通过 0 错误 0 警告

---

## REQ-020: 修复组件颜色识别 Bug - 双色谜题只能识别一种颜色

- [2026-02-14] 创建需求
- [2026-02-14] 完成开发：修复 PuzzleDetector.cs 第 943 行，将组件颜色处理从 `MatchNearest` 改为 `Register` + `Label`，与 Python 原型对齐。修复后双色/多色谜题能正确识别所有颜色组，编译通过 0 错误 0 警告

---

## REQ-019: 新增 Avalonia 跨平台 UI 版本

- [2026-02-14] 创建需求
- [2026-02-14] 完成开发：创建 Core 项目共享 ViewModel，新增 Avalonia 项目使用 Fluent 主题，实现与 WinUI 3 功能一致的 UI（打开截图、拖拽加载、Canvas 绘制棋盘），两个项目均编译通过
- [2026-02-14] 修复 Avalonia 版本：添加 UiTemplates 模板文件、修复 DataContext 绑定、修复拖拽事件处理
- [2026-02-14] 配置 Native AOT 发布：启用 PublishAot、PublishSingleFile，优化输出大小约 119MB

---

## REQ-018: 使用 ContentIsland 布局分离谜题组件

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：重构 MainWindow.xaml 为卡片式布局，将界面分为三个独立区域：谜题主体（左侧，带滚动条和独立边框）、谜题信息（右侧上方卡片）、元件列表（右侧下方卡片），使用浅灰背景和圆角边框区分区域

---

## REQ-017: 简化应用流程 - 图片输入自动求解

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：移除步骤回放UI控件（上一步、下一步、步骤文本、求解按钮）；修改 LoadFromScreenshotAsync 识别完成后自动调用求解器；修改 GetCurrentBoardSnapshot 直接返回最后一步结果；移除 code-behind 中步骤相关的事件绑定；Solver.fs 步骤记录功能保留但不在UI中暴露

---

## REQ-016: 使用 OpenCvSharp API 和 Linq 全面优化图像识别性能

- [2026-02-12] 创建需求：在 REQ-015 分析基础上，实际实施性能优化方案。核心原则是优先使用 OpenCvSharp API 和使用 Linq 替代手动列表操作，确保优化后的算法结果与 Python 原型保持一致，并进行性能和正确性验证。
- [2026-02-12] 完成开发：实施了所有关键性能优化，包括 SimilarityMapForColor 使用矩阵操作替代双重循环、MatchTemplate 使用 FindNonZero 替代双重循环、ClassifyCell 使用掩码操作和 Cv2.MeanStdDev、FindBars 使用掩码和矩阵求和、ParseComponentBlob 使用掩码操作统计像素。编译成功，0 个错误。
- [2026-02-12] 修复实现 bug：修复 FindBars 函数内存访问顺序错误导致的 System.AccessViolationException。问题根源是 Dispose 释放了 barChannels 后，后续代码又访问已销毁的 vBar 对象。修复方法：将 MeanStdDev 调用移到 Dispose 之前，确保访问时对象未被释放。

---

## REQ-015: 优化C#应用程序图像识别性能 - 使用OpenCvSharp和Linq替代手动循环

- [2026-02-12] 完成分析：深入分析了性能瓶颈，发现 C# 版本使用大量嵌套循环导致性能比 Python 慢数倍。主要瓶颈包括 SimilarityMapForColor、MatchTemplate、ClassifyCell、FindBars 等函数。由于 OpenCvSharp API 的限制（如 ConvertTo 返回 void 而非 Mat、FindNonZero 返回 Mat 而非 List），直接移植 Python 的 NumPy 向量化操作较为困难。建议后续使用更深入的 API 研究、unsafe 代码或绑定原生 OpenCV 函数来实现优化。

---

## REQ-014: WinUI3 图像识别算法与原型对齐

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：移植 prototype/detect.py 的完整识别算法到 WinUI3 项目；新增 DetectionConfig.cs 配置模型类；使用 .NET Configuration API + appsettings.json 替代 TOML 文件；调整检测顺序为 grid→components→tiles→requirements；实现按颜色相似度检测需求条；ColorGrouper 添加 MatchNearest 方法；锁定格和元件使用 MatchNearest 匹配颜色；添加元件模板检测（component-flame）并支持回退 blob 检测

---

## REQ-013: 优化原型的调试图像生成 - 识别区域叠加图

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：新增 _generate_region_overlay 函数，生成 5_all_regions.png 调试图，包含暗化背景（原图亮度50%）、各识别区域的彩色边框（网格绿色、元件蓝色、锁定格黄色、需求条红色/绿色），以及元件和需求条的二值化识别结果（使用对应的 HSV 颜色叠加）

---

## REQ-012: 配置文件系统 - 图像识别参数提取

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：创建 config.py 配置加载模块；创建 config.toml 默认配置文件（6大功能模块、60+参数）；重构 detect.py 使用配置系统；支持 --config 自定义配置文件

---

## REQ-011: 需求条按元件颜色相似度二值化检测

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：_similarity_map_for_color 按 Hue 相似度生成；_find_bars 按颜色循环检测；puzzle2 无误判 filled

---

## REQ-010: 取色逻辑改为仅在识别出的元件上取色

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：检测顺序 grid→components→tiles→requirements；ColorGrouper.match_nearest；puzzle2 仅颜色 A

---

## REQ-009: 使用 component-flame 模板检测元件 UI

- [2026-02-12] 创建需求
- [2026-02-12] 完成开发：实现 component_flame 多尺度模板匹配；y 轴上限过滤排除底部 UI；模板无匹配时回退 blob 检测；puzzle2 元件 5 误检已消除

---

## REQ-007: 元件形状可视化 & 棋盘元件区分

- [2026-02-11] 创建需求
- [2026-02-11] 完成开发：侧边栏用小方格矩阵展示元件原始形状；棋盘重构为两遍渲染，同一元件内部边框自动去除形成连通块

---

## REQ-006: 全页面接受图片拖入

- [2026-02-11] 创建需求
- [2026-02-11] 完成开发：拖拽区域从 Canvas Border 移至 RootGrid，全窗口均可接受图片拖入

---

## REQ-005: 拖拽图片支持 & 移除 JSON 导入

- [2026-02-11] 创建需求
- [2026-02-11] 完成开发：Canvas 区域支持拖入图片加载；移除 JSON 导入按钮及 ViewModel/code-behind 中所有相关代码

---

## REQ-004: 需求数据模型修复（已满足/总需求显示）

- [2026-02-11] 创建需求
- [2026-02-11] 完成开发：修复 ColorRequirement.Count 仅统计未满足数量导致算法误判无解的 bug；Count 改为存储总需求数，新增 FilledCount 字段；UI 需求显示改为"已满足/总数"格式

---

## REQ-003: WinUI 3 正式应用程序

- [2026-02-11] 创建解决方案结构：3 个项目（WinUI 3 UI、C# 识别库、F# 算法库）
- [2026-02-11] 定义共享数据模型 PuzzleModels.cs
- [2026-02-11] 移植图像识别到 C#（PuzzleDetector + ColorGrouper）
- [2026-02-11] 创建 F# 算法骨架（Solver.fs + 分步记录模板 StepRecorder）
- [2026-02-11] 构建 WinUI 3 UI（CommandBar + Canvas 棋盘 + 侧边栏 + 步骤回放）
- [2026-02-11] 修复 XAML 编译器崩溃：`CommandBarSeparator` → `AppBarSeparator`（WinUI 3 正确控件名）
- [2026-02-11] 修复 HSV→RGB 转换（OpenCV H=0-180, S/V=0-255 范围适配）
- [2026-02-11] 修复 MatchTemplate 返回类型（`double` → `float`）
- [2026-02-11] 添加 `global.json` 固定 .NET 9 SDK（避免 .NET 10 SDK 兼容性问题）
- [2026-02-11] 全部 3 个项目构建成功（0 错误）
- [2026-02-12] 完成算法实现：递归回溯 + 约束传播剪枝，支持分步记录求解过程

---

## REQ-002: 多分辨率与界面缩放支持

- [2026-02-10] 创建需求
- [2026-02-10] 完成开发：实现多尺度模板匹配自动探测 UI 缩放系数，支持 `--scale` 手动覆盖，并将核心像素阈值改为按缩放系数自适配

---

## REQ-001: 终末地源石电路模块解谜 - UI 识别原型

- [2026-02-10] 创建需求
- [2026-02-10] 完成开发：实现网格检测、格子分类、行列需求和元件检测的原型验证
- [2026-02-10] 优化行列需求检测：使用 Otsu 自适应阈值和变异系数（CV）判断，消除固定像素阈值依赖，提升对不同背景的鲁棒性

