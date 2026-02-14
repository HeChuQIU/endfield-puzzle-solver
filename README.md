# Endfield Puzzle Solver

终焉地理论拼图求解器 - Avalonia 跨平台版本

## 🚀 快速开始

### 下载发布版本

访问 [Releases 页面](../../releases) 下载最新版本的 ZIP 文件。

### 本地开发

```bash
# 克隆仓库
git clone <repository-url>
cd endfield-puzzle-solver

# 还原依赖
dotnet restore

# 运行 Avalonia 版本
dotnet run --project src/EndfieldPuzzleSolver.Avalonia/EndfieldPuzzleSolver.Avalonia.csproj
```

### 本地构建发布版本

```powershell
# 使用测试脚本（推荐）
.\scripts\test-release.ps1 -Version "v1.0.0"

# 或手动构建
dotnet publish src/EndfieldPuzzleSolver.Avalonia/EndfieldPuzzleSolver.Avalonia.csproj `
  -c Release -r win-x64 --self-contained -o publish/avalonia
```

## 📦 项目结构

```
endfield-puzzle-solver/
├── src/
│   ├── EndfieldPuzzleSolver.Avalonia/      # Avalonia UI 项目 (推荐)
│   ├── EndfieldPuzzleSolver/               # WinUI3 项目 (已弃用)
│   ├── EndfieldPuzzleSolver.Core/          # 核心业务逻辑
│   ├── EndfieldPuzzleSolver.Recognition/   # 图像识别模块
│   └── EndfieldPuzzleSolver.Algorithm/     # 求解算法 (F#)
├── HeChu-docs/                              # 需求文档
├── .github/workflows/                       # GitHub Actions
└── scripts/                                 # 构建脚本
```

## 🔧 技术栈

- **UI 框架**: Avalonia UI 11.2
- **运行时**: .NET 9.0
- **编译**: Native AOT
- **图像处理**: OpenCV (OpenCvSharp4)
- **算法**: F# 函数式编程
- **MVVM**: CommunityToolkit.Mvvm

## 📋 系统要求

### 开发环境
- .NET 9.0 SDK
- Visual Studio 2022 或 JetBrains Rider
- Windows 10/11 (推荐) 或 Linux/macOS

### 运行环境（发布版本）
- Windows 10/11 x64
- 无需安装 .NET 运行时

## 🎯 功能特性

- ✅ 自动识别拼图截图
- ✅ 智能求解拼图路径
- ✅ 支持拖拽图片
- ✅ 剪贴板粘贴图片（Ctrl+V）
- ✅ 实时显示求解步骤
- ✅ Native AOT 快速启动

## 🚢 发布流程

### 自动发布（推荐）

1. 进入 GitHub **Actions** 页面
2. 选择 **"Release Avalonia Build"**
3. 点击 **"Run workflow"**
4. 输入版本号（如 `v1.0.0`）
5. 等待构建完成
6. 在 **Releases** 页面下载

详细说明请查看 [HeChu-docs/RELEASE.md](HeChu-docs/RELEASE.md)

### 本地测试

```powershell
# 运行测试脚本
.\scripts\test-release.ps1

# 脚本会：
# 1. 构建项目
# 2. 移除不必要的文件
# 3. 创建 ZIP 包
# 4. 计算 SHA256 哈希
# 5. 询问是否运行测试
```

## 📊 发布包信息

- **大小**: ~38 MB (压缩), ~93 MB (解压)
- **文件数**: 13 个文件
- **不包含**: 
  - ❌ FFmpeg 视频库（27 MB）- 未使用
  - ❌ 调试符号文件（.pdb）
  - ❌ XML 文档文件

## ⚠️ 重要说明

### WinUI3 版本已弃用

WinUI3 版本因与 Native AOT 不兼容，无法正常运行，已停止维护。推荐使用 Avalonia 版本。

### OpenCV FFmpeg 说明

项目仅使用 OpenCV 的图像处理功能（`ImRead`, `MatchTemplate` 等），**不需要 FFmpeg 视频库**。发布版本已移除 `opencv_videoio_ffmpeg4110_64.dll` 以减小体积。

## 📚 文档

- **需求文档**: [HeChu-docs/](HeChu-docs/)
- **发布指南**: [HeChu-docs/RELEASE.md](HeChu-docs/RELEASE.md)
- **快速参考**: [HeChu-docs/RELEASE-QUICK.md](HeChu-docs/RELEASE-QUICK.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

[待添加]

---

**最后更新**: 2026-02-14
