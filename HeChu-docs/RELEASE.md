# 发布流程说明

## 自动发布 (GitHub Actions)

项目已配置自动发布流程，仅构建并发布 **Avalonia 版本**。

### 使用方法

1. 进入 GitHub 仓库的 **Actions** 标签页
2. 选择 **"Release Avalonia Build"** workflow
3. 点击 **"Run workflow"** 按钮
4. 输入版本号（例如：`v1.0.0`）
5. 点击 **"Run workflow"** 开始构建

### 构建流程

1. ✅ 检出代码
2. ✅ 设置 .NET 9.0 环境
3. ✅ 还原 NuGet 依赖
4. ✅ 发布 Avalonia 项目（Native AOT + 单文件）
5. ✅ 移除不必要的文件：
   - `*.pdb` (调试符号)
   - `*.xml` (文档文件)
   - `opencv_videoio_ffmpeg4110_64.dll` (未使用的 FFmpeg 库)
6. ✅ 打包为 ZIP 文件
7. ✅ 计算 SHA256 哈希值
8. ✅ 创建 GitHub Release 并上传附件

### 发布产物

**文件名**: `EndfieldPuzzleSolver-Avalonia-{版本号}-win-x64.zip`

**包含内容**:
- `EndfieldPuzzleSolver.Avalonia.exe` (主程序，约 19 MB)
- `OpenCvSharpExtern.dll` (OpenCV 原生库，约 59 MB)
- `libSkiaSharp.dll` (Avalonia 渲染引擎，约 9 MB)
- `av_libglesv2.dll` (GPU 渲染库，约 4 MB)
- `libHarfBuzzSharp.dll` (字体排版库，约 1.5 MB)
- `Assets/UiTemplates/*` (UI 模板图片)
- `appsettings.json` (配置文件)

**总大小**: 约 93 MB

### 发布特性

- ✅ **Native AOT 编译** - 启动速度快，无需 .NET 运行时
- ✅ **单文件发布** - exe 包含所有托管代码
- ✅ **自包含部署** - 无需用户安装任何依赖
- ✅ **移除 FFmpeg** - 项目不使用视频功能，已排除 27 MB 的 FFmpeg DLL
- ✅ **自动 SHA256 校验** - Release 页面包含文件哈希值

## WinUI3 版本状态

**已弃用** - WinUI3 与 Native AOT 不兼容，无法正常运行。

## 本地构建

如需本地构建发布版本：

```powershell
# 发布 Avalonia 版本
dotnet publish src/EndfieldPuzzleSolver.Avalonia/EndfieldPuzzleSolver.Avalonia.csproj `
  -c Release `
  -r win-x64 `
  --self-contained `
  -o publish/avalonia `
  -p:PublishAot=true `
  -p:PublishSingleFile=true

# 清理不必要的文件
Remove-Item publish/avalonia/*.pdb -Force
Remove-Item publish/avalonia/*.xml -Force
Remove-Item publish/avalonia/opencv_videoio_ffmpeg4110_64.dll -Force

# 创建 ZIP 包
Compress-Archive -Path publish/avalonia/* -DestinationPath EndfieldPuzzleSolver-Avalonia.zip
```

## 系统要求

- **操作系统**: Windows 10/11 x64
- **运行时**: 无需安装（自包含）
- **磁盘空间**: 约 100 MB（解压后）
- **显卡**: 支持 OpenGL 的显卡（用于 Avalonia 渲染）

## 版本命名规范

建议使用语义化版本号：

- `v1.0.0` - 首次正式发布
- `v1.1.0` - 添加新功能
- `v1.0.1` - 修复 bug
- `v2.0.0` - 重大更新（可能包含破坏性变更）

## 故障排除

### 发布失败

1. 检查 GitHub Actions 日志中的错误信息
2. 确认 .NET 9.0 SDK 可用
3. 检查项目文件是否正确配置 Native AOT

### ZIP 文件损坏

1. 检查构建日志中是否有文件缺失
2. 验证 SHA256 哈希值是否匹配

### 应用无法启动

1. 确认是在 Windows x64 系统上运行
2. 检查是否缺少必需的 DLL 文件
3. 尝试以管理员身份运行
