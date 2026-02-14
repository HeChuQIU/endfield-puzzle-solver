# 🚀 快速发布指南

## 发布新版本

1. **触发构建**
   ```
   GitHub → Actions → "Release Avalonia Build" → Run workflow
   ```

2. **输入版本号**
   ```
   例如: v1.0.0, v1.1.0, v2.0.0-beta
   ```

3. **等待构建完成** (约 3-5 分钟)

4. **检查 Releases 页面**
   ```
   GitHub → Releases → 找到新创建的版本
   ```

## 📦 发布内容

- **应用**: Avalonia 版本 (WinUI3 已弃用)
- **平台**: Windows x64
- **大小**: ~93 MB
- **特性**: Native AOT + 单文件

## 🔧 技术细节

### 已移除的文件
- ❌ `opencv_videoio_ffmpeg4110_64.dll` (27 MB) - 未使用视频功能
- ❌ `*.pdb` - 调试符号文件
- ❌ `*.xml` - 文档注释文件

### 保留的文件
- ✅ `EndfieldPuzzleSolver.Avalonia.exe` (19 MB)
- ✅ `OpenCvSharpExtern.dll` (59 MB) - 图像处理必需
- ✅ `libSkiaSharp.dll` (9 MB) - UI 渲染必需
- ✅ `av_libglesv2.dll` (4 MB) - GPU 加速必需
- ✅ `libHarfBuzzSharp.dll` (1.5 MB) - 字体渲染必需
- ✅ `Assets/` - UI 模板资源
- ✅ `appsettings.json` - 配置文件

## 📝 版本命名

| 类型 | 格式 | 示例 |
|------|------|------|
| 正式版本 | v{major}.{minor}.{patch} | v1.0.0 |
| 功能更新 | v{major}.{minor+1}.0 | v1.1.0 |
| Bug 修复 | v{major}.{minor}.{patch+1} | v1.0.1 |
| 测试版本 | v{version}-{tag} | v2.0.0-beta |

## ⚠️ 注意事项

1. **构建环境**: Windows runner (GitHub Actions)
2. **运行时**: 无需 .NET 运行时
3. **架构**: 仅支持 x64，不支持 x86/ARM
4. **OpenCV**: 已验证图像处理功能不需要 FFmpeg

## 🛠️ 本地测试

```powershell
# 构建
dotnet publish src/EndfieldPuzzleSolver.Avalonia/EndfieldPuzzleSolver.Avalonia.csproj `
  -c Release -r win-x64 --self-contained -o publish/avalonia

# 清理
Remove-Item publish/avalonia/*.pdb -Force
Remove-Item publish/avalonia/opencv_videoio_ffmpeg4110_64.dll -Force

# 打包
Compress-Archive -Path publish/avalonia/* -DestinationPath test-release.zip
```

## 📊 Avalonia vs WinUI3

| 特性 | Avalonia | WinUI3 |
|------|----------|--------|
| Native AOT | ✅ 支持 | ❌ 不兼容 |
| 文件数量 | 13 个 | 269 个 |
| 总大小 | 93 MB | 202 MB |
| 启动速度 | 快 | 慢 |
| 运行状态 | ✅ 正常 | ❌ 无法启动 |
| 推荐使用 | ✅ 是 | ❌ 已弃用 |

---

**更多详情**: 参见 [RELEASE.md](RELEASE.md)
