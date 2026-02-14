# 本地发布测试脚本
# 用于验证 GitHub Actions workflow 的本地版本

param(
    [string]$Version = "v0.0.0-test"
)

Write-Host "=== Endfield Puzzle Solver - 本地发布测试 ===" -ForegroundColor Cyan
Write-Host "版本: $Version`n" -ForegroundColor Yellow

# 清理旧的构建
Write-Host "[1/7] 清理旧构建..." -ForegroundColor Green
if (Test-Path "publish/avalonia") {
    Remove-Item "publish/avalonia" -Recurse -Force
}
if (Test-Path "*.zip") {
    Remove-Item "*.zip" -Force
}

# 还原依赖
Write-Host "[2/7] 还原依赖..." -ForegroundColor Green
dotnet restore
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 依赖还原失败" -ForegroundColor Red
    exit 1
}

# 发布项目
Write-Host "[3/7] 发布 Avalonia 项目 (Native AOT)..." -ForegroundColor Green
dotnet publish src/EndfieldPuzzleSolver.Avalonia/EndfieldPuzzleSolver.Avalonia.csproj `
    -c Release `
    -r win-x64 `
    --self-contained `
    -o publish/avalonia `
    -p:PublishAot=true `
    -p:PublishSingleFile=true `
    -p:IncludeNativeLibrariesForSelfExtract=true

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ 发布失败" -ForegroundColor Red
    exit 1
}

# 移除不必要的文件
Write-Host "[4/7] 移除不必要的文件..." -ForegroundColor Green
$removedFiles = @()
$filesToRemove = @(
    "publish/avalonia/*.pdb",
    "publish/avalonia/*.xml",
    "publish/avalonia/opencv_videoio_ffmpeg4110_64.dll"
)

foreach ($pattern in $filesToRemove) {
    $files = Get-ChildItem $pattern -ErrorAction SilentlyContinue
    if ($files) {
        foreach ($file in $files) {
            $removedFiles += $file.Name
            Remove-Item $file.FullName -Force
        }
    }
}

Write-Host "   移除了 $($removedFiles.Count) 个文件: $($removedFiles -join ', ')" -ForegroundColor Gray

# 列出最终文件
Write-Host "[5/7] 检查发布文件..." -ForegroundColor Green
$files = Get-ChildItem "publish/avalonia" -Recurse -File
$totalSize = ($files | Measure-Object -Property Length -Sum).Sum / 1MB
Write-Host "   文件数量: $($files.Count)" -ForegroundColor Gray
Write-Host "   总大小: $([math]::Round($totalSize, 2)) MB" -ForegroundColor Gray

# 创建 ZIP 包
Write-Host "[6/7] 创建 ZIP 包..." -ForegroundColor Green
$zipName = "EndfieldPuzzleSolver-Avalonia-$Version-win-x64.zip"
Compress-Archive -Path "publish/avalonia/*" -DestinationPath $zipName -Force
$zipSize = (Get-Item $zipName).Length / 1MB
Write-Host "   ZIP 文件: $zipName" -ForegroundColor Gray
Write-Host "   ZIP 大小: $([math]::Round($zipSize, 2)) MB" -ForegroundColor Gray

# 计算哈希
Write-Host "[7/7] 计算 SHA256 哈希..." -ForegroundColor Green
$hash = (Get-FileHash $zipName -Algorithm SHA256).Hash
Write-Host "   SHA256: $hash" -ForegroundColor Gray

# 保存发布信息
$releaseInfo = @"
=== 发布信息 ===
版本: $Version
文件: $zipName
大小: $([math]::Round($zipSize, 2)) MB
文件数: $($files.Count)
SHA256: $hash
构建时间: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")

=== 已移除的文件 ===
$($removedFiles -join "`n")

=== 包含的文件 ===
"@

Get-ChildItem "publish/avalonia" -File | ForEach-Object {
    $releaseInfo += "`n$($_.Name) - $([math]::Round($_.Length/1KB, 2)) KB"
}

$releaseInfo | Out-File "release-info.txt" -Encoding UTF8
Write-Host "`n✅ 构建完成！" -ForegroundColor Green
Write-Host "   发布信息已保存到: release-info.txt" -ForegroundColor Gray
Write-Host "   ZIP 文件: $zipName`n" -ForegroundColor Yellow

# 询问是否测试运行
$response = Read-Host "是否测试运行应用? (y/N)"
if ($response -eq 'y' -or $response -eq 'Y') {
    Write-Host "`n正在启动应用..." -ForegroundColor Cyan
    Start-Process "publish/avalonia/EndfieldPuzzleSolver.Avalonia.exe"
    Start-Sleep -Seconds 2
    
    $process = Get-Process "EndfieldPuzzleSolver.Avalonia" -ErrorAction SilentlyContinue
    if ($process) {
        Write-Host "✅ 应用已成功启动 (PID: $($process.Id))" -ForegroundColor Green
    } else {
        Write-Host "⚠️  无法检测到运行中的应用进程" -ForegroundColor Yellow
    }
}

Write-Host "`n🎉 本地发布测试完成！" -ForegroundColor Cyan
