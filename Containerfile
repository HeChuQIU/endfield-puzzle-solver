# 构建阶段
FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# 复制项目文件
COPY src/EndfieldPuzzleSolver.Api/EndfieldPuzzleSolver.Api.csproj EndfieldPuzzleSolver.Api/
COPY src/EndfieldPuzzleSolver.Recognition/EndfieldPuzzleSolver.Recognition.csproj EndfieldPuzzleSolver.Recognition/
COPY src/EndfieldPuzzleSolver.Algorithm/EndfieldPuzzleSolver.Algorithm.fsproj EndfieldPuzzleSolver.Algorithm/
COPY src/EndfieldPuzzleSolver.Core/EndfieldPuzzleSolver.Core.csproj EndfieldPuzzleSolver.Core/

# 还原依赖
RUN dotnet restore EndfieldPuzzleSolver.Api/EndfieldPuzzleSolver.Api.csproj

# 复制源代码
COPY src/EndfieldPuzzleSolver.Api/ EndfieldPuzzleSolver.Api/
COPY src/EndfieldPuzzleSolver.Recognition/ EndfieldPuzzleSolver.Recognition/
COPY src/EndfieldPuzzleSolver.Algorithm/ EndfieldPuzzleSolver.Algorithm/
COPY src/EndfieldPuzzleSolver.Core/ EndfieldPuzzleSolver.Core/

# 复制模板资源
COPY src/EndfieldPuzzleSolver/Assets/UiTemplates/ EndfieldPuzzleSolver/Assets/UiTemplates/

# 构建并发布
RUN dotnet publish EndfieldPuzzleSolver.Api/EndfieldPuzzleSolver.Api.csproj \
    -c Release \
    -o /app/publish \
    --no-restore

# 运行阶段
FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS runtime
WORKDIR /app

# 安装 OpenCvSharp 运行时依赖（原生库由 NuGet 包 OpenCvSharp4.official.runtime.linux-x64 提供）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdiplus \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 从构建阶段复制发布文件
COPY --from=build /app/publish .

# 创建模板目录
RUN mkdir -p /app/Assets/UiTemplates

# 暴露端口
EXPOSE 52000

# 设置环境变量
ENV ASPNETCORE_URLS=http://0.0.0.0:52000
ENV TemplateDir=/app/Assets/UiTemplates

ENTRYPOINT ["dotnet", "EndfieldPuzzleSolver.Api.dll"]
