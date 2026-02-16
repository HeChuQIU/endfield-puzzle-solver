FROM mcr.microsoft.com/dotnet/aspnet:9.0 AS base
WORKDIR /app
EXPOSE 8080

# 安装 OpenCvSharp 运行时依赖（原生库由 NuGet 包 OpenCvSharp4.official.runtime.linux-x64 提供）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdiplus \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

FROM mcr.microsoft.com/dotnet/sdk:9.0 AS build
WORKDIR /src

# 复制项目文件
COPY src/EndfieldPuzzleSolver.Api/EndfieldPuzzleSolver.Api.csproj src/EndfieldPuzzleSolver.Api/
COPY src/EndfieldPuzzleSolver.Recognition/EndfieldPuzzleSolver.Recognition.csproj src/EndfieldPuzzleSolver.Recognition/
COPY src/EndfieldPuzzleSolver.Algorithm/EndfieldPuzzleSolver.Algorithm.fsproj src/EndfieldPuzzleSolver.Algorithm/

# 还原依赖
RUN dotnet restore src/EndfieldPuzzleSolver.Api/EndfieldPuzzleSolver.Api.csproj

# 复制所有源码
COPY src/ src/

# 发布
WORKDIR /src/src/EndfieldPuzzleSolver.Api
RUN dotnet publish -c Release -o /app/publish --no-restore

FROM base AS final
WORKDIR /app
COPY --from=build /app/publish .
ENV ASPNETCORE_URLS=http://+:8080
ENTRYPOINT ["dotnet", "EndfieldPuzzleSolver.Api.dll"]
