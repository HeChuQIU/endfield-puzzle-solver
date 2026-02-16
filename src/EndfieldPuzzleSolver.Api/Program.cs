using EndfieldPuzzleSolver.Api;
using EndfieldPuzzleSolver.Recognition;
using EndfieldPuzzleSolver.Recognition.Config;
using EndfieldPuzzleSolver.Recognition.Models;
using EndfieldPuzzleSolver.Algorithm;
using Microsoft.AspNetCore.Http.Json;

var builder = WebApplication.CreateBuilder(args);

// 注册 2D 数组 JSON 转换器
builder.Services.Configure<JsonOptions>(options =>
{
    options.SerializerOptions.Converters.Add(new Array2DConverterFactory());
});

// 配置 CORS
builder.Services.AddCors(options =>
{
    options.AddDefaultPolicy(policy =>
    {
        var origins = builder.Configuration.GetSection("Cors:AllowedOrigins").Get<string[]>()
                      ?? ["*"];
        if (origins is ["*"])
        {
            policy.AllowAnyOrigin().AllowAnyHeader().AllowAnyMethod();
        }
        else
        {
            policy.WithOrigins(origins).AllowAnyHeader().AllowAnyMethod();
        }
    });
});

// 绑定识别配置
var detectionConfig = builder.Configuration.GetSection("Detection").Get<DetectionConfig>()
                      ?? new DetectionConfig();
builder.Services.AddSingleton(detectionConfig);

// 模板目录
var templateDir = builder.Configuration["TemplateDir"];
if (string.IsNullOrEmpty(templateDir))
    templateDir = Path.Combine(AppContext.BaseDirectory, "Assets", "UiTemplates");
builder.Services.AddKeyedSingleton("TemplateDir", templateDir);

var app = builder.Build();

app.UseCors();

// ─────────── Health check ───────────
app.MapGet("/api/health", () => Results.Ok(new { status = "ok", timestamp = DateTime.UtcNow }));

// ─────────── 识别 + 求解（一步完成）───────────
app.MapPost("/api/detect-and-solve", async (HttpRequest request,
    DetectionConfig config,
    [FromKeyedServices("TemplateDir")] string tplDir) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "请以 multipart/form-data 上传图片" });

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("image");
    if (file is null || file.Length == 0)
        return Results.BadRequest(new { error = "未找到 image 文件字段" });

    // 保存到临时文件（OpenCV 需要文件路径）
    var tempPath = Path.Combine(Path.GetTempPath(), $"eps_{Guid.NewGuid():N}{Path.GetExtension(file.FileName)}");
    try
    {
        await using (var fs = File.Create(tempPath))
        {
            await file.CopyToAsync(fs);
        }

        // 在线程池中运行识别（CPU 密集型）
        var (puzzleData, detectError) = await Task.Run(() =>
        {
            var detector = new PuzzleDetector(tplDir, config);
            var result = detector.Detect(tempPath);
            return (result.Data, result.Error);
        });

        if (puzzleData is null)
            return Results.UnprocessableEntity(new { error = detectError ?? "识别失败" });

        // 求解
        var solveResult = await Task.Run(() => Solver.solve(puzzleData));

        return Results.Ok(new DetectAndSolveResponse(puzzleData, solveResult));
    }
    finally
    {
        // 清理临时文件
        try { File.Delete(tempPath); } catch { /* ignore */ }
    }
}).DisableAntiforgery();

// ─────────── 仅识别 ───────────
app.MapPost("/api/detect", async (HttpRequest request,
    DetectionConfig config,
    [FromKeyedServices("TemplateDir")] string tplDir) =>
{
    if (!request.HasFormContentType)
        return Results.BadRequest(new { error = "请以 multipart/form-data 上传图片" });

    var form = await request.ReadFormAsync();
    var file = form.Files.GetFile("image");
    if (file is null || file.Length == 0)
        return Results.BadRequest(new { error = "未找到 image 文件字段" });

    var tempPath = Path.Combine(Path.GetTempPath(), $"eps_{Guid.NewGuid():N}{Path.GetExtension(file.FileName)}");
    try
    {
        await using (var fs = File.Create(tempPath))
        {
            await file.CopyToAsync(fs);
        }

        var result = await Task.Run(() =>
        {
            var detector = new PuzzleDetector(tplDir, config);
            return detector.Detect(tempPath);
        });

        if (result.Data is null)
            return Results.UnprocessableEntity(new { error = result.Error ?? "识别失败" });

        return Results.Ok(result.Data);
    }
    finally
    {
        try { File.Delete(tempPath); } catch { /* ignore */ }
    }
}).DisableAntiforgery();

// ─────────── 仅求解 ───────────
app.MapPost("/api/solve", (PuzzleData puzzle) =>
{
    var result = Solver.solve(puzzle);
    return Results.Ok(result);
});

app.Run();

// ──────────── DTO ────────────
record DetectAndSolveResponse(PuzzleData Puzzle, SolveResult Solution);
