using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;
using EndfieldPuzzleSolver.Web.Models;

namespace EndfieldPuzzleSolver.Web.Services;

/// <summary>
/// 后端 API 客户端，负责与 ASP.NET 后端通信。
/// </summary>
public class PuzzleApiClient
{
    private readonly HttpClient _http;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        PropertyNameCaseInsensitive = true,
        Converters = { new Array2DJsonConverter<bool>(), new Array2DJsonConverter<TileInfoDto>() }
    };

    public PuzzleApiClient(HttpClient http)
    {
        _http = http;
    }

    /// <summary>
    /// 上传截图，执行识别 + 求解，返回完整结果。
    /// </summary>
    public async Task<DetectAndSolveResult> DetectAndSolveAsync(Stream imageStream, string fileName)
    {
        using var content = new MultipartFormDataContent();
        using var streamContent = new StreamContent(imageStream);
        content.Add(streamContent, "image", fileName);

        var response = await _http.PostAsync("api/detect-and-solve", content);

        if (response.IsSuccessStatusCode)
        {
            var json = await response.Content.ReadAsStringAsync();
            var result = JsonSerializer.Deserialize<DetectAndSolveResponseDto>(json, JsonOptions);
            if (result?.Puzzle != null)
            {
                return new DetectAndSolveResult
                {
                    IsSuccess = true,
                    Puzzle = result.Puzzle,
                    Solution = result.Solution
                };
            }
        }

        // 解析错误
        try
        {
            var errorJson = await response.Content.ReadAsStringAsync();
            var error = JsonSerializer.Deserialize<ErrorResponse>(errorJson, JsonOptions);
            return new DetectAndSolveResult { IsSuccess = false, Error = error?.Error ?? $"HTTP {response.StatusCode}" };
        }
        catch
        {
            return new DetectAndSolveResult { IsSuccess = false, Error = $"HTTP {response.StatusCode}" };
        }
    }

    /// <summary>
    /// 加载预计算的示例谜题（从前端静态资源）。
    /// </summary>
    public async Task<DetectAndSolveResult> LoadSampleAsync(HttpClient wasmHttp, string sampleName)
    {
        try
        {
            var json = await wasmHttp.GetStringAsync($"samples/{sampleName}.json");
            var result = JsonSerializer.Deserialize<DetectAndSolveResponseDto>(json, JsonOptions);
            if (result?.Puzzle != null)
            {
                return new DetectAndSolveResult
                {
                    IsSuccess = true,
                    Puzzle = result.Puzzle,
                    Solution = result.Solution
                };
            }
            return new DetectAndSolveResult { IsSuccess = false, Error = "示例数据解析失败" };
        }
        catch (Exception ex)
        {
            return new DetectAndSolveResult { IsSuccess = false, Error = $"加载示例失败: {ex.Message}" };
        }
    }

    /// <summary>健康检查</summary>
    public async Task<bool> HealthCheckAsync()
    {
        try
        {
            var response = await _http.GetAsync("api/health");
            return response.IsSuccessStatusCode;
        }
        catch
        {
            return false;
        }
    }
}

public class DetectAndSolveResult
{
    public bool IsSuccess { get; init; }
    public PuzzleDataDto? Puzzle { get; init; }
    public SolveResultDto? Solution { get; init; }
    public string? Error { get; init; }
}

record ErrorResponse(string Error);
