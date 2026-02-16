namespace EndfieldPuzzleSolver.Web;

/// <summary>
/// 应用配置，从 appsettings.json 或环境变量加载。
/// </summary>
public class AppSettings
{
    /// <summary>后端 API 基地址（如 https://api.example.com）</summary>
    public string ApiBaseUrl { get; set; } = "";
}
