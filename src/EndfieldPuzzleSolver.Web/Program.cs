using EndfieldPuzzleSolver.Web;
using EndfieldPuzzleSolver.Web.Services;
using Microsoft.AspNetCore.Components.Web;
using Microsoft.AspNetCore.Components.WebAssembly.Hosting;
using Microsoft.FluentUI.AspNetCore.Components;

var builder = WebAssemblyHostBuilder.CreateDefault(args);
builder.RootComponents.Add<App>("#app");
builder.RootComponents.Add<HeadOutlet>("head::after");

// 读取配置
var appSettings = builder.Configuration.GetSection("App").Get<AppSettings>() ?? new AppSettings();

// 配置 HttpClient 指向后端 API
builder.Services.AddScoped(sp => 
{
    var client = new HttpClient { BaseAddress = new Uri(appSettings.ApiBaseUrl) };
    return client;
});

// 注册 API 客户端
builder.Services.AddScoped<PuzzleApiClient>();

// 注册 FluentUI 服务
builder.Services.AddFluentUIComponents();

await builder.Build().RunAsync();
