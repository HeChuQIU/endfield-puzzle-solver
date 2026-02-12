using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using EndfieldPuzzleSolver.Recognition;
using EndfieldPuzzleSolver.Recognition.Models;
using EndfieldPuzzleSolver.Recognition.Config;
using Microsoft.Extensions.Configuration;

namespace EndfieldPuzzleSolver.ViewModels;

/// <summary>
/// 主窗口的 ViewModel，使用 CommunityToolkit.Mvvm 的源生成器。
/// </summary>
public partial class MainViewModel : ObservableObject
{
    private static readonly string TemplateDir = Path.Combine(AppContext.BaseDirectory, "Assets", "UiTemplates");
    private static readonly IConfiguration Config = new ConfigurationBuilder()
        .SetBasePath(AppContext.BaseDirectory)
        .AddJsonFile("appsettings.json", optional: true, reloadOnChange: true)
        .Build();
    private static readonly DetectionConfig DetectionConfig = Config.GetSection("Detection").Get<DetectionConfig>()
        ?? new DetectionConfig();

    /// <summary>
    /// 由 MainWindow 设置，用于选择图片文件。
    /// </summary>
    public Func<Task<string?>>? PickImageAsync { get; set; }

    [ObservableProperty]
    private PuzzleData? _puzzleData;

    [ObservableProperty]
    private SolveResult? _solveResult;

    [ObservableProperty]
    private int _currentStepIndex;

    [ObservableProperty]
    private int _totalSteps;

    [ObservableProperty]
    private string _statusMessage = "就绪";

    [ObservableProperty]
    private bool _isLoading;

    public bool HasPuzzle => PuzzleData != null;

    public bool HasSolution => SolveResult is { IsSolved: true };

    public bool CanGoPrevious => TotalSteps > 0 && CurrentStepIndex > 0;

    public bool CanGoNext => TotalSteps > 0 && CurrentStepIndex < TotalSteps;

    public string StepNavigationText => TotalSteps > 0 ? $"步骤 {CurrentStepIndex + 1} / {TotalSteps}" : "步骤 0 / 0";

    /// <summary>
    /// 当需要重绘棋盘时触发（由 code-behind 订阅）。
    /// </summary>
    public event EventHandler? BoardSnapshotChanged;

    partial void OnPuzzleDataChanged(PuzzleData? value)
    {
        OnPropertyChanged(nameof(HasPuzzle));
        BoardSnapshotChanged?.Invoke(this, EventArgs.Empty);
    }

    partial void OnSolveResultChanged(SolveResult? value)
    {
        OnPropertyChanged(nameof(HasSolution));
        TotalSteps = value?.Steps?.Count ?? 0;
        CurrentStepIndex = TotalSteps > 0 ? TotalSteps - 1 : 0;
        OnPropertyChanged(nameof(CanGoPrevious));
        OnPropertyChanged(nameof(CanGoNext));
        OnPropertyChanged(nameof(StepNavigationText));
        BoardSnapshotChanged?.Invoke(this, EventArgs.Empty);
    }

    partial void OnCurrentStepIndexChanged(int value)
    {
        OnPropertyChanged(nameof(CanGoPrevious));
        OnPropertyChanged(nameof(CanGoNext));
        OnPropertyChanged(nameof(StepNavigationText));
        BoardSnapshotChanged?.Invoke(this, EventArgs.Empty);
    }

    partial void OnTotalStepsChanged(int value)
    {
        OnPropertyChanged(nameof(CanGoPrevious));
        OnPropertyChanged(nameof(CanGoNext));
        OnPropertyChanged(nameof(StepNavigationText));
    }

    [RelayCommand]
    public async Task OpenScreenshotAsync()
    {
        var path = PickImageAsync != null ? await PickImageAsync() : null;
        if (path != null)
            await LoadFromScreenshotAsync(path);
    }

    [RelayCommand(CanExecute = nameof(HasPuzzle))]
    public async Task SolvePuzzleAsync()
    {
        if (PuzzleData == null) return;
        IsLoading = true;
        StatusMessage = "正在求解...";
        try
        {
            var puzzle = PuzzleData;
            SolveResult = await Task.Run(() => EndfieldPuzzleSolver.Algorithm.Solver.solve(puzzle));
            StatusMessage = SolveResult.IsSolved ? "求解成功" : (SolveResult.Message ?? "无解");
        }
        catch (Exception ex)
        {
            SolveResult = null;
            StatusMessage = $"求解失败: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    [RelayCommand(CanExecute = nameof(CanGoPrevious))]
    public void GoToPreviousStep()
    {
        if (CanGoPrevious)
        {
            CurrentStepIndex--;
        }
    }

    [RelayCommand(CanExecute = nameof(CanGoNext))]
    public void GoToNextStep()
    {
        if (CanGoNext)
        {
            CurrentStepIndex++;
        }
    }

    /// <summary>
    /// 加载截图并识别谜题。由"打开截图"按钮和拖拽操作共同使用。
    /// </summary>
    public async Task LoadFromScreenshotAsync(string imagePath)
    {
        IsLoading = true;
        StatusMessage = "正在识别截图...";
        try
        {
            var result = await Task.Run(() =>
            {
                var detector = new PuzzleDetector(TemplateDir, DetectionConfig);
                return detector.Detect(imagePath);
            });

            if (result.IsSuccess)
            {
                PuzzleData = result.Data;
                SolveResult = null;
                StatusMessage = "识别完成";
            }
            else
            {
                PuzzleData = null;
                SolveResult = null;
                StatusMessage = $"识别失败: {result.Error}";
            }
        }
        catch (Exception ex)
        {
            PuzzleData = null;
            SolveResult = null;
            StatusMessage = $"识别异常: {ex.Message}";
        }
        finally
        {
            IsLoading = false;
        }
    }

    /// <summary>
    /// 获取当前步骤对应的棋盘快照（用于绘制）。
    /// 若有解且正在查看步骤，返回该步骤后的快照；否则返回谜题初始棋盘。
    /// </summary>
    public TileInfo[,]? GetCurrentBoardSnapshot()
    {
        if (PuzzleData == null) return null;
        if (SolveResult is not { IsSolved: true } result || result.Steps.Count == 0)
            return PuzzleData.Tiles;

        int idx = Math.Clamp(CurrentStepIndex, 0, result.Steps.Count - 1);
        return result.Steps[idx].BoardSnapshot;
    }

    public void NotifyStepNavigationChanged()
    {
        OnPropertyChanged(nameof(CanGoPrevious));
        OnPropertyChanged(nameof(CanGoNext));
        OnPropertyChanged(nameof(StepNavigationText));
    }
}
