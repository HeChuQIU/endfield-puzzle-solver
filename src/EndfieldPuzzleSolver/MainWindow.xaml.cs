using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using WinRT.Interop;
using EndfieldPuzzleSolver.Recognition.Models;
using EndfieldPuzzleSolver.ViewModels;

namespace EndfieldPuzzleSolver;

public sealed partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    private const double TileSize = 36;
    private const double RequirementMargin = 24;
    private const double ShapeCellSize = 14;

    public MainWindow()
    {
        InitializeComponent();
        SystemBackdrop = new Microsoft.UI.Xaml.Media.MicaBackdrop();


        // 设置文件选择回调
        ViewModel.PickImageAsync = PickImageAsync;
        ViewModel.BoardSnapshotChanged += OnBoardSnapshotChanged;

        RootGrid.Loaded += (_, _) =>
        {
            UpdatePuzzleInfo();
            DrawBoard();
        };

        OpenScreenshotBtn.Click += async (_, _) => await ViewModel.OpenScreenshotAsync();
        SolveBtn.Click += async (_, _) => await ViewModel.SolvePuzzleAsync();
        PrevStepBtn.Click += (_, _) => ViewModel.GoToPreviousStep();
        NextStepBtn.Click += (_, _) => ViewModel.GoToNextStep();

        ViewModel.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName is nameof(MainViewModel.StatusMessage))
                StatusInfoBar.Message = ViewModel.StatusMessage;
            if (e.PropertyName is nameof(MainViewModel.StepNavigationText) or nameof(MainViewModel.CurrentStepIndex) or nameof(MainViewModel.TotalSteps))
                StepTextBlock.Text = ViewModel.StepNavigationText;
        };
    }

    private async Task<string?> PickImageAsync()
    {
        var picker = new Windows.Storage.Pickers.FileOpenPicker
        {
            SuggestedStartLocation = Windows.Storage.Pickers.PickerLocationId.PicturesLibrary
        };
        picker.FileTypeFilter.Add(".png");
        picker.FileTypeFilter.Add(".jpg");
        picker.FileTypeFilter.Add(".jpeg");

        var hwnd = WindowNative.GetWindowHandle(this);
        InitializeWithWindow.Initialize(picker, hwnd);
        var file = await picker.PickSingleFileAsync();
        return file?.Path;
    }

    private static readonly HashSet<string> SupportedImageExtensions = [".png", ".jpg", ".jpeg"];

    private void BoardDropArea_DragOver(object sender, Microsoft.UI.Xaml.DragEventArgs e)
    {
        e.AcceptedOperation = Windows.ApplicationModel.DataTransfer.DataPackageOperation.Copy;
        e.DragUIOverride.Caption = "加载截图";
        e.DragUIOverride.IsCaptionVisible = true;
    }

    private async void BoardDropArea_Drop(object sender, Microsoft.UI.Xaml.DragEventArgs e)
    {
        if (!e.DataView.Contains(Windows.ApplicationModel.DataTransfer.StandardDataFormats.StorageItems))
            return;

        var items = await e.DataView.GetStorageItemsAsync();
        foreach (var item in items)
        {
            if (item is Windows.Storage.StorageFile file &&
                SupportedImageExtensions.Contains(file.FileType.ToLowerInvariant()))
            {
                await ViewModel.LoadFromScreenshotAsync(file.Path);
                return; // 只处理第一个有效图片
            }
        }
    }

    private void OnBoardSnapshotChanged(object? sender, EventArgs e)
    {
        UpdatePuzzleInfo();
        DrawBoard();
    }

    private void UpdatePuzzleInfo()
    {
        var vm = ViewModel;
        if (vm.PuzzleData == null)
        {
            GridSizeText.Text = "网格大小: - × -";
            ColorGroupsItems.Items.Clear();
            ComponentsItems.Items.Clear();
            PuzzleInfoPanel.Visibility = Visibility.Collapsed;
            PlaceholderText.Visibility = Visibility.Visible;
            return;
        }

        PuzzleInfoPanel.Visibility = Visibility.Visible;
        PlaceholderText.Visibility = Visibility.Collapsed;
        var p = vm.PuzzleData;
        GridSizeText.Text = $"网格大小: {p.Rows} × {p.Cols}";

        ColorGroupsItems.Items.Clear();
        foreach (var cg in p.ColorGroups)
        {
            var brush = HsvToBrush(cg.Hue, cg.Saturation, cg.Value);
            var panel = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 8 };
            panel.Children.Add(new Border
            {
                Width = 16,
                Height = 16,
                CornerRadius = new CornerRadius(2),
                Background = brush,
                BorderBrush = new SolidColorBrush(Microsoft.UI.Colors.Gray),
                BorderThickness = new Thickness(1)
            });
            panel.Children.Add(new TextBlock { Text = $"{cg.Label} (H{cg.Hue} S{cg.Saturation} V{cg.Value})", VerticalAlignment = VerticalAlignment.Center });
            ColorGroupsItems.Items.Add(panel);
        }

        ComponentsItems.Items.Clear();
        for (int i = 0; i < p.Components.Length; i++)
        {
            var comp = p.Components[i];
            var brush = GetColorGroupBrush(p, comp.ColorGroup);

            // 元件标签
            var label = new TextBlock
            {
                Text = $"元件 {i + 1}: {comp.ColorGroup} ({comp.TileCount} 格)",
                Foreground = brush,
                Margin = new Thickness(0, 4, 0, 2)
            };

            // 用小方格矩阵绘制元件原始形状
            var shapeCanvas = new Canvas
            {
                Width = comp.Cols * ShapeCellSize,
                Height = comp.Rows * ShapeCellSize,
                Margin = new Thickness(4, 0, 0, 4)
            };
            for (int sr = 0; sr < comp.Rows; sr++)
            {
                for (int sc = 0; sc < comp.Cols; sc++)
                {
                    if (comp.Shape[sr, sc])
                    {
                        var cell = new Rectangle
                        {
                            Width = ShapeCellSize - 1,
                            Height = ShapeCellSize - 1,
                            Fill = brush,
                            Stroke = new SolidColorBrush(Microsoft.UI.Colors.Black),
                            StrokeThickness = 0.5
                        };
                        Canvas.SetLeft(cell, sc * ShapeCellSize);
                        Canvas.SetTop(cell, sr * ShapeCellSize);
                        shapeCanvas.Children.Add(cell);
                    }
                }
            }

            var container = new StackPanel();
            container.Children.Add(label);
            container.Children.Add(shapeCanvas);
            ComponentsItems.Items.Add(container);
        }
    }

    private void DrawBoard()
    {
        BoardCanvas.Children.Clear();

        var snapshot = ViewModel.GetCurrentBoardSnapshot();
        var puzzle = ViewModel.PuzzleData;
        if (snapshot == null || puzzle == null)
            return;

        int rows = puzzle.Rows;
        int cols = puzzle.Cols;
        double gridW = cols * TileSize + 2 * RequirementMargin;
        double gridH = rows * TileSize + 2 * RequirementMargin;

        // 绘制行需求（左侧）—— 显示 "已满足/总数" 格式
        for (int r = 0; r < rows; r++)
        {
            var reqs = puzzle.RowRequirements[r];
            double y = RequirementMargin + r * TileSize + TileSize / 2;
            double x = RequirementMargin / 2;
            foreach (var req in reqs)
            {
                // 动态计算当前已满足数：统计当前快照中该行该颜色的 Lock 格子数
                int currentFilled = CountColorInRow(snapshot, r, req.ColorGroup);
                var tb = new TextBlock
                {
                    Text = $"{currentFilled}/{req.Count}",
                    Foreground = GetColorGroupBrush(puzzle, req.ColorGroup),
                    FontSize = 11,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                };
                Canvas.SetLeft(tb, x - 14);
                Canvas.SetTop(tb, y - 8);
                BoardCanvas.Children.Add(tb);
                x -= 28;
            }
        }

        // 绘制列需求（上方）—— 显示 "已满足/总数" 格式
        for (int c = 0; c < cols; c++)
        {
            var reqs = puzzle.ColumnRequirements[c];
            double x = RequirementMargin + c * TileSize + TileSize / 2;
            double y = RequirementMargin / 2;
            foreach (var req in reqs)
            {
                int currentFilled = CountColorInCol(snapshot, c, req.ColorGroup);
                var tb = new TextBlock
                {
                    Text = $"{currentFilled}/{req.Count}",
                    Foreground = GetColorGroupBrush(puzzle, req.ColorGroup),
                    FontSize = 11,
                    HorizontalAlignment = HorizontalAlignment.Center,
                    VerticalAlignment = VerticalAlignment.Center
                };
                Canvas.SetLeft(tb, x - 14);
                Canvas.SetTop(tb, y - 8);
                BoardCanvas.Children.Add(tb);
                y -= 16;
            }
        }

        // 第一遍：绘制所有格子的填充色（无边框）
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var tile = snapshot[r, c];
                double left = RequirementMargin + c * TileSize;
                double top = RequirementMargin + r * TileSize;

                SolidColorBrush? fill = tile.Type switch
                {
                    TileType.Disabled => new SolidColorBrush(Microsoft.UI.Colors.DarkGray),
                    TileType.Lock when tile.ColorGroup != null => GetColorGroupBrush(puzzle, tile.ColorGroup),
                    _ => null
                };

                if (fill != null)
                {
                    var rect = new Rectangle { Width = TileSize, Height = TileSize, Fill = fill };
                    Canvas.SetLeft(rect, left);
                    Canvas.SetTop(rect, top);
                    BoardCanvas.Children.Add(rect);
                }

                // 图标叠加
                if (tile.Type == TileType.Disabled)
                {
                    var symbol = new TextBlock
                    {
                        Text = "⊘", FontSize = 18,
                        Foreground = new SolidColorBrush(Microsoft.UI.Colors.White),
                        Width = TileSize, Height = TileSize,
                        TextAlignment = TextAlignment.Center,
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center
                    };
                    Canvas.SetLeft(symbol, left);
                    Canvas.SetTop(symbol, top);
                    BoardCanvas.Children.Add(symbol);
                }
                else if (tile.Type == TileType.Lock && tile.PlacedComponentIndex < 0 && tile.ColorGroup != null)
                {
                    var lockTb = new TextBlock
                    {
                        Text = "🔒", FontSize = 14,
                        Width = TileSize, Height = TileSize,
                        TextAlignment = TextAlignment.Center,
                        HorizontalAlignment = HorizontalAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center
                    };
                    Canvas.SetLeft(lockTb, left);
                    Canvas.SetTop(lockTb, top);
                    BoardCanvas.Children.Add(lockTb);
                }
            }
        }

        // 第二遍：绘制边框线（同一元件内部不画）
        var borderBrushDark = new SolidColorBrush(Microsoft.UI.Colors.Black);
        var borderBrushLight = new SolidColorBrush(Microsoft.UI.Colors.DarkGray);

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var tile = snapshot[r, c];
                double left = RequirementMargin + c * TileSize;
                double top = RequirementMargin + r * TileSize;

                var strokeBrush = tile.Type switch
                {
                    TileType.Empty => borderBrushLight,
                    TileType.Disabled => borderBrushLight,
                    _ => borderBrushDark
                };

                // 上边框：无上邻居 或 上邻居不是同一元件
                if (r == 0 || !IsSameComponent(tile, snapshot[r - 1, c]))
                    AddBorderLine(left, top, left + TileSize, top, strokeBrush);

                // 左边框：无左邻居 或 左邻居不是同一元件
                if (c == 0 || !IsSameComponent(tile, snapshot[r, c - 1]))
                    AddBorderLine(left, top, left, top + TileSize, strokeBrush);

                // 下边框：最后一行 或 下邻居不是同一元件
                if (r == rows - 1 || !IsSameComponent(tile, snapshot[r + 1, c]))
                    AddBorderLine(left, top + TileSize, left + TileSize, top + TileSize, strokeBrush);

                // 右边框：最后一列 或 右邻居不是同一元件
                if (c == cols - 1 || !IsSameComponent(tile, snapshot[r, c + 1]))
                    AddBorderLine(left + TileSize, top, left + TileSize, top + TileSize, strokeBrush);
            }
        }

        BoardCanvas.Width = gridW;
        BoardCanvas.Height = gridH;
    }

    /// <summary>判断两个格子是否属于同一个已放置元件</summary>
    private static bool IsSameComponent(TileInfo a, TileInfo b)
        => a.PlacedComponentIndex >= 0 && a.PlacedComponentIndex == b.PlacedComponentIndex;

    /// <summary>在 BoardCanvas 上绘制一条边框线</summary>
    private void AddBorderLine(double x1, double y1, double x2, double y2, SolidColorBrush stroke)
    {
        var line = new Line
        {
            X1 = x1, Y1 = y1,
            X2 = x2, Y2 = y2,
            Stroke = stroke,
            StrokeThickness = 1
        };
        BoardCanvas.Children.Add(line);
    }

    /// <summary>统计棋盘快照中某行某颜色组的 Lock 格子数量</summary>
    private static int CountColorInRow(TileInfo[,] board, int row, string colorGroup)
    {
        int count = 0;
        int cols = board.GetLength(1);
        for (int c = 0; c < cols; c++)
        {
            var tile = board[row, c];
            if (tile.Type == TileType.Lock && tile.ColorGroup == colorGroup)
                count++;
        }
        return count;
    }

    /// <summary>统计棋盘快照中某列某颜色组的 Lock 格子数量</summary>
    private static int CountColorInCol(TileInfo[,] board, int col, string colorGroup)
    {
        int count = 0;
        int rows = board.GetLength(0);
        for (int r = 0; r < rows; r++)
        {
            var tile = board[r, col];
            if (tile.Type == TileType.Lock && tile.ColorGroup == colorGroup)
                count++;
        }
        return count;
    }

    private static SolidColorBrush GetColorGroupBrush(PuzzleData puzzle, string colorGroup)
    {
        var cg = puzzle.ColorGroups.FirstOrDefault(x => x.Label == colorGroup);
        if (cg != null)
            return HsvToBrush(cg.Hue, cg.Saturation, cg.Value);
        return new SolidColorBrush(Microsoft.UI.Colors.Gray);
    }

    /// <summary>
    /// HSV 转 RGB 画笔。
    /// H: OpenCV 范围 0-180（自动转换为 0-360）
    /// S: OpenCV 范围 0-255（自动转换为 0-100）
    /// V: OpenCV 范围 0-255（自动转换为 0-100）
    /// </summary>
    private static SolidColorBrush HsvToBrush(int h, int s, int v)
    {
        var (r, g, b) = HsvToRgb(h, s, v);
        return new SolidColorBrush(Windows.UI.Color.FromArgb(255, r, g, b));
    }

    private static (byte R, byte G, byte B) HsvToRgb(int h, int s, int v)
    {
        // OpenCV HSV: H=0-180, S=0-255, V=0-255 → 标准: H=0-360, S=0-100, V=0-100
        double hDeg = Math.Clamp(h * 2.0, 0, 360);
        double sNorm = Math.Clamp(s / 255.0, 0, 1);
        double vNorm = Math.Clamp(v / 255.0, 0, 1);

        double c = vNorm * sNorm;
        double x = c * (1 - Math.Abs((hDeg / 60.0) % 2 - 1));
        double m = vNorm - c;

        double r1, g1, b1;
        if (hDeg < 60) { r1 = c; g1 = x; b1 = 0; }
        else if (hDeg < 120) { r1 = x; g1 = c; b1 = 0; }
        else if (hDeg < 180) { r1 = 0; g1 = c; b1 = x; }
        else if (hDeg < 240) { r1 = 0; g1 = x; b1 = c; }
        else if (hDeg < 300) { r1 = x; g1 = 0; b1 = c; }
        else { r1 = c; g1 = 0; b1 = x; }

        return (
            (byte)((r1 + m) * 255),
            (byte)((g1 + m) * 255),
            (byte)((b1 + m) * 255)
        );
    }
}
