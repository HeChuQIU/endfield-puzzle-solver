using Avalonia;
using Avalonia.Controls;
using Avalonia.Controls.Shapes;
using Avalonia.Input;
using Avalonia.Interactivity;
using Avalonia.Layout;
using Avalonia.Media;
using Avalonia.Platform.Storage;
using EndfieldPuzzleSolver.Core.ViewModels;
using EndfieldPuzzleSolver.Recognition.Models;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace EndfieldPuzzleSolver.Avalonia.Views;

public partial class MainWindow : Window
{
    public MainViewModel ViewModel { get; } = new();

    private const double TileSize = 36;
    private const double RequirementMargin = 24;
    private const double ShapeCellSize = 14;

    private static readonly HashSet<string> SupportedImageExtensions = [".png", ".jpg", ".jpeg"];

    public MainWindow()
    {
        InitializeComponent();
        
        // 设置 DataContext
        DataContext = ViewModel;

        // 启用拖拽（Window 级别）
        AddHandler(DragDrop.DragEnterEvent, OnDragEnter);
        AddHandler(DragDrop.DragLeaveEvent, OnDragLeave);
        AddHandler(DragDrop.DragOverEvent, OnDragOver);
        AddHandler(DragDrop.DropEvent, OnDrop);

        // 设置文件选择回调
        ViewModel.PickImageAsync = PickImageAsync;
        ViewModel.GetClipboardImagePathAsync = GetClipboardImagePathAsync;
        ViewModel.BoardSnapshotChanged += OnBoardSnapshotChanged;

        // 属性变更监听
        ViewModel.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(MainViewModel.StatusMessage))
                StatusTextBlock.Text = ViewModel.StatusMessage;
        };

        // 支持 Ctrl+V 快捷键粘贴
        KeyDown += async (_, e) =>
        {
            if (e.Key == global::Avalonia.Input.Key.V &&
                e.KeyModifiers.HasFlag(global::Avalonia.Input.KeyModifiers.Control))
            {
                await ViewModel.PasteFromClipboardAsync();
                e.Handled = true;
            }
        };

        Loaded += (_, _) =>
        {
            UpdatePuzzleInfo();
            DrawBoard();
        };
    }

    private void OnDragEnter(object? sender, DragEventArgs e)
    {
        if (e.Data.Contains(DataFormats.Files))
        {
            e.DragEffects = DragDropEffects.Copy;
        }
        else
        {
            e.DragEffects = DragDropEffects.None;
        }
    }

    private void OnDragLeave(object? sender, DragEventArgs e)
    {
        // 拖拽离开时的处理（可选）
    }

    private async Task<string?> PickImageAsync()
    {
        var storage = GetTopLevel(this)?.StorageProvider;
        if (storage == null) return null;

        var files = await storage.OpenFilePickerAsync(new FilePickerOpenOptions
        {
            Title = "选择截图",
            AllowMultiple = false,
            FileTypeFilter = new[]
            {
                new FilePickerFileType("图片文件") { Patterns = new[] { "*.png", "*.jpg", "*.jpeg" } }
            }
        });

        var file = files.FirstOrDefault();
        return file?.Path.LocalPath;
    }

    private async Task<string?> GetClipboardImagePathAsync()
    {
        var clipboard = TopLevel.GetTopLevel(this)?.Clipboard;
        if (clipboard == null) return null;

        try
        {
            var formats = await clipboard.GetFormatsAsync();

            // 尝试获取剪贴板中的图片数据（截图）
            foreach (var format in new[] { "PNG", "image/png", "image/bmp", "image/jpeg" })
            {
                if (!formats.Contains(format)) continue;
                var data = await clipboard.GetDataAsync(format);
                if (data is byte[] bytes && bytes.Length > 0)
                {
                    var ext = format.Contains("png") || format == "PNG" ? ".png" : ".bmp";
                    var tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"endfield_clipboard_{DateTime.Now:yyyyMMddHHmmss}{ext}");
                    await File.WriteAllBytesAsync(tempPath, bytes);
                    return tempPath;
                }
                if (data is Stream stream && stream.Length > 0)
                {
                    var ext = format.Contains("png") || format == "PNG" ? ".png" : ".bmp";
                    var tempPath = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"endfield_clipboard_{DateTime.Now:yyyyMMddHHmmss}{ext}");
                    await using var fs = File.Create(tempPath);
                    stream.Position = 0;
                    await stream.CopyToAsync(fs);
                    return tempPath;
                }
            }

            // 尝试获取剪贴板中复制的图片文件
            if (formats.Contains(DataFormats.Files))
            {
                var data = await clipboard.GetDataAsync(DataFormats.Files);
                if (data is IEnumerable<IStorageItem> files)
                {
                    foreach (var f in files)
                    {
                        if (f is IStorageFile storageFile)
                        {
                            var path = storageFile.Path.LocalPath;
                            var ext = System.IO.Path.GetExtension(path).ToLowerInvariant();
                            if (SupportedImageExtensions.Contains(ext))
                                return path;
                        }
                    }
                }
            }
        }
        catch
        {
            // 剪贴板访问可能失败
        }

        return null;
    }

    private void OnDragOver(object? sender, DragEventArgs e)
    {
        // 必须设置 DragEffects 才能接受拖拽
        if (e.Data.Contains(DataFormats.Files))
        {
            e.DragEffects = DragDropEffects.Copy;
        }
        else
        {
            e.DragEffects = DragDropEffects.None;
        }
    }

    private async void OnDrop(object? sender, DragEventArgs e)
    {
        if (!e.Data.Contains(DataFormats.Files))
        {
            e.DragEffects = DragDropEffects.None;
            return;
        }

        var files = e.Data.GetFiles();
        if (files == null) return;
        
        foreach (var file in files)
        {
            if (file is not IStorageFile storageFile) continue;
            
            var path = storageFile.Path.LocalPath;
            var ext = System.IO.Path.GetExtension(path).ToLowerInvariant();
            if (SupportedImageExtensions.Contains(ext))
            {
                await ViewModel.LoadFromScreenshotAsync(path);
                return;
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
            PuzzleInfoPanel.IsVisible = false;
            PlaceholderText.IsVisible = true;
            BoardPlaceholder.IsVisible = true;
            return;
        }

        PuzzleInfoPanel.IsVisible = true;
        PlaceholderText.IsVisible = false;
        BoardPlaceholder.IsVisible = false;

        var p = vm.PuzzleData;
        GridSizeText.Text = $"网格大小: {p.Rows} × {p.Cols}";

        // 颜色组：显示实际颜色 + HSV 值
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
                BorderBrush = Brushes.Gray,
                BorderThickness = new Thickness(1)
            });
            panel.Children.Add(new TextBlock
            {
                Text = $"{cg.Label} (H{cg.Hue} S{cg.Saturation} V{cg.Value})",
                VerticalAlignment = VerticalAlignment.Center
            });
            ColorGroupsItems.Items.Add(panel);
        }

        // 元件列表：编号 + 颜色组 + 格数 + 形状可视化
        ComponentsItems.Items.Clear();
        for (int i = 0; i < p.Components.Length; i++)
        {
            var comp = p.Components[i];
            var brush = GetColorGroupBrush(p, comp.ColorGroup);

            var label = new TextBlock
            {
                Text = $"元件 {i + 1}: {comp.ColorGroup} ({comp.TileCount} 格)",
                Foreground = brush,
                Margin = new Thickness(0, 4, 0, 2)
            };

            // 用小方格矩阵绘制元件形状
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
                            Stroke = Brushes.Black,
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

        // 绘制行需求（左侧）
        for (int r = 0; r < rows; r++)
        {
            var reqs = puzzle.RowRequirements[r];
            double y = RequirementMargin + r * TileSize + TileSize / 2;
            double x = RequirementMargin / 2;
            foreach (var req in reqs)
            {
                int currentFilled = CountColorInRow(snapshot, r, req.ColorGroup);
                var tb = new TextBlock
                {
                    Text = $"{currentFilled}/{req.Count}",
                    Foreground = GetColorGroupBrush(puzzle, req.ColorGroup),
                    FontSize = 11,
                };
                Canvas.SetLeft(tb, x - 14);
                Canvas.SetTop(tb, y - 8);
                BoardCanvas.Children.Add(tb);
                x -= 28;
            }
        }

        // 绘制列需求（上方）
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
                };
                Canvas.SetLeft(tb, x - 14);
                Canvas.SetTop(tb, y - 8);
                BoardCanvas.Children.Add(tb);
                y -= 16;
            }
        }

        // 第一遍：绘制所有格子的填充色
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var tile = snapshot[r, c];
                double left = RequirementMargin + c * TileSize;
                double top = RequirementMargin + r * TileSize;

                IBrush? fill = tile.Type switch
                {
                    TileType.Disabled => Brushes.DarkGray,
                    TileType.Lock when tile.ColorGroup != null => GetColorGroupBrush(puzzle, tile.ColorGroup),
                    _ => null
                };

                if (fill != null)
                {
                    var rect = new Rectangle
                    {
                        Width = TileSize,
                        Height = TileSize,
                        Fill = fill
                    };
                    Canvas.SetLeft(rect, left);
                    Canvas.SetTop(rect, top);
                    BoardCanvas.Children.Add(rect);
                }

                // 图标叠加
                if (tile.Type == TileType.Disabled)
                {
                    var symbol = new TextBlock
                    {
                        Text = "⊘",
                        FontSize = 18,
                        Foreground = Brushes.White,
                        Width = TileSize,
                        Height = TileSize,
                        TextAlignment = TextAlignment.Center,
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
                        Text = "🔒",
                        FontSize = 14,
                        Width = TileSize,
                        Height = TileSize,
                        TextAlignment = TextAlignment.Center,
                        VerticalAlignment = VerticalAlignment.Center
                    };
                    Canvas.SetLeft(lockTb, left);
                    Canvas.SetTop(lockTb, top);
                    BoardCanvas.Children.Add(lockTb);
                }
            }
        }

        // 第二遍：绘制边框线
        var borderBrushDark = Brushes.Black;
        var borderBrushLight = Brushes.DarkGray;

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

                // 上边框
                if (r == 0 || !IsSameComponent(tile, snapshot[r - 1, c]))
                    AddBorderLine(left, top, left + TileSize, top, strokeBrush);

                // 左边框
                if (c == 0 || !IsSameComponent(tile, snapshot[r, c - 1]))
                    AddBorderLine(left, top, left, top + TileSize, strokeBrush);

                // 下边框
                if (r == rows - 1 || !IsSameComponent(tile, snapshot[r + 1, c]))
                    AddBorderLine(left, top + TileSize, left + TileSize, top + TileSize, strokeBrush);

                // 右边框
                if (c == cols - 1 || !IsSameComponent(tile, snapshot[r, c + 1]))
                    AddBorderLine(left + TileSize, top, left + TileSize, top + TileSize, strokeBrush);
            }
        }

        BoardCanvas.Width = gridW;
        BoardCanvas.Height = gridH;
    }

    private static bool IsSameComponent(TileInfo a, TileInfo b)
        => a.PlacedComponentIndex >= 0 && a.PlacedComponentIndex == b.PlacedComponentIndex;

    private void AddBorderLine(double x1, double y1, double x2, double y2, IBrush stroke)
    {
        var line = new Line
        {
            StartPoint = new Point(x1, y1),
            EndPoint = new Point(x2, y2),
            Stroke = stroke,
            StrokeThickness = 1
        };
        BoardCanvas.Children.Add(line);
    }

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

    private static IBrush GetColorGroupBrush(PuzzleData puzzle, string colorGroup)
    {
        var cg = puzzle.ColorGroups.FirstOrDefault(x => x.Label == colorGroup);
        if (cg != null)
            return HsvToBrush(cg.Hue, cg.Saturation, cg.Value);
        return Brushes.Gray;
    }

    private static IBrush HsvToBrush(int h, int s, int v)
    {
        var (r, g, b) = HsvToRgb(h, s, v);
        return new SolidColorBrush(Color.FromRgb(r, g, b));
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
