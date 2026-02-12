using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Data;

namespace EndfieldPuzzleSolver.Converters;

/// <summary>
/// 将 bool 转换为 Visibility：true -> Visible，false -> Collapsed。
/// </summary>
public sealed partial class BoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, string language) =>
        value is bool b && b ? Visibility.Visible : Visibility.Collapsed;

    public object ConvertBack(object value, Type targetType, object parameter, string language) =>
        value is Visibility v && v == Visibility.Visible;
}

/// <summary>
/// 将 bool 转换为 Visibility：true -> Collapsed，false -> Visible。
/// 用于在无数据时显示占位文本。
/// </summary>
public sealed partial class InverseBoolToVisibilityConverter : IValueConverter
{
    public object Convert(object value, Type targetType, object parameter, string language) =>
        value is bool b && b ? Visibility.Collapsed : Visibility.Visible;

    public object ConvertBack(object value, Type targetType, object parameter, string language) =>
        value is Visibility v && v == Visibility.Collapsed;
}
