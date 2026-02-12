namespace EndfieldPuzzleSolver.Recognition.Config;

/// <summary>
/// 图像识别配置
/// 从 prototype/image/config.toml 迁移
/// </summary>
public class DetectionConfig
{
    /// <summary>颜色分组参数</summary>
    public ColorGroupingConfig ColorGrouping { get; set; } = new();

    /// <summary>自动缩放探测参数</summary>
    public AutoScalingConfig AutoScaling { get; set; } = new();

    /// <summary>网格检测参数</summary>
    public GridDetectionConfig GridDetection { get; set; } = new();

    /// <summary>格子分类参数</summary>
    public TileClassificationConfig TileClassification { get; set; } = new();

    /// <summary>需求条检测参数</summary>
    public RequirementDetectionConfig RequirementDetection { get; set; } = new();

    /// <summary>元件检测参数</summary>
    public ComponentDetectionConfig ComponentDetection { get; set; } = new();
}

public class ColorGroupingConfig
{
    /// <summary>Hue 差值小于此阈值视为同种颜色（0-180 范围）</summary>
    public int HueMergeDistance { get; set; } = 18;
}

public class AutoScalingConfig
{
    /// <summary>缩放系数搜索范围 [min, max]</summary>
    public double[] SearchRange { get; set; } = [0.5, 2.0];

    /// <summary>搜索步长，越小越精确但计算量越大</summary>
    public double StepSize { get; set; } = 0.05;

    /// <summary>ROI 水平范围比例 [start, end]</summary>
    public double[] RoiHorizontalRatio { get; set; } = [0.1667, 0.6667];

    /// <summary>ROI 垂直范围比例 [start, end]</summary>
    public double[] RoiVerticalRatio { get; set; } = [0.1667, 0.8333];
}

public class GridDetectionConfig
{
    /// <summary>模板匹配阈值（0.0-1.0）</summary>
    public double TemplateMatchThreshold { get; set; } = 0.80;

    /// <summary>使用的模板列表</summary>
    public string[] UsedTemplates { get; set; } = ["tile-empty", "tile-disable"];

    /// <summary>NMS IOU 阈值（0.0-1.0）</summary>
    public double NmsIouThreshold { get; set; } = 0.5;

    /// <summary>格子最小尺寸</summary>
    public int MinCellSize { get; set; } = 65;

    /// <summary>格子最大尺寸</summary>
    public int MaxCellSize { get; set; } = 100;

    /// <summary>格子间距最小值</summary>
    public int MinGapSize { get; set; } = 30;

    /// <summary>默认格子尺寸</summary>
    public int DefaultCellSize { get; set; } = 87;
}

public class TileClassificationConfig
{
    /// <summary>内部区域缩进比例</summary>
    public double InnerMarginRatio { get; set; } = 0.1;

    /// <summary>锁定格最小饱和度</summary>
    public int LockMinSaturation { get; set; } = 80;

    /// <summary>锁定格最小亮度</summary>
    public int LockMinValue { get; set; } = 80;

    /// <summary>锁定格最小像素占比</summary>
    public double LockMinPixelRatio { get; set; } = 0.10;

    /// <summary>锁定格最小平均亮度</summary>
    public int LockMinAvgValue { get; set; } = 100;

    /// <summary>锁定格最小平均饱和度</summary>
    public int LockMinAvgSaturation { get; set; } = 100;

    /// <summary>模板匹配最小得分</summary>
    public double TileMinScore { get; set; } = 0.35;

    /// <summary>空格平均亮度阈值</summary>
    public int EmptyMeanValueThreshold { get; set; } = 45;

    /// <summary>空格灰度方差阈值</summary>
    public int EmptyVarianceThreshold { get; set; } = 200;

    /// <summary>禁用格平均亮度阈值</summary>
    public int DisabledMeanValueThreshold { get; set; } = 60;

    /// <summary>禁用格灰度方差阈值</summary>
    public int DisabledVarianceThreshold { get; set; } = 100;
}

public class RequirementDetectionConfig
{
    /// <summary>搜索边距（像素）</summary>
    public int SearchMargin { get; set; } = 150;

    /// <summary>Hue 相似度宽度（0-180）</summary>
    public int HueSimilarityWidth { get; set; } = 10;

    /// <summary>最小饱和度</summary>
    public int MinSaturation { get; set; } = 50;

    /// <summary>最小亮度</summary>
    public int MinValue { get; set; } = 55;

    /// <summary>最小面积</summary>
    public int MinArea { get; set; } = 15;

    /// <summary>最大面积</summary>
    public int MaxArea { get; set; } = 800;

    /// <summary>最大长宽比</summary>
    public double MaxAspectRatio { get; set; } = 6.0;

    /// <summary>变异系数阈值</summary>
    public double FilledCvThreshold { get; set; } = 0.35;

    /// <summary>最小区域尺寸</summary>
    public int MinRegionSize { get; set; } = 5;
}

public class ComponentDetectionConfig
{
    /// <summary>右侧区域起始位置比例（0.0-1.0）</summary>
    public double SearchStartXRatio { get; set; } = 0.6;

    /// <summary>Y轴上限偏移（像素）</summary>
    public int YLimitOffset { get; set; } = 150;

    /// <summary>多尺度列表</summary>
    public double[] TemplateScales { get; set; } = [0.5, 0.6, 0.7, 0.8, 1.0];

    /// <summary>最小匹配得分</summary>
    public double TemplateMatchMinScore { get; set; } = 0.55;

    /// <summary>最大匹配得分</summary>
    public double TemplateMatchMaxScore { get; set; } = 0.75;

    /// <summary>NMS IOU 阈值</summary>
    public double TemplateNmsIouThreshold { get; set; } = 0.5;

    /// <summary>模板最小尺寸</summary>
    public int MinTemplateSize { get; set; } = 45;

    /// <summary>模板最大尺寸</summary>
    public int MaxTemplateSize { get; set; } = 130;

    /// <summary>Blob 最小饱和度</summary>
    public int BlobMinSaturation { get; set; } = 60;

    /// <summary>Blob 最小亮度</summary>
    public int BlobMinValue { get; set; } = 60;

    /// <summary>Blob 最小面积</summary>
    public int BlobMinArea { get; set; } = 500;

    /// <summary>Blob 最小尺寸</summary>
    public int BlobMinSize { get; set; } = 30;

    /// <summary>Blob 最大尺寸</summary>
    public int BlobMaxSize { get; set; } = 200;

    /// <summary>Blob 最大长宽比</summary>
    public double BlobMaxAspectRatio { get; set; } = 4.0;

    /// <summary>形状最小尺寸</summary>
    public int MinShapeSize { get; set; } = 10;

    /// <summary>形状最大尺寸</summary>
    public int MaxShapeSize { get; set; } = 45;

    /// <summary>最小总格子数</summary>
    public int MinTotalCells { get; set; } = 2;

    /// <summary>最大总格子数</summary>
    public int MaxTotalCells { get; set; } = 12;

    /// <summary>最小行数</summary>
    public int MinRows { get; set; } = 1;

    /// <summary>最大行数</summary>
    public int MaxRows { get; set; } = 4;

    /// <summary>最小列数</summary>
    public int MinCols { get; set; } = 1;

    /// <summary>最大列数</summary>
    public int MaxCols { get; set; } = 4;

    /// <summary>格子填充阈值</summary>
    public double CellFillThreshold { get; set; } = 0.40;

    /// <summary>清晰填充阈值</summary>
    public double ClearFilledThreshold { get; set; } = 0.60;

    /// <summary>清晰空阈值</summary>
    public double ClearEmptyThreshold { get; set; } = 0.15;

    /// <summary>最小颜色像素数</summary>
    public int MinColorPixels { get; set; } = 30;

    /// <summary>最小平均亮度</summary>
    public int MinAvgValue { get; set; } = 80;
}
