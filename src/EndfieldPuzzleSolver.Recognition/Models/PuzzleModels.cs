namespace EndfieldPuzzleSolver.Recognition.Models;

/// <summary>格子类型</summary>
public enum TileType
{
    /// <summary>空格 - 可放置元件</summary>
    Empty,
    /// <summary>禁用格 - 不可放置</summary>
    Disabled,
    /// <summary>锁定格 - 已预置颜色，不可移动但参与计数</summary>
    Lock
}

/// <summary>棋盘上的一个格子</summary>
public sealed class TileInfo
{
    public TileType Type { get; init; }

    /// <summary>颜色组标签（仅 Lock 类型有值，如 "A", "B"）</summary>
    public string? ColorGroup { get; init; }

    public static TileInfo Empty => new() { Type = TileType.Empty };
    public static TileInfo Disabled => new() { Type = TileType.Disabled };
    public static TileInfo Locked(string colorGroup) => new() { Type = TileType.Lock, ColorGroup = colorGroup };

    /// <summary>创建带有已放置元件信息的格子（用于求解后的棋盘快照）</summary>
    public static TileInfo Placed(string colorGroup, int componentIndex) =>
        new() { Type = TileType.Lock, ColorGroup = colorGroup, PlacedComponentIndex = componentIndex };

    /// <summary>如果此格子是求解过程中放置的元件，记录元件编号（-1 表示非放置）</summary>
    public int PlacedComponentIndex { get; init; } = -1;
}

/// <summary>单行或单列中某颜色组的需求数量</summary>
/// <param name="ColorGroup">颜色组标签</param>
/// <param name="Count">总需求数（包含已由预置锁定格满足的数量）</param>
/// <param name="FilledCount">已由预置锁定格满足的数量</param>
public sealed record ColorRequirement(string ColorGroup, int Count, int FilledCount = 0);

/// <summary>颜色组信息（用于 UI 渲染颜色）</summary>
public sealed record ColorGroupInfo(string Label, int Hue, int Saturation, int Value);

/// <summary>一个可放置的元件</summary>
public sealed class ComponentInfo
{
    /// <summary>形状矩阵：true = 占据，false = 空</summary>
    public required bool[,] Shape { get; init; }

    /// <summary>所属颜色组</summary>
    public required string ColorGroup { get; init; }

    /// <summary>获取形状的行数</summary>
    public int Rows => Shape.GetLength(0);

    /// <summary>获取形状的列数</summary>
    public int Cols => Shape.GetLength(1);

    /// <summary>获取占据的格子数</summary>
    public int TileCount
    {
        get
        {
            int count = 0;
            for (int r = 0; r < Rows; r++)
                for (int c = 0; c < Cols; c++)
                    if (Shape[r, c]) count++;
            return count;
        }
    }

    /// <summary>
    /// 获取旋转后的形状。
    /// rotation: 0=0°, 1=90°顺时针, 2=180°, 3=270°顺时针
    /// </summary>
    public bool[,] GetRotatedShape(int rotation)
    {
        var shape = Shape;
        for (int i = 0; i < (rotation % 4); i++)
            shape = RotateCW(shape);
        return shape;
    }

    private static bool[,] RotateCW(bool[,] m)
    {
        int rows = m.GetLength(0), cols = m.GetLength(1);
        var result = new bool[cols, rows];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[c, rows - 1 - r] = m[r, c];
        return result;
    }
}

/// <summary>完整的谜题数据</summary>
public sealed class PuzzleData
{
    /// <summary>网格行数</summary>
    public required int Rows { get; init; }

    /// <summary>网格列数</summary>
    public required int Cols { get; init; }

    /// <summary>棋盘格子 [row, col]</summary>
    public required TileInfo[,] Tiles { get; init; }

    /// <summary>每列的颜色需求，ColumnRequirements[col] = 该列各颜色需要的格子数</summary>
    public required ColorRequirement[][] ColumnRequirements { get; init; }

    /// <summary>每行的颜色需求，RowRequirements[row] = 该行各颜色需要的格子数</summary>
    public required ColorRequirement[][] RowRequirements { get; init; }

    /// <summary>可放置的元件列表</summary>
    public required ComponentInfo[] Components { get; init; }

    /// <summary>颜色组信息（用于 UI 渲染）</summary>
    public ColorGroupInfo[] ColorGroups { get; init; } = [];
}

// ──────────────── 求解结果类型 ────────────────

/// <summary>一步放置操作</summary>
public sealed class SolveStep
{
    /// <summary>放置的元件编号（对应 PuzzleData.Components 的索引）</summary>
    public required int ComponentIndex { get; init; }

    /// <summary>放置位置 - 行</summary>
    public required int Row { get; init; }

    /// <summary>放置位置 - 列</summary>
    public required int Col { get; init; }

    /// <summary>旋转角度：0=0°, 1=90°, 2=180°, 3=270°</summary>
    public required int Rotation { get; init; }

    /// <summary>此步骤执行后的棋盘快照</summary>
    public required TileInfo[,] BoardSnapshot { get; init; }
}

/// <summary>求解结果</summary>
public sealed class SolveResult
{
    /// <summary>是否求解成功</summary>
    public bool IsSolved { get; init; }

    /// <summary>求解步骤列表（按放置顺序）</summary>
    public IReadOnlyList<SolveStep> Steps { get; init; } = [];

    /// <summary>错误/提示信息</summary>
    public string? Message { get; init; }

    public static SolveResult Solved(IReadOnlyList<SolveStep> steps)
        => new() { IsSolved = true, Steps = steps, Message = "求解成功" };

    public static SolveResult NoSolution(string? reason = null)
        => new() { IsSolved = false, Message = reason ?? "此谜题无解" };
}
