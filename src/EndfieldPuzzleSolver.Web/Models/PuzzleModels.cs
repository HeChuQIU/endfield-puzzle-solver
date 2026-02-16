using System.Text.Json;
using System.Text.Json.Serialization;

namespace EndfieldPuzzleSolver.Web.Models;

// ──────────── 前端 DTO（不依赖后端 Recognition 库）────────────

public enum TileType
{
    Empty = 0,
    Disabled = 1,
    Lock = 2
}

public class TileInfoDto
{
    [JsonPropertyName("type")]
    public TileType Type { get; set; }

    [JsonPropertyName("colorGroup")]
    public string? ColorGroup { get; set; }

    [JsonPropertyName("placedComponentIndex")]
    public int PlacedComponentIndex { get; set; } = -1;
}

public class ColorRequirementDto
{
    [JsonPropertyName("colorGroup")]
    public string ColorGroup { get; set; } = "";

    [JsonPropertyName("count")]
    public int Count { get; set; }

    [JsonPropertyName("filledCount")]
    public int FilledCount { get; set; }
}

public class ColorGroupInfoDto
{
    [JsonPropertyName("label")]
    public string Label { get; set; } = "";

    [JsonPropertyName("hue")]
    public int Hue { get; set; }

    [JsonPropertyName("saturation")]
    public int Saturation { get; set; }

    [JsonPropertyName("value")]
    public int Value { get; set; }
}

public class ComponentInfoDto
{
    [JsonPropertyName("shape")]
    public bool[,] Shape { get; set; } = new bool[0, 0];

    [JsonPropertyName("colorGroup")]
    public string ColorGroup { get; set; } = "";

    public int Rows => Shape.GetLength(0);
    public int Cols => Shape.GetLength(1);

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
}

public class PuzzleDataDto
{
    [JsonPropertyName("rows")]
    public int Rows { get; set; }

    [JsonPropertyName("cols")]
    public int Cols { get; set; }

    [JsonPropertyName("tiles")]
    public TileInfoDto[,] Tiles { get; set; } = new TileInfoDto[0, 0];

    [JsonPropertyName("columnRequirements")]
    public ColorRequirementDto[][] ColumnRequirements { get; set; } = [];

    [JsonPropertyName("rowRequirements")]
    public ColorRequirementDto[][] RowRequirements { get; set; } = [];

    [JsonPropertyName("components")]
    public ComponentInfoDto[] Components { get; set; } = [];

    [JsonPropertyName("colorGroups")]
    public ColorGroupInfoDto[] ColorGroups { get; set; } = [];
}

public class SolveStepDto
{
    [JsonPropertyName("componentIndex")]
    public int ComponentIndex { get; set; }

    [JsonPropertyName("row")]
    public int Row { get; set; }

    [JsonPropertyName("col")]
    public int Col { get; set; }

    [JsonPropertyName("rotation")]
    public int Rotation { get; set; }

    [JsonPropertyName("boardSnapshot")]
    public TileInfoDto[,] BoardSnapshot { get; set; } = new TileInfoDto[0, 0];
}

public class SolveResultDto
{
    [JsonPropertyName("isSolved")]
    public bool IsSolved { get; set; }

    [JsonPropertyName("steps")]
    public List<SolveStepDto> Steps { get; set; } = [];

    [JsonPropertyName("message")]
    public string? Message { get; set; }
}

public class DetectAndSolveResponseDto
{
    [JsonPropertyName("puzzle")]
    public PuzzleDataDto? Puzzle { get; set; }

    [JsonPropertyName("solution")]
    public SolveResultDto? Solution { get; set; }
}

// ──────────── 2D 数组 JSON 转换器 ────────────

/// <summary>
/// 将 T[,] 序列化/反序列化为 JSON 的二维数组 [[...], [...], ...]
/// </summary>
public class Array2DJsonConverter<T> : JsonConverter<T[,]>
{
    public override T[,] Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartArray)
            throw new JsonException("Expected start of array");

        var rows = new List<List<T>>();
        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndArray)
                break;

            if (reader.TokenType == JsonTokenType.StartArray)
            {
                var row = new List<T>();
                while (reader.Read())
                {
                    if (reader.TokenType == JsonTokenType.EndArray)
                        break;
                    var item = JsonSerializer.Deserialize<T>(ref reader, options);
                    row.Add(item!);
                }
                rows.Add(row);
            }
        }

        if (rows.Count == 0)
            return new T[0, 0];

        int rowCount = rows.Count;
        int colCount = rows[0].Count;
        var result = new T[rowCount, colCount];
        for (int r = 0; r < rowCount; r++)
            for (int c = 0; c < Math.Min(colCount, rows[r].Count); c++)
                result[r, c] = rows[r][c];

        return result;
    }

    public override void Write(Utf8JsonWriter writer, T[,] value, JsonSerializerOptions options)
    {
        writer.WriteStartArray();
        int rows = value.GetLength(0);
        int cols = value.GetLength(1);
        for (int r = 0; r < rows; r++)
        {
            writer.WriteStartArray();
            for (int c = 0; c < cols; c++)
            {
                JsonSerializer.Serialize(writer, value[r, c], options);
            }
            writer.WriteEndArray();
        }
        writer.WriteEndArray();
    }
}
