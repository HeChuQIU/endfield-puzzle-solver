using System.Text.Json;
using System.Text.Json.Serialization;

namespace EndfieldPuzzleSolver.Api;

/// <summary>
/// System.Text.Json 转换器：将 T[,] 序列化为 JSON 二维数组 [[...], [...], ...]
/// </summary>
public class Array2DConverterFactory : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        return typeToConvert.IsArray && typeToConvert.GetArrayRank() == 2;
    }

    public override JsonConverter CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        var elementType = typeToConvert.GetElementType()!;
        var converterType = typeof(Array2DConverter<>).MakeGenericType(elementType);
        return (JsonConverter)Activator.CreateInstance(converterType)!;
    }
}

public class Array2DConverter<T> : JsonConverter<T[,]>
{
    public override T[,]? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartArray)
            throw new JsonException("Expected start of array for 2D array");

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
        int colCount = rows.Max(r => r.Count);
        var result = new T[rowCount, colCount];
        for (int r = 0; r < rowCount; r++)
            for (int c = 0; c < rows[r].Count; c++)
                result[r, c] = rows[r][c];

        return result;
    }

    public override void Write(Utf8JsonWriter writer, T[,] value, JsonSerializerOptions options)
    {
        int rows = value.GetLength(0);
        int cols = value.GetLength(1);

        writer.WriteStartArray();
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
