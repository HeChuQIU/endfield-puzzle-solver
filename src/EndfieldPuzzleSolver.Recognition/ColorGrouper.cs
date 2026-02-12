namespace EndfieldPuzzleSolver.Recognition;

/// <summary>
/// 收集所有检测到的 HSV 颜色，按 Hue 聚类，分配标签 A/B/C...
/// 不写死颜色名称，只区分「同一题目中不同的颜色」。
/// </summary>
public sealed class ColorGrouper
{
    /// <summary>Hue 差值小于此阈值视为同组</summary>
    public int HueMergeDist { get; } = 18;

    private readonly List<Cluster> _clusters = [];

    private sealed class Cluster
    {
        public int CenterHue;
        public int AvgS;
        public int AvgV;
        public readonly List<(int H, int S, int V)> Samples = [];
    }

    public ColorGrouper() { }
    public ColorGrouper(int hueMergeDist) => HueMergeDist = hueMergeDist;

    /// <summary>注册一个颜色，返回其所属组的索引</summary>
    public int Register(int h, int s, int v)
    {
        for (int i = 0; i < _clusters.Count; i++)
        {
            var cluster = _clusters[i];
            if (HueDistance(h, cluster.CenterHue) < HueMergeDist)
            {
                cluster.Samples.Add((h, s, v));

                var allH = cluster.Samples.Select(x => x.H).ToList();
                var allS = cluster.Samples.Select(x => x.S).ToList();
                var allV = cluster.Samples.Select(x => x.V).ToList();

                cluster.CenterHue = (int)Math.Round(allH.Average());
                cluster.AvgS = (int)Math.Round(allS.Average());
                cluster.AvgV = (int)Math.Round(allV.Average());

                return i;
            }
        }

        _clusters.Add(new Cluster
        {
            CenterHue = h,
            AvgS = s,
            AvgV = v,
            Samples = { (h, s, v) }
        });
        return _clusters.Count - 1;
    }

    /// <summary>返回组标签: A, B, C, ...</summary>
    public static string Label(int idx)
    {
        if (idx >= 0 && idx < 26)
            return ((char)('A' + idx)).ToString();
        return idx.ToString();
    }

    /// <summary>返回组的平均 HSV</summary>
    public (int H, int S, int V) ColorHsv(int idx)
    {
        if (idx >= 0 && idx < _clusters.Count)
        {
            var c = _clusters[idx];
            return (c.CenterHue, c.AvgS, c.AvgV);
        }
        return (0, 0, 0);
    }

    public int ClusterCount => _clusters.Count;

    /// <summary>匹配到已有 cluster 中 Hue 距离最近者，返回其 label；无 cluster 时临时 register</summary>
    public string MatchNearest(int h, int s, int v)
    {
        if (_clusters.Count == 0)
            return Label(Register(h, s, v));

        int bestIdx = 0;
        int bestDist = HueDistance(h, _clusters[0].CenterHue);
        for (int i = 1; i < _clusters.Count; i++)
        {
            int d = HueDistance(h, _clusters[i].CenterHue);
            if (d < bestDist)
            {
                bestDist = d;
                bestIdx = i;
            }
        }
        return Label(bestIdx);
    }

    /// <summary>返回组的采样次数</summary>
    public int GetSampleCount(int idx)
    {
        if (idx >= 0 && idx < _clusters.Count)
            return _clusters[idx].Samples.Count;
        return 0;
    }

    /// <summary>计算两个 OpenCV Hue(0-180) 之间的环形距离</summary>
    public static int HueDistance(int h1, int h2)
    {
        int d = Math.Abs(h1 - h2);
        return Math.Min(d, 180 - d);
    }
}
