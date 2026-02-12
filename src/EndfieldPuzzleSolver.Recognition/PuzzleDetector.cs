using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using OpenCvSharp;
using EndfieldPuzzleSolver.Recognition.Models;

namespace EndfieldPuzzleSolver.Recognition;

/// <summary>
/// 终末地源石电路模块解谜 - UI 识别
/// 从游戏截图中识别网格、格子类型、行列需求和可用元件
/// </summary>
public sealed class PuzzleDetector
{
    private readonly string _templateDir;
    private readonly Dictionary<string, Mat> _rawTemplates = new();
    private readonly Dictionary<string, Mat> _templates = new();
    private readonly bool _debug;
    private readonly string? _debugDir;
    private readonly double? _userScale;

    private Mat _image = null!;
    private Mat _gray = null!;
    private Mat _hsv = null!;
    private ColorGrouper _colorGrouper = null!;
    private double _scale = 1.0;

    public PuzzleDetector(string templateDir, bool debug = false, string? debugDir = null, double? scale = null)
    {
        _templateDir = templateDir;
        _debug = debug;
        _debugDir = debugDir;
        _userScale = scale;
        _scale = scale ?? 1.0;
        LoadTemplates();
    }

    private void LoadTemplates()
    {
        foreach (var name in new[] { "tile-empty.png", "tile-disable.png", "component-flame.png", "puzzle-area-corner-lt.png" })
        {
            var path = Path.Combine(_templateDir, name);
            if (!File.Exists(path)) continue;

            using var img = Cv2.ImRead(path);
            if (img.Empty()) continue;

            using var gray = new Mat();
            Cv2.CvtColor(img, gray, ColorConversionCodes.BGR2GRAY);
            var key = name.Replace(".png", "").Replace("-", "_");
            _rawTemplates[key] = gray.Clone();
        }
    }

    private void DebugSave(string name, Mat img)
    {
        if (!_debug || string.IsNullOrEmpty(_debugDir)) return;
        Directory.CreateDirectory(_debugDir);
        var path = Path.Combine(_debugDir, name);
        Cv2.ImWrite(path, img);
    }

    private static Mat ResizeTemplate(Mat tpl, double scale)
    {
        var h = tpl.Rows;
        var w = tpl.Cols;
        var nh = Math.Max(1, (int)Math.Round(h * scale));
        var nw = Math.Max(1, (int)Math.Round(w * scale));
        var resized = new Mat();
        Cv2.Resize(tpl, resized, new Size(nw, nh), 0, 0, InterpolationFlags.Linear);
        return resized;
    }

    private void BuildScaledTemplates(double scale)
    {
        foreach (var kv in _rawTemplates)
        {
            if (_templates.TryGetValue(kv.Key, out var old))
                old.Dispose();
            _templates[kv.Key] = ResizeTemplate(kv.Value, scale);
        }
    }

    private int Scaled(double value) => Math.Max(1, (int)Math.Round(value * _scale));
    private int ScaledArea(double value) => Math.Max(1, (int)Math.Round(value * _scale * _scale));

    private double? EstimateScale()
    {
        var h = _gray.Rows;
        var w = _gray.Cols;
        var roi = _gray[new Rect(w / 6, h / 6, 2 * w / 3 - w / 6, 5 * h / 6 - h / 6)];

        double bestScale = 1.0;
        double bestScore = -1.0;

        for (var s = 0.50; s <= 2.001; s += 0.05)
        {
            var scores = new List<double>();
            foreach (var key in new[] { "tile_empty", "tile_disable" })
            {
                if (!_rawTemplates.TryGetValue(key, out var tpl)) continue;

                var stpl = ResizeTemplate(tpl, s);
                try
                {
                    var th = stpl.Rows;
                    var tw = stpl.Cols;
                    var rh = roi.Rows;
                    var rw = roi.Cols;
                    if (th >= rh || tw >= rw) continue;

                    var res = new Mat();
                    Cv2.MatchTemplate(roi, stpl, res, TemplateMatchModes.CCoeffNormed);
                    if (res.Empty()) continue;

                    Cv2.MinMaxLoc(res, out _, out double maxVal, out _, out _);
                    scores.Add(maxVal);
                }
                finally
                {
                    stpl.Dispose();
                }
            }
            if (scores.Count == 0) continue;

            var score = scores.Average();
            if (score > bestScore)
            {
                bestScore = score;
                bestScale = s;
            }
        }

        if (bestScore < 0)
            return null;

        return bestScale;
    }

    /// <summary>检测结果</summary>
    public sealed record DetectResult(PuzzleData? Data, string? Error)
    {
        public bool IsSuccess => Data != null;
        public static DetectResult Success(PuzzleData data) => new(data, null);
        public static DetectResult Fail(string error) => new(null, error);
    }

    /// <summary>主检测入口（不抛异常，通过 DetectResult.Error 返回错误信息）</summary>
    public DetectResult Detect(string imagePath)
    {
        _image = Cv2.ImRead(imagePath);
        if (_image.Empty())
            return DetectResult.Fail($"无法读取图片: {imagePath}");

        _gray = new Mat();
        _hsv = new Mat();
        Cv2.CvtColor(_image, _gray, ColorConversionCodes.BGR2GRAY);
        Cv2.CvtColor(_image, _hsv, ColorConversionCodes.BGR2HSV);
        _colorGrouper = new ColorGrouper();

        var estimatedScale = _userScale ?? EstimateScale();
        if (estimatedScale == null)
            return DetectResult.Fail("自动缩放探测失败：未找到模板（请确认模板图片存在）");
        _scale = estimatedScale.Value;
        BuildScaledTemplates(_scale);

        var grid = DetectGrid();
        if (grid == null)
            return DetectResult.Fail("格子匹配数不足：无法识别棋盘网格（可能不是有效的谜题截图）");

        var g = grid.Value;
        var tiles = ClassifyTiles(g);
        var reqs = DetectRequirements(g);
        var comps = DetectComponents();

        var colorGroups = new List<ColorGroupInfo>();
        for (int i = 0; i < _colorGrouper.ClusterCount; i++)
        {
            var (h, s, v) = _colorGrouper.ColorHsv(i);
            colorGroups.Add(new ColorGroupInfo(ColorGrouper.Label(i), h, s, v));
        }

        return DetectResult.Success(new PuzzleData
        {
            Rows = g.Rows,
            Cols = g.Cols,
            Tiles = tiles,
            ColumnRequirements = reqs.Columns,
            RowRequirements = reqs.Rows,
            Components = comps,
            ColorGroups = colorGroups.ToArray()
        });
    }

    private (int Rows, int Cols, int OriginX, int OriginY, int CellW, int CellH)? DetectGrid()
    {
        var h = _gray.Rows;
        var w = _gray.Cols;
        var roi = _gray[new Rect(w / 6, h / 6, 2 * w / 3 - w / 6, 5 * h / 6 - h / 6)];
        var oxOff = w / 6;
        var oyOff = h / 6;

        var allPts = new List<(int X, int Y, int W, int H, double Score)>();
        foreach (var tplKey in new[] { "tile_empty", "tile_disable" })
        {
            if (!_templates.TryGetValue(tplKey, out var tpl)) continue;

            var pts = MatchTemplate(roi, tpl, 0.80);
            foreach (var (x, y, tw, th, s) in pts)
                allPts.Add((x + oxOff, y + oyOff, tw, th, s));
        }

        if (allPts.Count < 4)
            return null;

        allPts = Nms(allPts, 0.5);
        var minWh = Scaled(65);
        var maxWh = Scaled(100);
        allPts = allPts.Where(p => p.W > minWh && p.W < maxWh && p.H > minWh && p.H < maxWh).ToList();

        if (allPts.Count < 4)
            return null;

        var grid = InferGrid(allPts);
        return grid;
    }

    private List<(int X, int Y, int W, int H, double Score)> MatchTemplate(Mat imgGray, Mat tpl, double thresh)
    {
        var result = new List<(int, int, int, int, double)>();
        var th = tpl.Rows;
        var tw = tpl.Cols;

        using var res = new Mat();
        Cv2.MatchTemplate(imgGray, tpl, res, TemplateMatchModes.CCoeffNormed);

        for (int y = 0; y < res.Rows; y++)
        {
            for (int x = 0; x < res.Cols; x++)
            {
                var val = res.Get<float>(y, x);
                if (val >= thresh)
                    result.Add((x, y, tw, th, val));
            }
        }
        return result;
    }

    private List<(int X, int Y, int W, int H, double Score)> Nms(List<(int X, int Y, int W, int H, double Score)> boxes, double iouThresh)
    {
        if (boxes.Count == 0) return [];
        var sorted = boxes.OrderByDescending(b => b.Score).ToList();
        var keep = new List<(int X, int Y, int W, int H, double Score)>();

        while (sorted.Count > 0)
        {
            var best = sorted[0];
            sorted.RemoveAt(0);
            keep.Add(best);
            sorted = sorted.Where(b => Iou(best, b) < iouThresh).ToList();
        }
        return keep;
    }

    private static double Iou((int X, int Y, int W, int H, double _) a, (int X, int Y, int W, int H, double _) b)
    {
        var x1 = Math.Max(a.X, b.X);
        var y1 = Math.Max(a.Y, b.Y);
        var x2 = Math.Min(a.X + a.W, b.X + b.W);
        var y2 = Math.Min(a.Y + a.H, b.Y + b.H);
        var inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
        var union = a.W * a.H + b.W * b.H - inter;
        return union > 0 ? inter / union : 0;
    }

    private (int Rows, int Cols, int OriginX, int OriginY, int CellW, int CellH) InferGrid(List<(int X, int Y, int W, int H, double Score)> matches)
    {
        var centers = matches.Select(m => (m.X + m.W / 2, m.Y + m.H / 2)).ToList();
        var xs = centers.Select(c => c.Item1).Distinct().OrderBy(x => x).ToList();
        var ys = centers.Select(c => c.Item2).Distinct().OrderBy(y => y).ToList();

        var minGap = Scaled(30);
        var dxs = xs.Zip(xs.Skip(1), (a, b) => b - a).Where(d => d > minGap).ToList();
        var dys = ys.Zip(ys.Skip(1), (a, b) => b - a).Where(d => d > minGap).ToList();

        var defaultCell = Scaled(87);
        var cellW = dxs.Count > 0 ? (int)Median(dxs.Select(x => (double)x)) : defaultCell;
        var cellH = dys.Count > 0 ? (int)Median(dys.Select(x => (double)x)) : defaultCell;

        var minCx = centers.Min(c => c.Item1);
        var maxCx = centers.Max(c => c.Item1);
        var minCy = centers.Min(c => c.Item2);
        var maxCy = centers.Max(c => c.Item2);

        var rows = (int)Math.Round((maxCy - minCy) / (double)cellH) + 1;
        var cols = (int)Math.Round((maxCx - minCx) / (double)cellW) + 1;
        var originX = minCx - cellW / 2;
        var originY = minCy - cellH / 2;

        return (rows, cols, originX, originY, cellW, cellH);
    }

    private static double Median(IEnumerable<double> values)
    {
        var list = values.OrderBy(x => x).ToList();
        var n = list.Count;
        if (n == 0) return 0;
        if (n % 2 == 1) return list[n / 2];
        return (list[n / 2 - 1] + list[n / 2]) / 2;
    }

    private TileInfo[,] ClassifyTiles((int Rows, int Cols, int OriginX, int OriginY, int CellW, int CellH) grid)
    {
        var (rows, cols, ox, oy, cw, ch) = grid;
        var tiles = new TileInfo[rows, cols];

        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                var x = ox + c * cw;
                var y = oy + r * ch;
                var cellGray = _gray[new Rect(x, y, cw, ch)];
                var cellHsv = _hsv[new Rect(x, y, cw, ch)];
                tiles[r, c] = ClassifyCell(cellGray, cellHsv);
            }
        }
        return tiles;
    }

    private TileInfo ClassifyCell(Mat cellGray, Mat cellHsv)
    {
        var ch = cellGray.Rows;
        var cw = cellGray.Cols;
        var m = Math.Max(5, Math.Min(ch, cw) / 10);
        if (ch <= 2 * m || cw <= 2 * m)
            return new TileInfo { Type = TileType.Empty };

        var innerHsv = cellHsv[new Rect(m, m, cw - 2 * m, ch - 2 * m)];

        // Lock: 高饱和度 AND 高亮度 (HSV: Item0=H, Item1=S, Item2=V)
        int totalCount = 0;
        double sumH = 0, sumS = 0, sumV = 0;
        int lockCount = 0;
        for (int y = 0; y < innerHsv.Rows; y++)
        {
            for (int x = 0; x < innerHsv.Cols; x++)
            {
                var v = innerHsv.At<Vec3b>(y, x);
                var h = v.Item0;
                var sat = v.Item1;
                var val = v.Item2;
                totalCount++;
                if (sat > 80 && val > 80)
                {
                    lockCount++;
                    sumH += h;
                    sumS += sat;
                    sumV += val;
                }
            }
        }

        var lockRatio = totalCount > 0 ? (double)lockCount / totalCount : 0;
        if (lockRatio > 0.10)
        {
            var avgH = (int)(sumH / lockCount);
            var avgS = (int)(sumS / lockCount);
            var avgV = (int)(sumV / lockCount);
            if (avgV > 100 && avgS > 100)
            {
                var groupIdx = _colorGrouper.Register(avgH, avgS, avgV);
                return TileInfo.Locked(ColorGrouper.Label(groupIdx));
            }
        }

        // Disabled vs Empty: template competition
        var scDis = CellScore(cellGray, "tile_disable");
        var scEmp = CellScore(cellGray, "tile_empty");
        var bestScore = Math.Max(scDis, scEmp);
        if (bestScore > 0.35)
        {
            if (scDis >= scEmp)
                return TileInfo.Disabled;
            return TileInfo.Empty;
        }

        // Fallback: brightness and variance
        double meanV = 0, variance = 0;
        int vCount = 0;
        var innerGray = cellGray[new Rect(m, m, cw - 2 * m, ch - 2 * m)];
        for (int y = 0; y < innerGray.Rows; y++)
        {
            for (int x = 0; x < innerGray.Cols; x++)
            {
                var v = innerGray.At<byte>(y, x);
                meanV += v;
                variance += v * v;
                vCount++;
            }
        }
        if (vCount > 0)
        {
            meanV /= vCount;
            variance = (variance / vCount) - (meanV * meanV);
        }

        if (meanV < 45 && variance < 200)
            return TileInfo.Empty;
        if (meanV < 60 && variance > 100)
            return TileInfo.Disabled;

        return TileInfo.Empty;
    }

    private double CellScore(Mat cellGray, string tplKey)
    {
        if (!_templates.TryGetValue(tplKey, out var tpl)) return 0;
        var th = tpl.Rows;
        var tw = tpl.Cols;
        var ch = cellGray.Rows;
        var cw = cellGray.Cols;
        if (ch < th || cw < tw) return 0;

        using var res = new Mat();
        Cv2.MatchTemplate(cellGray, tpl, res, TemplateMatchModes.CCoeffNormed);
        if (res.Empty()) return 0;

        Cv2.MinMaxLoc(res, out _, out double maxVal, out _, out _);
        return maxVal;
    }

    private (ColorRequirement[][] Columns, ColorRequirement[][] Rows) DetectRequirements(
        (int Rows, int Cols, int OriginX, int OriginY, int CellW, int CellH) grid)
    {
        var (rows, cols, ox, oy, cw, ch) = grid;
        var gw = cw * cols;
        var gh = ch * rows;
        var margin = Scaled(150);

        var colRoi = _image[new Rect(ox, Math.Max(0, oy - margin), gw, Math.Max(1, oy - 2 - Math.Max(0, oy - margin)))];
        var colReqs = FindBars(colRoi, "col", ox, cw, cols, (ox, Math.Max(0, oy - margin)));

        var rowRoi = _image[new Rect(Math.Max(0, ox - margin), oy, Math.Max(1, ox - 2 - Math.Max(0, ox - margin)), gh)];
        var rowReqs = FindBars(rowRoi, "row", oy, ch, rows, (Math.Max(0, ox - margin), oy));

        return (colReqs, rowReqs);
    }

    private ColorRequirement[][] FindBars(Mat roiBgr, string orient, int gridStart, int cellSz, int nCells, (int X, int Y) regionOff)
    {
        var reqs = Enumerable.Range(0, nCells).Select(_ => new List<(string ColorGroup, bool Filled)>()).ToArray();
        if (roiBgr.Empty() || roiBgr.Total() == 0) return reqs.Select(r => r.GroupBy(x => x.ColorGroup).Select(g => new ColorRequirement(g.Key, g.Count(), g.Count(x => x.Filled))).ToArray()).ToArray();

        using var hsv = new Mat();
        Cv2.CvtColor(roiBgr, hsv, ColorConversionCodes.BGR2HSV);

        using var sat = new Mat();
        using var val = new Mat();
        Cv2.ExtractChannel(hsv, sat, 1);
        Cv2.ExtractChannel(hsv, val, 2);

        using var satBinary = new Mat();
        Cv2.Threshold(sat, satBinary, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);
        var mask = satBinary;

        var fgCount = Cv2.CountNonZero(mask);
        var fgRatio = (double)fgCount / mask.Total();
        if (fgRatio > 0.25)
        {
            var flat = new List<byte>();
            for (int y = 0; y < sat.Rows; y++)
                for (int x = 0; x < sat.Cols; x++)
                    flat.Add(sat.At<byte>(y, x));
            var percent96 = Percentile(flat, 96);
            Cv2.Threshold(sat, mask, percent96, 255, ThresholdTypes.Binary);
        }

        Cv2.FindContours(mask, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        foreach (var cnt in contours)
        {
            var rect = Cv2.BoundingRect(cnt);
            var x = rect.X;
            var y = rect.Y;
            var w = rect.Width;
            var h = rect.Height;
            var area = w * h;

            if (area < ScaledArea(15) || area > ScaledArea(800)) continue;
            var aspect = (double)Math.Max(w, h) / Math.Max(Math.Min(w, h), 1);
            if (aspect > 6) continue;

            var barMask = mask[rect];
            var barHsv = hsv[rect];
            using (var barVal = new Mat())
            {
                Cv2.ExtractChannel(barHsv, barVal, 2);

                var colorPixels = new List<(int H, int S, int V)>();
            for (int by = 0; by < barHsv.Rows; by++)
            {
                for (int bx = 0; bx < barHsv.Cols; bx++)
                {
                    if (barMask.At<byte>(by, bx) > 0)
                        colorPixels.Add((barHsv.At<Vec3b>(by, bx).Item0, barHsv.At<Vec3b>(by, bx).Item1, barHsv.At<Vec3b>(by, bx).Item2));
                }
            }
            if (colorPixels.Count < 5) continue;

            var avgH = (int)colorPixels.Average(p => p.H);
            var avgS = (int)colorPixels.Average(p => p.S);
            var avgV = (int)colorPixels.Average(p => p.V);

            double valCv = 0;
            bool filled = false;
            if (barVal.Rows >= 5 && barVal.Cols >= 5)
            {
                double sum = 0, sumSq = 0;
                int n = 0;
                for (int by = 0; by < barVal.Rows; by++)
                    for (int bx = 0; bx < barVal.Cols; bx++)
                    {
                        var v = (double)barVal.At<byte>(by, bx);
                        sum += v;
                        sumSq += v * v;
                        n++;
                    }
                var mean = sum / n;
                var std = Math.Sqrt(sumSq / n - mean * mean);
                valCv = mean > 1 ? std / mean : 0;
                filled = valCv < 0.35;
            }

            var groupIdx = _colorGrouper.Register(avgH, avgS, avgV);
            var colorGroup = ColorGrouper.Label(groupIdx);

            int idx;
            if (orient == "col")
            {
                var cx = regionOff.X + x + w / 2;
                idx = (int)Math.Round((cx - gridStart - cellSz / 2.0) / cellSz);
            }
            else
            {
                var cy = regionOff.Y + y + h / 2;
                idx = (int)Math.Round((cy - gridStart - cellSz / 2.0) / cellSz);
            }

            if (idx >= 0 && idx < nCells)
                reqs[idx].Add((colorGroup, filled));
            }
        }

        return reqs.Select(r => r.GroupBy(x => x.ColorGroup).Select(g => new ColorRequirement(g.Key, g.Count(), g.Count(x => x.Filled))).ToArray()).ToArray();
    }

    private static byte Percentile(List<byte> sorted, double p)
    {
        var list = sorted.OrderBy(x => x).ToList();
        var idx = (int)(p / 100 * (list.Count - 1));
        return list[Math.Clamp(idx, 0, list.Count - 1)];
    }

    private ComponentInfo[] DetectComponents()
    {
        var ih = _image.Rows;
        var iw = _image.Cols;
        var rx = 3 * iw / 5;
        var rHsv = _hsv[new Rect(rx, 0, iw - rx, ih)];

        using var sat = new Mat();
        using var val = new Mat();
        Cv2.ExtractChannel(rHsv, sat, 1);
        Cv2.ExtractChannel(rHsv, val, 2);
        using var rawMask = new Mat();
        Cv2.Compare(sat, new Scalar(60), rawMask, CmpType.GT);
        using var rawMask2 = new Mat();
        Cv2.Compare(val, new Scalar(60), rawMask2, CmpType.GT);
        using var combined = new Mat();
        Cv2.BitwiseAnd(rawMask, rawMask2, combined);

        Cv2.FindContours(combined, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        var blobs = new List<(int X, int Y, int W, int H, double Area)>();
        foreach (var cnt in contours)
        {
            var rect = Cv2.BoundingRect(cnt);
            var area = Cv2.ContourArea(cnt);
            var x = rect.X + rx;
            var y = rect.Y;
            var w = rect.Width;
            var h = rect.Height;

            if (area > ScaledArea(500) && w > Scaled(30) && w < Scaled(200) && h > Scaled(30) && h < Scaled(200))
            {
                var aspect = (double)Math.Max(w, h) / Math.Max(Math.Min(w, h), 1);
                if (aspect < 4)
                    blobs.Add((x, y, w, h, area));
            }
        }
        blobs = blobs.OrderBy(b => (b.Y, b.X)).ToList();

        var comps = new List<ComponentInfo>();
        foreach (var (x, y, w, h, _) in blobs)
        {
            var info = ParseComponentBlob(x, y, w, h);
            if (info != null)
                comps.Add(info);
        }
        return comps.ToArray();
    }

    private ComponentInfo? ParseComponentBlob(int x, int y, int w, int h)
    {
        var blobHsv = _hsv[new Rect(x, y, w, h)];
        using var sat = new Mat();
        using var val = new Mat();
        Cv2.ExtractChannel(blobHsv, sat, 1);
        Cv2.ExtractChannel(blobHsv, val, 2);
        using var m1 = new Mat();
        using var m2 = new Mat();
        Cv2.Compare(sat, new Scalar(60), m1, CmpType.GT);
        Cv2.Compare(val, new Scalar(60), m2, CmpType.GT);
        using var blobMask = new Mat();
        Cv2.BitwiseAnd(m1, m2, blobMask);

        var hProj = new double[h];
        var vProj = new double[w];
        for (int r = 0; r < h; r++)
            for (int c = 0; c < w; c++)
                if (blobMask.At<byte>(r, c) > 0)
                {
                    hProj[r] += 1;
                    vProj[c] += 1;
                }

        var hThresh = hProj.Max() > 0 ? hProj.Max() * 0.15 : 1;
        var vThresh = vProj.Max() > 0 ? vProj.Max() * 0.15 : 1;

        int rStart = 0, rEnd = h, cStart = 0, cEnd = w;
        for (int i = 0; i < h; i++) { if (hProj[i] > hThresh) { rStart = i; break; } }
        for (int i = h - 1; i >= 0; i--) { if (hProj[i] > hThresh) { rEnd = i + 1; break; } }
        for (int i = 0; i < w; i++) { if (vProj[i] > vThresh) { cStart = i; break; } }
        for (int i = w - 1; i >= 0; i--) { if (vProj[i] > vThresh) { cEnd = i + 1; break; } }

        var trimmed = blobMask[new Rect(cStart, rStart, cEnd - cStart, rEnd - rStart)];
        var th = trimmed.Rows;
        var tw = trimmed.Cols;
        if (th < Scaled(10) || tw < Scaled(10)) return null;

        int[][]? bestShape = null;
        double bestScore = -999;

        for (int nr = 1; nr <= 4; nr++)
        {
            for (int nc = 1; nc <= 4; nc++)
            {
                var total = nr * nc;
                if (total < 2 || total > 12) continue;

                var cellH = (double)th / nr;
                var cellW = (double)tw / nc;

                if (cellH < Scaled(12) || cellW < Scaled(12)) continue;
                if (cellH > Scaled(45) || cellW > Scaled(45)) continue;

                var shape = new int[nr][];
                var fills = new List<double>();
                for (int r = 0; r < nr; r++)
                {
                    shape[r] = new int[nc];
                    for (int c = 0; c < nc; c++)
                    {
                        var y1 = (int)(r * cellH);
                        var y2 = (int)((r + 1) * cellH);
                        var x1 = (int)(c * cellW);
                        var x2 = (int)((c + 1) * cellW);
                        var cell = trimmed[new Rect(x1, y1, x2 - x1, y2 - y1)];
                        var fill = cell.Total() > 0 ? (double)Cv2.CountNonZero(cell) / cell.Total() : 0;
                        fills.Add(fill);
                        shape[r][c] = fill > 0.40 ? 1 : 0;
                    }
                }

                var filled = shape.Sum(row => row.Sum());
                if (filled == 0 || filled == total) continue;

                var nClearFilled = fills.Count(f => f > 0.60);
                var nClearEmpty = fills.Count(f => f < 0.15);
                var nAmbiguous = total - nClearFilled - nClearEmpty;

                var clarity = (double)(nClearFilled + nClearEmpty) / total;
                var squareness = Math.Min(cellH, cellW) / Math.Max(cellH, cellW);
                var score = clarity * 10 + squareness * 5 - nAmbiguous * 8;

                if (score > bestScore)
                {
                    bestScore = score;
                    bestShape = shape;
                }
            }
        }

        if (bestShape == null)
            bestShape = [[1]];

        var maskCount = 0;
        for (int r = 0; r < blobMask.Rows; r++)
            for (int c = 0; c < blobMask.Cols; c++)
                if (blobMask.At<byte>(r, c) > 0) maskCount++;
        if (maskCount < ScaledArea(30)) return null;

        double sumH = 0, sumS = 0, sumV = 0;
        int pxCount = 0;
        for (int r = 0; r < blobHsv.Rows; r++)
        {
            for (int c = 0; c < blobHsv.Cols; c++)
            {
                if (blobMask.At<byte>(r, c) > 0)
                {
                    var v = blobHsv.At<Vec3b>(r, c);
                    sumH += v.Item0;
                    sumS += v.Item1;
                    sumV += v.Item2;
                    pxCount++;
                }
            }
        }
        var avgH = (int)(sumH / pxCount);
        var avgS = (int)(sumS / pxCount);
        var avgV = (int)(sumV / pxCount);
        if (avgV < 80) return null;

        var groupIdx = _colorGrouper.Register(avgH, avgS, avgV);
        var shapeBool = new bool[bestShape.Length, bestShape[0].Length];
        for (int r = 0; r < bestShape.Length; r++)
            for (int c = 0; c < bestShape[r].Length; c++)
                shapeBool[r, c] = bestShape[r][c] == 1;

        return new ComponentInfo
        {
            Shape = shapeBool,
            ColorGroup = ColorGrouper.Label(groupIdx)
        };
    }

    /// <summary>从 JSON 文件加载检测结果（用于测试，无需重新运行检测）</summary>
    public static PuzzleData LoadFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        var gridSize = root.GetProperty("grid_size");
        var rows = gridSize[0].GetInt32();
        var cols = gridSize[1].GetInt32();

        var tiles = new TileInfo[rows, cols];
        var tilesArr = root.GetProperty("tiles");
        for (int r = 0; r < rows; r++)
        {
            var row = tilesArr[r];
            for (int c = 0; c < cols; c++)
            {
                var cell = row[c];
                var typeStr = cell.GetProperty("type").GetString() ?? "empty";
                tiles[r, c] = typeStr switch
                {
                    "empty" => TileInfo.Empty,
                    "disabled" => TileInfo.Disabled,
                    "lock" => TileInfo.Locked(cell.GetProperty("color_group").GetString() ?? "A"),
                    _ => TileInfo.Empty
                };
            }
        }

        var reqs = root.GetProperty("requirements");
        var colBars = reqs.GetProperty("columns");
        var rowBars = reqs.GetProperty("rows");

        var columnReqs = new ColorRequirement[cols][];
        for (int c = 0; c < cols; c++)
        {
            var bars = colBars[c];
            var groups = bars.EnumerateArray()
                .GroupBy(b => b.GetProperty("color_group").GetString() ?? "?")
                .Select(g => new ColorRequirement(
                    g.Key,
                    g.Count(),
                    g.Count(b => b.GetProperty("filled").GetBoolean())))
                .ToArray();
            columnReqs[c] = groups;
        }

        var rowReqs = new ColorRequirement[rows][];
        for (int r = 0; r < rows; r++)
        {
            var bars = rowBars[r];
            var groups = bars.EnumerateArray()
                .GroupBy(b => b.GetProperty("color_group").GetString() ?? "?")
                .Select(g => new ColorRequirement(
                    g.Key,
                    g.Count(),
                    g.Count(b => b.GetProperty("filled").GetBoolean())))
                .ToArray();
            rowReqs[r] = groups;
        }

        var compsArr = root.GetProperty("components");
        var comps = new List<ComponentInfo>();
        foreach (var comp in compsArr.EnumerateArray())
        {
            var shapeArr = comp.GetProperty("shape");
            var shapeRows = shapeArr.GetArrayLength();
            var shapeCols = shapeArr[0].GetArrayLength();
            var shape = new bool[shapeRows, shapeCols];
            for (int sr = 0; sr < shapeRows; sr++)
                for (int sc = 0; sc < shapeCols; sc++)
                    shape[sr, sc] = shapeArr[sr][sc].GetInt32() == 1;

            comps.Add(new ComponentInfo
            {
                Shape = shape,
                ColorGroup = comp.GetProperty("color_group").GetString() ?? "A"
            });
        }

        var colorGroups = Array.Empty<ColorGroupInfo>();
        if (root.TryGetProperty("color_groups", out var cgEl))
        {
            colorGroups = cgEl.EnumerateArray().Select(g =>
            {
                var hsv = g.GetProperty("hsv");
                return new ColorGroupInfo(
                    g.GetProperty("label").GetString() ?? "?",
                    hsv[0].GetInt32(),
                    hsv[1].GetInt32(),
                    hsv[2].GetInt32());
            }).ToArray();
        }

        return new PuzzleData
        {
            Rows = rows,
            Cols = cols,
            Tiles = tiles,
            ColumnRequirements = columnReqs,
            RowRequirements = rowReqs,
            Components = comps.ToArray(),
            ColorGroups = colorGroups
        };
    }
}
