using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using OpenCvSharp;
using EndfieldPuzzleSolver.Recognition.Models;
using EndfieldPuzzleSolver.Recognition.Config;

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
    private readonly DetectionConfig _config;

    private Mat _image = null!;
    private Mat _gray = null!;
    private Mat _hsv = null!;
    private ColorGrouper _colorGrouper = null!;
    private double _scale = 1.0;

    public PuzzleDetector(string templateDir, DetectionConfig config, bool debug = false, string? debugDir = null, double? scale = null)
    {
        _templateDir = templateDir;
        _config = config;
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
        var roi = _gray[new Rect(
            (int)(w * _config.AutoScaling.RoiHorizontalRatio[0]),
            (int)(h * _config.AutoScaling.RoiVerticalRatio[0]),
            (int)(w * _config.AutoScaling.RoiHorizontalRatio[1] - w * _config.AutoScaling.RoiHorizontalRatio[0]),
            (int)(h * _config.AutoScaling.RoiVerticalRatio[1] - h * _config.AutoScaling.RoiVerticalRatio[0]))];

        double bestScale = 1.0;
        double bestScore = -1.0;
        var stepSize = _config.AutoScaling.StepSize;
        var searchRange = _config.AutoScaling.SearchRange;

        for (var s = searchRange[0]; s <= searchRange[1] + 0.001; s += stepSize)
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
        _colorGrouper = new ColorGrouper(_config.ColorGrouping.HueMergeDistance);

        var estimatedScale = _userScale ?? EstimateScale();
        if (estimatedScale == null)
            return DetectResult.Fail("自动缩放探测失败：未找到模板（请确认模板图片存在）");
        _scale = estimatedScale.Value;
        BuildScaledTemplates(_scale);

        var grid = DetectGrid();
        if (grid == null)
            return DetectResult.Fail("格子匹配数不足：无法识别棋盘网格（可能不是有效的谜题截图）");

        var g = grid.Value;
        // 计算网格底部用于组件检测的 y 限制
        var ih = _image.Rows;
        var gridBottom = g.OriginY + g.Rows * g.CellH;
        var yLimit = Math.Min(ih, gridBottom + Scaled(_config.ComponentDetection.YLimitOffset));

        var comps = DetectComponents(yLimit);
        var tiles = ClassifyTiles(g);
        var reqs = DetectRequirements(g);

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
        var roi = _gray[new Rect(
            (int)(w * _config.AutoScaling.RoiHorizontalRatio[0]),
            (int)(h * _config.AutoScaling.RoiVerticalRatio[0]),
            (int)(w * _config.AutoScaling.RoiHorizontalRatio[1] - w * _config.AutoScaling.RoiHorizontalRatio[0]),
            (int)(h * _config.AutoScaling.RoiVerticalRatio[1] - h * _config.AutoScaling.RoiVerticalRatio[0]))];
        var oxOff = (int)(w * _config.AutoScaling.RoiHorizontalRatio[0]);
        var oyOff = (int)(h * _config.AutoScaling.RoiVerticalRatio[0]);

        var allPts = new List<(int X, int Y, int W, int H, double Score)>();
        var tplKeyMap = new Dictionary<string, string>
        {
            ["tile-empty"] = "tile_empty",
            ["tile-disable"] = "tile_disable"
        };

        foreach (var tplName in _config.GridDetection.UsedTemplates)
        {
            if (!tplKeyMap.TryGetValue(tplName, out var tplKey)) continue;

            if (!_templates.TryGetValue(tplKey, out var tpl)) continue;

            var pts = MatchTemplate(roi, tpl, _config.GridDetection.TemplateMatchThreshold);
            foreach (var (x, y, tw, th, s) in pts)
                allPts.Add((x + oxOff, y + oyOff, tw, th, s));
        }

        if (allPts.Count < 4)
            return null;

        allPts = Nms(allPts, _config.GridDetection.NmsIouThreshold);
        var minWh = Scaled(_config.GridDetection.MinCellSize);
        var maxWh = Scaled(_config.GridDetection.MaxCellSize);
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

        // 使用阈值创建二值掩码
        var mask = new Mat();
        Cv2.Threshold(res, mask, (float)thresh, 255, ThresholdTypes.Binary);

        // 使用 FindNonZero 获取所有匹配点
        using var nz = mask.FindNonZero();
        if (nz.Empty() || nz.Rows == 0)
        {
            mask.Dispose();
            return result;
        }

        // 遍历所有匹配点
        for (int i = 0; i < nz.Rows; i++)
        {
            var pt = nz.At<Point>(i);
            var val = res.Get<float>(pt.Y, pt.X);
            result.Add((pt.X, pt.Y, tw, th, val));
        }

        mask.Dispose();
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

        var minGap = Scaled(_config.GridDetection.MinGapSize);
        var dxs = xs.Zip(xs.Skip(1), (a, b) => b - a).Where(d => d > minGap).ToList();
        var dys = ys.Zip(ys.Skip(1), (a, b) => b - a).Where(d => d > minGap).ToList();

        var defaultCell = Scaled(_config.GridDetection.DefaultCellSize);
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
        var innerMargin = Math.Max(5, Math.Min(ch, cw) / 10);
        innerMargin = Math.Max(innerMargin, (int)(Math.Min(ch, cw) * _config.TileClassification.InnerMarginRatio));

        if (ch <= 2 * innerMargin || cw <= 2 * innerMargin)
            return new TileInfo { Type = TileType.Empty };

        var innerHsv = cellHsv[new Rect(innerMargin, innerMargin, cw - 2 * innerMargin, ch - 2 * innerMargin)];

        // Lock: 高饱和度 AND 高亮度 (HSV: Item0=H, Item1=S, Item2=V)
        var lockMinSat = _config.TileClassification.LockMinSaturation;
        var lockMinVal = _config.TileClassification.LockMinValue;

        // 分离通道
        var channels = Cv2.Split(innerHsv);
        var hChannel = channels[0];
        var sChannel = channels[1];
        var vChannel = channels[2];

        // 创建饱和度和亮度掩码
        var satMask = new Mat();
        var valMask = new Mat();
        Cv2.Compare(sChannel, new Scalar((byte)lockMinSat), satMask, CmpType.GT);
        Cv2.Compare(vChannel, new Scalar((byte)lockMinVal), valMask, CmpType.GT);

        // 组合掩码
        var lockMask = new Mat();
        Cv2.BitwiseAnd(satMask, valMask, lockMask);

        // 统计符合条件的像素数
        var lockCount = Cv2.CountNonZero(lockMask);
        var totalCount = innerHsv.Total();

        var lockRatio = totalCount > 0 ? (double)lockCount / totalCount : 0;
        if (lockRatio > _config.TileClassification.LockMinPixelRatio)
        {
            // 计算符合条件的像素的平均 HSV 值
            using var maskedH = new Mat();
            using var maskedS = new Mat();
            using var maskedV = new Mat();
            Cv2.BitwiseAnd(hChannel, lockMask, maskedH);
            Cv2.BitwiseAnd(sChannel, lockMask, maskedS);
            Cv2.BitwiseAnd(vChannel, lockMask, maskedV);

            var avgH = Cv2.Sum(maskedH).Val0 / lockCount;
            var avgS = Cv2.Sum(maskedS).Val0 / lockCount;
            var avgV = Cv2.Sum(maskedV).Val0 / lockCount;

            // 真正的锁定格颜色明显：平均亮度应该 > lock_min_avg_val
            if (avgV > _config.TileClassification.LockMinAvgValue && avgS > _config.TileClassification.LockMinAvgSaturation)
            {
                var colorGroup = _colorGrouper.MatchNearest((int)avgH, (int)avgS, (int)avgV);
                foreach (var chItem in channels) chItem.Dispose();
                satMask.Dispose(); valMask.Dispose(); lockMask.Dispose();
                maskedH.Dispose(); maskedS.Dispose(); maskedV.Dispose();
                return TileInfo.Locked(colorGroup);
            }
        }

        // 释放临时 Mat
        foreach (var chItem in channels) chItem.Dispose();
        satMask.Dispose(); valMask.Dispose(); lockMask.Dispose();

        // Disabled vs Empty: template competition
        var scDis = CellScore(cellGray, "tile_disable");
        var scEmp = CellScore(cellGray, "tile_empty");
        var bestScore = Math.Max(scDis, scEmp);
        if (bestScore > _config.TileClassification.TileMinScore)
        {
            if (scDis >= scEmp)
                return TileInfo.Disabled;
            return TileInfo.Empty;
        }

        // Fallback: brightness and variance using MeanStdDev
        var innerGray = cellGray[new Rect(innerMargin, innerMargin, cw - 2 * innerMargin, ch - 2 * innerMargin)];
        Cv2.MeanStdDev(innerGray, out var mean, out var stddev);
        var meanV = mean.Val0;
        var variance = stddev.Val0 * stddev.Val0;

        if (meanV < _config.TileClassification.EmptyMeanValueThreshold && variance < _config.TileClassification.EmptyVarianceThreshold)
            return TileInfo.Empty;
        if (meanV < _config.TileClassification.DisabledMeanValueThreshold && variance > _config.TileClassification.DisabledVarianceThreshold)
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

        // 无元件颜色时返回空
        if (_colorGrouper.ClusterCount == 0)
            return reqs.Select(r => r.GroupBy(x => x.ColorGroup).Select(g => new ColorRequirement(g.Key, g.Count(), g.Count(x => x.Filled))).ToArray()).ToArray();

        // 按每种元件颜色分别计算相似度图并检测
        for (int ci = 0; ci < _colorGrouper.ClusterCount; ci++)
        {
            var (ch, cs, cv) = _colorGrouper.ColorHsv(ci);
            var grp = ColorGrouper.Label(ci);
            var simMap = SimilarityMapForColor(hsv, ch, cs, cv);

            // 二值化
            var simU8 = new Mat();
            simMap.ConvertTo(simU8, MatType.CV_8U, 255.0);
            var mask = new Mat();
            Cv2.Threshold(simU8, mask, 0, 255, ThresholdTypes.Binary | ThresholdTypes.Otsu);

            if (Cv2.CountNonZero(mask) < 10)
            {
                simU8.Dispose();
                mask.Dispose();
                continue;
            }

            var contours = Cv2.FindContoursAsArray(mask, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

            foreach (var cnt in contours)
            {
                var rect = Cv2.BoundingRect(cnt);
                var x = rect.X;
                var y = rect.Y;
                var w = rect.Width;
                var h = rect.Height;
                var area = w * h;

                if (area < ScaledArea(_config.RequirementDetection.MinArea) ||
                    area > ScaledArea(_config.RequirementDetection.MaxArea))
                    continue;
                var aspect = (double)Math.Max(w, h) / Math.Max(Math.Min(w, h), 1);
                if (aspect > _config.RequirementDetection.MaxAspectRatio)
                    continue;

                var barMask = mask[rect];
                var barHsv = hsv[rect];

                // 分离 HSV 通道
                var barChannels = Cv2.Split(barHsv);
                var hBar = barChannels[0];
                var sBar = barChannels[1];
                var vBar = barChannels[2];

                // 使用掩码提取颜色像素并计算平均值
                using var maskedH = new Mat();
                using var maskedS = new Mat();
                using var maskedV = new Mat();
                Cv2.BitwiseAnd(hBar, barMask, maskedH);
                Cv2.BitwiseAnd(sBar, barMask, maskedS);
                Cv2.BitwiseAnd(vBar, barMask, maskedV);

                var pxCount = Cv2.CountNonZero(barMask);
                if (pxCount < 5)
                {
                    foreach (var chItem in barChannels) chItem.Dispose();
                    maskedH.Dispose(); maskedS.Dispose(); maskedV.Dispose();
                    continue;
                }

                var avgH = Cv2.Sum(maskedH).Val0 / pxCount;
                var avgS = Cv2.Sum(maskedS).Val0 / pxCount;
                var avgV = Cv2.Sum(maskedV).Val0 / pxCount;

                // 判断 filled - 使用 MeanStdDev 计算标准差
                // 注意：必须在 Dispose barChannels 之前访问 vBar
                double valCv = 0;
                bool filled = false;
                if (vBar.Rows >= _config.RequirementDetection.MinRegionSize &&
                    vBar.Cols >= _config.RequirementDetection.MinRegionSize)
                {
                    Cv2.MeanStdDev(vBar, out var mean, out var stddev);
                    var meanVal = mean.Val0;
                    var stdVal = stddev.Val0;
                    valCv = meanVal > 1 ? stdVal / meanVal : 0;
                    filled = valCv < _config.RequirementDetection.FilledCvThreshold;
                }

                // 释放临时 Mat
                foreach (var chItem in barChannels) chItem.Dispose();
                maskedH.Dispose(); maskedS.Dispose(); maskedV.Dispose();

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
                    reqs[idx].Add((grp, filled));
            }

            simU8.Dispose();
            mask.Dispose();
        }

        return reqs.Select(r => r.GroupBy(x => x.ColorGroup).Select(g => new ColorRequirement(g.Key, g.Count(), g.Count(x => x.Filled))).ToArray()).ToArray();
    }

    private Mat SimilarityMapForColor(Mat hsv, int ch, int cs, int cv)
    {
        // 分离 HSV 通道
        var channels = Cv2.Split(hsv);
        var hChannel = channels[0];
        var sChannel = channels[1];
        var vChannel = channels[2];

        var hueWidth = _config.RequirementDetection.HueSimilarityWidth;
        var minSat = _config.RequirementDetection.MinSaturation;
        var minVal = _config.RequirementDetection.MinValue;

        // 转换为 float 类型
        var hFloat = new Mat();
        var sFloat = new Mat();
        var vFloat = new Mat();
        hChannel.ConvertTo(hFloat, MatType.CV_32F);
        sChannel.ConvertTo(sFloat, MatType.CV_32F);
        vChannel.ConvertTo(vFloat, MatType.CV_32F);

        // 创建目标色相矩阵
        var chMat = new Mat(hsv.Size(), MatType.CV_32F, new Scalar(ch));

        // 计算色相距离: |h - ch|
        var hueDist = new Mat();
        Cv2.Absdiff(hFloat, chMat, hueDist);

        // 计算最小距离: min(d, 180 - d)
        var hueDist2 = new Mat();
        Cv2.Subtract(new Scalar(180.0), hueDist, hueDist2);
        Cv2.Min(hueDist, hueDist2, hueDist);

        // 计算色相相似度: max(0, 1 - d / hueWidth)
        hueDist *= (1.0f / hueWidth);
        Cv2.Subtract(new Scalar(1.0f), hueDist, hueDist);
        Cv2.Max(hueDist, new Scalar(0.0f), hueDist);

        // 创建饱和度和亮度掩码
        var satMask = new Mat();
        var valMask = new Mat();
        Cv2.Compare(sFloat, new Scalar((float)minSat), satMask, CmpType.GE);
        Cv2.Compare(vFloat, new Scalar((float)minVal), valMask, CmpType.GE);

        // 组合有效区域掩码
        var validMask = new Mat();
        Cv2.BitwiseAnd(satMask, valMask, validMask);

        // 将掩码转换为 float 类型
        var validFloat = new Mat();
        validMask.ConvertTo(validFloat, MatType.CV_32F, 1.0 / 255.0);

        // 组合最终结果
        var result = new Mat();
        Cv2.Multiply(hueDist, validFloat, result);

        // 释放临时 Mat
        foreach (var chItem in channels) chItem.Dispose();
        hFloat.Dispose(); sFloat.Dispose(); vFloat.Dispose();
        chMat.Dispose(); hueDist2.Dispose();
        satMask.Dispose(); valMask.Dispose(); validMask.Dispose(); validFloat.Dispose();

        return result;
    }

    private ComponentInfo[] DetectComponents(int yLimit)
    {
        var ih = _image.Rows;
        var iw = _image.Cols;
        var rx = (int)(iw * _config.ComponentDetection.SearchStartXRatio);
        var roiGray = _gray[new Rect(rx, 0, iw - rx, ih)];
        var rHsv = _hsv[new Rect(rx, 0, iw - rx, ih)];

        // 优先使用 component_flame 模板匹配
        var blobs = DetectComponentsByTemplate(roiGray, rx, yLimit);
        if (blobs.Count < 1)
        {
            // 兜底：原有 HSV blob 检测
            blobs = DetectComponentsByBlob(rHsv, rx, yLimit);
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

    private List<(int X, int Y, int W, int H, double Area)> DetectComponentsByTemplate(Mat roiGray, int rx, int yLimit)
    {
        var result = new List<(int, int, int, int, double)>();
        var rawTpl = _rawTemplates.GetValueOrDefault("component_flame");
        if (rawTpl == null)
            return result;

        var rh = roiGray.Rows;
        var bestPts = new List<(int X, int Y, int W, int H, double Score)>();
        var bestN = 0;

        foreach (var compScale in _config.ComponentDetection.TemplateScales)
        {
            var tpl = ResizeTemplate(rawTpl, compScale * _scale);
            var th = tpl.Rows;
            var tw = tpl.Cols;
            if (th >= rh || tw >= roiGray.Cols)
            {
                tpl.Dispose();
                continue;
            }

            using var res = new Mat();
            Cv2.MatchTemplate(roiGray, tpl, res, TemplateMatchModes.CCoeffNormed);
            Cv2.MinMaxLoc(res, out _, out double maxScoreRes, out _, out _);

            var maxScore = (double)maxScoreRes;
            var thresh = Math.Max(_config.ComponentDetection.TemplateMatchMinScore,
                                  Math.Min(_config.ComponentDetection.TemplateMatchMaxScore, maxScore - 0.05));

            var pts = MatchTemplate(roiGray, tpl, thresh);
            var adjustedPts = pts.Select(p => (p.X + rx, p.Y, p.W, p.H, p.Score)).ToList();

            // 排除底部 UI
            adjustedPts = adjustedPts.Where(p => p.Y < yLimit).ToList();
            adjustedPts = Nms(adjustedPts, _config.ComponentDetection.TemplateNmsIouThreshold);

            var minTplSize = Scaled(_config.ComponentDetection.MinTemplateSize);
            var maxTplSize = Scaled(_config.ComponentDetection.MaxTemplateSize);
            adjustedPts = adjustedPts.Where(p => p.W > minTplSize && p.W < maxTplSize &&
                                                p.H > minTplSize && p.H < maxTplSize).ToList();

            if (adjustedPts.Count > bestN && adjustedPts.Count >= 1 && adjustedPts.Count <= 8)
            {
                bestN = adjustedPts.Count;
                bestPts = adjustedPts;
            }

            tpl.Dispose();
        }

        if (bestPts.Count == 0)
            return result;

        return bestPts.Select(p => (p.X, p.Y, p.W, p.H, (double)(p.W * p.H))).ToList();
    }

    private List<(int X, int Y, int W, int H, double Area)> DetectComponentsByBlob(Mat rHsv, int rx, int yLimit)
    {
        using var sat = new Mat();
        using var val = new Mat();
        Cv2.ExtractChannel(rHsv, sat, 1);
        Cv2.ExtractChannel(rHsv, val, 2);

        using var rawMask = new Mat();
        Cv2.Compare(sat, new Scalar(_config.ComponentDetection.BlobMinSaturation), rawMask, CmpType.GT);
        using var rawMask2 = new Mat();
        Cv2.Compare(val, new Scalar(_config.ComponentDetection.BlobMinValue), rawMask2, CmpType.GT);
        using var combined = new Mat();
        Cv2.BitwiseAnd(rawMask, rawMask2, combined);

        Cv2.FindContours(combined, out var contours, out _, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

        var blobs = new List<(int, int, int, int, double)>();
        foreach (var cnt in contours)
        {
            var rect = Cv2.BoundingRect(cnt);
            var area = Cv2.ContourArea(cnt);
            var x = rect.X + rx;
            var y = rect.Y;
            var w = rect.Width;
            var h = rect.Height;

            // 排除底部 UI
            if (y >= yLimit)
                continue;

            if (area > ScaledArea(_config.ComponentDetection.BlobMinArea) &&
                w > Scaled(_config.ComponentDetection.BlobMinSize) &&
                w < Scaled(_config.ComponentDetection.BlobMaxSize) &&
                h > Scaled(_config.ComponentDetection.BlobMinSize) &&
                h < Scaled(_config.ComponentDetection.BlobMaxSize))
            {
                var aspect = (double)Math.Max(w, h) / Math.Max(Math.Min(w, h), 1);
                if (aspect < _config.ComponentDetection.BlobMaxAspectRatio)
                {
                    blobs.Add((x, y, w, h, area));
                }
            }
        }
        return blobs;
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
        Cv2.Compare(sat, new Scalar(_config.ComponentDetection.BlobMinSaturation), m1, CmpType.GT);
        Cv2.Compare(val, new Scalar(_config.ComponentDetection.BlobMinValue), m2, CmpType.GT);
        using var blobMask = new Mat();
        Cv2.BitwiseAnd(m1, m2, blobMask);

        // 裁去边框辉光
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
        if (th < Scaled(_config.ComponentDetection.MinShapeSize) || tw < Scaled(_config.ComponentDetection.MinShapeSize))
            return null;

        // 尝试不同网格尺寸
        int[][]? bestShape = null;
        double bestScore = -999;

        for (int nr = _config.ComponentDetection.MinRows; nr <= _config.ComponentDetection.MaxRows; nr++)
        {
            for (int nc = _config.ComponentDetection.MinCols; nc <= _config.ComponentDetection.MaxCols; nc++)
            {
                var total = nr * nc;
                if (total < _config.ComponentDetection.MinTotalCells || total > _config.ComponentDetection.MaxTotalCells)
                    continue;

                var cellH = (double)th / nr;
                var cellW = (double)tw / nc;

                if (cellH < Scaled(_config.ComponentDetection.MinShapeSize) || cellW < Scaled(_config.ComponentDetection.MinShapeSize))
                    continue;
                if (cellH > Scaled(_config.ComponentDetection.MaxShapeSize) || cellW > Scaled(_config.ComponentDetection.MaxShapeSize))
                    continue;

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
                        shape[r][c] = fill > _config.ComponentDetection.CellFillThreshold ? 1 : 0;
                    }
                }

                var filled = shape.Sum(row => row.Sum());
                if (filled == 0 || filled == total) continue;

                var nClearFilled = fills.Count(f => f > _config.ComponentDetection.ClearFilledThreshold);
                var nClearEmpty = fills.Count(f => f < _config.ComponentDetection.ClearEmptyThreshold);
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

        // 提取颜色 - 使用掩码操作替代双重循环
        var maskCount = Cv2.CountNonZero(blobMask);
        if (maskCount < ScaledArea(_config.ComponentDetection.MinColorPixels))
            return null;

        // 分离 blobHsv 的通道
        var blobChannels = Cv2.Split(blobHsv);
        var blobH = blobChannels[0];
        var blobS = blobChannels[1];
        var blobV = blobChannels[2];

        // 使用掩码提取颜色像素并计算平均值
        using var maskedBlobH = new Mat();
        using var maskedBlobS = new Mat();
        using var maskedBlobV = new Mat();
        Cv2.BitwiseAnd(blobH, blobMask, maskedBlobH);
        Cv2.BitwiseAnd(blobS, blobMask, maskedBlobS);
        Cv2.BitwiseAnd(blobV, blobMask, maskedBlobV);

        var sumH = Cv2.Sum(maskedBlobH).Val0;
        var sumS = Cv2.Sum(maskedBlobS).Val0;
        var sumV = Cv2.Sum(maskedBlobV).Val0;
        var pxCount = maskCount;

        // 释放临时 Mat
        foreach (var chItem in blobChannels) chItem.Dispose();
        maskedBlobH.Dispose(); maskedBlobS.Dispose(); maskedBlobV.Dispose();

        var avgH = (int)(sumH / pxCount);
        var avgS = (int)(sumS / pxCount);
        var avgV = (int)(sumV / pxCount);
        if (avgV < _config.ComponentDetection.MinAvgValue)
            return null;

        var colorGroup = _colorGrouper.MatchNearest(avgH, avgS, avgV);
        var shapeBool = new bool[bestShape.Length, bestShape[0].Length];
        for (int r = 0; r < bestShape.Length; r++)
            for (int c = 0; c < bestShape[r].Length; c++)
                shapeBool[r, c] = bestShape[r][c] == 1;

        return new ComponentInfo
        {
            Shape = shapeBool,
            ColorGroup = colorGroup
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
