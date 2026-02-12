#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
终末地源石电路模块解谜 - UI 识别原型
从游戏截图中识别网格、格子类型、行列需求和可用元件
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from config import Config, load_default_config


# ──────────────────────── 颜色工具 ────────────────────────

def hue_distance(h1, h2):
    """计算两个 OpenCV Hue(0-180) 之间的环形距离"""
    d = abs(int(h1) - int(h2))
    return min(d, 180 - d)


def hsv_to_bgr(h, s, v):
    """HSV -> BGR (用于 OpenCV 绘制)"""
    px = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in bgr)


def hsv_to_rgb(h, s, v):
    """HSV -> RGB (用于 PIL 绘制)"""
    b, g, r = hsv_to_bgr(h, s, v)
    return (r, g, b)


# ──────────────────────── PIL 中文绘图 ────────────────────────

def cv2_to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))


def pil_to_cv2(pil_img):
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def get_font(size=18):
    for fp in [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
    ]:
        if os.path.exists(fp):
            return ImageFont.truetype(fp, size)
    return ImageFont.load_default()


# ──────────────────────── 颜色聚类 ────────────────────────

class ColorGrouper:
    """
    收集所有检测到的 HSV 颜色，按 Hue 聚类，分配标签 A/B/C...
    不写死颜色名称，只区分「同一题目中不同的颜色」。
    """

    def __init__(self, hue_merge_distance=18):
        self.clusters = []  # [(center_hue, avg_s, avg_v, [hue_samples])]
        self.hue_merge_distance = hue_merge_distance

    def register(self, h, s, v):
        """注册一个颜色，返回其所属组的索引"""
        for i, (ch, _, _, samples) in enumerate(self.clusters):
            if hue_distance(h, ch) < self.hue_merge_distance:
                samples.append((h, s, v))
                # 更新中心
                all_h = [x[0] for x in samples]
                all_s = [x[1] for x in samples]
                all_v = [x[2] for x in samples]
                self.clusters[i] = (
                    int(np.mean(all_h)),
                    int(np.mean(all_s)),
                    int(np.mean(all_v)),
                    samples,
                )
                return i
        # 新建组
        idx = len(self.clusters)
        self.clusters.append((h, s, v, [(h, s, v)]))
        return idx

    def label(self, idx):
        """返回组标签: A, B, C, ..."""
        if 0 <= idx < 26:
            return chr(ord('A') + idx)
        return str(idx)

    def match_nearest(self, h, s, v):
        """匹配到已有 cluster 中 Hue 距离最近者，返回其 label；无 cluster 时临时 register"""
        if not self.clusters:
            return self.label(self.register(h, s, v))
        best_idx = 0
        best_dist = hue_distance(h, self.clusters[0][0])
        for i, (ch, _, _, _) in enumerate(self.clusters):
            d = hue_distance(h, ch)
            if d < best_dist:
                best_dist = d
                best_idx = i
        return self.label(best_idx)

    def color_hsv(self, idx):
        """返回组的平均 HSV"""
        if 0 <= idx < len(self.clusters):
            h, s, v, _ = self.clusters[idx]
            return (h, s, v)
        return (0, 0, 0)


# ──────────────────────── 检测器 ────────────────────────

class PuzzleDetector:

    def __init__(self, template_dir, debug=False, debug_dir=None, scale=None, config=None):
        self.template_dir = Path(template_dir)
        self.raw_templates = {}
        self.templates = {}
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.user_scale = float(scale) if scale is not None else None
        self.scale = self.user_scale if self.user_scale is not None else 1.0

        # 加载配置
        if config is None:
            config = Config(Path(__file__).parent / "image" / "config.toml")
        self.config = config

        # 从配置加载颜色分组参数
        hue_merge_dist = config.get_int('color_grouping.hue_merge_distance', 18)

        # 初始化 ColorGrouper（稍后在 detect 中创建实例）
        self.color_grouper = None
        self._hue_merge_distance = hue_merge_dist

        self._load_templates()

    def _debug_save(self, name, img):
        """debug 模式下保存中间图像"""
        if not self.debug or self.debug_dir is None:
            return
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        path = self.debug_dir / name
        cv2.imwrite(str(path), img)
        print(f"  [debug] {path}")

    def _load_templates(self):
        for name in ['tile-empty.png', 'tile-disable.png',
                      'component-flame.png', 'puzzle-area-corner-lt.png']:
            path = self.template_dir / name
            if path.exists():
                img = cv2.imread(str(path))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                key = name.replace('.png', '').replace('-', '_')
                self.raw_templates[key] = gray
                print(f"[模板] {name}  {gray.shape}")

    def _resize_template(self, tpl, scale):
        h, w = tpl.shape
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        return cv2.resize(tpl, (nw, nh), interpolation=cv2.INTER_LINEAR)

    def _build_scaled_templates(self, scale):
        self.templates = {}
        for key, tpl in self.raw_templates.items():
            self.templates[key] = self._resize_template(tpl, scale)

    def _scaled(self, value):
        return max(1, int(round(value * self.scale)))

    def _scaled_area(self, value):
        return max(1, int(round(value * self.scale * self.scale)))

    def _estimate_scale(self):
        """多尺度模板匹配，实测当前截图中的 UI 缩放系数。"""
        h, w = self.gray.shape

        # 从配置加载参数
        roi_h_ratio = self.config.get_list('auto_scaling.roi_horizontal_ratio', [1/6, 2/3])
        roi_v_ratio = self.config.get_list('auto_scaling.roi_vertical_ratio', [1/6, 5/6])

        roi = self.gray[int(h * roi_v_ratio[0]):int(h * roi_v_ratio[1]),
                        int(w * roi_h_ratio[0]):int(w * roi_h_ratio[1])]

        search_range = self.config.get_list('auto_scaling.search_range', [0.5, 2.0])
        step_size = self.config.get_float('auto_scaling.step_size', 0.05)
        candidates = np.arange(search_range[0], search_range[1] + 0.001, step_size)

        best_scale = 1.0
        best_score = -1.0

        for scale in candidates:
            scores = []
            for key in ('tile_empty', 'tile_disable'):
                tpl = self.raw_templates.get(key)
                if tpl is None:
                    continue
                stpl = self._resize_template(tpl, float(scale))
                th, tw = stpl.shape
                rh, rw = roi.shape
                if th >= rh or tw >= rw:
                    continue
                res = cv2.matchTemplate(roi, stpl, cv2.TM_CCOEFF_NORMED)
                if res.size > 0:
                    scores.append(float(res.max()))
            if not scores:
                continue
            score = float(np.mean(scores))
            if score > best_score:
                best_score = score
                best_scale = float(scale)

        if best_score < 0:
            raise ValueError("自动缩放探测失败：未找到可用模板匹配结果")

        return best_scale

    # ─────────── 主流程 ───────────

    def detect(self, image_path):
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"无法读取: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)

        # 初始化 ColorGrouper，使用配置的 hue_merge_distance
        self.color_grouper = ColorGrouper(self._hue_merge_distance)

        if self.user_scale is None:
            self.scale = self._estimate_scale()
            print(f"[缩放] 自动探测 scale={self.scale:.2f}")
        else:
            self.scale = self.user_scale
            print(f"[缩放] 手动指定 scale={self.scale:.2f}")
        self._build_scaled_templates(self.scale)

        print(f"\n[图像] {Path(image_path).name}  {self.image.shape[1]}×{self.image.shape[0]}")

        grid = self._detect_grid()
        comps = self._detect_components(grid)
        tiles = self._classify_tiles(grid)
        reqs = self._detect_requirements(grid)

        # 输出颜色组信息
        n_groups = len(self.color_grouper.clusters)
        print(f"\n[颜色] 共识别 {n_groups} 种颜色")
        for i in range(n_groups):
            lbl = self.color_grouper.label(i)
            h, s, v = self.color_grouper.color_hsv(i)
            n = len(self.color_grouper.clusters[i][3])
            print(f"  颜色{lbl}: HSV=({h},{s},{v})  采样{n}次")

        return {
            'detected_scale': self.scale,
            'grid_size': grid['size'],
            'grid_origin_px': grid['origin'],
            'cell_size_px': grid['cell_size'],
            'tiles': tiles,
            'requirements': reqs,
            'components': comps,
            'color_groups': [
                {'label': self.color_grouper.label(i),
                 'hsv': list(self.color_grouper.color_hsv(i)),
                 'count': len(self.color_grouper.clusters[i][3])}
                for i in range(n_groups)
            ],
        }

    # ─────────── 1. 网格检测 ───────────

    def _detect_grid(self):
        print("\n[1/4] 网格检测")
        h, w = self.gray.shape

        # 从配置加载参数
        roi_h_ratio = self.config.get_list('auto_scaling.roi_horizontal_ratio', [1/6, 2/3])
        roi_v_ratio = self.config.get_list('auto_scaling.roi_vertical_ratio', [1/6, 5/6])
        roi = self.gray[int(h * roi_v_ratio[0]):int(h * roi_v_ratio[1]),
                        int(w * roi_h_ratio[0]):int(w * roi_h_ratio[1])]
        ox_off, oy_off = int(w * roi_h_ratio[0]), int(h * roi_v_ratio[0])

        all_pts = []
        used_templates = self.config.get_list('grid_detection.used_templates', ['tile-empty', 'tile-disable'])
        template_threshold = self.config.get_float('grid_detection.template_match_threshold', 0.80)

        tpl_key_map = {'tile-empty': 'tile_empty', 'tile-disable': 'tile_disable'}
        for tpl_name in used_templates:
            tpl_key = tpl_key_map.get(tpl_name)
            if not tpl_key:
                continue
            tpl = self.templates.get(tpl_key)
            if tpl is None:
                continue
            pts = self._match_template(roi, tpl, thresh=template_threshold)
            pts = [(x + ox_off, y + oy_off, tw, th, s) for x, y, tw, th, s in pts]
            all_pts.extend(pts)
            print(f"  {tpl_key}: {len(pts)} 匹配")

        if len(all_pts) < 4:
            raise ValueError("格子匹配数不足")

        nms_iou_thresh = self.config.get_float('grid_detection.nms_iou_threshold', 0.5)
        all_pts = self._nms(all_pts, iou_thresh=nms_iou_thresh)

        min_wh = self._scaled(self.config.get_int('grid_detection.min_cell_size', 65))
        max_wh = self._scaled(self.config.get_int('grid_detection.max_cell_size', 100))
        all_pts = [p for p in all_pts if min_wh < p[2] < max_wh and min_wh < p[3] < max_wh]
        print(f"  NMS+过滤: {len(all_pts)}")

        # debug: 在图上标出所有匹配点
        if self.debug:
            dbg = self.image.copy()
            for x, y, w_, h_, s in all_pts:
                cv2.rectangle(dbg, (x, y), (x + w_, y + h_), (0, 255, 0), 2)
                cv2.putText(dbg, f"{s:.2f}", (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            self._debug_save("1_grid_matches.png", dbg)

        grid = self._infer_grid(all_pts)
        print(f"  {grid['size'][0]}行×{grid['size'][1]}列  "
              f"原点={grid['origin']}  格子={grid['cell_size']}")

        # debug: 画出推断的网格
        if self.debug:
            dbg = self.image.copy()
            ox, oy = grid['origin']
            cw, ch = grid['cell_size']
            rows, cols = grid['size']
            for r in range(rows + 1):
                cv2.line(dbg, (ox, oy + r * ch), (ox + cols * cw, oy + r * ch), (0, 255, 0), 1)
            for c in range(cols + 1):
                cv2.line(dbg, (ox + c * cw, oy), (ox + c * cw, oy + rows * ch), (0, 255, 0), 1)
            self._debug_save("1_grid_inferred.png", dbg)

        return grid

    def _match_template(self, img_gray, tpl, thresh=0.8):
        th, tw = tpl.shape
        res = cv2.matchTemplate(img_gray, tpl, cv2.TM_CCOEFF_NORMED)
        locs = np.where(res >= thresh)
        return [(int(x), int(y), tw, th, float(res[y, x]))
                for y, x in zip(*locs)]

    def _nms(self, boxes, iou_thresh=0.3):
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
        keep = []
        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if self._iou(best, b) < iou_thresh]
        return keep

    @staticmethod
    def _iou(a, b):
        x1, y1_ = max(a[0], b[0]), max(a[1], b[1])
        x2, y2_ = min(a[0] + a[2], b[0] + b[2]), min(a[1] + a[3], b[1] + b[3])
        inter = max(0, x2 - x1) * max(0, y2_ - y1_)
        union = a[2] * a[3] + b[2] * b[3] - inter
        return inter / union if union else 0

    def _infer_grid(self, matches):
        centers = np.array([(x + w // 2, y + h // 2) for x, y, w, h, _ in matches])

        # 从配置加载参数
        min_gap = self._scaled(self.config.get_int('grid_detection.min_gap_size', 30))
        default_cell = self._scaled(self.config.get_int('grid_detection.default_cell_size', 87))

        # 用排序后的相邻间距的中位数来估算格子尺寸
        xs = sorted(set(centers[:, 0]))
        ys = sorted(set(centers[:, 1]))
        dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > min_gap]
        dys = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > min_gap]

        cell_w = int(np.median(dxs)) if dxs else default_cell
        cell_h = int(np.median(dys)) if dys else default_cell

        min_cx, max_cx = int(centers[:, 0].min()), int(centers[:, 0].max())
        min_cy, max_cy = int(centers[:, 1].min()), int(centers[:, 1].max())

        return {
            'size': [round((max_cy - min_cy) / cell_h) + 1,
                     round((max_cx - min_cx) / cell_w) + 1],
            'origin': [min_cx - cell_w // 2, min_cy - cell_h // 2],
            'cell_size': [cell_w, cell_h],
        }

    # ─────────── 2. 格子分类 ───────────

    def _classify_tiles(self, grid):
        print("\n[3/4] 格子分类")
        rows, cols = grid['size']
        ox, oy = grid['origin']
        cw, ch = grid['cell_size']

        counts = {'空': 0, '禁': 0, '锁': 0, '?': 0}
        tiles = []
        for r in range(rows):
            row = []
            for c in range(cols):
                x, y = ox + c * cw, oy + r * ch
                cell_gray = self.gray[y:y + ch, x:x + cw]
                cell_hsv = self.hsv[y:y + ch, x:x + cw]
                info = self._classify_cell(cell_gray, cell_hsv)
                row.append(info)
                key = {'empty': '空', 'disabled': '禁', 'lock': '锁'}.get(info['type'], '?')
                counts[key] += 1
            tiles.append(row)

        print(f"  {counts}")
        return tiles

    def _classify_cell(self, cell_gray, cell_hsv):
        """分类优先级：lock（亮+彩色） > disabled/empty（模板匹配竞争）"""
        ch, cw = cell_gray.shape

        # 从配置加载参数
        margin_ratio = self.config.get_float('tile_classification.inner_margin_ratio', 0.1)

        # 缩进比例，保留更多有效区域
        m = max(5, min(ch, cw) // 10)
        inner_margin = max(m, int(min(ch, cw) * margin_ratio))
        inner_hsv = cell_hsv[inner_margin:-inner_margin, inner_margin:-inner_margin]

        if inner_hsv.size == 0:
            return {'type': 'unknown'}

        sat = inner_hsv[:, :, 1].astype(float)
        val = inner_hsv[:, :, 2].astype(float)

        # ── 1. 锁定格：高饱和度 AND 高亮度 ──
        lock_min_sat = self.config.get_int('tile_classification.lock_min_saturation', 80)
        lock_min_val = self.config.get_int('tile_classification.lock_min_value', 80)
        lock_min_ratio = self.config.get_float('tile_classification.lock_min_pixel_ratio', 0.10)
        lock_min_avg_val = self.config.get_int('tile_classification.lock_min_avg_value', 100)
        lock_min_avg_sat = self.config.get_int('tile_classification.lock_min_avg_saturation', 100)

        lock_mask = (sat > lock_min_sat) & (val > lock_min_val)
        lock_ratio = np.sum(lock_mask) / lock_mask.size
        if lock_ratio > lock_min_ratio:
            px = inner_hsv[lock_mask]
            avg_h = int(np.mean(px[:, 0]))
            avg_s = int(np.mean(px[:, 1]))
            avg_v = int(np.mean(px[:, 2]))
            # 真正的锁定格颜色明显：平均亮度应该 > lock_min_avg_val
            if avg_v > lock_min_avg_val and avg_s > lock_min_avg_sat:
                grp = self.color_grouper.match_nearest(avg_h, avg_s, avg_v)
                return {
                    'type': 'lock',
                    'color': [avg_h, avg_s, avg_v],
                    'color_group': grp,
                }

        # ── 2. 禁用格 vs 空格：竞争匹配，选得分更高的 ──
        sc_dis = self._cell_score(cell_gray, 'tile_disable')
        sc_emp = self._cell_score(cell_gray, 'tile_empty')

        # 只要有一个模板匹配得不错就分类
        tile_min_score = self.config.get_float('tile_classification.tile_min_score', 0.35)
        best_score = max(sc_dis, sc_emp)
        if best_score > tile_min_score:
            if sc_dis >= sc_emp:
                return {'type': 'disabled'}
            else:
                return {'type': 'empty'}

        # ── 3. 兜底：用亮度和纹理 ──
        mean_v = float(np.mean(val))
        # 禁用格有对角条纹，灰度方差大于纯黑空格
        gray_inner = cell_gray[inner_margin:-inner_margin, inner_margin:-inner_margin]
        variance = float(np.var(gray_inner))

        empty_mean_v = self.config.get_int('tile_classification.empty_mean_value_threshold', 45)
        empty_var = self.config.get_int('tile_classification.empty_variance_threshold', 200)
        disabled_mean_v = self.config.get_int('tile_classification.disabled_mean_value_threshold', 60)
        disabled_var = self.config.get_int('tile_classification.disabled_variance_threshold', 100)

        if mean_v < empty_mean_v and variance < empty_var:
            return {'type': 'empty'}
        if mean_v < disabled_mean_v and variance > disabled_var:
            return {'type': 'disabled'}

        return {'type': 'unknown'}

    def _cell_score(self, cell_gray, tpl_key):
        """在格子区域内搜索原始尺寸模板（不 resize），返回最大匹配分"""
        tpl = self.templates.get(tpl_key)
        if tpl is None:
            return 0.0
        th, tw = tpl.shape
        ch, cw = cell_gray.shape
        # 格子区域必须大于模板
        if ch < th or cw < tw:
            return 0.0
        res = cv2.matchTemplate(cell_gray, tpl, cv2.TM_CCOEFF_NORMED)
        return float(res.max()) if res.size else 0.0

    # ─────────── 3. 行列需求 ───────────

    def _detect_requirements(self, grid):
        print("\n[4/4] 行列需求")
        ox, oy = grid['origin']
        cw, ch = grid['cell_size']
        rows, cols = grid['size']
        gw, gh = cw * cols, ch * rows

        # 从配置加载参数
        search_margin = self._scaled(self.config.get_int('requirement_detection.search_margin', 150))

        # 搜索范围扩大到 search_margin，但 mask 用 sat+val 联合阈值排除暗色背景辉光
        col_roi = self.image[max(0, oy - search_margin):oy - 2, ox:ox + gw]
        col_debug = col_roi.copy() if self.debug else None
        col_reqs = self._find_bars(col_roi, 'col', ox, cw, cols,
                                    (ox, max(0, oy - search_margin)),
                                    debug_img=col_debug, debug_mask_key='col')

        row_roi = self.image[oy:oy + gh, max(0, ox - search_margin):ox - 2]
        row_debug = row_roi.copy() if self.debug else None
        row_reqs = self._find_bars(row_roi, 'row', oy, ch, rows,
                                    (max(0, ox - search_margin), oy),
                                    debug_img=row_debug, debug_mask_key='row')

        ct = sum(len(c) for c in col_reqs)
        rt = sum(len(r) for r in row_reqs)
        print(f"  列: {cols}组 共{ct}条  行: {rows}组 共{rt}条")

        # debug: 标注后的 ROI（绿=filled, 红=unfilled）；sim_mask 由 _find_bars 保存
        if self.debug:
            if col_debug is not None and col_debug.size > 0:
                self._debug_save("3_col_bars.png", col_debug)
            if row_debug is not None and row_debug.size > 0:
                self._debug_save("3_row_bars.png", row_debug)

        return {'columns': col_reqs, 'rows': row_reqs}

    def _similarity_map_for_color(self, hsv, ch, cs, cv, hue_width=10, min_sat=50, min_val=55):
        """按元件颜色生成相似度图：越接近元件色越接近 1"""
        # 从配置加载参数
        hue_width = self.config.get_int('requirement_detection.hue_similarity_width', 10)
        min_sat = self.config.get_int('requirement_detection.min_saturation', 50)
        min_val = self.config.get_int('requirement_detection.min_value', 55)

        h_arr = hsv[:, :, 0].astype(np.float32)
        s_arr = hsv[:, :, 1].astype(np.float32)
        v_arr = hsv[:, :, 2].astype(np.float32)
        d = np.abs(h_arr - ch)
        d = np.minimum(d, 180 - d)
        hue_sim = np.maximum(0.0, 1.0 - d / hue_width)
        valid = (s_arr >= min_sat) & (v_arr >= min_val)
        return np.where(valid, hue_sim, 0.0).astype(np.float32)

    def _find_bars(self, roi_bgr, orient, grid_start, cell_sz, n_cells, region_off,
                   debug_img=None, debug_mask_key=None):
        if roi_bgr.size == 0:
            return [[] for _ in range(n_cells)]

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        reqs = [[] for _ in range(n_cells)]

        # 无元件颜色时返回空
        if not self.color_grouper.clusters:
            return reqs

        # 从配置加载参数
        min_area = self._scaled_area(self.config.get_int('requirement_detection.min_area', 15))
        max_area = self._scaled_area(self.config.get_int('requirement_detection.max_area', 800))
        max_aspect = self.config.get_float('requirement_detection.max_aspect_ratio', 6.0)
        filled_cv_thresh = self.config.get_float('requirement_detection.filled_cv_threshold', 0.35)
        min_region_size = self.config.get_int('requirement_detection.min_region_size', 5)

        # 按每种元件颜色分别计算相似度图并检测
        combined_mask = None
        for ci, (ch, cs, cv, _) in enumerate(self.color_grouper.clusters):
            grp = self.color_grouper.label(ci)
            sim_map = self._similarity_map_for_color(hsv, ch, cs, cv)
            sim_u8 = (np.clip(sim_map, 0, 1) * 255).astype(np.uint8)
            _, mask = cv2.threshold(sim_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if np.sum(mask > 0) < 10:
                continue
            if combined_mask is None:
                combined_mask = mask.copy()
            else:
                combined_mask = np.maximum(combined_mask, mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if area < min_area or area > max_area:
                    continue
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect > max_aspect:
                    continue

                bar_mask = mask[y:y + h, x:x + w]
                bar_hsv = hsv[y:y + h, x:x + w]
                bar_val = bar_hsv[:, :, 2]
                color_pixels = bar_hsv[bar_mask > 0]
                if len(color_pixels) < 5:
                    continue

                avg_h = int(np.mean(color_pixels[:, 0]))
                avg_s = int(np.mean(color_pixels[:, 1]))
                avg_v = int(np.mean(color_pixels[:, 2]))

                bh, bw = bar_val.shape
                val_cv = 0.0
                if bh >= min_region_size and bw >= min_region_size:
                    bar_val_f = bar_val.astype(float)
                    val_cv = float(np.std(bar_val_f) / max(np.mean(bar_val_f), 1))
                    filled = val_cv < filled_cv_thresh
                else:
                    filled = False

                if orient == 'col':
                    cx = region_off[0] + x + w // 2
                    idx = int(round((cx - grid_start - cell_sz / 2) / cell_sz))
                else:
                    cy = region_off[1] + y + h // 2
                    idx = int(round((cy - grid_start - cell_sz / 2) / cell_sz))

                if 0 <= idx < n_cells:
                    reqs[idx].append({
                        'color': [avg_h, avg_s, avg_v],
                        'color_group': grp,
                        'filled': filled,
                    })

                if debug_img is not None:
                    color = (0, 255, 0) if filled else (0, 0, 255)
                    cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(debug_img, f"{grp} cv{val_cv:.2f}", (x, y - 3),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        if debug_mask_key and combined_mask is not None and self.debug and self.debug_dir:
            self._debug_save(f"3_{debug_mask_key}_sim_mask.png", combined_mask)

        return reqs

    # ─────────── 4. 元件检测 ───────────

    def _detect_components_by_template(self, roi_gray, rx, y_limit):
        """使用 component_flame 模板匹配定位元件卡片边框"""
        raw_tpl = self.raw_templates.get('component_flame')
        if raw_tpl is None:
            return []

        # 从配置加载参数
        template_scales = self.config.get_list('component_detection.template_scales', [0.5, 0.6, 0.7, 0.8, 1.0])
        min_score = self.config.get_float('component_detection.template_match_min_score', 0.55)
        max_score = self.config.get_float('component_detection.template_match_max_score', 0.75)
        nms_iou = self.config.get_float('component_detection.template_nms_iou_threshold', 0.5)
        min_tpl_size = self._scaled(self.config.get_int('component_detection.min_template_size', 45))
        max_tpl_size = self._scaled(self.config.get_int('component_detection.max_template_size', 130))

        rh, rw = roi_gray.shape
        best_pts = []
        best_n = 0

        for comp_scale in template_scales:
            tpl = self._resize_template(raw_tpl, comp_scale * self.scale)
            th, tw = tpl.shape
            if th >= rh or tw >= rw:
                continue
            res = cv2.matchTemplate(roi_gray, tpl, cv2.TM_CCOEFF_NORMED)
            max_score_res = float(res.max()) if res.size else 0
            thresh = max(min_score, min(max_score, max_score_res - 0.05))
            pts = self._match_template(roi_gray, tpl, thresh=thresh)
            pts = [(x + rx, y, tw, th, float(s)) for x, y, _, _, s in pts]
            pts = [p for p in pts if p[1] < y_limit]  # 排除底部 UI
            pts = self._nms(pts, iou_thresh=nms_iou)
            pts = [p for p in pts if min_tpl_size < p[2] < max_tpl_size and min_tpl_size < p[3] < max_tpl_size]
            if len(pts) > best_n and 1 <= len(pts) <= 8:
                best_n = len(pts)
                best_pts = pts

        if not best_pts:
            return []
        return [(x, y, w, h, w * h) for x, y, w, h, _ in best_pts]

    def _detect_components_by_blob(self, r_hsv, rx, y_limit):
        """原有 HSV blob 检测（兜底），排除 y >= y_limit 的底部 UI"""
        # 从配置加载参数
        blob_min_sat = self.config.get_int('component_detection.blob_min_saturation', 60)
        blob_min_val = self.config.get_int('component_detection.blob_min_value', 60)
        blob_min_area = self._scaled_area(self.config.get_int('component_detection.blob_min_area', 500))
        blob_min_size = self._scaled(self.config.get_int('component_detection.blob_min_size', 30))
        blob_max_size = self._scaled(self.config.get_int('component_detection.blob_max_size', 200))
        blob_max_aspect = self.config.get_float('component_detection.blob_max_aspect_ratio', 4.0)

        raw_mask = ((r_hsv[:, :, 1] > blob_min_sat) & (r_hsv[:, :, 2] > blob_min_val)).astype(np.uint8)
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        blobs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if y >= y_limit:
                continue  # 排除底部 UI（机器人、修门等）
            area = cv2.contourArea(cnt)
            if area > blob_min_area and blob_min_size < w < blob_max_size and blob_min_size < h < blob_max_size:
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect < blob_max_aspect:
                    blobs.append((x + rx, y, w, h, area))
        return blobs

    def _detect_components(self, grid):
        print("\n[2/4] 元件检测")
        ih, iw = self.image.shape[:2]

        # 从配置加载参数
        search_start_ratio = self.config.get_float('component_detection.search_start_x_ratio', 0.6)
        rx = int(iw * search_start_ratio)
        roi_gray = self.gray[:, rx:]
        r_hsv = self.hsv[:, rx:]

        # 元件区 y 轴上限：排除底部 UI（机器人、修门等）
        ox, oy = grid['origin']
        rows, cols = grid['size']
        cw, ch = grid['cell_size']
        grid_bottom = oy + rows * ch
        y_limit_offset = self._scaled(self.config.get_int('component_detection.y_limit_offset', 150))
        y_limit = grid_bottom + y_limit_offset

        # 优先使用 component_flame 模板匹配（元件卡片边框）
        blobs = self._detect_components_by_template(roi_gray, rx, y_limit)
        if len(blobs) < 1:
            # 兜底：原有 HSV blob 检测
            blobs = self._detect_components_by_blob(r_hsv, rx, y_limit)
            print(f"  模板匹配无结果，回退 blob 检测: {len(blobs)}")

        blobs.sort(key=lambda b: (b[1], b[0]))
        print(f"  元件 blob: {len(blobs)}")

        # debug: 右侧区域 mask 和检测到的 blob
        if self.debug:
            blob_min_sat = self.config.get_int('component_detection.blob_min_saturation', 60)
            blob_min_val = self.config.get_int('component_detection.blob_min_value', 60)
            raw_mask = ((r_hsv[:, :, 1] > blob_min_sat) & (r_hsv[:, :, 2] > blob_min_val)).astype(np.uint8)
            self._debug_save("4_comp_mask.png", raw_mask * 255)
            dbg = self.image.copy()
            for x, y, w, h, area in blobs:
                cv2.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(dbg, f"a={area}", (x, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            self._debug_save("4_comp_blobs.png", dbg)

        comps = []
        for i, (x, y, w, h, area) in enumerate(blobs):
            info = self._parse_component_blob(x, y, w, h)
            if info:
                comps.append(info)
                # debug: 每个元件的 mask
                if self.debug:
                    blob_hsv = self.hsv[y:y + h, x:x + w]
                    blob_min_sat = self.config.get_int('component_detection.blob_min_saturation', 60)
                    blob_min_val = self.config.get_int('component_detection.blob_min_value', 60)
                    blob_mask = ((blob_hsv[:, :, 1] > blob_min_sat) &
                                 (blob_hsv[:, :, 2] > blob_min_val)).astype(np.uint8) * 255
                    self._debug_save(f"4_comp_{i}_mask.png", blob_mask)

        print(f"  解析: {len(comps)}")
        return comps

    def _parse_component_blob(self, x, y, w, h):
        """对单个元件 blob，用网格搜索推断最佳形状矩阵"""
        # 从配置加载参数
        blob_min_sat = self.config.get_int('component_detection.blob_min_saturation', 60)
        blob_min_val = self.config.get_int('component_detection.blob_min_value', 60)
        min_shape_size = self._scaled(self.config.get_int('component_detection.min_shape_size', 10))
        max_shape_size = self._scaled(self.config.get_int('component_detection.max_shape_size', 45))
        min_total_cells = self.config.get_int('component_detection.min_total_cells', 2)
        max_total_cells = self.config.get_int('component_detection.max_total_cells', 12)
        min_rows = self.config.get_int('component_detection.min_rows', 1)
        max_rows = self.config.get_int('component_detection.max_rows', 4)
        min_cols = self.config.get_int('component_detection.min_cols', 1)
        max_cols = self.config.get_int('component_detection.max_cols', 4)
        cell_fill_thresh = self.config.get_float('component_detection.cell_fill_threshold', 0.40)
        clear_filled_thresh = self.config.get_float('component_detection.clear_filled_threshold', 0.60)
        clear_empty_thresh = self.config.get_float('component_detection.clear_empty_threshold', 0.15)
        min_color_pixels = self._scaled_area(self.config.get_int('component_detection.min_color_pixels', 30))
        min_avg_val = self.config.get_int('component_detection.min_avg_value', 80)

        blob_hsv = self.hsv[y:y + h, x:x + w]
        blob_mask = ((blob_hsv[:, :, 1] > blob_min_sat) &
                     (blob_hsv[:, :, 2] > blob_min_val)).astype(np.uint8)

        # ── 1. 裁去边框辉光：找到内容边界 ──
        h_proj = np.sum(blob_mask, axis=1)
        v_proj = np.sum(blob_mask, axis=0)
        h_thresh = max(h_proj) * 0.15 if max(h_proj) > 0 else 1
        v_thresh = max(v_proj) * 0.15 if max(v_proj) > 0 else 1

        r_start = next((i for i in range(len(h_proj)) if h_proj[i] > h_thresh), 0)
        r_end = next((i for i in range(len(h_proj) - 1, -1, -1)
                      if h_proj[i] > h_thresh), len(h_proj) - 1) + 1
        c_start = next((i for i in range(len(v_proj)) if v_proj[i] > v_thresh), 0)
        c_end = next((i for i in range(len(v_proj) - 1, -1, -1)
                      if v_proj[i] > v_thresh), len(v_proj) - 1) + 1

        trimmed = blob_mask[r_start:r_end, c_start:c_end]
        th, tw = trimmed.shape
        if th < min_shape_size or tw < min_shape_size:
            return None

        # ── 2. 尝试不同网格尺寸，选择最清晰分离的 ──
        best_shape = None
        best_score = -999

        for nr in range(min_rows, max_rows + 1):
            for nc in range(min_cols, max_cols + 1):
                total = nr * nc
                if total < min_total_cells or total > max_total_cells:
                    continue

                cell_h = th / nr
                cell_w = tw / nc

                # 格子尺寸不能太小也不能太大
                if cell_h < min_shape_size or cell_w < min_shape_size:
                    continue
                if cell_h > max_shape_size or cell_w > max_shape_size:
                    continue

                shape = []
                fills = []
                for r in range(nr):
                    row = []
                    for c in range(nc):
                        y1 = int(r * cell_h)
                        y2 = int((r + 1) * cell_h)
                        x1 = int(c * cell_w)
                        x2 = int((c + 1) * cell_w)
                        cell = trimmed[y1:y2, x1:x2]
                        fill = float(np.sum(cell) / cell.size) if cell.size > 0 else 0.0
                        fills.append(fill)
                        row.append(1 if fill > cell_fill_thresh else 0)
                    shape.append(row)

                filled = sum(sum(r) for r in shape)
                if filled == 0 or filled == total:
                    continue  # 全空或全满没意义

                # 评分：综合清晰度 + 格子方形度 + 复杂度惩罚
                n_clear_filled = sum(1 for f in fills if f > clear_filled_thresh)
                n_clear_empty = sum(1 for f in fills if f < clear_empty_thresh)
                n_ambiguous = total - n_clear_filled - n_clear_empty

                # 清晰度（归一化）
                clarity = (n_clear_filled + n_clear_empty) / total
                # 格子方形度（1.0=正方形，0=极细长）
                squareness = min(cell_h, cell_w) / max(cell_h, cell_w)
                # 总分：清晰度高 + 格子越方正越好 + 模糊格子重罚
                score = clarity * 10 + squareness * 5 - n_ambiguous * 8

                if score > best_score:
                    best_score = score
                    best_shape = shape

        if best_shape is None:
            # 无法找到好的网格，当作 1×1
            best_shape = [[1]]

        # ── 3. 提取颜色 ──
        px_mask = blob_mask.astype(bool)
        if np.sum(px_mask) < min_color_pixels:
            return None

        px = blob_hsv[px_mask]
        avg_h = int(np.mean(px[:, 0]))
        avg_s = int(np.mean(px[:, 1]))
        avg_v = int(np.mean(px[:, 2]))

        if avg_v < min_avg_val:
            return None

        group_idx = self.color_grouper.register(avg_h, avg_s, avg_v)

        return {
            'shape': best_shape,
            'color': [avg_h, avg_s, avg_v],
            'color_group': self.color_grouper.label(group_idx),
            'slot_pos': [x, y, w, h],
        }

    # ─────────── 可视化 ───────────

    def visualize(self, result, output_path):
        vis = self.image.copy()
        ox, oy = result['grid_origin_px']
        cw, ch = result['cell_size_px']
        rows, cols = result['grid_size']

        # 网格外框（OpenCV 画，PIL 不好画粗线）
        cv2.rectangle(vis, (ox, oy), (ox + cw * cols, oy + ch * rows), (0, 255, 0), 3)

        # 转 PIL 绘中文
        pil = cv2_to_pil(vis)
        draw = ImageDraw.Draw(pil)
        font = get_font(20)
        font_sm = get_font(16)
        font_lg = get_font(24)

        # 网格标签放在列需求文本上方
        max_col_groups = max((len(set(b.get('color_group', '?') for b in bars))
                              for bars in result['requirements']['columns'] if bars),
                             default=0)
        label_offset = 30 + max_col_groups * 18
        draw.text((ox, oy - label_offset), f"网格 {rows}×{cols}", fill=(0, 255, 0), font=font_lg)

        # 格子
        tiles = result['tiles']
        for r in range(rows):
            for c in range(cols):
                x, y = ox + c * cw, oy + r * ch
                tile = tiles[r][c]
                t = tile['type']

                if t == 'empty':
                    rgb = (100, 180, 255)
                    label = "空"
                elif t == 'disabled':
                    rgb = (160, 160, 160)
                    label = "禁"
                elif t == 'lock':
                    h_v, s_v, v_v = tile.get('color', [0, 0, 0])
                    rgb = hsv_to_rgb(h_v, max(s_v, 150), max(v_v, 150))  # 提亮以便看清
                    grp = tile.get('color_group', '?')
                    label = f"锁{grp}"
                else:
                    rgb = (255, 80, 80)
                    label = "?"

                draw.rectangle([(x + 1, y + 1), (x + cw - 1, y + ch - 1)], outline=rgb, width=2)
                draw.text((x + 4, y + 2), label, fill=rgb, font=font_sm)

        # 列需求（文本形式：颜色组 已满足/总数）
        for c, bars in enumerate(result['requirements']['columns']):
            if not bars:
                continue
            # 按颜色组分组统计
            groups = {}
            for bar in bars:
                g = bar.get('color_group', '?')
                if g not in groups:
                    groups[g] = {'total': 0, 'filled': 0, 'color': bar.get('color', [0, 0, 0])}
                groups[g]['total'] += 1
                if bar.get('filled', False):
                    groups[g]['filled'] += 1
            # 画文本
            lines = []
            for g, info in groups.items():
                h_v, s_v, v_v = info['color']
                rgb = hsv_to_rgb(h_v, max(s_v, 150), max(v_v, 150))
                lines.append((f"{g}:{info['filled']}/{info['total']}", rgb))
            cx = ox + c * cw + 2
            for li, (txt, rgb) in enumerate(lines):
                draw.text((cx, oy - 20 - li * 16), txt, fill=rgb, font=font_sm)

        # 行需求（文本形式：颜色组 已满足/总数）
        for r, bars in enumerate(result['requirements']['rows']):
            if not bars:
                continue
            groups = {}
            for bar in bars:
                g = bar.get('color_group', '?')
                if g not in groups:
                    groups[g] = {'total': 0, 'filled': 0, 'color': bar.get('color', [0, 0, 0])}
                groups[g]['total'] += 1
                if bar.get('filled', False):
                    groups[g]['filled'] += 1
            lines = []
            for g, info in groups.items():
                h_v, s_v, v_v = info['color']
                rgb = hsv_to_rgb(h_v, max(s_v, 150), max(v_v, 150))
                lines.append((f"{g}:{info['filled']}/{info['total']}", rgb))
            txt_full = " ".join(t for t, _ in lines)
            # 取第一个颜色作为文本颜色（混色时用白色）
            txt_rgb = lines[0][1] if len(lines) == 1 else (255, 255, 255)
            # 多颜色组时逐个绘制
            cy = oy + r * ch + ch // 2 - 8
            cur_x = ox - 8
            for txt, rgb in reversed(lines):
                tw = draw.textlength(txt, font=font_sm)
                cur_x -= int(tw) + 4
                draw.text((cur_x, cy), txt, fill=rgb, font=font_sm)

        # 元件（按形状矩阵画各个格子）
        for i, comp in enumerate(result['components']):
            sx, sy, sw, sh = comp['slot_pos']
            grp = comp.get('color_group', '?')
            shape = comp.get('shape', [])
            n_tiles = sum(sum(r) for r in shape)
            h_v, s_v, v_v = comp.get('color', [0, 0, 0])
            rgb = hsv_to_rgb(h_v, max(s_v, 150), max(v_v, 150))
            # 半透明填充色（用稍暗的颜色模拟）
            fill_rgb = tuple(max(0, c // 2) for c in rgb)

            n_rows = len(shape)
            n_cols = max((len(r) for r in shape), default=1)

            # 在 slot 区域上方留空标注文字，形状画在 slot 下方偏移
            label_y = sy - 20
            draw.text((sx, label_y), f"元件{i + 1} 颜色{grp} {n_tiles}格",
                       fill=rgb, font=font_sm)

            # 计算绘制区域：在 slot 位置画形状
            # 使用 slot 的尺寸来确定格子大小
            cell_draw_w = sw // n_cols if n_cols > 0 else sw
            cell_draw_h = sh // n_rows if n_rows > 0 else sh
            # 保持方形
            cell_draw = min(cell_draw_w, cell_draw_h)
            cell_draw = max(cell_draw, 12)

            for r in range(n_rows):
                for c in range(n_cols):
                    if c < len(shape[r]) and shape[r][c] == 1:
                        cx = sx + c * cell_draw
                        cy = sy + r * cell_draw
                        draw.rectangle([(cx, cy), (cx + cell_draw - 2, cy + cell_draw - 2)],
                                       fill=fill_rgb, outline=rgb, width=2)

        # 颜色图例（右下角）
        cg = result.get('color_groups', [])
        if cg:
            ly = oy + ch * rows + 10
            draw.text((ox, ly), "颜色组:", fill=(255, 255, 255), font=font)
            for ci, g in enumerate(cg):
                gh, gs, gv = g['hsv']
                rgb = hsv_to_rgb(gh, max(gs, 150), max(gv, 150))
                lbl = g['label']
                cnt = g['count']
                draw.rectangle([(ox + 80 + ci * 100, ly), (ox + 95 + ci * 100, ly + 15)],
                                fill=rgb)
                draw.text((ox + 100 + ci * 100, ly), f"{lbl}(×{cnt})",
                           fill=rgb, font=font_sm)

        vis = pil_to_cv2(pil)
        cv2.imwrite(str(output_path), vis)
        print(f"[保存] {output_path}")


# ──────────────────────── 主函数 ────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="终末地源石电路模块解谜 - UI 识别原型")
    parser.add_argument("image", help="游戏截图路径")
    parser.add_argument("--debug", action="store_true", help="输出中间调试图像")
    parser.add_argument("--scale", type=float, default=None,
                        help="手动指定 UI 缩放系数（不指定则自动探测）")
    parser.add_argument("--config", type=str, default=None,
                        help="指定配置文件路径（默认使用 image/config.toml）")
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"错误: 不存在 - {image_path}")
        sys.exit(1)

    output_dir = Path(__file__).parent / "image" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    debug_dir = output_dir / f"{stem}_debug" if args.debug else None

    if args.scale is not None and args.scale <= 0:
        print("错误: --scale 必须大于 0")
        sys.exit(1)

    # 加载配置
    config = None
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"错误: 配置文件不存在 - {config_path}")
            sys.exit(1)
        config = Config(config_path)
        print(f"[配置] 使用自定义配置: {config_path}")
    else:
        # 使用默认配置
        config = load_default_config()
        print(f"[配置] 使用默认配置: {config._config_path}")

    template_dir = Path(__file__).parent / "image" / "ui-template"
    detector = PuzzleDetector(template_dir, debug=args.debug, debug_dir=debug_dir,
                               scale=args.scale, config=config)
    result = detector.detect(image_path)

    def to_json(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_json(i) for i in obj]
        return obj

    jpath = output_dir / f"{stem}_result.json"
    with open(jpath, 'w', encoding='utf-8') as f:
        json.dump(to_json(result), f, indent=2, ensure_ascii=False)
    print(f"\n[JSON] {jpath}")

    vpath = output_dir / f"{stem}_annotated.png"
    detector.visualize(result, vpath)
    print("[完成]")


if __name__ == '__main__':
    main()
