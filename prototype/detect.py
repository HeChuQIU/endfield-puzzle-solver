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

    HUE_MERGE_DIST = 18  # Hue 差值小于此阈值视为同组

    def __init__(self):
        self.clusters = []  # [(center_hue, avg_s, avg_v, [hue_samples])]

    def register(self, h, s, v):
        """注册一个颜色，返回其所属组的索引"""
        for i, (ch, _, _, samples) in enumerate(self.clusters):
            if hue_distance(h, ch) < self.HUE_MERGE_DIST:
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

    def color_hsv(self, idx):
        """返回组的平均 HSV"""
        if 0 <= idx < len(self.clusters):
            h, s, v, _ = self.clusters[idx]
            return (h, s, v)
        return (0, 0, 0)


# ──────────────────────── 检测器 ────────────────────────

class PuzzleDetector:

    def __init__(self, template_dir, debug=False, debug_dir=None):
        self.template_dir = Path(template_dir)
        self.templates = {}
        self.debug = debug
        self.debug_dir = Path(debug_dir) if debug_dir else None
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
                self.templates[key] = gray
                print(f"[模板] {name}  {gray.shape}")

    # ─────────── 主流程 ───────────

    def detect(self, image_path):
        self.image = cv2.imread(str(image_path))
        if self.image is None:
            raise ValueError(f"无法读取: {image_path}")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        self.color_grouper = ColorGrouper()

        print(f"\n[图像] {Path(image_path).name}  {self.image.shape[1]}×{self.image.shape[0]}")

        grid = self._detect_grid()
        tiles = self._classify_tiles(grid)
        reqs = self._detect_requirements(grid)
        comps = self._detect_components()

        # 输出颜色组信息
        n_groups = len(self.color_grouper.clusters)
        print(f"\n[颜色] 共识别 {n_groups} 种颜色")
        for i in range(n_groups):
            lbl = self.color_grouper.label(i)
            h, s, v = self.color_grouper.color_hsv(i)
            n = len(self.color_grouper.clusters[i][3])
            print(f"  颜色{lbl}: HSV=({h},{s},{v})  采样{n}次")

        return {
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
        roi = self.gray[h // 6:5 * h // 6, w // 6:2 * w // 3]
        ox_off, oy_off = w // 6, h // 6

        all_pts = []
        for tpl_key in ('tile_empty', 'tile_disable'):
            tpl = self.templates.get(tpl_key)
            if tpl is None:
                continue
            pts = self._match_template(roi, tpl, thresh=0.80)
            pts = [(x + ox_off, y + oy_off, tw, th, s) for x, y, tw, th, s in pts]
            all_pts.extend(pts)
            print(f"  {tpl_key}: {len(pts)} 匹配")

        if len(all_pts) < 4:
            raise ValueError("格子匹配数不足")

        all_pts = self._nms(all_pts, iou_thresh=0.5)
        all_pts = [p for p in all_pts if 65 < p[2] < 100 and 65 < p[3] < 100]
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

        # 用排序后的相邻间距的中位数来估算格子尺寸
        xs = sorted(set(centers[:, 0]))
        ys = sorted(set(centers[:, 1]))
        dxs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1) if xs[i + 1] - xs[i] > 30]
        dys = [ys[i + 1] - ys[i] for i in range(len(ys) - 1) if ys[i + 1] - ys[i] > 30]

        cell_w = int(np.median(dxs)) if dxs else 87
        cell_h = int(np.median(dys)) if dys else 87

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
        print("\n[2/4] 格子分类")
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
        # 缩进 10%，保留更多有效区域
        m = max(5, min(ch, cw) // 10)
        inner_hsv = cell_hsv[m:-m, m:-m]
        if inner_hsv.size == 0:
            return {'type': 'unknown'}

        sat = inner_hsv[:, :, 1].astype(float)
        val = inner_hsv[:, :, 2].astype(float)

        # ── 1. 锁定格：高饱和度 AND 高亮度 ──
        lock_mask = (sat > 80) & (val > 80)
        lock_ratio = np.sum(lock_mask) / lock_mask.size
        if lock_ratio > 0.10:
            px = inner_hsv[lock_mask]
            avg_h = int(np.mean(px[:, 0]))
            avg_s = int(np.mean(px[:, 1]))
            avg_v = int(np.mean(px[:, 2]))
            # 真正的锁定格颜色明显：平均亮度应该 > 100
            if avg_v > 100 and avg_s > 100:
                group_idx = self.color_grouper.register(avg_h, avg_s, avg_v)
                return {
                    'type': 'lock',
                    'color': [avg_h, avg_s, avg_v],
                    'color_group': self.color_grouper.label(group_idx),
                }

        # ── 2. 禁用格 vs 空格：竞争匹配，选得分更高的 ──
        sc_dis = self._cell_score(cell_gray, 'tile_disable')
        sc_emp = self._cell_score(cell_gray, 'tile_empty')

        # 只要有一个模板匹配得不错就分类
        best_score = max(sc_dis, sc_emp)
        if best_score > 0.35:
            if sc_dis >= sc_emp:
                return {'type': 'disabled'}
            else:
                return {'type': 'empty'}

        # ── 3. 兜底：用亮度和纹理 ──
        mean_v = float(np.mean(val))
        # 禁用格有对角条纹，灰度方差大于纯黑空格
        gray_inner = cell_gray[m:-m, m:-m]
        variance = float(np.var(gray_inner))
        if mean_v < 45 and variance < 200:
            return {'type': 'empty'}
        if mean_v < 60 and variance > 100:
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
        print("\n[3/4] 行列需求")
        ox, oy = grid['origin']
        cw, ch = grid['cell_size']
        rows, cols = grid['size']
        gw, gh = cw * cols, ch * rows

        # 搜索范围扩大到 150px，但 mask 用 sat+val 联合阈值排除暗色背景辉光
        margin = 150
        col_roi = self.image[max(0, oy - margin):oy - 2, ox:ox + gw]
        col_debug = col_roi.copy() if self.debug else None
        col_reqs = self._find_bars(col_roi, 'col', ox, cw, cols,
                                    (ox, max(0, oy - margin)),
                                    debug_img=col_debug)

        row_roi = self.image[oy:oy + gh, max(0, ox - margin):ox - 2]
        row_debug = row_roi.copy() if self.debug else None
        row_reqs = self._find_bars(row_roi, 'row', oy, ch, rows,
                                    (max(0, ox - margin), oy),
                                    debug_img=row_debug)

        ct = sum(len(c) for c in col_reqs)
        rt = sum(len(r) for r in row_reqs)
        print(f"  列: {cols}组 共{ct}条  行: {rows}组 共{rt}条")

        # debug: 标注后的 ROI（绿=filled, 红=unfilled）+ Otsu mask
        if self.debug:
            if col_debug is not None and col_debug.size > 0:
                self._debug_save("3_col_bars.png", col_debug)
                col_hsv = cv2.cvtColor(col_roi, cv2.COLOR_BGR2HSV)
                _, col_otsu = cv2.threshold(col_hsv[:, :, 1], 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self._debug_save("3_col_otsu_mask.png", col_otsu)
            if row_debug is not None and row_debug.size > 0:
                self._debug_save("3_row_bars.png", row_debug)
                row_hsv = cv2.cvtColor(row_roi, cv2.COLOR_BGR2HSV)
                _, row_otsu = cv2.threshold(row_hsv[:, :, 1], 0, 255,
                                            cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                self._debug_save("3_row_otsu_mask.png", row_otsu)

        return {'columns': col_reqs, 'rows': row_reqs}

    def _find_bars(self, roi_bgr, orient, grid_start, cell_sz, n_cells, region_off,
                   debug_img=None):
        if roi_bgr.size == 0:
            return [[] for _ in range(n_cells)]

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        # ── 自适应前景分离（替代固定阈值）──
        # Otsu 在饱和度通道自动分离彩色前景（bars）和模糊背景
        otsu_thresh, sat_binary = cv2.threshold(sat, 0, 255,
                                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (sat_binary > 0).astype(np.uint8)

        # 安全检查：Otsu 前景过多说明分布不够双峰，用百分位回退
        fg_ratio = np.sum(mask) / mask.size
        if fg_ratio > 0.25:
            otsu_thresh = float(np.percentile(sat, 96))
            mask = (sat > otsu_thresh).astype(np.uint8)

        # 背景亮度基准：用 mask 外的像素中位数
        bg_pixels = val[mask == 0]
        bg_val = float(np.median(bg_pixels)) if bg_pixels.size > 100 else 0.0

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        reqs = [[] for _ in range(n_cells)]
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w * h < 15 or w * h > 800:
                continue
            # 需求条形状：不能太方也不能太细长
            aspect = max(w, h) / max(min(w, h), 1)
            if aspect > 6:
                continue

            # 颜色提取：用 mask 内的像素（与检测一致）
            bar_mask = mask[y:y + h, x:x + w]
            bar_hsv = hsv[y:y + h, x:x + w]
            bar_val = bar_hsv[:, :, 2]
            color_pixels = bar_hsv[bar_mask > 0]
            if len(color_pixels) < 5:
                continue

            avg_h = int(np.mean(color_pixels[:, 0]))
            avg_s = int(np.mean(color_pixels[:, 1]))
            avg_v = int(np.mean(color_pixels[:, 2]))

            # ── filled 判断：亮度变异系数（完全自适应，无绝对阈值）──
            # 实心条：内部亮度均匀 → CV 低 (< 0.35)
            # 空心条：边框亮 + 中心暗 → CV 高 (> 0.4)
            bh, bw = bar_val.shape
            val_cv = 0.0
            if bh >= 5 and bw >= 5:
                bar_val_f = bar_val.astype(float)
                val_cv = float(np.std(bar_val_f) / max(np.mean(bar_val_f), 1))
                filled = val_cv < 0.35
            else:
                filled = False

            group_idx = self.color_grouper.register(avg_h, avg_s, avg_v)

            if orient == 'col':
                cx = region_off[0] + x + w // 2
                idx = int(round((cx - grid_start - cell_sz / 2) / cell_sz))
            else:
                cy = region_off[1] + y + h // 2
                idx = int(round((cy - grid_start - cell_sz / 2) / cell_sz))

            if 0 <= idx < n_cells:
                reqs[idx].append({
                    'color': [avg_h, avg_s, avg_v],
                    'color_group': self.color_grouper.label(group_idx),
                    'filled': filled,
                })

            # debug: 在 ROI 图上标注每个 bar
            if debug_img is not None:
                color = (0, 255, 0) if filled else (0, 0, 255)
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), color, 2)
                label = f"cv{val_cv:.2f}"
                cv2.putText(debug_img, label, (x, y - 3),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return reqs

    # ─────────── 4. 元件检测 ───────────

    def _detect_components(self):
        print("\n[4/4] 元件检测")
        ih, iw = self.image.shape[:2]
        rx = 3 * iw // 5
        r_hsv = self.hsv[:, rx:]

        # 原始 mask：高饱和+高亮度
        raw_mask = ((r_hsv[:, :, 1] > 60) & (r_hsv[:, :, 2] > 60)).astype(np.uint8)

        # 找连通域（每个元件是一个大 blob）
        contours, _ = cv2.findContours(raw_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)

        blobs = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = cv2.contourArea(cnt)
            # 过滤：面积适中，宽高合理（排除细长线条和噪点）
            if area > 500 and 30 < w < 200 and 30 < h < 200:
                aspect = max(w, h) / max(min(w, h), 1)
                if aspect < 4:  # 不要太狭长
                    blobs.append((x + rx, y, w, h, area))

        blobs.sort(key=lambda b: (b[1], b[0]))
        print(f"  元件 blob: {len(blobs)}")

        # debug: 右侧区域 mask 和检测到的 blob
        if self.debug:
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
                    blob_mask = ((blob_hsv[:, :, 1] > 60) &
                                 (blob_hsv[:, :, 2] > 60)).astype(np.uint8) * 255
                    self._debug_save(f"4_comp_{i}_mask.png", blob_mask)

        print(f"  解析: {len(comps)}")
        return comps

    def _parse_component_blob(self, x, y, w, h):
        """对单个元件 blob，用网格搜索推断最佳形状矩阵"""
        blob_hsv = self.hsv[y:y + h, x:x + w]
        blob_mask = ((blob_hsv[:, :, 1] > 60) &
                     (blob_hsv[:, :, 2] > 60)).astype(np.uint8)

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
        if th < 10 or tw < 10:
            return None

        # ── 2. 尝试不同网格尺寸，选择最清晰分离的 ──
        best_shape = None
        best_score = -999

        for nr in range(1, 5):
            for nc in range(1, 5):
                total = nr * nc
                if total < 2 or total > 12:
                    continue

                cell_h = th / nr
                cell_w = tw / nc

                # 格子尺寸不能太小也不能太大
                if cell_h < 12 or cell_w < 12:
                    continue
                if cell_h > 45 or cell_w > 45:
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
                        row.append(1 if fill > 0.40 else 0)
                    shape.append(row)

                filled = sum(sum(r) for r in shape)
                if filled == 0 or filled == total:
                    continue  # 全空或全满没意义

                # 评分：综合清晰度 + 格子方形度 + 复杂度惩罚
                n_clear_filled = sum(1 for f in fills if f > 0.60)
                n_clear_empty = sum(1 for f in fills if f < 0.15)
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
        if np.sum(px_mask) < 30:
            return None

        px = blob_hsv[px_mask]
        avg_h = int(np.mean(px[:, 0]))
        avg_s = int(np.mean(px[:, 1]))
        avg_v = int(np.mean(px[:, 2]))

        if avg_v < 80:
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
    args = parser.parse_args()

    image_path = args.image
    if not os.path.exists(image_path):
        print(f"错误: 不存在 - {image_path}")
        sys.exit(1)

    output_dir = Path(__file__).parent / "image" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem

    debug_dir = output_dir / f"{stem}_debug" if args.debug else None

    template_dir = Path(__file__).parent / "image" / "ui-template"
    detector = PuzzleDetector(template_dir, debug=args.debug, debug_dir=debug_dir)
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
