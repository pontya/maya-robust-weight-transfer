# -*- coding: utf-8 -*-
"""
Robust Skin Weights Transfer for Maya  (1ファイル版 / Python 2.7・3 両対応)
=========================================================================
Abdrashitov et al., "Robust Skin Weights Transfer via Weight Inpainting"
(SIGGRAPH Asia 2023 Technical Communications) を論文ベースで独自実装したもの。

  1. Closest point match : ソース表面の最近接点からウェイトを取得。
     距離・法線角度の条件を満たす頂点だけを「確定(known)」とする。
  2. Weight inpainting   : 残りの頂点は Q = -L + L M^-1 L (L: cotan Laplacian,
     M: 質量行列) の二次エネルギー最小化で補間する。

1ファイル版の特徴:
  - このファイルを Documents/maya/scripts に置くだけで動く(プラグイン不要)。
  - Undo には対応しない(転写すると Undo 履歴はクリアされる)。
  - 計算に numpy / scipy を使う。
      Maya が Python 3 で numpy / scipy を読める   -> Maya の中で計算
      Maya 2022 の Python 2 モードなど、読めない   -> Maya 付属の Python 3 (mayapy)
                                                       を別プロセスで起動して計算
    numpy / scipy は Python 3 の mayapy に pip で入れる:
      "C:\\Program Files\\Autodesk\\Maya2022\\bin\\mayapy.exe" -m pip install --user numpy scipy

使い方:
    import robust_weight_transfer as rwt
    rwt.show()                       # UI
    rwt.transfer("base", "cloth")    # スクリプトから
"""
from __future__ import division, print_function, absolute_import

import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from array import array

_WORKER_FLAG = "--rwt-worker"
_IN_WORKER = _WORKER_FLAG in sys.argv

try:
    import numpy as np
    HAS_NUMPY = True
except Exception:  # Maya 2022 の Python 2 モードなど
    np = None
    HAS_NUMPY = False

try:  # あれば直接法で高速に解く
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    HAS_SCIPY = True
except Exception:
    _sp = _spla = None
    HAS_SCIPY = False

if not _IN_WORKER:
    import maya.cmds as cmds
    import maya.api.OpenMaya as om
    import maya.api.OpenMayaAnim as oma
else:  # 計算用の別プロセス(Maya を読み込まない)
    cmds = om = oma = None

PY2 = sys.version_info[0] == 2
if PY2:  # pragma: no cover
    text_type = unicode  # noqa: F821
else:
    text_type = str

__version__ = "3.1.0"
EDITION = u"1ファイル版"
WINDOW = "robustWeightTransferWin"
_last_inpainted = {"mesh": None, "ids": [], "unsolved": []}
LAYER_CLOSEST = "RWT_Closest"
LAYER_INPAINT = "RWT_Inpaint"


def _err_text(e):
    """例外を表示用の文字列に(Python 2 で日本語のメッセージでも落ちないように)。"""
    try:
        return text_type(e)
    except Exception:
        try:
            return str(e).decode("utf-8", "replace")
        except Exception:
            return repr(e)


def _print(msg):
    try:
        print(u"[RobustWT] " + msg)
    except Exception:
        pass


# ==========================================================================
# 計算部分(numpy / scipy)。Maya の中、または別プロセスの mayapy で動く
# ==========================================================================
def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


def _weld(pts, tol):
    q = np.round(pts / tol).astype(np.int64)
    _, inv = np.unique(q, axis=0, return_inverse=True)
    return inv.reshape(-1)


def _cotan_laplacian(pts, tris, clamp=False):
    """L (正半定値, 行和0) を COO で、質量 M(ボロノイ混合面積)を対角で返す。

    公式実装(libigl: L = 2*cotmatrix, massmatrix VORONOI)と同じ値になる。
    clamp=True で負の cot を 0 に丸める(破綻したメッシュ向けの安全策)。
    """
    nv = len(pts)
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area2 = np.linalg.norm(cross, axis=1)
    good = area2 > 1e-12
    tris, a, b, c, area2 = tris[good], a[good], b[good], c[good], area2[good]

    def dot(u, v):
        return np.einsum("ij,ij->i", u, v)

    cot0 = dot(b - a, c - a) / area2  # 頂点0の角 -> 辺(1,2)
    cot1 = dot(c - b, a - b) / area2  # 頂点1の角 -> 辺(2,0)
    cot2 = dot(a - c, b - c) / area2  # 頂点2の角 -> 辺(0,1)
    cots = np.stack([cot0, cot1, cot2], 1)
    if clamp:
        cots = np.clip(cots, 0.0, 1e3)

    i = np.concatenate([tris[:, 1], tris[:, 2], tris[:, 0]])
    j = np.concatenate([tris[:, 2], tris[:, 0], tris[:, 1]])
    w = np.concatenate([cots[:, 0], cots[:, 1], cots[:, 2]])  # = 2 * (0.5 cot)
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    vals = np.concatenate([-w, -w])
    diag = np.bincount(rows, weights=np.concatenate([w, w]), minlength=nv)
    rows = np.concatenate([rows, np.arange(nv)])
    cols = np.concatenate([cols, np.arange(nv)])
    vals = np.concatenate([vals, diag])

    # ボロノイ混合面積 (Meyer et al. 2003 / libigl MASSMATRIX_TYPE_VORONOI)
    area = area2 * 0.5
    l2 = np.stack([dot(c - b, c - b), dot(a - c, a - c), dot(b - a, b - a)], 1)  # 対辺長^2
    obtuse = np.stack([cot0 < 0, cot1 < 0, cot2 < 0], 1)
    vor = np.stack([
        (l2[:, 1] * cot1 + l2[:, 2] * cot2) / 8.0,
        (l2[:, 2] * cot2 + l2[:, 0] * cot0) / 8.0,
        (l2[:, 0] * cot0 + l2[:, 1] * cot1) / 8.0,
    ], 1)
    any_obt = obtuse.any(1)
    vor[any_obt] = np.where(obtuse[any_obt], area[any_obt, None] / 2.0,
                            area[any_obt, None] / 4.0)
    mass = np.bincount(tris.reshape(-1), weights=vor.reshape(-1), minlength=nv)
    mass = np.maximum(mass, max(mass.mean(), 1e-12) * 1e-6)
    return _coalesce(rows, cols, vals, nv), mass, tris


def _pointcloud_laplacian(pts, k=24):
    """メッシュの辺を使わない点群ラプラシアン(簡易版)。

    k近傍をガウス重みでつなぎ、平面上でメッシュ(cot)ラプラシアンと同じスケールに
    なるよう行ごとに正規化する(sum_j w_ij |x_j-x_i|^2 = 4 m_i)。
    別パーツ(ボタン・ポーチ)やシームで割れた部分も、近ければ自然につながる。
    戻り値は _cotan_laplacian と同じ形式。
    """
    if not HAS_SCIPY:
        raise RuntimeError(u"点群モードには scipy が必要です")
    from scipy.spatial import cKDTree
    n = len(pts)
    k = int(max(3, min(k, n - 1)))
    d, idx = cKDTree(pts).query(pts, k=k + 1)
    d, idx = d[:, 1:], idx[:, 1:]
    h = np.maximum(d[:, -1], 1e-12)                 # 局所スケール(k番目の近傍までの距離)
    mass = np.pi * h * h / k                          # 1点が受け持つ面積の目安
    w = np.exp(-2.0 * (d / h[:, None]) ** 2)
    norm = (w * d * d).sum(1)
    w *= (4.0 * mass / np.maximum(norm, 1e-30))[:, None]
    rows = np.repeat(np.arange(n), k)
    cols = idx.reshape(-1)
    w = w.reshape(-1)
    ok = rows != cols
    rows, cols, w = rows[ok], cols[ok], w[ok]
    # 対称化((i,j) と (j,i) の平均)
    r2 = np.concatenate([rows, cols])
    c2 = np.concatenate([cols, rows])
    w2 = np.concatenate([w, w]) * 0.5
    r2, c2, w2 = _coalesce(r2, c2, w2, n)
    diag = np.bincount(r2, weights=w2, minlength=n)
    rows = np.concatenate([r2, np.arange(n)])
    cols = np.concatenate([c2, np.arange(n)])
    vals = np.concatenate([-w2, diag])
    return _coalesce(rows, cols, vals, n), mass


def _coalesce(rows, cols, vals, n):
    key = rows * n + cols
    order = np.argsort(key, kind="stable")
    key, vals = key[order], vals[order]
    uk, start = np.unique(key, return_index=True)
    v = np.add.reduceat(vals, start)
    return (uk // n, uk % n, v)


class _Sparse(object):
    """numpyだけで行列ベクトル積をする最小のCSR。"""

    def __init__(self, rows, cols, vals, n):
        order = np.lexsort((cols, rows))
        self.r, self.c, self.v = rows[order], cols[order], vals[order]
        self.n = n
        self.start = np.searchsorted(self.r, np.arange(n))

    def dot(self, x):
        prod = self.v[:, None] * x[self.c]
        return np.add.reduceat(prod, self.start, axis=0)


def _components(n, rows, cols):
    lab = np.arange(n)
    while True:
        m = np.minimum(lab[rows], lab[cols])
        new = lab.copy()
        np.minimum.at(new, rows, m)
        np.minimum.at(new, cols, m)
        new = new[new]
        if np.array_equal(new, lab):
            return lab
        lab = new


def _inpaint(pts, tris, known, w_known, max_iter=4000, tol=1e-7, log=_print, clamp=False,
             lap=None):
    """known 以外の頂点のウェイトを Q = -L + L M^-1 L の最小化で求める。"""
    nv = len(pts)
    if lap is None:
        (lr, lc, lv), mass, tris = _cotan_laplacian(pts, tris, clamp=clamp)
    else:
        (lr, lc, lv), mass = lap
    W = np.zeros((nv, w_known.shape[1]))
    W[known] = w_known

    # 既知点を一つも持たない連結成分は inpaint 不能 -> 呼び出し側で最近接値を使う
    offd = lr != lc
    lab = _components(nv, lr[offd], lc[offd])
    has_known = np.zeros(nv, bool)
    has_known[np.unique(lab[known])] = True
    orphan = ~has_known[lab]
    unknown = (~known) & (~orphan)
    if not unknown.any():
        return W, orphan

    # Q = L + L M^-1 L  (Lは正半定値の符号規約。論文/公式の -L + L M^-1 L と同じ)
    if HAS_SCIPY:
        L = _sp.csr_matrix((lv, (lr, lc)), shape=(nv, nv))
        Q = (L + L.dot(_sp.diags(1.0 / mass)).dot(L)).tocsr()
        U = np.where(unknown)[0]
        K = np.where(~unknown)[0]
        Quu = Q[U][:, U].tocsc()
        cols_used = np.where(np.abs(W[known]).sum(0) > 0)[0]
        rhs = -(Q[U][:, K].dot(W[np.ix_(K, cols_used)]))
        log(u"  scipy splu で解いています (未知 %d 頂点 / %d インフルエンス)"
            % (len(U), len(cols_used)))
        lu = _spla.splu(Quu)
        W[np.ix_(U, cols_used)] = lu.solve(np.asfortranarray(rhs))
        return W, orphan

    L = _Sparse(lr, lc, lv, nv)
    minv = 1.0 / mass

    def Qdot(x):
        lx = L.dot(x)
        return lx + L.dot(lx * minv[:, None])

    # Jacobi 前処理: diag(Q)_i = L_ii + sum_j L_ij^2 / M_j
    ldiag = np.zeros(nv)
    ldiag[lr[lr == lc]] = lv[lr == lc]
    qdiag = ldiag + np.bincount(lr, weights=lv * lv * minv[lc], minlength=nv)
    pinv = np.where(unknown, 1.0 / np.maximum(qdiag, 1e-30), 0.0)[:, None]
    umask = unknown[:, None].astype(np.float64)

    # 列ごとの独立CG を「影響のあるインフルエンスだけ」に絞ってまとめて回す
    cols_used = np.where(np.abs(W[known]).sum(0) > 0)[0]
    log(u"  numpy CG で解いています (未知 %d 頂点 / %d インフルエンス)"
        % (unknown.sum(), len(cols_used)))
    chunk = 32
    for s in range(0, len(cols_used), chunk):
        cs = cols_used[s:s + chunk]
        Wk = W[:, cs] * (~unknown)[:, None]
        b = -Qdot(Wk) * umask
        x = np.zeros_like(b)
        r = b - Qdot(x) * umask
        z = r * pinv
        p = z.copy()
        rz = (r * z).sum(0)
        bn = np.maximum(np.sqrt((b * b).sum(0)), 1e-30)
        for it in range(max_iter):
            Ap = Qdot(p) * umask
            alpha = rz / np.maximum((p * Ap).sum(0), 1e-300)
            x += p * alpha
            r -= Ap * alpha
            res = np.sqrt((r * r).sum(0)) / bn
            if res.max() < tol:
                break
            z = r * pinv
            rz_new = (r * z).sum(0)
            beta = rz_new / np.maximum(rz, 1e-300)
            rz = rz_new
            p = z + p * beta
        log(u"    influences %d-%d: %d iter, residual %.2e"
            % (s, s + len(cs) - 1, it + 1, res.max()))
        W[np.ix_(unknown, cs)] = x[unknown]
    return W, orphan


def _area_weighted_normals(pts, tris):
    """libigl per_vertex_normals(既定=面積重み)と同じ頂点法線。公式実装と判定を揃えるため。"""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    fn = np.cross(b - a, c - a)  # 長さ = 2 * 面積 -> 面積重み
    out = np.zeros_like(pts)
    for k in range(3):
        np.add.at(out, tris[:, k], fn)
    return _normalize(out)


def _adjacency(n, tris):
    """igl.adjacency_list 相当(重複なし・昇順)を CSR 形式で返す。"""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]])
    e = np.concatenate([e, e[:, ::-1]])
    e = np.unique(e, axis=0)
    e = e[e[:, 0] != e[:, 1]]
    start = np.searchsorted(e[:, 0], np.arange(n + 1))
    return start, e[:, 1]


def _smooth_official(pts, tris, W, matched, radius, iterations, alpha):
    """公式 Python 実装 utils.smooth の移植(対象範囲の選び方・更新順まで同じ)。

    対象 = 補間頂点 + 各補間頂点からメッシュの接続をたどって radius 以内にある頂点。
    頂点番号順に、その場で(Gauss-Seidel 的に)更新する。
    """
    n = len(pts)
    start, nbr = _adjacency(n, tris)
    region = ~matched.copy()
    for vid in np.where(~matched)[0]:  # 公式と同じ順序・条件で領域を広げる
        stack = [vid]
        p0 = pts[vid]
        while stack:
            vv = stack.pop()
            for nn in nbr[start[vv]:start[vv + 1]]:
                if not region[nn] and np.linalg.norm(p0 - pts[nn]) < radius:
                    region[nn] = True
                    stack.append(nn)
    cols = np.where(np.abs(W).sum(0) > 0)[0]
    X = W[:, cols].copy()
    idx = np.where(region)[0]
    for _ in range(int(iterations)):
        for i in idx:
            nb = nbr[start[i]:start[i + 1]]
            if len(nb) == 0:
                continue
            X[i] = (1.0 - alpha) * X[i] + X[nb].sum(0) * (alpha / len(nb))
    W = W.copy()
    W[:, cols] = X
    return W, region


def _smooth_region(pts, seeds, radius):
    """公式実装と同じく、補間頂点 + それらから radius 以内の頂点をスムース対象にする。"""
    region = seeds.copy()
    if not seeds.any() or seeds.all():
        return region
    if HAS_SCIPY:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(pts[seeds]).query(pts[~seeds], k=1, distance_upper_bound=radius)
        idx = np.where(~seeds)[0]
        region[idx[d < radius]] = True
    return region


def _smooth(W, edges, movable, iterations, strength):
    """一様ラプラシアンによるスムース。movable=False の頂点は固定。"""
    n = W.shape[0]
    edges = np.unique(np.sort(edges, axis=1), axis=0)  # 重複辺を除く(近傍を1回ずつ数える)
    edges = edges[edges[:, 0] != edges[:, 1]]
    a = np.concatenate([edges[:, 0], edges[:, 1]])
    b = np.concatenate([edges[:, 1], edges[:, 0]])
    deg = np.bincount(a, minlength=n).astype(np.float64)
    ok = movable & (deg > 0)
    cols = np.where(np.abs(W).sum(0) > 0)[0]
    X = W[:, cols].copy()
    if HAS_SCIPY:
        A = _sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
        nbr_sum = lambda Y: A.dot(Y)
    else:
        def nbr_sum(Y):
            out = np.zeros_like(Y)
            np.add.at(out, a, Y[b])
            return out
    for _ in range(int(iterations)):
        avg = nbr_sum(X)
        avg[ok] /= deg[ok][:, None]
        X[ok] = X[ok] * (1.0 - strength) + avg[ok] * strength
    W = W.copy()
    W[:, cols] = X
    return W


def _postprocess(W, max_inf, prune=1e-4):
    W = np.clip(W, 0.0, None)  # 論文 3.2: 負の値は 0 にして正規化
    if prune:
        W[W < prune] = 0.0
    if max_inf and max_inf > 0 and W.shape[1] > max_inf:
        # 上位 max_inf 個だけ残す
        idx = np.argsort(-W, axis=1)[:, max_inf:]
        np.put_along_axis(W, idx, 0.0, axis=1)
    s = W.sum(1, keepdims=True)
    bad = s[:, 0] < 1e-8
    s[bad] = 1.0
    return W / s, bad



def compute(d, p, log=_print):
    """最近接点の判定〜補間〜スムース〜後処理。Maya を使わない(numpy / scipy のみ)。

    d : Maya 側で集めた配列(numpy)
        src_pts, src_nrm, src_tris, tgt_pts, tgt_nrm, tgt_tris,
        tri_v (n,3) 最近接三角形の頂点, uv (n,2) その重心座標,
        used_src (k,) 参照されるソース頂点, wsrc (k, ninf) そのウェイト, mask (m,)
    p : 設定(transfer の引数を解決したもの)
    戻り値: (配列の dict, 情報の dict)
    """
    paper = p["mode"] == "paper"
    weld, interp, clamp_cot = p["weld"], p["interp"], p["clamp_cot"]
    if paper:
        weld, interp, clamp_cot = False, "mesh", False
    if interp == "pointcloud" and not HAS_SCIPY:
        raise RuntimeError(u"拡張モードの点群補間には scipy が必要です")
    max_influences = int(p["max_influences"] or 0)
    smooth_iter, smooth_strength = int(p["smooth_iter"]), float(p["smooth_strength"])

    src_pts = d["src_pts"]
    tgt_pts = d["tgt_pts"]
    src_tris = d["src_tris"].astype(np.int64)
    tgt_tris = d["tgt_tris"].astype(np.int64)
    if paper:  # 公式と同じ面積重みの頂点法線で判定する
        src_nrm = _area_weighted_normals(src_pts, src_tris)
        tgt_nrm = _area_weighted_normals(tgt_pts, tgt_tris)
    else:
        src_nrm = _normalize(d["src_nrm"])
        tgt_nrm = _normalize(d["tgt_nrm"])
    bb = tgt_pts.max(0) - tgt_pts.min(0)  # 公式実装と同じくターゲット基準
    diag = float(np.linalg.norm(bb))
    # Maya API のワールド座標は内部単位(cm)なので dist_abs も cm
    dist_thresh = float(p["dist_abs"]) if p["dist_abs"] else float(p["dist_ratio"]) * diag
    log(u"[%s] 距離しきい値 %.3f cm / 角度 %.1f度 / 最大インフルエンス %s"
        % (u"論文準拠" if paper else u"拡張", dist_thresh, p["angle"],
           max_influences or u"制限なし"))

    # --- Step1: 最近接点(探索は Maya 側。ここでは条件の判定) ---
    t = time.time()
    tri_v = d["tri_v"].astype(np.int64)
    uv = d["uv"]
    bary = np.stack([uv[:, 0], uv[:, 1], 1.0 - uv[:, 0] - uv[:, 1]], 1)
    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
    cp = np.einsum("ij,ijk->ik", bary, src_pts[tri_v])
    cn = _normalize(np.einsum("ij,ijk->ik", bary, src_nrm[tri_v]))
    dist = np.linalg.norm(cp - tgt_pts, axis=1)
    dots = np.einsum("ij,ij->i", cn, tgt_nrm)
    if p["allow_flip"]:
        dots = np.abs(dots)
    known = (dist <= dist_thresh) & (dots >= math.cos(math.radians(p["angle"])))
    log(u"Step1 最近接点: 確定 %d / %d 頂点" % (known.sum(), len(known)))
    masked = np.zeros(len(known), bool)
    if len(d["mask"]):
        masked[d["mask"].astype(np.int64)] = True
        known[masked] = False
        log(u"  マスク: %d 頂点を強制的に補間対象にしました" % masked.sum())
    known_step1 = known.copy()

    used_src = d["used_src"].astype(np.int64)
    Wsrc_used = d["wsrc"]
    ninf = Wsrc_used.shape[1]
    remap = np.full(len(src_pts), -1, np.int64)
    remap[used_src] = np.arange(len(used_src))
    W_cp = np.einsum("ij,ijk->ik", bary, Wsrc_used[remap[tri_v]])  # 全頂点の最近接ウェイト

    # --- Step2: inpainting (溶接メッシュ上) ---
    t = time.time()
    if weld:
        wid = _weld(tgt_pts, max(diag * 1e-6, 1e-6))
    else:
        wid = np.arange(len(tgt_pts))
    nw = wid.max() + 1
    cnt = np.bincount(wid, minlength=nw).astype(np.float64)
    w_pts = np.zeros((nw, 3))
    np.add.at(w_pts, wid, tgt_pts)
    w_pts /= cnt[:, None]
    w_tris = wid[tgt_tris]
    w_tris = w_tris[(w_tris[:, 0] != w_tris[:, 1]) & (w_tris[:, 1] != w_tris[:, 2])
                    & (w_tris[:, 0] != w_tris[:, 2])]
    w_dist = np.full(nw, np.inf)
    np.minimum.at(w_dist, wid, dist)

    # 確定点を1つも持たない島(ポーチ・バックル等の別パーツ)の扱い
    #   論文準拠: 何もしない(補間できないので最近接の値を入れ、報告する)
    #   拡張    : 法線条件を外して距離が近い頂点を確定にする。無ければ島内で最も近い 10%。
    #             ただしマスクで「必ず補間」と指定された頂点は確定に戻さない。
    known = known.copy()
    w_known = np.zeros(nw, bool)
    w_known[wid[known]] = True
    w_masked = np.zeros(nw, bool)
    w_masked[wid[masked]] = True
    e = np.concatenate([w_tris[:, [0, 1]], w_tris[:, [1, 2]], w_tris[:, [2, 0]]])
    lap = None
    if interp == "pointcloud":
        lap = _pointcloud_laplacian(w_pts)
        (lr_, lc_, _), _ = lap
        off = lr_ != lc_
        lab = _components(nw, lr_[off], lc_[off])  # 近傍グラフでつながりを判定
        log(u"  補間モード: 点群(近傍グラフで別パーツもつなぐ)")
    else:
        lab = _components(nw, e[:, 0], e[:, 1])
        log(u"  補間モード: メッシュ(cot ラプラシアン)")
    comp_has = np.zeros(nw, bool)
    comp_has[lab[w_known]] = True
    orphan_w = ~comp_has[lab]
    if orphan_w.any() and not paper:
        n_islands = 0
        for c in np.unique(lab[orphan_w]):
            members = np.where(lab == c)[0]
            cand = members[~w_masked[members]]
            if len(cand) == 0:
                continue  # 全部マスク -> 救済しない(未解決として報告)
            sel = cand[w_dist[cand] <= dist_thresh]
            if len(sel) == 0:
                k = max(1, int(math.ceil(len(cand) * 0.1)))
                sel = cand[np.argsort(w_dist[cand])[:k]]
            w_known[sel] = True
            n_islands += 1
        known = w_known[wid]
        log(u"  確定点のない島 %d 個を距離のみで補強" % n_islands)

    kcount = np.bincount(wid[known], minlength=nw).astype(np.float64)
    Wk = np.zeros((nw, ninf))
    np.add.at(Wk, wid[known], W_cp[known])
    Wk[w_known] /= kcount[w_known][:, None]

    W_w, orphan = _inpaint(w_pts, w_tris, w_known, Wk[w_known], log=log, clamp=clamp_cot,
                           lap=lap)

    # --- 後処理: 補間頂点だけをスムース(脇・股などの境目をなじませる) ---
    if smooth_iter > 0 and smooth_strength > 0:
        w_step1 = np.zeros(nw, bool)
        w_step1[wid[known_step1]] = True
        if paper:
            W_w, movable = _smooth_official(w_pts, w_tris, W_w, w_step1, dist_thresh,
                                            smooth_iter, smooth_strength)
        else:
            movable = _smooth_region(w_pts, ~w_step1, dist_thresh)
            W_w = _smooth(W_w, e, movable, smooth_iter, smooth_strength)
        log(u"  スムース %d 回 (強さ %.2f) を %d 頂点に適用"
            % (smooth_iter, smooth_strength, movable.sum()))
    W = W_w[wid]
    orphan_v = orphan[wid]
    W[orphan_v] = W_cp[orphan_v]  # 補間できない頂点は最近接ウェイトで代用(報告する)
    n_parts = len(np.unique(lab[orphan])) if orphan.any() else 0
    log(u"Step2 inpainting: 補間 %d 頂点 / 補間できなかった別パーツ %d 個 (%d 頂点) (%.1fs)"
        % ((~known).sum() - orphan_v.sum(), n_parts, orphan_v.sum(), time.time() - t))

    prune = 0.0 if paper else 1e-4
    W, bad = _postprocess(W, max_influences, prune=prune)
    if bad.any():  # 全部 0 になった頂点は最近接コピーで代用(インフルエンス数の制限もかける)
        W[bad], _ = _postprocess(W_cp[bad], max_influences, prune=prune)

    cols = np.where(W.sum(0) > 0)[0]
    out = {
        "cols": cols.astype(np.int32),
        "W": np.ascontiguousarray(W[:, cols]),
        "known": known_step1.astype(np.int8),
        "orphan": orphan_v.astype(np.int8),
    }
    if p["output"] == "ngskin":
        W_close, _ = _postprocess(W_cp, max_influences, prune=prune)
        cols_c = np.where(W_close.sum(0) > 0)[0]
        out["cols_close"] = cols_c.astype(np.int32)
        out["W_close"] = np.ascontiguousarray(W_close[:, cols_c])
    return out, {"n_parts": int(n_parts)}


# --------------------------------------------------------------------------
# Maya と計算部分の受け渡し(配列は array.array <-> numpy、別プロセスはファイル経由)
# --------------------------------------------------------------------------
_NP_DTYPE = {"d": "<f8", "i": "<i4", "b": "i1"}
_TC_OF = {"float64": "d", "int32": "i", "int8": "b"}


def _array_from_bytes(tc, raw):
    a = array(tc)
    if hasattr(a, "frombytes"):
        a.frombytes(raw)
    else:  # Python 2
        a.fromstring(raw)
    return a


def _np_from_array(a, shape):
    return np.frombuffer(a, dtype=_NP_DTYPE[a.typecode]).reshape(shape).copy()


def _array_from_np(x):
    tc = _TC_OF[str(x.dtype)]
    return _array_from_bytes(tc, np.ascontiguousarray(x, dtype=_NP_DTYPE[tc]).tobytes())


def _worker_main(argv):
    """別プロセス(Python 3 の mayapy)側の入口:  mayapy このファイル --rwt-worker <dir>"""
    work = argv[argv.index(_WORKER_FLAG) + 1]

    def log(msg):
        sys.stdout.write(u"LOG " + msg.replace(u"\n", u" ") + u"\n")
        sys.stdout.flush()

    with io.open(os.path.join(work, "in.json"), "r", encoding="utf-8") as f:
        meta = json.load(f)
    data = {}
    for name, (tc, shape) in meta["arrays"].items():
        x = np.fromfile(os.path.join(work, "in_%s.bin" % name), dtype=_NP_DTYPE[tc])
        data[name] = x.reshape(shape)
    log(u"計算プロセス: Python %d.%d / numpy %s / scipy %s"
        % (sys.version_info[0], sys.version_info[1], np.__version__,
           u"ok" if HAS_SCIPY else u"なし"))
    out, info = compute(data, meta["params"], log)
    res = {"arrays": {}, "info": info}
    for name, x in out.items():
        tc = _TC_OF[str(x.dtype)]
        np.ascontiguousarray(x, dtype=_NP_DTYPE[tc]).tofile(
            os.path.join(work, "out_%s.bin" % name))
        res["arrays"][name] = [tc, list(x.shape)]
    with io.open(os.path.join(work, "out.json"), "w", encoding="utf-8") as f:
        f.write(text_type(json.dumps(res)))
    log(u"計算プロセス: 完了")
    return 0


# ==========================================================================
# 計算の実行方法(Maya の中 / 別プロセス)
# ==========================================================================
_OPT = "robustWeightTransfer3_"
_OPT_PY3 = _OPT + "python3"
_py3_cache = {}


def _default_mayapy():
    """この Maya に付属する Python 3 の mayapy のパス(候補)。"""
    exe = "mayapy.exe" if os.name == "nt" else "mayapy"
    dirs = []
    loc = os.environ.get("MAYA_LOCATION")
    if loc:
        dirs.append(os.path.join(loc, "bin"))
    dirs.append(os.path.dirname(sys.executable))
    for dr in dirs:
        cand = os.path.join(dr, exe)
        if os.path.isfile(cand):
            return os.path.normpath(cand)
    return os.path.join(dirs[0], exe) if dirs else exe


def _python3_candidates():
    out = []
    env = os.environ.get("RWT_PYTHON3")
    if env:
        out.append(env)
    if cmds is not None and cmds.optionVar(exists=_OPT_PY3):
        v = cmds.optionVar(q=_OPT_PY3)
        if v:
            out.append(v)
    out.append(_default_mayapy())
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def _native(path):
    """Python 2 の subprocess に渡すパスはバイト列にする。"""
    if PY2 and isinstance(path, text_type):
        return path.encode(sys.getfilesystemencoding() or "mbcs")
    return path


def _popen_kwargs():
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW(コンソールを出さない)
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    for k in ("PYTHONHOME", "PYTHONPATH"):  # Maya(Python 2)の設定を持ち込まない
        env.pop(k, None)
    kw["env"] = env
    return kw


def check_python3(path):
    """path の Python が Python 3 で numpy / scipy を読めるか。(ok, 説明) を返す。"""
    if path in _py3_cache:
        return _py3_cache[path]
    if not os.path.isfile(path):
        res = (False, u"見つかりません")
    else:
        code = ("import sys, numpy, scipy; "
                "print('%d.%d %s %s' % (sys.version_info[0], sys.version_info[1], "
                "numpy.__version__, scipy.__version__))")
        try:
            proc = subprocess.Popen([_native(path), "-c", code], stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, **_popen_kwargs())
            out, err = proc.communicate()
            out = out.decode("utf-8", "replace").strip()
            if proc.returncode == 0 and out and out[0] == "3":
                ver, npv, spv = out.split()[:3]
                res = (True, u"Python %s / numpy %s / scipy %s" % (ver, npv, spv))
            elif proc.returncode == 0:
                res = (False, u"Python 3 ではありません (%s)" % out)
            else:
                last = err.decode("utf-8", "replace").strip().splitlines()
                res = (False, last[-1] if last else u"起動に失敗しました")
        except Exception as e:
            res = (False, _err_text(e))
    _py3_cache[path] = res
    return res


def _find_python3():
    for c in _python3_candidates():
        ok, _ = check_python3(c)
        if ok:
            return c
    return None


def pip_command():
    return u'"%s" -m pip install --user numpy scipy' % _default_mayapy()


def _need_libs_message():
    return (u"計算に必要な numpy / scipy が見つかりません。\n"
            u"コマンドプロンプトで次を実行してから、もう一度転写してください"
            u"(Maya の再起動は不要です):\n\n%s\n\n"
            u"※ Python 2 モードの Maya でも、入れる先は Python 3 の mayapy.exe です"
            u"(mayapy2.exe ではありません)。" % pip_command())


def _choose_backend(need_scipy, backend="auto"):
    """("inproc", None) か ("worker", python3 のパス) を返す。"""
    backend = os.environ.get("RWT_BACKEND", backend)
    if backend == "inproc":
        if not HAS_NUMPY or (need_scipy and not HAS_SCIPY):
            raise RuntimeError(_need_libs_message())
        return "inproc", None
    if backend != "worker" and HAS_NUMPY and HAS_SCIPY:
        return "inproc", None
    py = _find_python3()
    if py:
        return "worker", py
    if backend != "worker" and HAS_NUMPY and not need_scipy:
        return "inproc", None  # numpy だけでも論文準拠は解ける(遅い)
    raise RuntimeError(_need_libs_message())


def _this_source():
    path = os.path.abspath(__file__)
    if path.endswith((".pyc", ".pyo")):
        path = path[:-1]
    if not os.path.isfile(path):
        raise RuntimeError(u"スクリプト本体 (.py) が見つかりません: %s" % path)
    return path


def _run_worker(py, data, params, log):
    work = tempfile.mkdtemp(prefix="rwt_")
    try:
        meta = {"params": params, "arrays": {}}
        for name, (a, shape) in data.items():
            with open(os.path.join(work, "in_%s.bin" % name), "wb") as f:
                a.tofile(f)
            meta["arrays"][name] = [a.typecode, list(shape)]
        with io.open(os.path.join(work, "in.json"), "w", encoding="utf-8") as f:
            f.write(text_type(json.dumps(meta)))
        script = os.path.join(work, "rwt_worker.py")  # ASCII のパスで起動する
        shutil.copyfile(_this_source(), script)
        errpath = os.path.join(work, "stderr.txt")
        with open(errpath, "wb") as errf:
            proc = subprocess.Popen([_native(py), _native(script), _WORKER_FLAG, _native(work)], stdout=subprocess.PIPE,
                                    stderr=errf, **_popen_kwargs())
            for line in iter(proc.stdout.readline, b""):
                line = line.decode("utf-8", "replace").rstrip()
                if line.startswith(u"LOG "):
                    log(line[4:])
                elif line:
                    _print(line)
            proc.wait()
        if proc.returncode != 0 or not os.path.isfile(os.path.join(work, "out.json")):
            with io.open(errpath, "r", encoding="utf-8", errors="replace") as f:
                err = f.read().strip()
            _print(err)
            last = err.splitlines()[-1] if err else u"(出力なし)"
            raise RuntimeError(u"計算プロセスでエラーが発生しました: %s" % last)
        with io.open(os.path.join(work, "out.json"), "r", encoding="utf-8") as f:
            res = json.load(f)
        out = {}
        for name, (tc, shape) in res["arrays"].items():
            n = 1
            for s in shape:
                n *= s
            with open(os.path.join(work, "out_%s.bin" % name), "rb") as f:
                a = _array_from_bytes(str(tc), f.read())
            if len(a) != n:
                raise RuntimeError(u"計算結果の読み込みに失敗しました: %s" % name)
            out[name] = (a, shape)
        return out, res["info"]
    finally:
        if not os.environ.get("RWT_KEEP_TMP"):
            shutil.rmtree(work, ignore_errors=True)


def _run_compute(data, params, log, backend="auto"):
    need_scipy = params["mode"] != "paper" and params["interp"] == "pointcloud"
    kind, py = _choose_backend(need_scipy, backend)
    if kind == "worker":
        log(u"計算: 別プロセス (%s)" % py)
        return _run_worker(py, data, params, log)
    log(u"計算: Maya 内 (numpy %s%s)" % (np.__version__, u" / scipy" if HAS_SCIPY else u""))
    nd = dict((k, _np_from_array(a, shape)) for k, (a, shape) in data.items())
    out, info = compute(nd, params, log)
    return dict((k, (_array_from_np(x), list(x.shape))) for k, x in out.items()), info


def environment_report():
    """計算環境の確認結果(文字列)。"""
    lines = [u"Maya %s / Python %d.%d" % (cmds.about(v=True), sys.version_info[0],
                                         sys.version_info[1])]
    if HAS_NUMPY:
        lines.append(u"Maya 内: numpy %s / scipy %s"
                     % (np.__version__, u"あり" if HAS_SCIPY else u"なし"))
    else:
        lines.append(u"Maya 内: numpy なし")
    for c in _python3_candidates():
        ok, msg = check_python3(c)
        lines.append(u"%s %s\n    %s" % (u"○" if ok else u"×", c, msg))
    try:
        kind, py = _choose_backend(False)
        lines.append(u"\n→ 計算は %s で行います。"
                     % (u"Maya 内" if kind == "inproc" else u"別プロセス (Python 3 の mayapy)"))
        if kind == "inproc" and not HAS_SCIPY:
            lines.append(u"  scipy が無いため遅く、拡張モードは使えません。")
    except RuntimeError:
        lines.append(u"\n→ 計算できません。次のコマンドで numpy / scipy を入れてください:\n"
                     + pip_command())
    return u"\n".join(lines)


# ==========================================================================
# Maya 側(numpy を使わない。Python 2 / 3 両対応)
# ==========================================================================
def has_ngskin():
    """ngSkinTools2 が使えるか(Maya 2027 は標準同梱)。"""
    try:
        if not cmds.pluginInfo("ngSkinTools2", q=True, loaded=True):
            cmds.loadPlugin("ngSkinTools2", quiet=True)
        import ngSkinTools2.api  # noqa: F401
        return cmds.pluginInfo("ngSkinTools2", q=True, loaded=True)
    except Exception:
        return False


class _NoUndo(object):
    """Undo を記録しない(終わると Undo 履歴はクリアされる)。"""

    def __enter__(self):
        self.was = cmds.undoInfo(q=True, state=True)
        cmds.undoInfo(state=False)
        return self

    def __exit__(self, *exc):
        cmds.undoInfo(state=self.was)
        return False


def _mesh_shape(node):
    if cmds.nodeType(node) == "mesh":
        return node
    shapes = cmds.listRelatives(node, s=True, ni=True, type="mesh", f=True) or []
    if not shapes:
        raise RuntimeError(u"メッシュが見つかりません: %s" % node)
    return shapes[0]


def _dag(name):
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDagPath(0)


def _obj(name):
    sel = om.MSelectionList()
    sel.add(name)
    return sel.getDependNode(0)


def find_skin_cluster(node):
    shape = _mesh_shape(node)
    hist = cmds.listHistory(shape, pdo=True) or []
    for h in hist:
        if cmds.nodeType(h) == "skinCluster":
            return h
    return None


def _int_array(seq):
    return array("i", [int(i) for i in seq])


def _mesh_data(shape):
    """頂点位置・頂点法線・三角形を array で返す。"""
    dag = _dag(shape)
    fn = om.MFnMesh(dag)
    pts = array("d")
    for p in fn.getPoints(om.MSpace.kWorld):
        pts.extend((p.x, p.y, p.z))
    nrm = array("d")
    for n in fn.getVertexNormals(True, om.MSpace.kWorld):
        nrm.extend((n.x, n.y, n.z))
    _, tri_ids = fn.getTriangles()
    tris = _int_array(tri_ids)
    return dag, pts, nrm, tris


def _vertex_component(ids):
    fn = om.MFnSingleIndexedComponent()
    comp = fn.create(om.MFn.kMeshVertComponent)
    fn.addElements(om.MIntArray([int(i) for i in ids]))
    return comp


def _complete_component(n):
    fn = om.MFnSingleIndexedComponent()
    comp = fn.create(om.MFn.kMeshVertComponent)
    fn.setCompleteData(int(n))
    return comp


def _gather(src_shape, tgt_shape, src_skin, mask_vertices, log):
    """計算に必要なデータを Maya から集める(最近接点の探索もここで行う)。"""
    t = time.time()
    _, src_pts, src_nrm, src_tris = _mesh_data(src_shape)
    _, tgt_pts, tgt_nrm, tgt_tris = _mesh_data(tgt_shape)
    ns, nt = len(src_pts) // 3, len(tgt_pts) // 3

    # 最近接点(三角形の頂点と重心座標)
    dag = _dag(src_shape)
    fn = om.MFnMesh(dag)
    inter = om.MMeshIntersector()
    inter.create(dag.node(), dag.inclusiveMatrix())
    tri_v = array("i")
    uv = array("d")
    tri_cache = {}
    for i in range(nt):
        pom = inter.getClosestPoint(om.MPoint(tgt_pts[3 * i], tgt_pts[3 * i + 1],
                                              tgt_pts[3 * i + 2]))
        key = (pom.face, pom.triangle)
        tv = tri_cache.get(key)
        if tv is None:
            tv = tuple(int(v) for v in fn.getPolygonTriangleVertices(pom.face, pom.triangle))
            tri_cache[key] = tv
        tri_v.extend(tv)
        u, v = pom.barycentricCoords
        uv.extend((u, v))
    log(u"最近接点を検索しました (%.1fs)" % (time.time() - t))

    # ソースウェイトは参照される頂点ぶんだけ読む
    used = sorted(set(tri_v))
    fn_skin = oma.MFnSkinCluster(_obj(src_skin))
    w_flat, ninf = fn_skin.getWeights(dag, _vertex_component(used))
    wsrc = array("d", list(w_flat))
    mask = _int_array(mask_vertices or [])
    return {
        "src_pts": (src_pts, [ns, 3]), "src_nrm": (src_nrm, [ns, 3]),
        "src_tris": (src_tris, [len(src_tris) // 3, 3]),
        "tgt_pts": (tgt_pts, [nt, 3]), "tgt_nrm": (tgt_nrm, [nt, 3]),
        "tgt_tris": (tgt_tris, [len(tgt_tris) // 3, 3]),
        "tri_v": (tri_v, [nt, 3]), "uv": (uv, [nt, 2]),
        "used_src": (_int_array(used), [len(used)]),
        "wsrc": (wsrc, [len(used), int(ninf)]),
        "mask": (mask, [len(mask)]),
    }


def transfer(source, target, dist_ratio=0.05, angle=30.0, allow_flip=False,
             max_influences=None, smooth_iter=10, smooth_strength=0.2,
             mask_vertices=None, output="skin", weld=True, clamp_cot=False,
             interp="mesh", dist_abs=None, mode="paper", log=None, backend="auto"):
    """source のスキンウェイトを target に Robust Skin Weights Transfer で転写する。

    ※ 1ファイル版は Undo できません(実行すると Undo 履歴はクリアされます)。

    mode : "paper"    = 論文準拠。公式 Python 実装(rin-23/RobustSkinWeightsTransferCode の
                        src/utils.py)と同じ処理だけを行う。独自処理(溶接・別パーツ救済・
                        インフルエンス数の制限・小さい値の削除・点群補間・安全モード)は使わない。
                        weld / interp / clamp_cot / max_influences は無視される。
           "extended" = 拡張。溶接・別パーツ救済・点群補間・インフルエンス数の制限などを使う。
    既定値は公式 Python 実装に合わせている(角度 30°。論文本文では 35°)。

    dist_ratio : ターゲットのバウンディングボックス対角長に対する距離しきい値(公式 0.05)
    dist_abs   : 距離しきい値を cm で直接指定(指定時は dist_ratio より優先。Blender版は 5cm)
    angle      : 法線の角度しきい値(度)(公式 30)
    allow_flip : 裏向き法線も一致とみなす(論文の任意オプション。重ね着・厚みのある服向け)
    max_influences : 1頂点あたりの最大インフルエンス数(None=ソースと同じ)
    smooth_iter / smooth_strength : 補間頂点とその距離しきい値内の近傍に掛ける
                    ラプラシアンスムース(論文3.3。公式 10回 / 0.2)
    clamp_cot  : 負の cot 重みを 0 に丸める(破綻メッシュ向けの安全策。論文とは別物になる)
    interp     : "mesh"       = メッシュの cot ラプラシアンで補間(論文・公式実装どおり)
                 "pointcloud" = 点群ラプラシアンで補間(Blender版の方式に近い簡易版)
    mask_vertices : 強制的に補間させる頂点番号(論文3.4のマスク。襟が顎に付く等の対策)
    output : "skin"   = skinCluster に直接書き込む
             "ngskin" = ngSkinTools2 のレイヤー2枚で出力
    backend: "auto" / "inproc"(Maya 内で計算) / "worker"(Python 3 の mayapy で計算)
    """
    log = log or _print
    if output == "ngskin" and not has_ngskin():
        raise RuntimeError(u"ngSkinTools2 が読み込めません")
    with _NoUndo():
        return _transfer(source, target, dist_ratio, angle, allow_flip, max_influences,
                         smooth_iter, smooth_strength, mask_vertices, output, weld,
                         clamp_cot, interp, dist_abs, mode, log, backend)


def _transfer(source, target, dist_ratio, angle, allow_flip, max_influences,
              smooth_iter, smooth_strength, mask_vertices, output, weld, clamp_cot,
              interp, dist_abs, mode, log, backend):
    paper = mode == "paper"
    t0 = time.time()
    src_shape = _mesh_shape(source)
    tgt_shape = _mesh_shape(target)
    src_skin = find_skin_cluster(src_shape)
    if not src_skin:
        raise RuntimeError(u"ソースにskinClusterがありません: %s" % source)
    influences = cmds.skinCluster(src_skin, q=True, inf=True)
    if paper:
        max_influences = 0  # 論文準拠では制限しない
    elif max_influences is None:
        max_influences = cmds.getAttr(src_skin + ".maxInfluences")

    # --- 計算(失敗してもシーンは変わらないよう、バインドより先に行う) ---
    data = _gather(src_shape, tgt_shape, src_skin, mask_vertices, log)
    params = {
        "mode": "paper" if paper else "extended", "dist_ratio": float(dist_ratio),
        "dist_abs": float(dist_abs) if dist_abs else None, "angle": float(angle),
        "allow_flip": bool(allow_flip), "max_influences": int(max_influences or 0),
        "smooth_iter": int(smooth_iter), "smooth_strength": float(smooth_strength),
        "weld": bool(weld), "clamp_cot": bool(clamp_cot), "interp": interp,
        "output": output,
    }
    res, info = _run_compute(data, params, log, backend)
    nverts = data["tgt_pts"][1][0]

    # --- ターゲットのskinCluster準備(なければ作る) ---
    tgt_skin = find_skin_cluster(tgt_shape)
    if not tgt_skin:
        tgt_xform = cmds.listRelatives(tgt_shape, p=True)[0]
        tgt_skin = cmds.skinCluster(influences, tgt_xform, toSelectedBones=True,
                                    bindMethod=0, maximumInfluences=1,
                                    obeyMaxInfluences=False, normalizeWeights=1,
                                    name=tgt_xform + "_robustSkin")[0]
        log(u"ターゲットをバインドしました: %s" % tgt_skin)
    else:
        have = set(cmds.ls(cmds.skinCluster(tgt_skin, q=True, inf=True), l=True))
        missing = [j for j in influences if cmds.ls(j, l=True)[0] not in have]
        for j in missing:
            cmds.skinCluster(tgt_skin, e=True, ai=j, wt=0.0, lw=False)
        if missing:
            log(u"ターゲットにインフルエンスを %d 個追加しました" % len(missing))

    # --- 書き込み ---
    t = time.time()
    W = (res["cols"][0], res["W"][0])
    known = res["known"][0]
    if output == "ngskin":
        W_close = (res["cols_close"][0], res["W_close"][0])
        mask = [0.0 if k else 1.0 for k in known]
        _write_ngskin_layers(tgt_skin, tgt_shape, influences, W_close, W, mask, nverts, log)
    else:
        _write_weights(tgt_skin, tgt_shape, influences, W, nverts)
    log(u"書き込み (%.1fs)" % (time.time() - t))

    inpainted = [i for i, k in enumerate(known) if not k]
    unsolved = [i for i, o in enumerate(res["orphan"][0]) if o]
    _last_inpainted["mesh"] = cmds.listRelatives(tgt_shape, p=True, f=True)[0]
    _last_inpainted["ids"] = inpainted
    _last_inpainted["unsolved"] = unsolved
    log(u"完了: %s -> %s (合計 %.1fs)" % (source, target, time.time() - t0))
    return {"skinCluster": tgt_skin, "known": nverts - len(inpainted),
            "inpainted": len(inpainted), "orphan": len(unsolved),
            "unsolved_parts": int(info.get("n_parts", 0)), "unsolved": len(unsolved),
            "mesh": _last_inpainted["mesh"], "mode": mode}


def _double_array(values):
    try:
        return om.MDoubleArray(values)
    except Exception:
        return om.MDoubleArray(list(values))


def _write_weights(tgt_skin, tgt_shape, src_infs, W, nverts):
    """W = (使うソースインフルエンスの番号, 頂点ごとに並べた値)。他のインフルエンスは 0。"""
    cols, values = W
    for j in cmds.skinCluster(tgt_skin, q=True, inf=True):
        cmds.setAttr(j + ".lockInfluenceWeights", 0)
    fn = oma.MFnSkinCluster(_obj(tgt_skin))
    tgt_paths = fn.influenceObjects()
    tgt_index = dict((cmds.ls(p.fullPathName(), l=True)[0], i) for i, p in enumerate(tgt_paths))
    idx = [tgt_index[cmds.ls(src_infs[c], l=True)[0]] for c in cols]
    dag = _dag(tgt_shape)
    comp = _complete_component(nverts)
    n_all = len(tgt_paths)
    # 書き込み中は自動正規化を止める(0 にした直後に Maya が値を補うのを防ぐ)
    cmds.setAttr(tgt_skin + ".normalizeWeights", 0)
    try:
        fn.setWeights(dag, comp, om.MIntArray(list(range(n_all))),
                      om.MDoubleArray(int(nverts) * n_all, 0.0), False)
        if idx:
            fn.setWeights(dag, comp, om.MIntArray(idx), _double_array(values), False)
    finally:
        cmds.setAttr(tgt_skin + ".normalizeWeights", 1)


def _layer_from_weights_via_temp(tgt_shape, src_infs, W, nverts, name, log):
    """W を持つ ngSkin レイヤーを作って tgt_shape に追加する(高速版)。

    ngSkin の per-influence 書き込みは重いので、一時メッシュに skinCluster を作って
    一括で書き、「skinCluster からレイヤー作成」(C++)→ 頂点番号で transfer_layers する。
    """
    from ngSkinTools2 import api as ng

    xf = cmds.listRelatives(tgt_shape, p=True, f=True)[0]
    tmp = cmds.duplicate(xf, n="rwt_tmp_layer")[0]
    try:
        for sh in cmds.listRelatives(tmp, s=True, f=True) or []:
            if cmds.getAttr(sh + ".intermediateObject"):
                cmds.delete(sh)
        tmp_shape = _mesh_shape(tmp)
        tmp_skin = cmds.skinCluster(src_infs, tmp, toSelectedBones=True, bindMethod=0,
                                    maximumInfluences=1, obeyMaxInfluences=False,
                                    normalizeWeights=1)[0]
        _write_weights(tmp_skin, tmp_shape, src_infs, W, nverts)
        ng.init_layers(tmp_skin).add(name)  # 最初のレイヤーは skinCluster から作られる
        ng.transfer_layers(tmp, xf, vertex_transfer_mode="vertexId")
    finally:
        if cmds.objExists(tmp):
            cmds.delete(tmp)


def _write_ngskin_layers(tgt_skin, tgt_shape, src_infs, W_closest, W_robust, inpaint_mask,
                         nverts, log):
    """ngSkinTools2 にレイヤー2枚で書き出す。

    下: RWT_Closest  … 最近接コピー(Maya標準相当)
    上: RWT_Inpaint  … 本手法の結果。マスク=補間した頂点(白)/確定した頂点(黒)
    既存の他のレイヤーには触らない(RWT_* だけ作り直す)。
    """
    from ngSkinTools2 import api as ng

    t = time.time()
    _write_weights(tgt_skin, tgt_shape, src_infs, W_robust, nverts)
    had_layers = bool(cmds.listConnections(tgt_skin, type="ngst2SkinLayerData"))
    layers = ng.init_layers(tgt_skin)
    for layer in layers.list():  # 再実行時は作り直す
        if layer.name in (LAYER_CLOSEST, LAYER_INPAINT):
            layers.delete(layer)
    others = layers.list()

    if not others:
        _write_weights(tgt_skin, tgt_shape, src_infs, W_closest, nverts)
        layers.add(LAYER_CLOSEST)  # レイヤーが無いときは skinCluster から作られる
    else:
        _layer_from_weights_via_temp(tgt_shape, src_infs, W_closest, nverts,
                                     LAYER_CLOSEST, log)
    _layer_from_weights_via_temp(tgt_shape, src_infs, W_robust, nverts, LAYER_INPAINT, log)

    top = [l for l in ng.Layers(tgt_skin).list() if l.name == LAYER_INPAINT][-1]
    top.set_weights(ng.NamedPaintTarget.MASK, list(inpaint_mask))
    top.set_current()
    log(u"ngSkinTools2 レイヤーに出力: %s(下) / %s(上, マスク付き) (%.1fs)%s"
        % (LAYER_CLOSEST, LAYER_INPAINT, time.time() - t,
           u" ※既存レイヤーの上に追加" if (had_layers and others) else u""))


def select_inpainted():
    mesh, ids = _last_inpainted["mesh"], _last_inpainted["ids"]
    if not mesh or not ids:
        cmds.warning(u"まだ転写していません")
        return
    cmds.select(["%s.vtx[%d]" % (mesh, i) for i in ids], r=True)


def select_unsolved():
    """補間できなかった頂点(コピーできる頂点が1つも無い別パーツ)を選択する。"""
    mesh, ids = _last_inpainted["mesh"], _last_inpainted.get("unsolved", [])
    if not mesh or not ids:
        cmds.warning(u"補間できなかった頂点はありません")
        return
    cmds.select(["%s.vtx[%d]" % (mesh, i) for i in ids], r=True)


# --------------------------------------------------------------------------
# UI (Qt / PySide2・PySide6 両対応)
# --------------------------------------------------------------------------
MODE_PAPER, MODE_EXT = 0, 1
MODE_NAMES = [u"論文準拠", u"拡張・小パーツ対応"]
# モードごとの既定値
#   論文準拠: 公式 Python 実装(src/utils.py, sphere_to_plane_transfer.py)と同じ値
#   拡張    : 距離 5cm / 裏面あり / スムースなし / 最大 8 本(Blender アドオンの既定値を参考)
MODE_DEFAULTS = {
    MODE_PAPER: {
        "dist": 5.0, "dist_unit": 0, "angle": 30.0, "flip": False, "smooth_iter": 10,
        "smooth_strength": 0.2, "max_inf": 0, "mask": False, "clamp": False, "output": 0,
        "select_after": False,
    },
    MODE_EXT: {
        "dist": 5.0, "dist_unit": 1, "angle": 30.0, "flip": True, "smooth_iter": 0,
        "smooth_strength": 0.2, "max_inf": 8, "mask": False, "clamp": False, "output": 0,
        "select_after": False,
    },
}
DEFAULTS = MODE_DEFAULTS[MODE_PAPER]

# オプション説明(ツールチップとヘルプ画面で共用)
HELP = [
    (u"ソース(体)", u"ウェイトのコピー元。スキン済みのメッシュを指定します。"),
    (u"ターゲット(服)", u"コピー先。スキンが無ければ、体と同じジョイントで自動バインドします。"),
    (u"モード",
     u"・論文準拠: 論文の公式 Python 実装と同じ処理だけを行う。比較・検証の基準。\n"
     u"・拡張・小パーツ対応: 点群補間・頂点の溶接・別パーツの救済・インフルエンス数の制限を加える。"
     u"ボタンやポーチ等も下の布と一緒に動きやすい。\n"
     u"設定はモードごとに保存されます。"),
    (u"距離",
     u"服の頂点から体の表面までがこの距離以内なら、体のウェイトをそのままコピー。"
     u"遠い頂点は補間に回す。単位は % (服の大きさに対する割合) か cm。\n"
     u"上げる→コピーが増え最近接コピーに近づく / 下げる→補間が増える。"),
    (u"角度",
     u"服と体の向き(法線)がこの角度以上ずれた頂点は、近くてもコピーせず補間に回す。"
     u"股や脇で反対側の脚・胴のウェイトを拾うのを防ぐ。\n"
     u"既定 30°(公式 Python 実装の値。論文本文では 35°)。"),
    (u"裏向きの面もコピー",
     u"厚みのある服・裏地の内側の面(体と逆向き)もコピー対象にする。"
     u"補間される頂点が多すぎるときに試す。"),
    (u"スムース",
     u"補間で埋めた部分とその周りだけを平均化して、コピー部分との境目をなじませる。"
     u"0回で無効。上げすぎるとウェイトがぼやける。"),
    (u"最大インフルエンス",
     u"1頂点あたりのジョイント数の上限。超えた分は小さい順に削る。"
     u"0 = 体の skinCluster の設定(maxInfluences)。論文準拠モードでは使わない(制限なし)。"),
    (u"選択頂点を必ず補間(マスク)",
     u"オンにして服の頂点を選択した状態で転写すると、その頂点は体からコピーせず周りから埋める。"
     u"襟が顎のウェイトを拾うとき等に。"),
    (u"安全モード",
     u"メッシュが壊れていて(潰れた三角形など)結果が乱れるときだけ使う。"
     u"計算を安定させる代わりに論文の式から少し外れる。拡張モードのみ。"),
    (u"補間できなかった頂点",
     u"コピーできる頂点が1つも無い別パーツは補間できないため、最近接コピーの値を入れて報告します。"
     u"拡張モードでは距離だけで基準点を補って補間します(マスク指定した頂点は除く)。"),
    (u"出力先",
     u"・skinCluster: 普通に書き込む。\n"
     u"・ngSkinレイヤー: 下=最近接コピー / 上=補間結果+マスクの2枚。"
     u"マスクを塗って部位ごとにどちらを使うか選べる。"),
    (u"完了後に補間頂点を選択", u"転写後、補間で埋めた頂点を選択して確認できるようにする。"),
    (u"元に戻す", u"1ファイル版は Undo できません。転写すると Undo 履歴もクリアされます。"
                 u"必要なら転写の前にシーンを保存してください。"),
]
_TIPS = dict(HELP)


HELP_WINDOW = "robustWeightTransferHelp"
_UNDO_NOTE = u"Undo はできません(Undo 履歴もクリアされます)。"
PAPER_URL = "https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/"

HELP_DETAIL = [
    (u"使い方", [
        u"1. Maya で 体(スキン済み) → 服 の順に選択します。服は Ctrl / Shift で複数追加できます。",
        u"2. モードを選んで「転写」を押します。",
        u"服にスキンが無い場合は、体と同じジョイントで自動的にバインドします。",
        u"1ファイル版は Undo できません。転写すると Undo 履歴もクリアされます。必要なら転写の前にシーンを保存してください。",
        u"バインドポーズの状態で実行してください。",
    ]),
    (u"計算環境(1ファイル版)", [
        u"計算には numpy / scipy を使います。Maya の Python で読めれば Maya の中で計算します。",
        u"Maya 2022 の Python 2 モードなど、Maya の中で読めない場合は、Maya に付属する Python 3 の mayapy を"
        u"別プロセスで起動して計算します(結果は同じです)。",
        u"numpy / scipy は Python 3 の mayapy に pip で入れます(mayapy2.exe ではありません)。例:",
        u'"C:\\Program Files\\Autodesk\\Maya2022\\bin\\mayapy.exe" -m pip install --user numpy scipy',
        u"ヘルプ → 計算環境を確認 で、使われる Python と numpy / scipy の有無を確認できます。"
        u"mayapy が別の場所にある場合は ヘルプ → Python 3 の場所を指定 で選びます(環境変数 RWT_PYTHON3 でも可)。",
    ]),
    (u"しくみ", [
        u"Robust Skin Weights Transfer via Weight Inpainting(Abdrashitov et al., SIGGRAPH Asia 2023)の実装です。",
        u"① 体に近く、向きもそろっている服の頂点は、体の表面のウェイトをそのままコピーします。",
        u"② それ以外の頂点(ゆったりした布、股・脇まわりなど)は、①の頂点から滑らかに補間します"
        u"(Q = -L + L M^-1 L の二次エネルギー最小化)。",
    ]),
    (u"モード: 論文準拠", [
        u"論文の公式 Python 実装(rin-23/RobustSkinWeightsTransferCode の src/utils.py)と同じ処理だけを行います。"
        u"結果が気になったときの比較の基準に使います。",
        u"・マッチング: 面積重みの頂点法線、ターゲットの対角長 × 距離(%)、角度 30°",
        u"・補間: L = 2 × cot ラプラシアン、M = ボロノイ混合面積(公式 Python 版と同じ。C++ 版は L = cot ラプラシアン)",
        u"・スムース: 公式と同じく、補間頂点とそこからメッシュをたどって距離以内の頂点を、頂点番号順に 10回・0.2",
        u"・最後に負の値を 0 にして正規化(論文 3.2 の処理)",
        u"使わない処理: 頂点の溶接、別パーツの救済、インフルエンス数の制限、小さい値の削除、点群補間、安全モード。",
        u"公式の球→平面データで、マッチング・補間・スムースとも公式 Python 版と一致することを確認しています"
        u"(Maya 上の最終ウェイトで差 1.2e-8 以下。Maya の頂点座標が単精度のため)。"
        u"実際の衣装では、Maya と libigl の最近接点計算の違いなどで差が出る可能性があります。",
        u"FBX 由来の衣装など、UV の境目で頂点が分かれたメッシュでは、分かれた部分の多くが"
        u"「補間できなかった頂点」になります。実用には拡張モードを使ってください。",
    ]),
    (u"モード: 拡張・小パーツ対応", [
        u"論文の処理に、実用向けの独自処理を加えたモードです。",
        u"・点群補間: メッシュの辺ではなく近くの点どうしをつないで補間。ボタン・ポーチ等の別パーツも下の布と一緒に動く",
        u"・頂点の溶接: 同じ位置にある頂点をまとめて計算(UV の境目で割れない)",
        u"・別パーツの救済: コピーできる頂点が無いパーツは、距離だけで基準点を補う",
        u"・インフルエンス数の制限と、0.0001 未満の値の削除",
        u"注意: 補間方法が服全体で変わります。左右の脚・脇・重なった布など、近いけれど別々に動かしたい場所も"
        u"つながりやすくなります。体の近くの硬い物(バックパック等)は周りに引っ張られやすくなります。",
        u"Blender アドオン robust-weight-transfer を参考にしていますが、点群補間は独自の簡易実装で、"
        u"同じ結果にはなりません。",
    ]),
    (u"距離", [
        u"服の頂点から体の表面までがこの距離以内なら、体のウェイトをそのままコピーします。"
        u"これより遠い頂点は補間に回します。",
        u"単位は右のメニューで選びます。% = 服のバウンディングボックス対角長に対する割合(論文の既定 5%)、cm = 絶対距離。",
        u"上げる → コピーが増え、普通の最近接コピーに近づきます。下げる → 補間が増えます。",
    ]),
    (u"角度 (°)", [
        u"服と体の向き(法線)のずれがこの角度以上の頂点は、距離が近くてもコピーせず補間に回します。",
        u"既定 30° は公式 Python 実装の値です。論文本文では 35° と書かれています。",
        u"股や脇で、反対側の脚や胴のウェイトを拾ってしまうのを防ぐための条件です。",
    ]),
    (u"スムース回数 / スムース強さ", [
        u"補間した部分とその周辺だけを周りの頂点と平均して、コピーした部分との境目をなじませます。回数 0 で無効。",
        u"論文準拠: 公式と同じ範囲・順序で計算(既定 10回 / 0.2)。",
        u"拡張: 補間頂点から空間的な距離以内の頂点を、まとめて更新(既定はオフ)。",
    ]),
    (u"最大インフルエンス", [
        u"1頂点あたりのインフルエンス(ジョイント)数の上限です。超えた分は小さい順に削って正規化します。",
        u"0 のときは体の skinCluster の設定(maxInfluences)に合わせます。論文準拠モードでは使いません(制限なし)。",
    ]),
    (u"出力先", [
        u"skinCluster: 服の skinCluster に直接書き込みます。",
        u"ngSkin レイヤー: ngSkinTools2 のレイヤー2枚で出力します。"
        u"下 = RWT_Closest(最近接コピー)、上 = RWT_Inpaint(補間結果 + 補間領域のマスク)。"
        u"マスクを塗ると、部位ごとに「最近接」と「補間」を選べます。再実行時は RWT_* の2枚だけ作り直します。",
    ]),
    (u"裏面もコピー", [
        u"厚みのある服や裏地の内側の面は、体と逆向きになります。オンにすると、こうした面もコピーの対象にします"
        u"(論文の任意オプション)。補間される頂点が多すぎるときに試してください。",
    ]),
    (u"選択頂点を補間", [
        u"オンにして、服の頂点を選択した状態(体 → 服の頂点 の順)で転写すると、選んだ頂点は体からコピーせず、"
        u"周りから補間します(論文 3.4 のマスク)。どちらのモードでも必ず守られます。",
        u"例: 襟が顎のウェイトを拾ってしまうとき、襟の頂点を選んで使います。",
    ]),
    (u"補間できなかった頂点", [
        u"コピーできる頂点が1つも無い別パーツ(ボタン等)は、補間の基準が無いため補間できません。"
        u"その頂点には最近接コピーの値を入れ、結果欄に件数を表示します。「補間できなかった頂点を選択」で確認できます。",
        u"拡張モードでは、距離だけで基準点を補って補間します。ただしマスクで指定した頂点は基準点にしません。",
    ]),
    (u"安全モード(拡張のみ)", [
        u"潰れた三角形などでメッシュが壊れていて結果が乱れるときだけ使います。",
        u"計算を安定させる代わりに、論文の式から少し外れます。",
    ]),
    (u"完了後に補間頂点を選択", [
        u"転写後、補間で埋めた頂点を選択します。結果欄の「補間頂点を選択」ボタンでも選べます。",
    ]),
    (u"既定値に戻す", [
        u"編集メニュー、または詳細設定の「リセット」で、今のモードの設定を既定値に戻します。",
    ]),
]


def show_help():
    """各項目の説明(Maya 標準 UI)。"""
    if cmds.window(HELP_WINDOW, exists=True):
        cmds.deleteUI(HELP_WINDOW)
    cmds.window(HELP_WINDOW, title=u"Robust Skin Weights Transfer - ヘルプ",
                widthHeight=(520, 640))
    cmds.scrollLayout(cr=True)
    cmds.columnLayout(adj=True, rs=4, co=("both", 6))
    cmds.text(l=u"Robust Skin Weights Transfer  v%s  (%s)" % (__version__, EDITION), al="left",
              fn="boldLabelFont", h=26)
    for title, lines in HELP_DETAIL:
        cmds.frameLayout(l=title, cll=True, cl=False, mw=8, mh=6)
        cmds.columnLayout(adj=True, rs=4)
        for line in lines:
            # 折り返し後の行数から高さを決める(固定高さだと最終行が切れるため)
            n = int(math.ceil(sum(2 if ord(ch) > 0x2E7F else 1 for ch in line) / 76.0))
            cmds.text(l=line, al="left", ww=True, h=18 * max(1, n) + 2)
        cmds.setParent("..")
        cmds.setParent("..")
    cmds.separator(h=10, style="none")
    cmds.rowLayout(nc=2, adj=1)
    cmds.button(l=u"論文ページを開く", c=lambda *_: cmds.launch(web=PAPER_URL))
    cmds.button(l=u"閉じる", w=90, c=lambda *_: cmds.deleteUI(HELP_WINDOW))
    cmds.setParent("..")
    cmds.showWindow(HELP_WINDOW)


def show_about():
    cmds.confirmDialog(
        t=u"バージョン情報",
        m=(u"Robust Skin Weights Transfer  v%s\n\n"
           u"Abdrashitov, Raichstat, Monsen, Hill.\n"
           u"Robust Skin Weights Transfer via Weight Inpainting.\n"
           u"SIGGRAPH Asia 2023 Technical Communications.\n\n"
           u"論文をもとにした独自実装です。" % (__version__ + u" " + EDITION)),
        b=["OK"])


def _qt():
    try:
        from PySide6 import QtWidgets, QtCore, QtGui
        from shiboken6 import wrapInstance
    except ImportError:
        from PySide2 import QtWidgets, QtCore, QtGui
        from shiboken2 import wrapInstance
    return QtWidgets, QtCore, QtGui, wrapInstance


def _maya_main_window():
    import maya.OpenMayaUI as omui
    QtWidgets, _, _, wrapInstance = _qt()
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget) if ptr else None


def _opt_get(key, mode):
    name = "%sm%d_%s" % (_OPT, mode, key)
    d = MODE_DEFAULTS[mode][key]
    if cmds.optionVar(exists=name):
        v = cmds.optionVar(q=name)
        return bool(v) if isinstance(d, bool) else type(d)(v)
    return d


def _opt_set(key, value, mode):
    name = "%sm%d_%s" % (_OPT, mode, key)
    if isinstance(value, (bool, int)):
        cmds.optionVar(iv=(name, int(value)))
    else:
        cmds.optionVar(fv=(name, float(value)))


def _selected_target_vertices(target):
    sel = cmds.ls(sl=True, fl=True) or []
    ids = []
    shape = _mesh_shape(target)
    xf = cmds.listRelatives(shape, p=True, f=True)[0]
    for c in cmds.ls(cmds.polyListComponentConversion(sel, tv=True) or [], fl=True):
        node, _, rest = c.partition(".vtx[")
        if rest and cmds.ls(node, l=True)[0] in (xf, cmds.ls(shape, l=True)[0]):
            ids.append(int(rest[:-1]))
    return ids


def _as_transform(node):
    if cmds.nodeType(node) == "mesh":
        node = cmds.listRelatives(node, p=True, f=True)[0]
    return cmds.ls(node)[0]  # 一意になる最短名


def _selection_meshes():
    """選択順に、メッシュを持つトランスフォームを重複なしで返す(コンポーネント選択も可)。"""
    out, seen = [], set()
    for n in cmds.ls(sl=True, o=True, l=True) or []:
        try:
            shape = _mesh_shape(n)
        except Exception:
            continue
        xf = cmds.listRelatives(shape, p=True, f=True)[0]
        if xf not in seen:
            seen.add(xf)
            out.append(xf)
    return out


def _short(node):
    return node.split("|")[-1]


def _validate(src, tgt):
    if not src or not tgt:
        return u"ソース(体)とターゲット(服)を指定してください。"
    for n in (src, tgt):
        if not cmds.objExists(n):
            return u"見つかりません: %s" % n
        if len(cmds.ls(n)) > 1:
            return (u"同じ名前のノードが複数あります: %s\n"
                    u"シーンで選択して「選択から」で指定し直してください。" % n)
        try:
            _mesh_shape(n)
        except RuntimeError as e:
            return _err_text(e)
    if not find_skin_cluster(src):
        return u"ソース(体)に skinCluster がありません: %s" % src
    if cmds.ls(cmds.listRelatives(_mesh_shape(src), p=True, f=True), l=True) == \
            cmds.ls(cmds.listRelatives(_mesh_shape(tgt), p=True, f=True), l=True):
        return u"ソースとターゲットが同じです。"
    return None


# ログの文言 -> 進捗(%)
_PROGRESS_KEYS = [
    (u"バインド", 5), (u"距離しきい値", 10), (u"Step1", 30), (u"補間モード", 40),
    (u"確定点のない島", 45), (u"で解いています", 55), (u"スムース", 75), (u"Step2", 82),
    (u"ngSkinTools2", 95), (u"書き込み", 95), (u"完了", 100),
]

_STYLE = """
QWidget#rwtRoot { font-size: 12px; }
QLabel#rwtTitle { font-size: 16px; font-weight: bold; }
QLabel#rwtSub { color: #9a9a9a; }
QLabel#rwtSection { color: #8fb8de; font-weight: bold; padding-top: 4px; }
QLabel#rwtHint { color: #9a9a9a; font-size: 11px; }
QFrame#rwtCard { background: #2f3336; border: 1px solid #3c4246; border-radius: 6px; }
QPushButton#rwtRun { background: #3d7ab8; color: white; font-size: 14px; font-weight: bold;
                     border: none; border-radius: 5px; padding: 9px; }
QPushButton#rwtRun:hover { background: #4a8bcc; }
QPushButton#rwtRun:disabled { background: #44515d; color: #9aa; }
QPushButton#rwtFlat { border: 1px solid #555; border-radius: 4px; padding: 4px 10px; }
QToolButton#rwtFold { border: none; color: #8fb8de; font-weight: bold; }
QProgressBar { border: none; background: #2f3336; border-radius: 3px; }
QProgressBar::chunk { background: #3d7ab8; border-radius: 3px; }
"""

_window = None


def show():
    """ツールを開く。体 → 服の順に選択してから開くと自動で入る。"""
    global _window
    QtWidgets, QtCore, QtGui, _ = _qt()
    if _window is not None:
        try:
            _window.close()
            _window.deleteLater()
        except Exception:
            pass
    _window = _RwtWindow(_maya_main_window())
    _window.show()
    return _window


def _build_window_class():
    QtWidgets, QtCore, QtGui, _ = _qt()

    class RwtWindow(QtWidgets.QDialog):
        def __init__(self, parent=None):
            super(RwtWindow, self).__init__(parent)
            self.setObjectName("rwtRoot")
            self.setWindowTitle(u"Robust Skin Weights Transfer")
            self.setWindowFlags(self.windowFlags() | QtCore.Qt.Tool)
            self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
            self.setStyleSheet(_STYLE)
            self.setMinimumWidth(430)
            self._build()
            self._load_settings()
            self._job = cmds.scriptJob(e=["SelectionChanged", self._refresh_selection])
            self._refresh_selection()

        # ---------------- 構築 ----------------
        def _section(self, layout, text):
            lab = QtWidgets.QLabel(text)
            lab.setObjectName("rwtSection")
            layout.addWidget(lab)

        def _build(self):
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(12, 4, 12, 12)
            root.setSpacing(8)

            # メニュー
            bar = QtWidgets.QMenuBar(self)
            m_edit = bar.addMenu(u"編集")
            m_edit.addAction(u"このモードの既定値に戻す", self._reset)
            m_help = bar.addMenu(u"ヘルプ")
            m_help.addAction(u"各項目の説明...", show_help)
            m_help.addAction(u"論文ページを開く", lambda: cmds.launch(web=PAPER_URL))
            m_help.addSeparator()
            m_help.addAction(u"計算環境を確認...", self._env_check)
            m_help.addAction(u"Python 3 の場所を指定...", self._pick_python3)
            m_help.addSeparator()
            m_help.addAction(u"バージョン情報", show_about)
            root.setMenuBar(bar)

            # モード
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(u"モード")
            lab.setToolTip(_TIPS[u"モード"])
            self.mode = QtWidgets.QComboBox()
            self.mode.addItems(MODE_NAMES)
            self.mode.setToolTip(_TIPS[u"モード"])
            row.addWidget(lab)
            row.addWidget(self.mode, 1)
            root.addLayout(row)

            # 詳細設定(折りたたみ)
            self.fold = QtWidgets.QToolButton()
            self.fold.setObjectName("rwtFold")
            self.fold.setText(u"詳細設定")
            self.fold.setCheckable(True)
            self.fold.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            self.fold.setArrowType(QtCore.Qt.RightArrow)
            self.fold.toggled.connect(self._toggle_adv)
            root.addWidget(self.fold)
            self.adv = QtWidgets.QFrame()
            self.adv.setObjectName("rwtCard")
            form = QtWidgets.QFormLayout(self.adv)
            form.setContentsMargins(10, 8, 10, 8)
            form.setLabelAlignment(QtCore.Qt.AlignRight)
            form.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)

            def add(label, w, key):
                # 入力欄・チェックボックスとも右の列に揃え、幅いっぱいに広げる
                w.setToolTip(_TIPS[key])
                w.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
                lab = QtWidgets.QLabel(label)
                lab.setToolTip(_TIPS[key])
                form.addRow(lab, w)

            def dspin(lo, hi, dec, step):
                w = QtWidgets.QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setDecimals(dec)
                w.setSingleStep(step)
                return w

            def ispin(lo, hi):
                w = QtWidgets.QSpinBox()
                w.setRange(lo, hi)
                return w

            self.dist = dspin(0.01, 1000.0, 2, 0.5)
            self.dist_unit = QtWidgets.QComboBox()
            self.dist_unit.addItems([u"%", u"cm"])
            self.dist_unit.setToolTip(u"% = 服のバウンディングボックス対角長に対する割合 / cm = 絶対距離")
            dist_box = QtWidgets.QWidget()
            dl = QtWidgets.QHBoxLayout(dist_box)
            dl.setContentsMargins(0, 0, 0, 0)
            dl.setSpacing(4)
            dl.addWidget(self.dist, 1)
            dl.addWidget(self.dist_unit)
            self.dist.setToolTip(_TIPS[u"距離"])
            dist_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            lab = QtWidgets.QLabel(u"距離")
            lab.setToolTip(_TIPS[u"距離"])
            form.addRow(lab, dist_box)
            self.angle = dspin(1.0, 90.0, 1, 1.0)
            add(u"角度 (°)", self.angle, u"角度")
            self.smooth_iter = ispin(0, 50)
            add(u"スムース回数", self.smooth_iter, u"スムース")
            self.smooth_strength = dspin(0.0, 1.0, 2, 0.05)
            add(u"スムース強さ", self.smooth_strength, u"スムース")
            self.max_inf = ispin(0, 16)
            add(u"最大インフルエンス", self.max_inf, u"最大インフルエンス")
            self.output = QtWidgets.QComboBox()
            self.output.addItems([u"skinCluster", u"ngSkin レイヤー"])
            add(u"出力先", self.output, u"出力先")
            self.ng_ok = has_ngskin()
            if not self.ng_ok:
                self.output.model().item(1).setEnabled(False)
            self.flip = QtWidgets.QCheckBox(u"裏面もコピー")
            add(u"", self.flip, u"裏向きの面もコピー")
            self.mask = QtWidgets.QCheckBox(u"選択頂点を補間")
            add(u"", self.mask, u"選択頂点を必ず補間(マスク)")
            self.clamp = QtWidgets.QCheckBox(u"安全モード")
            add(u"", self.clamp, u"安全モード")
            self.select_after = QtWidgets.QCheckBox(u"完了後に補間頂点を選択")
            add(u"", self.select_after, u"完了後に補間頂点を選択")
            reset = QtWidgets.QPushButton(u"リセット")
            reset.setObjectName("rwtFlat")
            reset.setToolTip(u"今のモードの設定を既定値に戻す")
            reset.clicked.connect(self._reset)
            rb = QtWidgets.QHBoxLayout()
            rb.addStretch(1)
            rb.addWidget(reset)
            form.addRow(rb)
            self.adv.setVisible(False)
            root.addWidget(self.adv)

            # 実行
            self.run_btn = QtWidgets.QPushButton(u"転写")
            self.run_btn.setObjectName("rwtRun")
            self.run_btn.setToolTip(u"Maya で 体 → 服 の順に選択して実行(服は複数可)。\n"
                                    + _UNDO_NOTE)
            self.run_btn.clicked.connect(self._run)
            root.addWidget(self.run_btn)
            self.progress = QtWidgets.QProgressBar()
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress.setTextVisible(False)
            self.progress.setFixedHeight(6)
            root.addWidget(self.progress)
            self.status = QtWidgets.QLabel(u"")
            self.status.setObjectName("rwtHint")
            self.status.setWordWrap(True)
            root.addWidget(self.status)

            # 結果
            self.result = QtWidgets.QFrame()
            self.result.setObjectName("rwtCard")
            rl = QtWidgets.QHBoxLayout(self.result)
            rl.setContentsMargins(10, 6, 10, 6)
            self.result_label = QtWidgets.QLabel()
            self.result_label.setWordWrap(True)
            rl.addWidget(self.result_label, 1)
            sel_btn = QtWidgets.QPushButton(u"補間頂点を選択")
            sel_btn.setObjectName("rwtFlat")
            sel_btn.setToolTip(_TIPS[u"完了後に補間頂点を選択"])
            sel_btn.clicked.connect(lambda: select_inpainted())
            btns = QtWidgets.QVBoxLayout()
            btns.addWidget(sel_btn)
            self.unsolved_btn = QtWidgets.QPushButton(u"補間できなかった頂点を選択")
            self.unsolved_btn.setObjectName("rwtFlat")
            self.unsolved_btn.setToolTip(_TIPS[u"補間できなかった頂点"])
            self.unsolved_btn.clicked.connect(lambda: select_unsolved())
            btns.addWidget(self.unsolved_btn)
            rl.addLayout(btns)
            self.result.setVisible(False)
            root.addWidget(self.result)
            root.addStretch(1)

        # ---------------- 設定 ----------------
        def _widgets(self):
            return {
                "dist": self.dist, "dist_unit": self.dist_unit,
                "angle": self.angle, "flip": self.flip,
                "smooth_iter": self.smooth_iter, "smooth_strength": self.smooth_strength,
                "max_inf": self.max_inf, "mask": self.mask, "clamp": self.clamp,
                "output": self.output, "select_after": self.select_after,
            }

        @staticmethod
        def _val(w):
            if isinstance(w, QtWidgets.QCheckBox):
                return w.isChecked()
            if isinstance(w, QtWidgets.QComboBox):
                return w.currentIndex()
            return w.value()

        @staticmethod
        def _put(w, v):
            if isinstance(w, QtWidgets.QCheckBox):
                w.setChecked(bool(v))
            elif isinstance(w, QtWidgets.QComboBox):
                w.setCurrentIndex(int(v))
            else:
                w.setValue(v)

        def _cur_mode(self):
            return self.mode.currentIndex()

        def _load_mode_settings(self, mode):
            for k, w in self._widgets().items():
                try:
                    self._put(w, _opt_get(k, mode))
                except Exception:
                    self._put(w, MODE_DEFAULTS[mode][k])
            if not self.ng_ok and self.output.currentIndex() == 1:
                self.output.setCurrentIndex(0)
            self._update_mode_ui()

        def _save_mode_settings(self, mode):
            for k, w in self._widgets().items():
                _opt_set(k, self._val(w), mode)

        def _load_settings(self):
            m = cmds.optionVar(q=_OPT + "mode") if cmds.optionVar(exists=_OPT + "mode") else 0
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(int(m) if int(m) in MODE_DEFAULTS else 0)
            self.mode.blockSignals(False)
            self._shown_mode = self._cur_mode()
            self._load_mode_settings(self._shown_mode)
            self.mode.currentIndexChanged.connect(self._on_mode)
            self.fold.setChecked(bool(cmds.optionVar(q=_OPT + "advOpen"))
                                 if cmds.optionVar(exists=_OPT + "advOpen") else False)

        def _save_settings(self):
            self._save_mode_settings(self._shown_mode)
            cmds.optionVar(iv=(_OPT + "mode", self._cur_mode()))
            cmds.optionVar(iv=(_OPT + "advOpen", int(self.fold.isChecked())))

        def _on_mode(self, idx):
            self._save_mode_settings(self._shown_mode)  # 切り替え前のモードの設定を保存
            self._shown_mode = idx
            self._load_mode_settings(idx)
            cmds.optionVar(iv=(_OPT + "mode", idx))
            self._set_status(u"")

        def _update_mode_ui(self):
            paper = self._cur_mode() == MODE_PAPER
            for w in (self.max_inf, self.clamp):
                w.setEnabled(not paper)

        def _reset(self):
            self._load_mode_defaults()

        def _load_mode_defaults(self):
            mode = self._cur_mode()
            for k, w in self._widgets().items():
                self._put(w, MODE_DEFAULTS[mode][k])
            if not self.ng_ok and self.output.currentIndex() == 1:
                self.output.setCurrentIndex(0)
            self._save_settings()
            self._set_status(u"")

        def closeEvent(self, event):
            self._kill_job()
            self._save_settings()
            super(RwtWindow, self).closeEvent(event)

        # ---------------- 操作 ----------------
        def _env_check(self):
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                text = environment_report()
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle(u"計算環境")
            box.setText(text)
            box.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            box.exec_()

        def _pick_python3(self):
            start = _default_mayapy()
            path = QtWidgets.QFileDialog.getOpenFileName(
                self, u"Python 3 の mayapy を選択", os.path.dirname(start),
                u"Python (mayapy*.exe python*.exe mayapy python*);;すべて (*)")[0]
            if not path:
                return
            path = os.path.normpath(path)
            _py3_cache.pop(path, None)
            ok, msg = check_python3(path)
            if ok:
                cmds.optionVar(sv=(_OPT_PY3, path))
                QtWidgets.QMessageBox.information(self, u"計算環境", u"設定しました。\n%s\n%s"
                                                  % (path, msg))
            else:
                QtWidgets.QMessageBox.warning(self, u"計算環境",
                                              u"使えません: %s\n%s\n\n%s"
                                              % (path, msg, pip_command()))

        def _toggle_adv(self, on):
            self.fold.setArrowType(QtCore.Qt.DownArrow if on else QtCore.Qt.RightArrow)
            self.adv.setVisible(on)
            QtCore.QTimer.singleShot(0, self.adjustSize)

        def _kill_job(self):
            job = getattr(self, "_job", None)
            if job and cmds.scriptJob(exists=job):
                cmds.scriptJob(kill=job, force=True)
            self._job = None

        def _refresh_selection(self):
            try:
                meshes = _selection_meshes()
            except Exception:
                meshes = []
            self._meshes = meshes
            ok = len(meshes) >= 2
            self.run_btn.setEnabled(ok)
            if ok:
                tip = u"元: %s\n服: %s" % (_short(meshes[0]), u", ".join(_short(m) for m in meshes[1:]))
            else:
                tip = u"Maya で 体 → 服 の順に選択してください(服は複数可)"
            self.run_btn.setToolTip(tip + u"\n\n" + _UNDO_NOTE)

        def _set_status(self, text):
            self.status.setText(text)
            QtWidgets.QApplication.processEvents()

        def _run(self):
            meshes = list(getattr(self, "_meshes", []) or _selection_meshes())
            if len(meshes) < 2:
                QtWidgets.QMessageBox.warning(self, u"Robust Skin Weights Transfer",
                                              u"体 → 服 の順に2つ以上選択してください。")
                return
            src, targets = meshes[0], meshes[1:]
            for tgt in targets:
                err = _validate(src, tgt)
                if err:
                    QtWidgets.QMessageBox.warning(self, u"Robust Skin Weights Transfer", err)
                    return
            self._save_settings()
            # マスク: 各ターゲットで選択中の頂点(実行前に読む)
            masks = {t: (_selected_target_vertices(t) if self.mask.isChecked() else None)
                     for t in targets}
            base_kw = dict(
                dist_ratio=self.dist.value() / 100.0,
                dist_abs=self.dist.value() if self.dist_unit.currentIndex() == 1 else None,
                angle=self.angle.value(),
                allow_flip=self.flip.isChecked(),
                smooth_iter=self.smooth_iter.value(),
                smooth_strength=self.smooth_strength.value(),
                clamp_cot=self.clamp.isChecked(),
                output="ngskin" if self.output.currentIndex() == 1 else "skin",
                max_influences=self.max_inf.value() or None,
                mode="paper" if self._cur_mode() == MODE_PAPER else "extended",
                interp="pointcloud",  # 拡張モードでのみ使われる
            )
            n = len(targets)
            state = {"i": 0}

            def log(msg):
                print(u"[RobustWT] " + msg)
                for key, pct in _PROGRESS_KEYS:
                    if key in msg:
                        v = int((state["i"] * 100 + pct) / n)
                        self.progress.setValue(max(self.progress.value(), v))
                        break
                prefix = (u"[%d/%d %s] " % (state["i"] + 1, n, _short(targets[state["i"]]))
                          if n > 1 else u"")
                self._set_status(prefix + msg.strip())

            self.run_btn.setEnabled(False)
            self.result.setVisible(False)
            self.progress.setValue(2)
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            t0 = time.time()
            results = []
            sel_before = cmds.ls(sl=True, l=True)
            tgt = targets[0]
            try:
                with _NoUndo():
                    for i, tgt in enumerate(targets):
                        state["i"] = i
                        t = time.time()
                        res = transfer(src, tgt, log=log, mask_vertices=masks[tgt], **base_kw)
                        results.append((tgt, res, time.time() - t))
                    if sel_before:  # 選択を元に戻す
                        cmds.select(sel_before, r=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.progress.setValue(0)
                msg = _err_text(e)
                self._set_status(u"エラー: %s" % msg.splitlines()[0] if msg else u"エラー")
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(
                    self, u"Robust Skin Weights Transfer",
                    u"転写に失敗しました(%s)。\n\n%s\n\n詳細はスクリプトエディタを確認してください。"
                    % (_short(tgt), msg))
                return
            finally:
                self._refresh_selection()
            QtWidgets.QApplication.restoreOverrideCursor()
            self.progress.setValue(100)
            lines = [u"<b>完了</b>  %.1f 秒  (%s)" % (time.time() - t0,
                                                      MODE_NAMES[self._cur_mode()])]
            any_unsolved = False
            for tgt, res, dt in results:
                total = res["known"] + res["inpainted"]
                lines.append(u"%s  コピー %.0f%% / 補間 %.0f%%"
                             % (_short(tgt), 100.0 * res["known"] / max(total, 1),
                                100.0 * res["inpainted"] / max(total, 1)))
                if res.get("unsolved"):
                    any_unsolved = True
                    lines.append(u"<span style='color:#e0b050'>  ⚠ 補間できなかった別パーツ %d 個"
                                 u"(%d 頂点)は最近接コピー</span>"
                                 % (res["unsolved_parts"], res["unsolved"]))
            self.unsolved_btn.setVisible(any_unsolved)
            self.result_label.setText(u"<br>".join(lines))
            self.result.setVisible(True)
            self._set_status(u"")
            cmds.inViewMessage(amg=u"Robust Skin Weights Transfer: 完了", pos="topCenter",
                               fade=True)
            if self.select_after.isChecked():
                select_inpainted()
            QtCore.QTimer.singleShot(0, self.adjustSize)

    return RwtWindow


class _LazyWindow(object):
    """Qt クラスは Maya 上で初めて使うときに作る(モジュール読み込みを軽く保つ)。"""
    _cls = None

    def __call__(self, *a, **k):
        if _LazyWindow._cls is None:
            _LazyWindow._cls = _build_window_class()
        return _LazyWindow._cls(*a, **k)


_RwtWindow = _LazyWindow()


def install_shelf():
    """現在のシェルフにボタンを追加する。"""
    import maya.mel as mel
    shelf = mel.eval("$_tmp = $gShelfTopLevel")
    current = cmds.tabLayout(shelf, q=True, st=True)
    cmds.shelfButton(p=current, l="RobustWT", iol="RWT", i="commandButton.png",
                     ann=u"Robust Skin Weights Transfer",
                     c="import robust_weight_transfer as rwt; rwt.show()", stp="python")
    print(u"シェルフ '%s' に RobustWT ボタンを追加しました" % current)


if __name__ == "__main__" and _IN_WORKER:
    sys.exit(_worker_main(sys.argv))
