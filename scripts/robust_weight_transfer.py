# -*- coding: utf-8 -*-
"""
Robust Skin Weights Transfer for Maya
=====================================
Abdrashitov et al., "Robust Skin Weights Transfer via Weight Inpainting"
(SIGGRAPH Asia 2023 Technical Communications) を論文ベースで独自実装したもの。

  1. Closest point match : ソース表面の最近接点からウェイトを取得。
     距離・法線角度の条件を満たす頂点だけを「確定(known)」とする。
  2. Weight inpainting   : 残りの頂点は Q = -L + L M^-1 L (L: cotan Laplacian,
     M: 質量行列) の二次エネルギー最小化で補間する。

- ターゲットがスキニングされていなくてもOK(ソースのインフルエンスで自動バインド)。
- UVシーム等で分割された頂点は位置で溶接してから補間する。
- numpyのみで動作(scipyがあれば高速な直接法を使う)。

使い方:
    import robust_weight_transfer as rwt
    rwt.show()                       # UI
    rwt.transfer("base", "cloth")    # スクリプトから
"""
from __future__ import annotations

import math
import time

import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

try:  # 任意: あれば直接法で高速に解く
    import scipy.sparse as _sp
    import scipy.sparse.linalg as _spla
    HAS_SCIPY = True
except Exception:  # pragma: no cover
    HAS_SCIPY = False

__version__ = "2.4.0"
WINDOW = "robustWeightTransferWin"
_last_inpainted = {"mesh": None, "ids": []}
_PENDING = {}  # rwtSetSkinWeights コマンドへの受け渡し用
_PLUGIN = "rwt_undo_cmd"
LAYER_CLOSEST = "RWT_Closest"
LAYER_INPAINT = "RWT_Inpaint"


def has_ngskin():
    """ngSkinTools2 が使えるか(Maya 2027 は標準同梱)。"""
    try:
        if not cmds.pluginInfo("ngSkinTools2", q=True, loaded=True):
            cmds.loadPlugin("ngSkinTools2", quiet=True)
        import ngSkinTools2.api  # noqa: F401
        return cmds.pluginInfo("ngSkinTools2", q=True, loaded=True)
    except Exception:
        return False


def _ensure_plugin():
    """Undo対応の書き込みコマンド(同じフォルダの rwt_undo_cmd.py)を読み込む。"""
    import os
    if cmds.pluginInfo(_PLUGIN, q=True, loaded=True):
        return
    # MAYA_PLUG_IN_PATH 上(例: Documents/maya/plug-ins)にあれば名前で読む
    try:
        cmds.loadPlugin(_PLUGIN + ".py", quiet=True)
    except RuntimeError:
        here = os.path.dirname(os.path.abspath(__file__))
        for cand in (os.path.join(here, _PLUGIN + ".py"),
                     os.path.join(here, os.pardir, "plug-ins", _PLUGIN + ".py")):
            if os.path.exists(cand):
                cmds.loadPlugin(os.path.normpath(cand), quiet=True)
                break
    if not cmds.pluginInfo(_PLUGIN, q=True, loaded=True):
        raise RuntimeError(u"Undo用プラグイン %s.py を読み込めませんでした。"
                           u"プラグインマネージャで許可してください。" % _PLUGIN)


# --------------------------------------------------------------------------
# Maya helpers
# --------------------------------------------------------------------------
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


def _mesh_arrays(shape):
    dag = _dag(shape)
    fn = om.MFnMesh(dag)
    pts = np.array([(p.x, p.y, p.z) for p in fn.getPoints(om.MSpace.kWorld)], dtype=np.float64)
    nrm = np.array([(n.x, n.y, n.z) for n in fn.getVertexNormals(True, om.MSpace.kWorld)],
                   dtype=np.float64)
    _, tri_ids = fn.getTriangles()
    tris = np.array(tri_ids, dtype=np.int64).reshape(-1, 3)
    return dag, fn, pts, _normalize(nrm), tris


def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return v / n


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


# --------------------------------------------------------------------------
# Step 1: closest point matching
# --------------------------------------------------------------------------
def _closest_point_match(src_shape, src_pts, src_nrm, tgt_pts, tgt_nrm,
                         dist_thresh, angle_deg, allow_flip):
    dag = _dag(src_shape)
    fn = om.MFnMesh(dag)
    inter = om.MMeshIntersector()
    inter.create(dag.node(), dag.inclusiveMatrix())

    n = len(tgt_pts)
    tri_v = np.zeros((n, 3), dtype=np.int64)
    bary = np.zeros((n, 3), dtype=np.float64)
    tri_cache = {}
    for i in range(n):
        p = tgt_pts[i]
        pom = inter.getClosestPoint(om.MPoint(p[0], p[1], p[2]))
        key = (pom.face, pom.triangle)
        tv = tri_cache.get(key)
        if tv is None:
            tv = tuple(fn.getPolygonTriangleVertices(pom.face, pom.triangle))
            tri_cache[key] = tv
        u, v = pom.barycentricCoords
        tri_v[i] = tv
        bary[i] = (u, v, 1.0 - u - v)

    bary = np.clip(bary, 0.0, 1.0)
    bary /= np.maximum(bary.sum(1, keepdims=True), 1e-12)
    cp = np.einsum("ij,ijk->ik", bary, src_pts[tri_v])
    cn = _normalize(np.einsum("ij,ijk->ik", bary, src_nrm[tri_v]))
    dist = np.linalg.norm(cp - tgt_pts, axis=1)
    dots = np.einsum("ij,ij->i", cn, tgt_nrm)
    if allow_flip:
        dots = np.abs(dots)
    cos_t = math.cos(math.radians(angle_deg))
    known = (dist <= dist_thresh) & (dots >= cos_t)
    return tri_v, bary, known, dist


# --------------------------------------------------------------------------
# Step 2: inpainting
# --------------------------------------------------------------------------
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


def _inpaint(pts, tris, known, w_known, max_iter=4000, tol=1e-7, log=print, clamp=False,
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
        Q = (L + L @ _sp.diags(1.0 / mass) @ L).tocsr()
        U = np.where(unknown)[0]
        K = np.where(~unknown)[0]
        Quu = Q[U][:, U].tocsc()
        cols_used = np.where(np.abs(W[known]).sum(0) > 0)[0]
        rhs = -(Q[U][:, K] @ W[np.ix_(K, cols_used)])
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
        nbr_sum = lambda Y: A @ Y
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
    W = np.clip(W, 0.0, None)
    W[W < prune] = 0.0
    if max_inf and max_inf > 0 and W.shape[1] > max_inf:
        # 上位 max_inf 個だけ残す
        idx = np.argsort(-W, axis=1)[:, max_inf:]
        np.put_along_axis(W, idx, 0.0, axis=1)
    s = W.sum(1, keepdims=True)
    bad = s[:, 0] < 1e-8
    s[bad] = 1.0
    return W / s, bad


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def transfer(source, target, dist_ratio=0.05, angle=30.0, allow_flip=False,
             max_influences=None, smooth_iter=10, smooth_strength=0.2,
             mask_vertices=None, output="skin", weld=True, clamp_cot=False,
             interp="mesh", dist_abs=None, log=None):
    """source のスキンウェイトを target に Robust Skin Weights Transfer で転写する。

    既定値は論文の公式実装(rin-23/RobustSkinWeightsTransferCode)に合わせている。

    dist_ratio : ターゲットのバウンディングボックス対角長に対する距離しきい値(公式 0.05)
    dist_abs   : 距離しきい値を cm で直接指定(指定時は dist_ratio より優先。Blender版は 5cm)
    angle      : 法線の角度しきい値(度)(公式 30)
    allow_flip : 裏向き法線も一致とみなす(論文の任意オプション。重ね着・厚みのある服向け)
    max_influences : 1頂点あたりの最大インフルエンス数(None=ソースと同じ)
    smooth_iter / smooth_strength : 補間頂点とその距離しきい値内の近傍に掛ける
                    ラプラシアンスムース(論文3.3。公式 10回 / 0.2)
    clamp_cot  : 負の cot 重みを 0 に丸める(破綻メッシュ向けの安全策。論文とは別物になる)
    interp     : "mesh"       = メッシュの cot ラプラシアンで補間(論文・公式実装どおり)
                 "pointcloud" = 点群ラプラシアンで補間(Blender版の方式に近い簡易版)。
                                別パーツやシームを越えて、近くの布からウェイトが流れる
    mask_vertices : 強制的に補間させる頂点番号(論文3.4のマスク。襟が顎に付く等の対策)
    output : "skin"   = skinCluster に直接書き込む
             "ngskin" = ngSkinTools2 のレイヤー2枚で出力
                        (RWT_Closest: 最近接コピー / RWT_Inpaint: 本手法 + 補間領域マスク)
    """
    log = log or (lambda s: print(u"[RobustWT] " + s))
    _ensure_plugin()
    if output == "ngskin" and not has_ngskin():
        raise RuntimeError(u"ngSkinTools2 が読み込めません")
    # バインド・インフルエンス追加・ウェイト書き込みをまとめて1回のUndoで戻せるようにする
    cmds.undoInfo(openChunk=True, chunkName="RobustWeightTransfer")
    try:
        return _transfer(source, target, dist_ratio, angle, allow_flip, max_influences,
                         smooth_iter, smooth_strength, mask_vertices, output, weld,
                         clamp_cot, interp, dist_abs, log)
    finally:
        cmds.undoInfo(closeChunk=True)


def _transfer(source, target, dist_ratio, angle, allow_flip, max_influences,
              smooth_iter, smooth_strength, mask_vertices, output, weld, clamp_cot,
              interp, dist_abs, log):
    t0 = time.time()
    src_shape = _mesh_shape(source)
    tgt_shape = _mesh_shape(target)
    src_skin = find_skin_cluster(src_shape)
    if not src_skin:
        raise RuntimeError(u"ソースにskinClusterがありません: %s" % source)

    influences = cmds.skinCluster(src_skin, q=True, inf=True)
    if max_influences is None:
        max_influences = cmds.getAttr(src_skin + ".maxInfluences")

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

    # --- ジオメトリ ---
    _, _, src_pts, src_nrm, _ = _mesh_arrays(src_shape)
    tgt_dag, _, tgt_pts, tgt_nrm, tgt_tris = _mesh_arrays(tgt_shape)
    bb = tgt_pts.max(0) - tgt_pts.min(0)  # 公式実装と同じくターゲット基準
    diag = float(np.linalg.norm(bb))
    # API のワールド座標は内部単位(cm)なので dist_abs も cm
    dist_thresh = float(dist_abs) if dist_abs else dist_ratio * diag
    log(u"距離しきい値 %.3f / 角度 %.1f度 / 最大インフルエンス %d"
        % (dist_thresh, angle, max_influences))

    # --- Step1: 最近接点 ---
    t = time.time()
    tri_v, bary, known, dist = _closest_point_match(
        src_shape, src_pts, src_nrm, tgt_pts, tgt_nrm, dist_thresh, angle, allow_flip)
    log(u"Step1 最近接点: 確定 %d / %d 頂点 (%.1fs)" % (known.sum(), len(known), time.time() - t))
    if mask_vertices is not None and len(mask_vertices):
        known[np.asarray(mask_vertices, dtype=np.int64)] = False
        log(u"  マスク: %d 頂点を強制的に補間対象にしました" % len(mask_vertices))
    known_step1 = known.copy()

    # ソースウェイトは参照される頂点ぶんだけ読む
    src_dag = _dag(src_shape)
    src_fn_skin = oma.MFnSkinCluster(_obj(src_skin))
    used_src = np.unique(tri_v)
    w_flat, ninf = src_fn_skin.getWeights(src_dag, _vertex_component(used_src))
    Wsrc_used = np.array(list(w_flat), dtype=np.float64).reshape(len(used_src), ninf)
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

    # 確定点を1つも持たない島(ポーチ・バックル等の別パーツ)は、法線条件を外して
    # 距離が近い頂点を確定にする。それも無ければ島内で最も近い 10% を確定にする。
    # -> 島の中も inpainting で滑らかに補間され、パーツがちぎれにくくなる。
    known = known.copy()
    w_known = np.zeros(nw, bool)
    w_known[wid[known]] = True
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
        log(u"  補間モード: メッシュ(論文どおり)")
    comp_has = np.zeros(nw, bool)
    comp_has[lab[w_known]] = True
    orphan_w = ~comp_has[lab]
    n_islands = 0
    if orphan_w.any():
        for c in np.unique(lab[orphan_w]):
            members = np.where(lab == c)[0]
            sel = members[w_dist[members] <= dist_thresh]
            if len(sel) == 0:
                k = max(1, int(math.ceil(len(members) * 0.1)))
                sel = members[np.argsort(w_dist[members])[:k]]
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
        movable = _smooth_region(w_pts, ~w_step1, dist_thresh)
        W_w = _smooth(W_w, e, movable, smooth_iter, smooth_strength)
        log(u"  スムース %d 回 (強さ %.2f) を %d 頂点に適用(補間頂点+距離しきい値内の近傍)"
            % (smooth_iter, smooth_strength, movable.sum()))
    W = W_w[wid]
    orphan_v = orphan[wid]
    W[orphan_v] = W_cp[orphan_v]  # 既知点のない島は最近接ウェイトで代用
    log(u"Step2 inpainting: 補間 %d 頂点 / 既知点なしの島 %d 頂点 (%.1fs)"
        % ((~known).sum() - orphan_v.sum(), orphan_v.sum(), time.time() - t))

    W, bad = _postprocess(W, max_influences)
    if bad.any():
        W[bad] = W_cp[bad] / np.maximum(W_cp[bad].sum(1, keepdims=True), 1e-12)

    # --- 書き込み ---
    t = time.time()
    if output == "ngskin":
        W_close, bad_c = _postprocess(W_cp, max_influences)
        _write_ngskin_layers(tgt_skin, tgt_shape, tgt_dag, src_skin, influences,
                             W_close, W, ~known_step1, log)
    else:
        _write_weights(tgt_skin, tgt_shape, tgt_dag, src_skin, influences, W)
    log(u"書き込み (%.1fs)" % (time.time() - t))

    inpainted = np.where(~known_step1)[0]
    _last_inpainted["mesh"] = cmds.listRelatives(tgt_shape, p=True, f=True)[0]
    _last_inpainted["ids"] = inpainted.tolist()
    log(u"完了: %s -> %s (合計 %.1fs)" % (source, target, time.time() - t0))
    return {"skinCluster": tgt_skin, "known": int(known_step1.sum()),
            "inpainted": int(len(inpainted)), "orphan": int(orphan_v.sum())}


def _write_weights(tgt_skin, tgt_shape, tgt_dag, src_skin, src_infs, W):
    tgt_infs = cmds.skinCluster(tgt_skin, q=True, inf=True)
    for j in tgt_infs:
        cmds.setAttr(j + ".lockInfluenceWeights", 0)
    fn = oma.MFnSkinCluster(_obj(tgt_skin))
    tgt_paths = fn.influenceObjects()
    tgt_index = {cmds.ls(p.fullPathName(), l=True)[0]: i for i, p in enumerate(tgt_paths)}
    src_to_tgt = np.array([tgt_index[cmds.ls(j, l=True)[0]] for j in src_infs])

    cols = np.where(W.sum(0) > 0)[0]
    vals = np.ascontiguousarray(W[:, cols]).ravel().tolist()
    _PENDING["write"] = {
        "skin": _obj(tgt_skin),
        "dag": tgt_dag,
        "comp": _complete_component(W.shape[0]),
        "num_verts": int(W.shape[0]),
        "all_idx": om.MIntArray(list(range(len(tgt_paths)))),
        "idx": om.MIntArray([int(src_to_tgt[c]) for c in cols]),
        "values": om.MDoubleArray(vals),
    }
    try:
        cmds.rwtSetSkinWeights()  # Undo 対応
    finally:
        _PENDING.pop("write", None)
    cmds.setAttr(tgt_skin + ".normalizeWeights", 1)


def _layer_from_weights_via_temp(tgt_shape, src_skin, src_infs, W, name, log):
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
        _write_weights(tmp_skin, tmp_shape, _dag(tmp_shape), src_skin, src_infs, W)
        ng.init_layers(tmp_skin).add(name)  # 最初のレイヤーは skinCluster から作られる
        ng.transfer_layers(tmp, xf, vertex_transfer_mode="vertexId")
    finally:
        if cmds.objExists(tmp):
            cmds.delete(tmp)


def _write_ngskin_layers(tgt_skin, tgt_shape, tgt_dag, src_skin, src_infs,
                         W_closest, W_robust, inpaint_mask, log):
    """ngSkinTools2 にレイヤー2枚で書き出す。

    下: RWT_Closest  … 最近接コピー(Maya標準相当)
    上: RWT_Inpaint  … 本手法の結果。マスク=補間した頂点(白)/確定した頂点(黒)
    マスクを塗り替えれば、部位ごとに「最近接」と「補間」を選べる。
    既存の他のレイヤーには触らない(RWT_* だけ作り直す)。
    """
    from ngSkinTools2 import api as ng

    t = time.time()
    # skinCluster にも結果を書いておく(レイヤー作成前の状態として)
    _write_weights(tgt_skin, tgt_shape, tgt_dag, src_skin, src_infs, W_robust)
    had_layers = bool(cmds.listConnections(tgt_skin, type="ngst2SkinLayerData"))
    layers = ng.init_layers(tgt_skin)
    for layer in layers.list():  # 再実行時は作り直す
        if layer.name in (LAYER_CLOSEST, LAYER_INPAINT):
            layers.delete(layer)
    others = layers.list()

    # 下: RWT_Closest
    if not others:
        _write_weights(tgt_skin, tgt_shape, tgt_dag, src_skin, src_infs, W_closest)
        layers.add(LAYER_CLOSEST)  # レイヤーが無いときは skinCluster から作られる
    else:
        _layer_from_weights_via_temp(tgt_shape, src_skin, src_infs, W_closest,
                                     LAYER_CLOSEST, log)
    # 上: RWT_Inpaint
    _layer_from_weights_via_temp(tgt_shape, src_skin, src_infs, W_robust, LAYER_INPAINT, log)

    top = [l for l in ng.Layers(tgt_skin).list() if l.name == LAYER_INPAINT][-1]
    top.set_weights(ng.NamedPaintTarget.MASK, inpaint_mask.astype(np.float64).tolist())
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


# --------------------------------------------------------------------------
# UI (Qt / PySide2・PySide6 両対応)
# --------------------------------------------------------------------------
# 論文の公式実装と同じ既定値
DEFAULTS = {
    "dist": 5.0, "angle": 30.0, "flip": False, "smooth_iter": 10, "smooth_strength": 0.2,
    "max_inf": 0, "mask": False, "clamp": False, "output": 0, "select_after": False,
    "interp": 0, "dist_unit": 0,
}
# Blender版(SentFromSpace robust-weight-transfer)の既定値に相当する設定
#  距離 0.05m(=5cm, 絶対値) / 角度30 / 裏向き法線あり / 点群モード / スムースなし / 最大4本
PRESET_BLENDER = {
    "dist": 5.0, "dist_unit": 1, "angle": 30.0, "flip": True, "smooth_iter": 0,
    "smooth_strength": 0.2, "max_inf": 4, "mask": False, "clamp": False, "output": 0,
    "select_after": False, "interp": 1,
}
_OPT = "robustWeightTransfer2_"

# オプション説明(ツールチップとヘルプ画面で共用)
HELP = [
    (u"ソース(体)", u"ウェイトのコピー元。スキン済みのメッシュを指定します。"),
    (u"ターゲット(服)", u"コピー先。スキンが無ければ、体と同じジョイントで自動バインドします。"),
    (u"補間方法",
     u"体から遠い頂点を、どうやって埋めるか。\n"
     u"・標準(論文どおり): メッシュのつながりに沿って埋める。ほとんどの服はこちら。\n"
     u"・小パーツも布に追従: 近くの点どうしで埋める。ボタンやポーチ等の別パーツも"
     u"下の布と一緒に動く。体の近くの硬い物(バックパック等)は周りに引っ張られやすい。"),
    (u"距離",
     u"服の頂点から体の表面までがこの距離以内なら、体のウェイトをそのままコピー。"
     u"遠い頂点は補間に回す。単位は % (服の大きさに対する割合) か cm。\n"
     u"上げる→コピーが増え最近接コピーに近づく / 下げる→補間が増える。"),
    (u"角度",
     u"服と体の向き(法線)がこの角度以上ずれた頂点は、近くてもコピーせず補間に回す。"
     u"股や脇で反対側の脚・胴のウェイトを拾うのを防ぐ。"),
    (u"裏向きの面もコピー",
     u"厚みのある服・裏地の内側の面(体と逆向き)もコピー対象にする。"
     u"補間される頂点が多すぎるときに試す。"),
    (u"スムース",
     u"補間で埋めた部分とその周りだけを平均化して、コピー部分との境目をなじませる。"
     u"0回で無効。上げすぎるとウェイトがぼやける。"),
    (u"最大インフルエンス",
     u"1頂点あたりのジョイント数の上限。超えた分は小さい順に削る。"
     u"エンジンの制限(4本・8本など)に合わせる。「体と同じ」は体のskinClusterの設定。"),
    (u"選択頂点を必ず補間(マスク)",
     u"オンにして服の頂点を選択した状態で転写すると、その頂点は体からコピーせず周りから埋める。"
     u"襟が顎のウェイトを拾うとき等に。"),
    (u"安全モード",
     u"メッシュが壊れていて(潰れた三角形など)結果が乱れるときだけ使う。"
     u"計算を安定させる代わりに論文の式から少し外れる。"),
    (u"出力先",
     u"・skinCluster: 普通に書き込む。\n"
     u"・ngSkinレイヤー: 下=最近接コピー / 上=補間結果+マスクの2枚。"
     u"マスクを塗って部位ごとにどちらを使うか選べる。"),
    (u"完了後に補間頂点を選択", u"転写後、補間で埋めた頂点を選択して確認できるようにする。"),
    (u"元に戻す", u"転写はまるごと Ctrl+Z 1回で元に戻せます。"),
]
_TIPS = dict(HELP)


HELP_WINDOW = "robustWeightTransferHelp"
PAPER_URL = "https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/"

HELP_DETAIL = [
    (u"使い方", [
        u"1. Maya で 体(スキン済み) → 服 の順に選択します。服は Ctrl / Shift で複数追加できます。",
        u"2. 「転写」を押します。",
        u"服にスキンが無い場合は、体と同じジョイントで自動的にバインドします。",
        u"転写は Ctrl+Z 1回でまとめて元に戻せます(服を複数選んだ場合も1回)。",
        u"バインドポーズの状態で実行してください。",
    ]),
    (u"しくみ", [
        u"Robust Skin Weights Transfer via Weight Inpainting(Abdrashitov et al., SIGGRAPH Asia 2023)の実装です。",
        u"① 体に近く、向きもそろっている服の頂点は、体の表面のウェイトをそのままコピーします。",
        u"② それ以外の頂点(ゆったりした布、股・脇まわりなど)は、①の頂点から滑らかに補間します。",
        u"既定値は論文の公式実装と同じで、補間結果は公式実装と数値的に一致することを確認済みです。",
    ]),
    (u"補間方法", [
        u"標準: メッシュのつながりに沿って補間します(論文どおり)。ほとんどの服はこちらで問題ありません。",
        u"小パーツ追従: メッシュのつながりを使わず、近くの点どうしで補間します。"
        u"ボタン・ポーチ・バックル等の別パーツにも、すぐ下の布のウェイトが流れて一緒に動きます。",
        u"注意: 小パーツ追従では、体の近くにある硬い物(バックパック等)が周囲の動きに引っ張られやすくなります。",
    ]),
    (u"距離", [
        u"服の頂点から体の表面までがこの距離以内なら、体のウェイトをそのままコピーします。"
        u"これより遠い頂点は補間に回します。",
        u"単位は右のメニューで選びます。% = 服のバウンディングボックス対角長に対する割合(論文の既定 5%)、"
        u"cm = 絶対距離(Blender版の既定 5cm)。",
        u"上げる → コピーが増え、普通の最近接コピーに近づきます。",
        u"下げる → 補間が増え、体から離れた布がなめらかになります。",
    ]),
    (u"角度 (°)", [
        u"服と体の向き(法線)のずれがこの角度以上の頂点は、距離が近くてもコピーせず補間に回します。既定 30。",
        u"股や脇で、反対側の脚や胴のウェイトを拾ってしまうのを防ぐための条件です。",
    ]),
    (u"スムース回数 / スムース強さ", [
        u"補間した部分とその周辺だけを周りの頂点と平均して、コピーした部分との境目をなじませます。",
        u"既定は 10回 / 0.20(論文の公式実装と同じ)。回数 0 で無効になります。",
        u"上げすぎるとウェイトがぼやけます。",
    ]),
    (u"最大インフルエンス", [
        u"1頂点あたりのインフルエンス(ジョイント)数の上限です。超えた分は小さい順に削って正規化します。",
        u"0 のときは体の skinCluster の設定(maxInfluences)に合わせます。",
        u"ゲームエンジンの制限(4本・8本など)に合わせるときに使います。",
    ]),
    (u"出力先", [
        u"skinCluster: 服の skinCluster に直接書き込みます。",
        u"ngSkin レイヤー: ngSkinTools2 のレイヤー2枚で出力します。"
        u"下 = RWT_Closest(最近接コピー)、上 = RWT_Inpaint(補間結果 + 補間領域のマスク)。"
        u"マスクを塗ると、部位ごとに「最近接」と「補間」を選べます。再実行時は RWT_* の2枚だけ作り直します。",
    ]),
    (u"裏面もコピー", [
        u"厚みのある服や裏地の内側の面は、体と逆向きになります。オンにすると、こうした面もコピーの対象にします。",
        u"補間される頂点が多すぎるときに試してください。既定オフ。",
    ]),
    (u"選択頂点を補間", [
        u"オンにして、服の頂点を選択した状態(体 → 服の頂点 の順)で転写すると、選んだ頂点は体からコピーせず、"
        u"周りから補間します。",
        u"例: 襟が顎のウェイトを拾ってしまうとき、襟の頂点を選んで使います。",
    ]),
    (u"安全モード", [
        u"潰れた三角形などでメッシュが壊れていて結果が乱れるときだけ使います。",
        u"計算を安定させる代わりに、論文の式から少し外れます。既定オフ。",
    ]),
    (u"完了後に補間頂点を選択", [
        u"転写後、補間で埋めた頂点を選択します。どこが補間になったかの確認用です。",
        u"結果欄の「補間頂点を選択」ボタンでも、あとから選択できます。",
    ]),
    (u"設定プリセット(編集メニュー)", [
        u"論文の既定値にする: 公式実装と同じ値(距離 5% / 角度 30 / 裏面なし / 標準 / スムース 10回・0.2 / "
        u"最大インフルエンスは体と同じ)。詳細設定の「リセット」も同じです。",
        u"Blender版の設定にする: Blender アドオン robust-weight-transfer の既定値に相当"
        u"(距離 5cm / 角度 30 / 裏面もコピー / 小パーツ追従 / スムースなし / 最大インフルエンス 4)。",
        u"違い: Blender版はインフルエンス数を周囲となじませながら削り、最後の正規化はしません。"
        u"このツールは小さい順に削って正規化します。",
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
    cmds.text(l=u"Robust Skin Weights Transfer  v" + __version__, al="left",
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
           u"論文をもとにした独自実装です。" % __version__),
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


def _opt_get(key):
    name = _OPT + key
    if cmds.optionVar(exists=name):
        v = cmds.optionVar(q=name)
        d = DEFAULTS[key]
        return bool(v) if isinstance(d, bool) else type(d)(v)
    return DEFAULTS[key]


def _opt_set(key, value):
    if isinstance(value, (bool, int)):
        cmds.optionVar(iv=(_OPT + key, int(value)))
    else:
        cmds.optionVar(fv=(_OPT + key, float(value)))


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
            return str(e)
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
            m_edit.addAction(u"論文の既定値にする", self._reset)
            m_edit.addAction(u"Blender版の設定にする", self._preset_blender)
            m_help = bar.addMenu(u"ヘルプ")
            m_help.addAction(u"各項目の説明...", show_help)
            m_help.addAction(u"論文ページを開く", lambda: cmds.launch(web=PAPER_URL))
            m_help.addSeparator()
            m_help.addAction(u"バージョン情報", show_about)
            root.setMenuBar(bar)

            # 補間方法
            row = QtWidgets.QHBoxLayout()
            lab = QtWidgets.QLabel(u"補間方法")
            lab.setToolTip(_TIPS[u"補間方法"])
            self.interp = QtWidgets.QComboBox()
            self.interp.addItems([u"標準", u"小パーツ追従"])
            self.interp.setToolTip(_TIPS[u"補間方法"])
            row.addWidget(lab)
            row.addWidget(self.interp, 1)
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
            reset.setToolTip(u"既定値(論文の公式実装と同じ値)に戻す")
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
                                    u"Ctrl+Z 1回で元に戻せます。")
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
            rl.addWidget(sel_btn)
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
                "interp": self.interp,
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

        def _load_settings(self):
            for k, w in self._widgets().items():
                try:
                    self._put(w, _opt_get(k))
                except Exception:
                    self._put(w, DEFAULTS[k])
            if not self.ng_ok and self.output.currentIndex() == 1:
                self.output.setCurrentIndex(0)
            self.fold.setChecked(bool(cmds.optionVar(q=_OPT + "advOpen"))
                                 if cmds.optionVar(exists=_OPT + "advOpen") else False)
            self._update_hint()

        def _save_settings(self):
            for k, w in self._widgets().items():
                _opt_set(k, self._val(w))
            cmds.optionVar(iv=(_OPT + "advOpen", int(self.fold.isChecked())))

        def _reset(self):
            self._apply(DEFAULTS)

        def _preset_blender(self):
            self._apply(PRESET_BLENDER)

        def _apply(self, values):
            for k, w in self._widgets().items():
                self._put(w, values.get(k, DEFAULTS[k]))
            if not self.ng_ok and self.output.currentIndex() == 1:
                self.output.setCurrentIndex(0)
            self._save_settings()
            self._set_status(u"")

        def closeEvent(self, event):
            self._kill_job()
            self._save_settings()
            super(RwtWindow, self).closeEvent(event)

        # ---------------- 操作 ----------------
        def _update_hint(self, *_):
            pass

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
            self.run_btn.setToolTip(tip + u"\n\nCtrl+Z 1回で元に戻せます。")

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
                interp="pointcloud" if self.interp.currentIndex() == 1 else "mesh",
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
            cmds.undoInfo(openChunk=True, chunkName="RobustWeightTransfer_All")
            try:
                for i, tgt in enumerate(targets):
                    state["i"] = i
                    t = time.time()
                    res = transfer(src, tgt, log=log, mask_vertices=masks[tgt], **base_kw)
                    results.append((tgt, res, time.time() - t))
                if sel_before:  # 選択を元に戻す(Undo のまとまりの中で)
                    cmds.select(sel_before, r=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.progress.setValue(0)
                self._set_status(u"エラー: %s" % e)
                QtWidgets.QApplication.restoreOverrideCursor()
                QtWidgets.QMessageBox.critical(
                    self, u"Robust Skin Weights Transfer",
                    u"転写に失敗しました(%s)。\n\n%s\n\n詳細はスクリプトエディタを確認してください。"
                    % (_short(tgt), e))
                return
            finally:
                cmds.undoInfo(closeChunk=True)
                self._refresh_selection()
            QtWidgets.QApplication.restoreOverrideCursor()
            self.progress.setValue(100)
            lines = [u"<b>完了</b>  %.1f 秒" % (time.time() - t0)]
            for tgt, res, dt in results:
                total = res["known"] + res["inpainted"]
                lines.append(u"%s  コピー %.0f%% / 補間 %.0f%%"
                             % (_short(tgt), 100.0 * res["known"] / max(total, 1),
                                100.0 * res["inpainted"] / max(total, 1)))
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
