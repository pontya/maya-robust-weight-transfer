# -*- coding: utf-8 -*-
"""
robust_weight_transfer 用の Undo 対応ウェイト書き込みコマンド (Maya Python API 2.0 プラグイン)。

    cmds.rwtSetSkinWeights()

書き込むデータは robust_weight_transfer._PENDING に積んでから呼ぶ(コマンド引数で
大きな配列を渡せないため)。実行前の全ウェイトを保持し、undoIt で書き戻す。
"""
import maya.api.OpenMaya as om
import maya.api.OpenMayaAnim as oma

CMD_NAME = "rwtSetSkinWeights"


def maya_useNewAPI():
    pass


class RwtSetSkinWeights(om.MPxCommand):
    def __init__(self):
        om.MPxCommand.__init__(self)
        self._data = None
        self._old = None

    @staticmethod
    def creator():
        return RwtSetSkinWeights()

    def isUndoable(self):
        return True

    def doIt(self, args):
        import robust_weight_transfer as rwt
        data = rwt._PENDING.pop("write", None)
        if data is None:
            raise RuntimeError("rwtSetSkinWeights: pending data がありません")
        self._data = data
        fn = oma.MFnSkinCluster(data["skin"])
        # 実行前の全インフルエンス分のウェイトを C++ 配列のまま保持
        self._old, _ = fn.getWeights(data["dag"], data["comp"])
        self.redoIt()

    def redoIt(self):
        d = self._data
        fn = oma.MFnSkinCluster(d["skin"])
        n_all = len(d["all_idx"])
        # 全インフルエンスを 0 に -> 使うインフルエンスだけ書く
        fn.setWeights(d["dag"], d["comp"], d["all_idx"],
                      om.MDoubleArray(d["num_verts"] * n_all, 0.0), False, False)
        fn.setWeights(d["dag"], d["comp"], d["idx"], d["values"], False, False)

    def undoIt(self):
        d = self._data
        fn = oma.MFnSkinCluster(d["skin"])
        fn.setWeights(d["dag"], d["comp"], d["all_idx"], self._old, False, False)


def initializePlugin(obj):
    om.MFnPlugin(obj, "pontya", "1.0").registerCommand(CMD_NAME, RwtSetSkinWeights.creator)


def uninitializePlugin(obj):
    om.MFnPlugin(obj).deregisterCommand(CMD_NAME)
