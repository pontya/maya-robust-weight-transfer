# -*- coding: utf-8 -*-
"""
Robust Skin Weights Transfer インストーラー

このファイルを Maya のビューポートにドラッグ&ドロップしてください。
  1. このフォルダを Maya モジュールとして登録 (Documents/maya/modules/RobustWeightTransfer.mod)
  2. 必要な Python パッケージ(numpy / scipy)が無ければ pip でインストール
  3. 現在のシェルフに起動ボタンを追加
"""
import os
import subprocess
import sys

MOD_NAME = "RobustWeightTransfer"


def _here():
    return os.path.dirname(os.path.abspath(__file__)).replace("\\", "/")


def _mayapy():
    exe_dir = os.path.dirname(sys.executable)
    for name in ("mayapy.exe", "mayapy"):
        p = os.path.join(exe_dir, name)
        if os.path.exists(p):
            return p
    return None


def _missing_packages():
    missing = []
    for pkg in ("numpy", "scipy"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def _pip_install(pkgs):
    mayapy = _mayapy()
    if not mayapy:
        raise RuntimeError("mayapy が見つかりません: %s" % sys.executable)
    # Maya 2022 (Python 3.7) は scipy 1.7 系まで
    specs = []
    for p in pkgs:
        if p == "scipy" and sys.version_info < (3, 8):
            specs.append("scipy<1.8")
        elif p == "numpy" and sys.version_info < (3, 8):
            specs.append("numpy<1.22")
        else:
            specs.append(p)
    cmd = [mayapy, "-m", "pip", "install", "--user"] + specs
    flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
    r = subprocess.run(cmd, capture_output=True, text=True, creationflags=flags)
    print(r.stdout[-3000:])
    print(r.stderr[-3000:])
    if r.returncode != 0:
        raise RuntimeError("pip install に失敗しました:\n" + " ".join(cmd))


def install():
    import maya.cmds as cmds
    import maya.mel as mel

    here = _here()

    # 1. モジュール登録
    mod_dir = os.path.join(cmds.internalVar(uad=True), "modules")
    if not os.path.isdir(mod_dir):
        os.makedirs(mod_dir)
    mod_path = os.path.join(mod_dir, MOD_NAME + ".mod")
    with open(mod_path, "w") as f:
        f.write("+ %s 1.0 %s\n" % (MOD_NAME, here))
        f.write("scripts: scripts\n")
        f.write("plug-ins: plug-ins\n")

    # 今のセッションでもすぐ使えるようにパスを通す
    scripts = here + "/scripts"
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    plug = here + "/plug-ins"
    paths = os.environ.get("MAYA_PLUG_IN_PATH", "")
    if plug not in paths:
        os.environ["MAYA_PLUG_IN_PATH"] = plug + os.pathsep + paths

    # 2. 依存パッケージ
    missing = _missing_packages()
    if missing:
        ans = cmds.confirmDialog(
            t="Robust Skin Weights Transfer",
            m=("次の Python パッケージが見つかりません: %s\n\n"
               "Maya の Python (mayapy) に pip --user でインストールしますか?\n"
               "(インターネット接続が必要です)" % ", ".join(missing)),
            b=["インストール", "あとで"], db="インストール", cb="あとで")
        if ans == "インストール":
            cmds.waitCursor(state=True)
            try:
                _pip_install(missing)
            finally:
                cmds.waitCursor(state=False)

    # 3. シェルフボタン
    try:
        shelf_top = mel.eval("$_rwt_tmp = $gShelfTopLevel")
        current = cmds.tabLayout(shelf_top, q=True, st=True)
        exists = [b for b in (cmds.shelfLayout(current, q=True, ca=True) or [])
                  if cmds.objectTypeUI(b) == "shelfButton"
                  and cmds.shelfButton(b, q=True, l=True) == "RobustWT"]
        if not exists:
            cmds.shelfButton(p=current, l="RobustWT", iol="RWT", i="commandButton.png",
                             ann="Robust Skin Weights Transfer",
                             c="import robust_weight_transfer as rwt; rwt.show()", stp="python")
    except Exception as e:  # シェルフが無い環境など
        print("シェルフボタンの追加をスキップ: %s" % e)

    msg = ("インストールしました。\n\n"
           "シェルフの「RWT」ボタン、または次のコマンドで起動できます:\n"
           "import robust_weight_transfer as rwt; rwt.show()")
    if _missing_packages():
        msg += ("\n\n※ numpy / scipy がまだ使えません。Maya を再起動するか、"
                "README の手順で手動インストールしてください。")
    else:
        msg += "\n\n※ pip でインストールした場合は Maya の再起動をおすすめします。" if missing else ""
    cmds.confirmDialog(t="Robust Skin Weights Transfer", m=msg, b=["OK"])


def onMayaDroppedPythonFile(*_):
    install()
