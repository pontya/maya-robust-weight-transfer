# Maya Robust Skin Weights Transfer

体(スキン済みメッシュ)のスキンウェイトを、服へ転写する Maya ツールです。
[Robust Skin Weights Transfer via Weight Inpainting](https://www.dgp.toronto.edu/~rinat/projects/RobustSkinWeightsTransfer/)(Abdrashitov et al., SIGGRAPH Asia 2023)を論文ベースで実装しています。

- 体に近く、向きもそろっている服の頂点 → 体のウェイトをそのままコピー
- それ以外(ゆったりした布、股・脇まわりなど) → コピーした頂点から滑らかに補間

Maya 標準の `Copy Skin Weights`(最近接)で起きやすい、股で布が左右に引き裂かれる・脇が胴に引っ張られるといった破綻を抑えます。

<img src="docs/ui.png" width="380">

## 特長

- 服がスキニングされていなくても OK(体と同じジョイントで自動バインド)
- Maya で **体 → 服** の順に選ぶだけ。服は複数まとめて転写可能
- **Ctrl+Z 1回で元に戻せる**(複数の服を転写した場合も1回)
- 補間方法を2種類から選択
  - **標準**: 論文どおりのメッシュ補間
  - **小パーツ追従**: 点群補間。ボタン・ポーチなどの別パーツも下の布と一緒に動く
- 出力先を **skinCluster** / **ngSkinTools2 のレイヤー**(最近接コピー + 補間結果 + マスク)から選択
- 7万頂点クラスの衣装で 10〜20 秒程度

## 動作環境

| Maya | 状態 | 必要なもの |
|---|---|---|
| 2027 | 動作確認済み(Windows) | scipy |
| 2023〜2026 | 動作する見込み(未検証) | scipy(numpy が無ければ numpy も) |
| 2022 | 未検証 | numpy, scipy(1.7 系) |

- **numpy**: 最近の Maya には同梱(2027 は同梱を確認済み)。無い場合はインストーラーが pip で入れます
- **scipy**: 追加インストールが必要(インストーラーが pip で入れます)
- **ngSkinTools2**: 任意。「ngSkin レイヤー」出力を使うときだけ必要(Maya 2027 は同梱。それ以前は [ngskintools.com](https://www.ngskintools.com/) から)

## インストール

1. このリポジトリをダウンロードして、好きな場所に置きます
   - 緑の **Code** ボタン → **Download ZIP** → 展開
   - または `git clone https://github.com/pontya/maya-robust-weight-transfer.git`
2. Maya を起動し、フォルダ内の **`install.py` をビューポートにドラッグ&ドロップ**します
   - Maya モジュールとして登録(`Documents/maya/modules/RobustWeightTransfer.mod`)
   - numpy / scipy が無ければ、pip でインストールするか確認されます(インターネット接続が必要)
   - 現在のシェルフに **RWT** ボタンを追加
3. scipy をインストールした場合は Maya を再起動します

> 初回の転写時に、Undo 用プラグイン `rwt_undo_cmd.py` の読み込み許可を求められることがあります。**許可(ロード)** を選んでください。

### 手動でインストールする場合

1. scipy を Maya の Python に入れます(コマンドプロンプトで実行。パスは Maya のバージョンに合わせて変更)
   ```bat
   "C:\Program Files\Autodesk\Maya2027\bin\mayapy.exe" -m pip install --user scipy
   ```
   Maya 2022 の場合:
   ```bat
   "C:\Program Files\Autodesk\Maya2022\bin\mayapy.exe" -m pip install --user "numpy<1.22" "scipy<1.8"
   ```
2. `Documents/maya/modules/` に `RobustWeightTransfer.mod` を作り、次の内容を書きます(パスは置いた場所に変更)
   ```
   + RobustWeightTransfer 1.0 C:/path/to/maya-robust-weight-transfer
   scripts: scripts
   plug-ins: plug-ins
   ```
3. Maya を再起動

## 使い方

1. Maya で **体(スキン済み)→ 服** の順に選択します(服は Ctrl / Shift で複数追加可)
2. シェルフの **RWT** ボタン、または次のコマンドでツールを開きます
   ```python
   import robust_weight_transfer as rwt
   rwt.show()
   ```
3. **転写** を押します

バインドポーズの状態で実行してください。各項目の説明は、マウスオーバーのツールチップと **ヘルプ → 各項目の説明** にあります。

スクリプトから使う場合:

```python
import robust_weight_transfer as rwt
rwt.transfer("body", "cloth")                         # 論文の既定値
rwt.transfer("body", "cloth", dist_abs=5.0, angle=30,  # Blender版相当
             allow_flip=True, interp="pointcloud", smooth_iter=0, max_influences=8)
```

## 設定項目

| 項目 | 内容 | 論文の既定値 |
|---|---|---|
| 補間方法 | 標準(メッシュ)/ 小パーツ追従(点群) | 標準 |
| 距離 | この距離以内の頂点は体からコピー。単位は % (服の大きさに対する割合) か cm | 5 % |
| 角度 | 体と服の向きがこれ以上ずれた頂点は補間に回す | 30° |
| スムース回数 / 強さ | 補間部分の境目をなじませる | 10 / 0.20 |
| 最大インフルエンス | 1頂点あたりのジョイント数の上限(0 = 体と同じ) | 0 |
| 出力先 | skinCluster / ngSkin レイヤー | skinCluster |
| 裏面もコピー | 厚み・裏地のある服の内側の面もコピー対象に | オフ |
| 選択頂点を補間 | 選択した服の頂点を必ず補間(襟が顎に付く等の対策) | オフ |
| 安全モード | 壊れたメッシュで結果が乱れるとき用 | オフ |

**編集** メニューから「論文の既定値」と「Blender版の設定」(Blender アドオン [robust-weight-transfer](https://github.com/sentfromspacevr/robust-weight-transfer) の既定値に相当: 5cm / 30° / 裏面あり / 小パーツ追従 / スムースなし)を切り替えられます。

## 論文の実装との対応

- 3.1 最近接点マッチング、3.2 ウェイトの補間(Q = −L + LM⁻¹L)、3.3 スムース、3.4 マスクを実装
- 論文の公式実装 [rin-23/RobustSkinWeightsTransferCode](https://github.com/rin-23/RobustSkinWeightsTransferCode) の球→平面データで比較
  - マッチング結果: 一致
  - 補間結果: 誤差 1e-14 で一致
  - スムース: 最大 0.0014 の差(公式は頂点を1つずつ順番に、本ツールは全頂点まとめて更新するため)
- 論文から追加した処理
  - UV シーム等で分割された頂点を位置で溶接してから補間
  - コピーできる頂点が1つも無い別パーツは、距離だけで基準の頂点を補う
  - 補間後に負のウェイトを 0 にして正規化

## 既知の制限

- 「小パーツ追従」は点群ラプラシアンの簡易な自前実装です(Blender版が使う robust_laplacian とは別物)
- 最大インフルエンスは小さい順に打ち切ります(周囲となじませながら削る処理はありません)
- ウェイトの書き込みには Undo 用の小さなプラグイン(`plug-ins/rwt_undo_cmd.py`)を使います

## アンインストール

`Documents/maya/modules/RobustWeightTransfer.mod` とダウンロードしたフォルダを削除し、シェルフの RWT ボタンを消してください。

## ライセンス

MIT License(`LICENSE` を参照)

論文:

```bibtex
@inproceedings{abdrashitov2023robust,
  author    = {Abdrashitov, Rinat and Raichstat, Kim and Monsen, Jared and Hill, David},
  title     = {Robust Skin Weights Transfer via Weight Inpainting},
  booktitle = {SIGGRAPH Asia 2023 Technical Communications},
  year      = {2023},
  doi       = {10.1145/3610543.3626180}
}
```
