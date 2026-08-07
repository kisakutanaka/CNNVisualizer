# STEP.md

CNN Visualizer を小さく動く単位で段階的に実装していくための計画。各STEPは「完了したらブラウザで動作確認できる」粒度にしてある。上から順番に進める。詳細な背景・目的・制約は [Plan.md](Plan.md) と [CLAUDE.md](CLAUDE.md) を参照。

## 技術スタック

- UI: React + TypeScript + Vite
- 推論: TensorFlow.js（学習済みモデル / 中間層の活性化取得に使用）
- ホスティング: GitHub Pages（静的ビルドのみ、バックエンドなし）

## 進め方のルール

- 1STEPずつ実装し、完了したら動作確認してからチェックを付けて次に進む
- 各STEPはMVP範囲（[Plan.md](Plan.md) 参照）を超えて機能を先取りしない
- 手戻りが出たら、このファイルのSTEPを見直して更新する

---

## STEP 0: プロジェクト雛形

- [x] Vite + React + TypeScript でプロジェクトを作成
- [x] GitHub Pages向けのbase pathを設定できる状態にする（vite.config.ts、base: '/CNNVisualizer/'）
- [x] `npm run dev` でローカル起動し、空のページが表示されることを確認（`npm run build` も成功確認済み）

## STEP 1: 画像入力（アップロード）

- [x] 画像ファイルをアップロードするUIを実装（src/App.tsx）
- [x] アップロードした画像をプレビュー表示する

## STEP 2: 画像入力（カメラ撮影）

- [x] `<input type="file" accept="image/*" capture="environment">` でカメラ撮影に対応（iPhone/iOS Safari想定、src/App.tsx）
- [x] 撮影した画像もSTEP1と同じプレビューに表示できるようにする

## STEP 3: TensorFlow.jsの導入と1モデルでの推論

- [x] TensorFlow.jsをセットアップ（@tensorflow/tfjs）
- [x] 学習済みモデル（MobileNet v1 alpha 0.25、Keras/LayersModel形式）を1つ読み込む
- [x] アップロード/撮影した画像を推論し、分類結果（ラベルと確率）をリスト表示する
- 備考: 当初`@tensorflow-models/mobilenet`（GraphModel）を使っていたが、STEP5で中間層の活性化を取得する必要が生じたため`tf.loadLayersModel()`に切り替えた（GraphModelは`classify()`/`infer()`しか公開しておらず、任意の層の出力を取り出せなかったため）。分類ロジック・前処理・ImageNetラベル一覧（`src/data/imagenetClasses.ts`、元パッケージから抽出、Apache License 2.0）は自前実装に置き換え済み（`src/mobilenet.ts`）

## STEP 4: モデル切り替え機能

- [ ] 切り替え可能な学習済みモデルを2〜3種類用意する（LayersModel形式で中間層アクセスできるものを選定）
- [ ] モデル選択UI（ドロップダウン等）を実装
- [ ] 選択中のモデルで推論結果が切り替わることを確認する

## STEP 5: 中間層の活性化を取得する

※STEP4より先に実施（ユーザー判断、2026-08-05）

- [x] 選択中モデルの各層（中間層）の出力を取得する仕組みを実装（`buildActivationModel`/`getActivationShapes`、`src/mobilenet.ts`）
- [x] まずはコンソール等で、層ごとの出力テンソルの形状を確認できる状態にする（56層、空間サイズ112→7・チャンネル数8→256と進む様子をブラウザ実行で確認済み）

## STEP 6: 活性化を画像として可視化する

- [x] 1つの層の活性化マップをcanvasに描画する（`getLayerHeatmap`でチャンネル平均→0〜1正規化、`drawActivationOverlay`で青→緑→赤のヒートマップ色に変換、src/mobilenet.ts, src/heatmap.ts）
- [x] 元画像の上にオーバーレイ表示する（透過重ね合わせ。活性化が弱い部分ほどalphaを下げて元画像を透かす。現在は固定で`conv_pw_5_relu`層を表示、層切り替えはSTEP7で対応）

## STEP 7: 層の切り替えUI（スワイプ）

- [x] 複数層の活性化オーバーレイを、スワイプ（またはタブ/矢印操作含む）で切り替えられるようにする（`OVERLAY_LAYERS`＝MobileNetV1の各ブロック出力14層に絞ってswipe対象に。左右スワイプ＋矢印ボタンの両方に対応、src/App.tsx, src/mobilenet.ts）
- [x] 現在表示している層が何層目かをUIで分かるようにする（「層 X / 14」インジケーター表示）

## STEP 7.5: 個別ニューロンの活性化グリッド表示（MVP範囲外の追加機能、ユーザー要望により2026-08-05実施）

チャンネル平均のオーバーレイだけでなく、選択中の層の各チャンネル（ニューロン）の活性化を個別のタイルとして並べて見られるようにした。

- [x] 「個別ニューロンを見る」ボタンでオーバーレイ表示とグリッド表示を切り替え（src/App.tsx）
- [x] 各チャンネルをチャンネルごとに独立して0〜1正規化し、タイル状に描画（`getChannelHeatmaps`, src/mobilenet.ts / `drawChannelTile`, src/heatmap.ts）
- [x] タイルは`minmax(64px, 1fr)`のCSS Gridで視認性を確保しつつ、チャンネル数が多い層（最大256）も縦スクロールで全件表示
- [x] ブラウザ実行（iPhone 13ビューポート、Playwright）で8チャンネル/256チャンネルどちらも件数・タイルサイズ（約67px四方）を確認済み
- [x] 層が深くなるほど活性化マップの解像度が下がり何に反応しているか分かりづらくなるため、各タイルの背景にモノクロ化した元画像を敷いてヒートマップを半透明合成（`drawChannelTile`, src/heatmap.ts）。グレースケール変換はタイルの数（最大256）だけやり直すと重いため、画像ごとに1回だけ生成してcanvasにキャッシュし、各タイルへはその縮小コピー（drawImageのスケーリング）のみで済ませるよう最適化（ユーザー指摘により2026-08-05対応）

## STEP 8: UI/UXの磨き込み

- [ ] シンプルさを保ったまま、全体のレイアウト・配色・操作性を整える
- [ ] モデル選択〜画像入力〜結果確認〜層プレビューの一連の流れがスムーズか確認する

## STEP 9: GitHub Pagesへのデプロイ

- [x] GitHub Actionsで自動デプロイするworkflowを作成（.github/workflows/deploy.yml）※iPhone実機テストのためSTEP2完了時点で前倒しで実施
- [x] リポジトリ設定でPagesのSourceを「GitHub Actions」にする
- [x] 本番ビルドを作成し、GitHub Pages上で正しく動作することを確認する（iPhone実機でカメラ撮影〜プレビューまで確認済み）
- [ ] READMEに使い方・デプロイURLを記載する（MVP完成時にまとめて記載）

---

## MVP完了後のバックログ（優先度未定、着手はMVP完成後）

- 分類結果ラベルの日本語化（現状はImageNetの英語ラベルをそのまま表示。ユーザーコメント2026-08-05: 「結果が英語だからいずれ日本語にしたい」）

---

MVP完了後の追加機能はここには含めない。必要になった時点で別途検討する。
