# 三井物産商品予測チャレンジ (MITSUI&CO. Commodity Prediction Challenge)

- https://www.kaggle.com/competitions/mitsui-commodity-prediction-challenge

## 🎯 コンペティション概要

### 目的
過去のマーケットデータ（ロンドン金属取引所（LME）、日本取引所グループ（JPX）、米国株式市場、および外国為替市場の過去のデータ）を用い、金属・先物・米国株・為替を含む複数アセットの 翌営業日リターンを順位付けで予測し、安定かつ高精度なモデルを作るコンペ。

### データ構成
- 世界中の市場から取得された複数（貴金属、先物、米国株、外国為替など）の金融商品に関する金融時系列データで構成されている

### 評価指標
シャープ比の一種で、予測値と目標値の間のスピアマン順位相関の平均を標準偏差で割ることによって算出。
- https://www.kaggle.com/code/metric/mitsui-co-commodity-prediction-metric?scriptVersionId=250883469&cellId=1

### 提出ファイル
このコンペティションへの応募には、提供されている評価APIを使用する必要がある。これにより、モデルが将来の時点を覗き込むことがなくなる。
- https://www.kaggle.com/code/sohier/mitsui-demo-submission?scriptVersionId=251889403&cellId=1

## プロジェクト構成

```
repo/
├── README.md            # プロジェクト概要・実行手順・命名規約
├── .gitignore
├── pyproject.toml
├── notebooks/
│
├── data/
│   ├── raw/             # 生データ（ダウンロード直後の状態）
│   ├── interim/         # 前処理途中の一時データ
│   └── processed/       # モデル入力に使う最終データ
│
├── src/
│   ├── __init__.py
│   ├── features/        # 前処理・特徴量関数
│   ├── models/          # 学習・推論ロジック（train.py など）
│   ├── metrics/         # 公式 score.py とローカル検証ユーティリティを格納
│   ├── optimization/    # Optuna 関連コード（objective, callback）
│   └── pipelines/       # 学習・推論・提出をワークフロー化
│   └── utils/           # 汎用関数（ログ出力など）
│
├── configs/
│   ├── default.yaml     # ベース設定（モデル種別・データパスなど）
│   └── optuna_space.yaml # Optuna 用：探索するハイパラ範囲を宣言
│
├── scripts/
│   ├── run_exp.py       # 単発実験：configs/*.yaml を読み込んで学習
│   ├── run_optuna.py    # 自動探索：Optuna で n_trials を実行
│   └── export_best.py   # Study からベスト trial を取り出し experiments へ格納
│
├── optuna_studies/
│   ├── 20250729_lgbm.db  # SQLite: Study 名＝<日付>_<algo>.db
│   └── plots/           # optimization_history.png など自動生成図
│
├── experiments/
│   ├── summary.csv      # run_id, val_auc などを自動追記する一覧表
│   └── 20250729_1413_lgbm/  # run_id＝日時+出力関数の引数
│       ├── config.yaml   # その run の設定
│       ├── metrics.json  # 評価指標
│       ├── artifacts/    # 学習済みモデルや図表
│       └── log.txt       # 標準出力・エラー全文
│
└── submissions/          # 提出用csvファイルを管理
```

## 🚀 セットアップ手順


<!-- ## 📈 アプローチ戦略

### 1. データ理解
- 商品価格の時系列パターン分析
- 季節性、トレンド、周期性の確認
- 経済指標との相関関係分析
- 欠損値・外れ値の処理戦略

### 2. 特徴量エンジニアリング
- **時系列特徴量**: ラグ特徴量、移動平均、差分系列
- **統計的特徴量**: ローリング統計量（平均、標準偏差、歪度、尖度）
- **技術指標**: RSI、MACD、ボリンジャーバンドなど
- **外部データ**: 経済指標、為替レート、原油価格など

### 3. モデリング戦略
- **ベースライン**: ARIMA、線形回帰、移動平均
- **機械学習**: LightGBM、XGBoost、Random Forest
- **深層学習**: LSTM、GRU、Transformer、Prophet
- **アンサンブル**: スタッキング、ブレンディング、重み付き平均

### 4. 検証戦略
- **時系列分割**: 過去データでの訓練、未来データでの検証
- **Walk-Forward検証**: 段階的な時間窓での検証
- **複数期間での評価**: 短期・中期・長期予測の精度確認

## 🛠️ 使用技術・ライブラリ

### データ処理・分析
- `pandas`, `numpy`: データ操作
- `matplotlib`, `seaborn`, `plotly`: 可視化
- `scipy`, `statsmodels`: 統計分析

### 機械学習
- `scikit-learn`: 基本的な機械学習アルゴリズム
- `lightgbm`, `xgboost`: 勾配ブースティング
- `optuna`: ハイパーパラメータ最適化

### 時系列分析
- `prophet`: Facebook Prophet
- `pmdarima`: 自動ARIMA
- `sktime`: 時系列機械学習

### 深層学習
- `torch`: PyTorch

## 📝 実験管理

### MLflow/Weights & Biases
- 実験パラメータの記録
- モデル性能の追跡
- 可視化・比較

### Git管理
- ブランチ戦略: feature/experiment-name
- コミットメッセージ: [FEAT/FIX/EXP] 詳細説明

## 🎯 評価・提出

### ローカル検証
```bash
# 交差検証の実行
python src/evaluate.py --model lightgbm --cv-splits 5

# 予測の生成
python src/predict.py --model best_model --output submissions/
```

### 提出準備
```bash
# 提出ファイルの形式確認
python src/utils/check_submission.py submissions/submission.csv
```

## 📚 参考資料

- [時系列予測のベストプラクティス](https://example.com)
- [商品価格予測の研究論文](https://example.com)
- [Kaggle時系列コンペの過去ソリューション](https://example.com)

## 🤝 貢献

1. 新しい特徴量やモデルのアイデア
2. コードの改善・最適化
3. ドキュメントの追加・修正 -->

## 📄 ライセンス
