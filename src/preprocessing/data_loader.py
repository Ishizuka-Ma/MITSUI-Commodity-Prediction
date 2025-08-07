"""
データ読み込みと前処理のためのモジュール
- データファイルの読み込み
- 欠損値処理
- データ型変換
- ラベルとの結合
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from IPython.display import display
import warnings
warnings.filterwarnings('ignore')

# 定数定義
SOLUTION_NULL_FILLER = -999999


class DataLoader:
    """
    データ読み込みと前処理を行うクラス
    """
    
    def __init__(self, data_dir: str):
        """
        コンストラクタ
        
        Args:
            data_dir (str): データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self.train_df = None
        self.train_labels_df = None
        self.test_df = None
        self.target_pairs_df = None
        
    def load_all_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        全てのデータファイルを読み込む
        
        Returns:
            Dict[str, pd.DataFrame]: 読み込まれたデータフレームの辞書
        """
        self.train_df = pd.read_csv(self.data_dir / "train.csv")
        self.train_labels_df = pd.read_csv(self.data_dir / "train_labels.csv")
        self.test_df = pd.read_csv(self.data_dir / "test.csv")
        self.target_pairs_df = pd.read_csv(self.data_dir / "target_pairs.csv")
        
        return self.train_df, self.train_labels_df, self.test_df, self.target_pairs_df
    
    def get_data_info(self, df: pd.DataFrame) -> None:
        """
        読み込まれたデータの基本情報を取得
        """
        print(f"df.shape: {df.shape}")
    
    def process_missing_values(self, strategy: str) -> None:
        """
        欠損値の処理
        
        Args:
            strategy (str): 欠損値処理戦略
                - 'forward_fill': 前方向埋め
                - 'backward_fill': 後方向埋め
                - 'drop': 欠損行を削除
                - 'interpolate': 線形補間
        """
        if strategy == 'forward_fill':
            self.train_df = self.train_df.fillna(method='ffill')
            self.test_df = self.test_df.fillna(method='ffill')
            
        elif strategy == 'backward_fill':
            self.train_df = self.train_df.fillna(method='bfill')
            self.test_df = self.test_df.fillna(method='bfill')
            
        elif strategy == 'interpolate':
            # 数値列のみ線形補間
            numeric_cols = self.train_df.select_dtypes(include=[np.number]).columns
            self.train_df[numeric_cols] = self.train_df[numeric_cols].interpolate()
            self.test_df[numeric_cols] = self.test_df[numeric_cols].interpolate()
            
        elif strategy == 'drop':
            initial_train_len = len(self.train_df)
            self.train_df = self.train_df.dropna()
            self.train_labels_df = self.train_labels_df.loc[self.train_df.index]
            print(f"    train行数: {initial_train_len} → {len(self.train_df)}")
    
    def handle_solution_null_filler(self) -> None:
        """
        SOLUTION_NULL_FILLERの処理
        評価時にはこの値はNaNとして扱われるため、適切に処理する
        """
        # train_labels内のSOLUTION_NULL_FILLERをNaNに変換
        target_cols = [col for col in self.train_labels_df.columns if col.startswith('target_')]
        
        filler_count = 0
        for col in target_cols:
            mask = self.train_labels_df[col] == SOLUTION_NULL_FILLER
            filler_count += mask.sum()
            self.train_labels_df.loc[mask, col] = np.nan
            
        print(f"  - {filler_count}個のSOLUTION_NULL_FILLERをNaNに変換")
    
    def prepare_features_and_targets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        特徴量とターゲットの準備
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (特徴量, ターゲット)
        """
        # date_idで結合
        if 'date_id' not in self.train_df.columns or 'date_id' not in self.train_labels_df.columns:
            raise ValueError("date_id列が見つかりません")
        
        # 結合
        merged_df = pd.merge(
            self.train_df, 
            self.train_labels_df, 
            on='date_id', 
            how='inner'
        )
        
        # 特徴量とターゲットを分離
        target_cols = [col for col in merged_df.columns if col.startswith('target_')]
        feature_cols = [col for col in merged_df.columns if not col.startswith('target_')]
        
        features = merged_df[feature_cols]
        targets = merged_df[target_cols]
        
        return features, targets
    
def main():
    """
    テスト用のメイン関数
    """
    # データディレクトリのパス
    data_dir = Path(__file__).parent.parent.parent / "data" / "input"
    
    # DataLoaderの初期化
    loader = DataLoader(data_dir)
    
    # データ読み込み
    train_df, train_labels_df, test_df, target_pairs_df = loader.load_all_data()
    
    # データ情報の表示
    print("\n📊 データ情報:")
    print("train_df")
    loader.get_data_info(train_df)
    print("train_labels_df")
    loader.get_data_info(train_labels_df)
    print("test_df")
    loader.get_data_info(test_df)
    print("target_pairs_df")
    loader.get_data_info(target_pairs_df)
    
    # 欠損値処理
    # loader.process_missing_values(strategy='forward_fill')
    
    # SOLUTION_NULL_FILLER処理
    loader.handle_solution_null_filler()
    
    # 特徴量とターゲットの準備
    features, targets = loader.prepare_features_and_targets()
    
    print("\n✅ DataLoaderテスト完了!")

if __name__ == "__main__":
    main()