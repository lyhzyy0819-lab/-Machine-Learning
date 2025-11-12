"""
特征工程模块 - 客户流失预测
特征编码、特征创建、特征选择和特征缩放
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif

import sys
sys.path.append(str(Path(__file__).parent.parent))
import config
from src.utils import Timer, save_model


class FeatureEngineer:
    """
    特征工程类
    处理特征编码、特征创建和特征选择
    """

    def __init__(self):
        """初始化特征工程器"""
        self.logger = logging.getLogger("ChurnPrediction")
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.encoded_features = []

    def encode_binary_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        编码二值特征

        Args:
            df: 输入DataFrame

        Returns:
            编码后的DataFrame
        """
        self.logger.info("编码二值特征...")
        df_encoded = df.copy()

        # 处理目标变量
        if config.TARGET_COL in df_encoded.columns:
            df_encoded[config.TARGET_COL] = df_encoded[config.TARGET_COL].map({'Yes': 1, 'No': 0})
            self.logger.info(f"  目标变量 '{config.TARGET_COL}' 已编码: Yes=1, No=0")

        # 处理二值特征
        for feature in config.BINARY_FEATURES:
            if feature not in df_encoded.columns:
                continue

            unique_values = df_encoded[feature].unique()

            # 性别特征
            if feature == 'gender':
                df_encoded[feature] = df_encoded[feature].map({'Male': 1, 'Female': 0})
                self.logger.info(f"  {feature}: Male=1, Female=0")

            # SeniorCitizen已经是0/1
            elif feature == 'SeniorCitizen':
                self.logger.info(f"  {feature}: 已是数值型 (0/1)")

            # 其他Yes/No特征
            elif 'Yes' in unique_values or 'No' in unique_values:
                df_encoded[feature] = df_encoded[feature].map({'Yes': 1, 'No': 0})
                self.logger.info(f"  {feature}: Yes=1, No=0")

        return df_encoded

    def encode_categorical_features(self, df: pd.DataFrame,
                                   method: str = 'onehot') -> pd.DataFrame:
        """
        编码多分类特征

        Args:
            df: 输入DataFrame
            method: 编码方法 ('onehot' 或 'label')

        Returns:
            编码后的DataFrame
        """
        self.logger.info(f"使用 {method} 编码多分类特征...")
        df_encoded = df.copy()

        if method == 'onehot':
            # One-Hot编码
            for feature in config.CATEGORICAL_FEATURES:
                if feature not in df_encoded.columns:
                    continue

                # 创建One-Hot编码
                dummies = pd.get_dummies(df_encoded[feature],
                                        prefix=feature,
                                        drop_first=True)  # 避免多重共线性

                # 添加到DataFrame
                df_encoded = pd.concat([df_encoded, dummies], axis=1)

                # 删除原始列
                df_encoded.drop(feature, axis=1, inplace=True)

                self.logger.info(f"  {feature}: 创建 {len(dummies.columns)} 个虚拟变量")
                self.encoded_features.extend(dummies.columns.tolist())

        elif method == 'label':
            # Label编码
            for feature in config.CATEGORICAL_FEATURES:
                if feature not in df_encoded.columns:
                    continue

                le = LabelEncoder()
                df_encoded[feature] = le.fit_transform(df_encoded[feature].astype(str))
                self.label_encoders[feature] = le

                self.logger.info(f"  {feature}: Label编码完成 ({len(le.classes_)} 个类别)")

        else:
            raise ValueError(f"不支持的编码方法: {method}")

        return df_encoded

    def create_tenure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建在网时长相关特征

        Args:
            df: 输入DataFrame

        Returns:
            添加特征后的DataFrame
        """
        self.logger.info("创建在网时长特征...")
        df_feat = df.copy()

        if 'tenure' not in df_feat.columns:
            self.logger.warning("'tenure' 列不存在，跳过")
            return df_feat

        # 在网时长分组
        df_feat['TenureGroup'] = pd.cut(
            df_feat['tenure'],
            bins=config.TENURE_BINS,
            labels=config.TENURE_LABELS,
            include_lowest=True
        )
        self.logger.info("  创建特征: TenureGroup")

        # 在网时长平方（捕捉非线性关系）
        df_feat['TenureSquared'] = df_feat['tenure'] ** 2
        self.logger.info("  创建特征: TenureSquared")

        # 在网时长对数（平滑极端值）
        df_feat['TenureLog'] = np.log1p(df_feat['tenure'])
        self.logger.info("  创建特征: TenureLog")

        # 是否新客户（在网时长≤12个月）
        df_feat['IsNewCustomer'] = (df_feat['tenure'] <= 12).astype(int)
        self.logger.info("  创建特征: IsNewCustomer")

        # 是否长期客户（在网时长≥48个月）
        df_feat['IsLongTermCustomer'] = (df_feat['tenure'] >= 48).astype(int)
        self.logger.info("  创建特征: IsLongTermCustomer")

        return df_feat

    def create_charge_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建费用相关特征

        Args:
            df: 输入DataFrame

        Returns:
            添加特征后的DataFrame
        """
        self.logger.info("创建费用特征...")
        df_feat = df.copy()

        required_cols = ['MonthlyCharges', 'TotalCharges', 'tenure']
        if not all(col in df_feat.columns for col in required_cols):
            self.logger.warning("缺少必需列，跳过部分特征创建")
            return df_feat

        # 平均月费用（总费用/在网时长）
        df_feat['AvgMonthlyCharges'] = df_feat.apply(
            lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
            axis=1
        )
        self.logger.info("  创建特征: AvgMonthlyCharges")

        # 费用增长率
        df_feat['ChargeGrowthRate'] = df_feat.apply(
            lambda row: (row['MonthlyCharges'] - row['AvgMonthlyCharges']) / row['AvgMonthlyCharges']
            if row['AvgMonthlyCharges'] > 0 else 0,
            axis=1
        )
        self.logger.info("  创建特征: ChargeGrowthRate")

        # 月费用分组
        df_feat['MonthlyChargesGroup'] = pd.cut(
            df_feat['MonthlyCharges'],
            bins=config.MONTHLY_CHARGES_BINS,
            labels=config.MONTHLY_CHARGES_LABELS,
            include_lowest=True
        )
        self.logger.info("  创建特征: MonthlyChargesGroup")

        # 高费用标记（月费用高于80美元）
        df_feat['IsHighCharges'] = (df_feat['MonthlyCharges'] > 80).astype(int)
        self.logger.info("  创建特征: IsHighCharges")

        # 费用/在网时长比率
        df_feat['ChargeTenureRatio'] = df_feat['TotalCharges'] / (df_feat['tenure'] + 1)
        self.logger.info("  创建特征: ChargeTenureRatio")

        # 月费用对数
        df_feat['MonthlyChargesLog'] = np.log1p(df_feat['MonthlyCharges'])
        self.logger.info("  创建特征: MonthlyChargesLog")

        return df_feat

    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建服务相关特征

        Args:
            df: 输入DataFrame

        Returns:
            添加特征后的DataFrame
        """
        self.logger.info("创建服务特征...")
        df_feat = df.copy()

        # 服务数量统计
        service_cols = [
            'PhoneService', 'MultipleLines', 'InternetService',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        available_service_cols = [col for col in service_cols if col in df_feat.columns]

        if available_service_cols:
            # 总服务数量
            df_feat['TotalServices'] = 0
            for col in available_service_cols:
                # 如果服务已启用，计数+1
                if df_feat[col].dtype == 'int64':
                    df_feat['TotalServices'] += df_feat[col]
                else:
                    df_feat['TotalServices'] += (df_feat[col] == 'Yes').astype(int)

            self.logger.info(f"  创建特征: TotalServices (基于 {len(available_service_cols)} 个服务)")

        # 是否有互联网服务
        if 'InternetService' in df_feat.columns:
            df_feat['HasInternet'] = (df_feat['InternetService'] != 'No').astype(int)
            self.logger.info("  创建特征: HasInternet")

        # 是否有电话服务
        if 'PhoneService' in df_feat.columns:
            if df_feat['PhoneService'].dtype == 'int64':
                df_feat['HasPhone'] = df_feat['PhoneService']
            else:
                df_feat['HasPhone'] = (df_feat['PhoneService'] == 'Yes').astype(int)
            self.logger.info("  创建特征: HasPhone")

        # 安全服务数量（OnlineSecurity + DeviceProtection）
        security_cols = ['OnlineSecurity', 'DeviceProtection']
        available_security = [col for col in security_cols if col in df_feat.columns]
        if available_security:
            df_feat['SecurityServices'] = 0
            for col in available_security:
                if df_feat[col].dtype == 'int64':
                    df_feat['SecurityServices'] += df_feat[col]
                else:
                    df_feat['SecurityServices'] += (df_feat[col] == 'Yes').astype(int)
            self.logger.info("  创建特征: SecurityServices")

        # 流媒体服务数量（StreamingTV + StreamingMovies）
        streaming_cols = ['StreamingTV', 'StreamingMovies']
        available_streaming = [col for col in streaming_cols if col in df_feat.columns]
        if available_streaming:
            df_feat['StreamingServices'] = 0
            for col in available_streaming:
                if df_feat[col].dtype == 'int64':
                    df_feat['StreamingServices'] += df_feat[col]
                else:
                    df_feat['StreamingServices'] += (df_feat[col] == 'Yes').astype(int)
            self.logger.info("  创建特征: StreamingServices")

        return df_feat

    def create_contract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建合同相关特征

        Args:
            df: 输入DataFrame

        Returns:
            添加特征后的DataFrame
        """
        self.logger.info("创建合同特征...")
        df_feat = df.copy()

        # 是否月付合同
        if 'Contract' in df_feat.columns:
            df_feat['IsMonthToMonth'] = (df_feat['Contract'] == 'Month-to-month').astype(int)
            self.logger.info("  创建特征: IsMonthToMonth")

            # 是否长期合同（1年或2年）
            df_feat['IsLongTermContract'] = (
                (df_feat['Contract'] == 'One year') |
                (df_feat['Contract'] == 'Two year')
            ).astype(int)
            self.logger.info("  创建特征: IsLongTermContract")

        # 是否自动支付
        if 'PaymentMethod' in df_feat.columns:
            df_feat['IsAutoPayment'] = (
                df_feat['PaymentMethod'].str.contains('automatic', case=False, na=False)
            ).astype(int)
            self.logger.info("  创建特征: IsAutoPayment")

        # 是否无纸化账单
        if 'PaperlessBilling' in df_feat.columns:
            if df_feat['PaperlessBilling'].dtype != 'int64':
                df_feat['IsPaperlessBilling'] = (df_feat['PaperlessBilling'] == 'Yes').astype(int)
                self.logger.info("  创建特征: IsPaperlessBilling")

        return df_feat

    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建人口统计学特征

        Args:
            df: 输入DataFrame

        Returns:
            添加特征后的DataFrame
        """
        self.logger.info("创建人口统计学特征...")
        df_feat = df.copy()

        # 家庭组合特征
        if 'Partner' in df_feat.columns and 'Dependents' in df_feat.columns:
            # 是否有家庭成员
            if df_feat['Partner'].dtype == 'int64':
                df_feat['HasFamily'] = ((df_feat['Partner'] == 1) | (df_feat['Dependents'] == 1)).astype(int)
            else:
                df_feat['HasFamily'] = (
                    (df_feat['Partner'] == 'Yes') | (df_feat['Dependents'] == 'Yes')
                ).astype(int)
            self.logger.info("  创建特征: HasFamily")

            # 家庭规模评分
            if df_feat['Partner'].dtype == 'int64':
                df_feat['FamilySize'] = df_feat['Partner'] + df_feat['Dependents']
            else:
                df_feat['FamilySize'] = (
                    (df_feat['Partner'] == 'Yes').astype(int) +
                    (df_feat['Dependents'] == 'Yes').astype(int)
                )
            self.logger.info("  创建特征: FamilySize")

        # 老年人且无家庭（高风险群体）
        if all(col in df_feat.columns for col in ['SeniorCitizen', 'Partner', 'Dependents']):
            if df_feat['Partner'].dtype == 'int64':
                df_feat['SeniorAlone'] = (
                    (df_feat['SeniorCitizen'] == 1) &
                    (df_feat['Partner'] == 0) &
                    (df_feat['Dependents'] == 0)
                ).astype(int)
            else:
                df_feat['SeniorAlone'] = (
                    (df_feat['SeniorCitizen'] == 1) &
                    (df_feat['Partner'] == 'No') &
                    (df_feat['Dependents'] == 'No')
                ).astype(int)
            self.logger.info("  创建特征: SeniorAlone")

        return df_feat

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        创建交互特征

        Args:
            df: 输入DataFrame

        Returns:
            添加特征后的DataFrame
        """
        self.logger.info("创建交互特征...")
        df_feat = df.copy()

        # 在网时长 × 月费用
        if 'tenure' in df_feat.columns and 'MonthlyCharges' in df_feat.columns:
            df_feat['TenureMonthlyCharges'] = df_feat['tenure'] * df_feat['MonthlyCharges']
            self.logger.info("  创建特征: TenureMonthlyCharges")

        # 服务数量 × 月费用
        if 'TotalServices' in df_feat.columns and 'MonthlyCharges' in df_feat.columns:
            df_feat['ServicesCharges'] = df_feat['TotalServices'] * df_feat['MonthlyCharges']
            self.logger.info("  创建特征: ServicesCharges")

        # 每项服务的平均费用
        if 'TotalServices' in df_feat.columns and 'MonthlyCharges' in df_feat.columns:
            df_feat['ChargesPerService'] = df_feat.apply(
                lambda row: row['MonthlyCharges'] / row['TotalServices'] if row['TotalServices'] > 0 else row['MonthlyCharges'],
                axis=1
            )
            self.logger.info("  创建特征: ChargesPerService")

        return df_feat

    def engineer_features(self, df: pd.DataFrame,
                         encoding_method: str = 'onehot') -> pd.DataFrame:
        """
        完整的特征工程流程

        Args:
            df: 输入DataFrame
            encoding_method: 编码方法 ('onehot' 或 'label')

        Returns:
            完成特征工程后的DataFrame
        """
        self.logger.info("=" * 60)
        self.logger.info("开始特征工程")
        self.logger.info("=" * 60)

        with Timer("特征工程"):
            # 1. 创建新特征（在编码之前）
            df_feat = self.create_tenure_features(df)
            df_feat = self.create_charge_features(df_feat)
            df_feat = self.create_service_features(df_feat)
            df_feat = self.create_contract_features(df_feat)
            df_feat = self.create_demographic_features(df_feat)

            if config.CREATE_INTERACTION_FEATURES:
                df_feat = self.create_interaction_features(df_feat)

            # 2. 编码二值特征
            df_feat = self.encode_binary_features(df_feat)

            # 3. 编码多分类特征
            df_feat = self.encode_categorical_features(df_feat, method=encoding_method)

            # 4. 处理分组特征（TenureGroup, MonthlyChargesGroup）
            group_cols = ['TenureGroup', 'MonthlyChargesGroup']
            for col in group_cols:
                if col in df_feat.columns:
                    # One-Hot编码分组特征
                    dummies = pd.get_dummies(df_feat[col], prefix=col, drop_first=True)
                    df_feat = pd.concat([df_feat, dummies], axis=1)
                    df_feat.drop(col, axis=1, inplace=True)
                    self.logger.info(f"  {col}: 创建 {len(dummies.columns)} 个虚拟变量")

            # 保存特征名称
            self.feature_names = df_feat.columns.tolist()

            self.logger.info(f"\n特征工程完成！")
            self.logger.info(f"最终特征数量: {df_feat.shape[1]}")
            self.logger.info(f"数据形状: {df_feat.shape}")

        return df_feat

    def select_features(self, X: pd.DataFrame, y: pd.Series,
                       method: str = 'mutual_info',
                       k: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """
        特征选择

        Args:
            X: 特征DataFrame
            y: 目标Series
            method: 选择方法 ('chi2', 'f_classif', 'mutual_info')
            k: 选择的特征数量

        Returns:
            选择后的特征DataFrame和特征名称列表
        """
        self.logger.info(f"使用 {method} 进行特征选择...")
        self.logger.info(f"从 {X.shape[1]} 个特征中选择 top {k} 个")

        # 确保所有值都是非负的（chi2要求）
        if method == 'chi2':
            X_positive = X - X.min() + 1e-10
        else:
            X_positive = X

        # 选择评分函数
        if method == 'chi2':
            score_func = chi2
        elif method == 'f_classif':
            score_func = f_classif
        elif method == 'mutual_info':
            score_func = mutual_info_classif
        else:
            raise ValueError(f"不支持的特征选择方法: {method}")

        # 特征选择
        selector = SelectKBest(score_func=score_func, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X_positive, y)

        # 获取选中的特征名称
        selected_indices = selector.get_support(indices=True)
        selected_features = X.columns[selected_indices].tolist()

        self.logger.info(f"已选择 {len(selected_features)} 个特征")
        self.logger.info(f"Top 10 特征: {selected_features[:10]}")

        return X[selected_features], selected_features

    def scale_features(self, X_train: pd.DataFrame,
                      X_test: Optional[pd.DataFrame] = None) -> Tuple:
        """
        特征缩放

        Args:
            X_train: 训练集特征
            X_test: 测试集特征（可选）

        Returns:
            缩放后的特征
        """
        self.logger.info("执行特征缩放...")

        # 拟合并转换训练集
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_scaled = pd.DataFrame(X_train_scaled,
                                      columns=X_train.columns,
                                      index=X_train.index)

        self.logger.info(f"  训练集形状: {X_train_scaled.shape}")

        if X_test is not None:
            # 转换测试集
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled,
                                        columns=X_test.columns,
                                        index=X_test.index)
            self.logger.info(f"  测试集形状: {X_test_scaled.shape}")
            return X_train_scaled, X_test_scaled

        return X_train_scaled

    def save_feature_engineer(self, save_dir: Path = None) -> None:
        """
        保存特征工程器

        Args:
            save_dir: 保存目录
        """
        if save_dir is None:
            save_dir = config.MODELS_DIR

        # 保存标签编码器
        if self.label_encoders:
            save_model(self.label_encoders, save_dir / 'label_encoders.pkl')
            self.logger.info(f"标签编码器已保存")

        # 保存缩放器
        save_model(self.scaler, config.SCALER_PATH)
        self.logger.info(f"特征缩放器已保存")

        # 保存特征名称
        save_model(self.feature_names, config.FEATURE_NAMES_PATH)
        self.logger.info(f"特征名称已保存")


# ==================== 模块级别的包装函数 ====================
# 这些函数供 main.py 等外部模块导入使用

def engineer_features(df: pd.DataFrame, encoding_method: str = 'onehot') -> pd.DataFrame:
    """
    特征工程包装函数

    Args:
        df: 输入DataFrame
        encoding_method: 编码方法 ('onehot' 或 'label')

    Returns:
        完成特征工程后的DataFrame
    """
    engineer = FeatureEngineer()
    return engineer.engineer_features(df, encoding_method=encoding_method)


def select_features(X: pd.DataFrame, y: pd.Series,
                   method: str = 'mutual_info',
                   k: int = 20) -> pd.DataFrame:
    """
    特征选择包装函数

    Args:
        X: 特征DataFrame
        y: 目标Series
        method: 选择方法 ('chi2', 'f_classif', 'mutual_info')
        k: 选择的特征数量

    Returns:
        选择后的特征DataFrame
    """
    engineer = FeatureEngineer()
    X_selected, _ = engineer.select_features(X, y, method=method, k=k)
    return X_selected


if __name__ == '__main__':
    # 测试特征工程模块
    from src.utils import setup_logger
    from src.data_loader import load_data
    from src.data_preprocessing import preprocess_data

    logger = setup_logger("Churn_Prediction", config.LOG_FILE, "INFO")

    print("=" * 60)
    print("特征工程模块测试")
    print("=" * 60)

    # 1. 加载数据
    print("\n1. 加载数据")
    df = load_data()
    print(f"原始数据形状: {df.shape}")

    # 2. 数据预处理
    print("\n2. 数据预处理")
    df_clean = preprocess_data(df)
    print(f"预处理后数据形状: {df_clean.shape}")

    # 3. 特征工程
    print("\n3. 执行特征工程")
    engineer = FeatureEngineer()
    df_feat = engineer.engineer_features(df_clean, encoding_method='onehot')

    print(f"\n特征工程后数据形状: {df_feat.shape}")
    print(f"\n前10个特征:")
    for i, feat in enumerate(engineer.feature_names[:10], 1):
        print(f"  {i}. {feat}")

    # 4. 查看特征统计
    print("\n4. 数值特征统计")
    numeric_features = df_feat.select_dtypes(include=[np.number]).columns
    print(df_feat[numeric_features[:5]].describe())

    # 5. 特征选择测试
    if config.TARGET_COL in df_feat.columns:
        print("\n5. 特征选择测试")
        X = df_feat.drop(columns=[config.TARGET_COL, config.ID_COL], errors='ignore')
        y = df_feat[config.TARGET_COL]

        X_selected, selected_features = engineer.select_features(X, y, method='mutual_info', k=15)
        print(f"选择的特征数量: {len(selected_features)}")
        print(f"选择的特征: {selected_features}")

    # 6. 保存特征工程器
    print("\n6. 保存特征工程器")
    engineer.save_feature_engineer()

    print("\n特征工程模块测试完成！")
