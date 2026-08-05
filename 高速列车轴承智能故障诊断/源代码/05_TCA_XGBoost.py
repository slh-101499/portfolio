import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.manifold import TSNE 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
import torch 
import torch.nn as nn 
import warnings 
import matplotlib.pyplot as plt 
from matplotlib.font_manager import FontProperties, fontManager 
import os 

warnings.filterwarnings('ignore') 

# 设置中文字体和图形样式 
# 注册字体 
font_path = '/System/Library/Fonts/STHeiti Medium.ttc' 

# 使用 FontProperties 获取字体名称 
custom_font = FontProperties(fname=font_path) 
font_name = custom_font.get_name() 

# 设置全局字体 
plt.rcParams['font.sans-serif'] = [font_name, 'Arial', 'DejaVu Sans'] 
plt.rcParams['axes.unicode_minus'] = False 
sns.set_style("whitegrid") 
sns.set_palette("husl") 


class TCA: 
    """Transfer Component Analysis (TCA) 实现""" 

    def __init__(self, kernel_type='linear', dim=30, lamb=1, gamma=1): 
        self.kernel_type = kernel_type 
        self.dim = dim 
        self.lamb = lamb 
        self.gamma = gamma 

    def fit(self, Xs, Xt): 
        """训练TCA变换""" 
        X = np.hstack((Xs.T, Xt.T)) 
        X /= np.linalg.norm(X, axis=0) 
        m, n = X.shape 

        ns, nt = len(Xs), len(Xt) 
        self.ns, self.nt = ns, nt 

        # 构建MMD矩阵 
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1)))) 
        M = e * e.T 

        # 构建中心矩阵 
        H = np.eye(n) - 1 / n * np.ones((n, n)) 

        # 核矩阵 
        K = self.kernel(self.kernel_type, X, None, self.gamma) 

        # 求解特征值问题 
        A = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(m) 
        B = np.linalg.multi_dot([K, H, K.T]) 

        # 广义特征值分解 
        eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(B).dot(A)) 
        idx = np.argsort(eigvals)[:self.dim] 

        self.W = eigvecs[:, idx] 
        return self 

    def transform(self, Xs, Xt): 
        """应用TCA变换""" 
        X = np.hstack((Xs.T, Xt.T)) 
        X /= np.linalg.norm(X, axis=0) 
        K = self.kernel(self.kernel_type, X, None, self.gamma) 
        X_new = np.dot(K.T, self.W) 

        Xs_new = X_new[:self.ns] 
        Xt_new = X_new[self.ns:] 

        return Xs_new, Xt_new 

    def kernel(self, kernel_type, X, Y, gamma): 
        """核函数""" 
        if kernel_type == 'linear': 
            if Y is None: 
                return np.dot(X.T, X) 
            else: 
                return np.dot(X.T, Y) 
        elif kernel_type == 'rbf': 
            if Y is None: 
                Y = X 
            n = X.shape[1] 
            m = Y.shape[1] 
            K = np.zeros((n, m)) 
            for i in range(n): 
                for j in range(m): 
                    K[i, j] = np.exp(-gamma * np.linalg.norm(X[:, i] - Y[:, j]) ** 2) 
            return K 
        else: 
            raise ValueError('不支持的核类型') 


def load_and_prepare_data(): 
    """加载和准备源域和目标域数据""" 

    # 加载源域数据（训练集和测试集） 
    train_src = pd.read_csv('/Users/slh/Downloads/数据集/train_features_standardized.csv') 
    test_src = pd.read_csv('/Users/slh/Downloads/数据集/test_features_standardized.csv') 

    # 合并源域数据 
    X_src = pd.concat([train_src, test_src], ignore_index=True) 

    # 移除不需要的列，保留特征 
    src_features_to_remove = ['filename', 'fault_size', 'load'] 
    X_src_features = X_src.drop(columns=src_features_to_remove, errors='ignore') 

    # 分离特征和标签 
    y_src = X_src_features['fault_type'] 

    X_src_final = X_src_features.drop('fault_type', axis=1) 

    # 加载目标域数据 
    target_data = pd.read_csv('/Users/slh/Downloads/数据集/target_features_standardized.csv') 

    # 移除目标域不需要的列 
    tar_features_to_remove = ['filename', 'fault_type', 'fault_size', 'load'] 
    X_tar = target_data.drop(columns=tar_features_to_remove, errors='ignore') 

    # 获取目标域文件名（A-P） 
    target_filenames = target_data['filename'].tolist() 

    print(f"源域数据形状: {X_src_final.shape}") 
    print(f"源域标签形状: {y_src.shape}") 
    print(f"目标域数据形状: {X_tar.shape}") 
    print(f"目标域文件数量: {len(target_filenames)}") 

    return X_src_final, y_src, X_tar, target_filenames 


def prepare_tca_data(X_src_train, X_tar_scaled): 
    """准备TCA数据""" 

    # 合并源域和目标域特征 
    X_combined = np.vstack([X_src_train, X_tar_scaled]) 

    # 创建域标签：源域为1，目标域为0 
    domain_labels = np.zeros(X_combined.shape[0]) 
    domain_labels[:len(X_src_train)] = 1  # 源域样本标记为1 

    print(f"合并后数据形状: {X_combined.shape}") 
    print(f"源域样本数: {len(X_src_train)}") 
    print(f"目标域样本数: {len(X_tar_scaled)}") 
    print(f"域标签分布 - 源域(1): {np.sum(domain_labels == 1)}, 目标域(0): {np.sum(domain_labels == 0)}") 

    return X_combined, domain_labels 


def train_tca_xgboost(X_src_train, y_src_train, X_tar_scaled): 
    """训练TCA + XGBoost模型""" 

    # 准备TCA数据 
    X_combined, domain_labels = prepare_tca_data(X_src_train, X_tar_scaled) 

    # 训练TCA变换器 
    print("训练TCA变换器...") 
    tca = TCA(kernel_type='linear', dim=min(30, X_combined.shape[1]), lamb=1, gamma=1) 
    tca.fit(X_src_train.values, X_tar_scaled.values) 

    # 应用TCA变换 
    X_src_tca, X_tar_tca = tca.transform(X_src_train.values, X_tar_scaled.values) 

    print(f"TCA变换后源域形状: {X_src_tca.shape}") 
    print(f"TCA变换后目标域形状: {X_tar_tca.shape}") 

    # 在TCA空间训练XGBoost分类器 
    print("训练XGBoost分类器...") 
    xgb_model = xgb.XGBClassifier( 
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        eval_metric='mlogloss' 
    ) 

    # 分割源域数据用于验证 
    X_train, X_val, y_train, y_val = train_test_split( 
        X_src_tca, y_src_train, test_size=0.2, random_state=42, stratify=y_src_train 
    ) 


    # 训练模型 
    xgb_model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=10, 
                  verbose=False) 

    # 在验证集上评估 
    y_val_pred = xgb_model.predict(X_val) 
    val_accuracy = accuracy_score(y_val, y_val_pred) 
    print(f"验证集准确率: {val_accuracy:.4f}") 

    return tca, xgb_model, X_src_tca, X_tar_tca, X_combined, domain_labels 


def predict_target_domain_tca(xgb_model, X_tar_tca, target_filenames): 
    """预测目标域数据""" 

    # 预测目标域 
    y_tar_pred = xgb_model.predict(X_tar_tca) 
    y_tar_pred_proba = xgb_model.predict_proba(X_tar_tca) 

    # 获取置信度（最大概率） 
    confidence_scores = np.max(y_tar_pred_proba, axis=1) 

    print("目标域预测完成!") 
    return y_tar_pred, confidence_scores, y_tar_pred_proba 


def plot_tsne_comparison_tca(X_src_train, X_tar_scaled, y_src_train, 
                             X_src_tca, X_tar_tca, target_filenames, 
                             y_tar_pred, le): 
    """绘制TCA迁移前后的t-SNE对比图""" 

    # 迁移前：原始特征 
    print("正在进行迁移前t-SNE降维...") 
    tsne_original = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_src_train) // 4)) 

    # 合并源域和目标域原始特征 
    original_combined = np.vstack([X_src_train.values, X_tar_scaled.values]) 
    original_tsne = tsne_original.fit_transform(original_combined) 

    # 迁移后：TCA特征 
    print("正在进行迁移后t-SNE降维...") 
    tsne_tca = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_src_tca) // 4)) 
    tca_combined = np.vstack([X_src_tca, X_tar_tca]) 
    tca_tsne = tsne_tca.fit_transform(tca_combined) 

    # 创建图形 
    fig, axes = plt.subplots(2, 3, figsize=(24, 16)) 

    # 迁移前t-SNE可视化 
    split_point = len(X_src_train) 
    src_original = original_tsne[:split_point] 
    tar_original = original_tsne[split_point:] 

    # 子图1：迁移前（按域着色） 
    axes[0, 0].scatter(src_original[:, 0], src_original[:, 1], 
                       c='blue', alpha=0.6, label='Source Domain', s=50) 
    axes[0, 0].scatter(tar_original[:, 0], tar_original[:, 1], 
                       c='red', alpha=0.6, label='Target Domain', s=50) 
    axes[0, 0].set_title('Before TCA Adaptation\n(Colored by Domain)', fontsize=14, fontweight='bold') 
    axes[0, 0].set_xlabel('t-SNE Component 1') 
    axes[0, 0].set_ylabel('t-SNE Component 2') 
    axes[0, 0].legend() 
    axes[0, 0].grid(True, alpha=0.3) 

    # 子图2：迁移前（按故障类型着色） 
    unique_labels = le.classes_ 
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels))) 

    for i, label in enumerate(unique_labels): 
        mask = (y_src_train == label) 
        if mask.any(): 

            axes[0, 1].scatter(src_original[mask, 0], src_original[mask, 1], 
                               c=[colors[i]], label=label, alpha=0.7, s=50) 

    # 标记目标域样本 
    for i, filename in enumerate(target_filenames): 
        axes[0, 1].scatter(tar_original[i, 0], tar_original[i, 1], 
                           c='black', marker='*', s=150, alpha=0.8, label=f'Target {filename}' if i == 0 else "") 

    axes[0, 1].set_title('Before TCA Adaptation\n(Colored by Fault Type)', fontsize=14, fontweight='bold') 
    axes[0, 1].set_xlabel('t-SNE Component 1') 
    axes[0, 1].set_ylabel('t-SNE Component 2') 
    axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    axes[0, 1].grid(True, alpha=0.3) 

    # 子图3：迁移前特征分布密度 
    src_density = np.linalg.norm(src_original, axis=1) 
    tar_density = np.linalg.norm(tar_original, axis=1) 

    axes[0, 2].hist(src_density, bins=30, alpha=0.7, label='Source Domain', color='blue', density=True) 
    axes[0, 2].hist(tar_density, bins=30, alpha=0.7, label='Target Domain', color='red', density=True) 
    axes[0, 2].set_title('Feature Distribution Before TCA', fontsize=14, fontweight='bold') 
    axes[0, 2].set_xlabel('Feature Norm') 
    axes[0, 2].set_ylabel('Density') 
    axes[0, 2].legend() 
    axes[0, 2].grid(True, alpha=0.3) 

    # 迁移后t-SNE可视化 
    src_tca = tca_tsne[:len(X_src_tca)] 
    tar_tca = tca_tsne[len(X_src_tca):] 

    # 子图4：迁移后（按域着色） 
    axes[1, 0].scatter(src_tca[:, 0], src_tca[:, 1], 
                       c='blue', alpha=0.6, label='Source Domain', s=50) 
    axes[1, 0].scatter(tar_tca[:, 0], tar_tca[:, 1], 
                       c='red', alpha=0.6, label='Target Domain', s=50) 
    axes[1, 0].set_title('After TCA Adaptation\n(Colored by Domain)', fontsize=14, fontweight='bold') 
    axes[1, 0].set_xlabel('t-SNE Component 1') 
    axes[1, 0].set_ylabel('t-SNE Component 2') 
    axes[1, 0].legend() 
    axes[1, 0].grid(True, alpha=0.3) 

    # 子图5：迁移后（按故障类型着色） 
    for i, label in enumerate(unique_labels): 
        mask = (y_src_train == label) 
        if mask.any(): 
            axes[1, 1].scatter(src_tca[mask, 0], src_tca[mask, 1], 
                               c=[colors[i]], label=label, alpha=0.7, s=50) 

    # 标记目标域样本及其预测标签 
    for i, (filename, pred_label) in enumerate(zip(target_filenames, y_tar_pred)): 
        color_idx = np.where(unique_labels == pred_label)[0][0] 
        axes[1, 1].scatter(tar_tca[i, 0], tar_tca[i, 1], 
                           c=[colors[color_idx]], marker='*', s=150, alpha=0.8, 
                           label=f'Target {filename} ({pred_label})' if i == 0 else "") 

    axes[1, 1].set_title('After TCA Adaptation\n(Colored by Predicted Fault Type)', fontsize=14, fontweight='bold') 
    axes[1, 1].set_xlabel('t-SNE Component 1') 
    axes[1, 1].set_ylabel('t-SNE Component 2') 
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    axes[1, 1].grid(True, alpha=0.3) 

    # 子图6：迁移后特征分布密度 
    src_tca_density = np.linalg.norm(src_tca, axis=1) 
    tar_tca_density = np.linalg.norm(tar_tca, axis=1) 

    axes[1, 2].hist(src_tca_density, bins=30, alpha=0.7, label='Source Domain', color='blue', density=True) 
    axes[1, 2].hist(tar_tca_density, bins=30, alpha=0.7, label='Target Domain', color='red', density=True) 
    axes[1, 2].set_title('Feature Distribution After TCA', fontsize=14, fontweight='bold') 
    axes[1, 2].set_xlabel('Feature Norm') 
    axes[1, 2].set_ylabel('Density') 
    axes[1, 2].legend() 
    axes[1, 2].grid(True, alpha=0.3) 

    plt.tight_layout() 

    plt.show() 

    return original_tsne, tca_tsne 


def create_prediction_table_tca(target_filenames, predicted_labels, confidence_scores, le): 
    """创建TCA预测结果表格""" 

    results_df = pd.DataFrame({ 
        'Filename': target_filenames, 
        'Predicted_Fault_Type': predicted_labels, 
        'Confidence': confidence_scores 
    }) 

    # 按文件名排序（A-P） 
    results_df['File_Index'] = results_df['Filename'].apply( 
        lambda x: ord(x[0]) if isinstance(x, str) and len(x) > 0 else 0) 
    results_df = results_df.sort_values('File_Index').drop('File_Index', axis=1) 

    # 创建精美的表格可视化 
    plt.figure(figsize=(14, 10)) 
    ax = plt.subplot(111, frame_on=False) 
    ax.xaxis.set_visible(False) 
    ax.yaxis.set_visible(False) 

    # 创建表格数据 
    table_data = [] 
    for _, row in results_df.iterrows(): 
        table_data.append([row['Filename'], row['Predicted_Fault_Type'], f"{row['Confidence']:.4f}"]) 

    # 创建表格 
    table = plt.table(cellText=table_data, 
                      colLabels=['Filename', 'Predicted Fault Type', 'Confidence'], 
                      colWidths=[0.15, 0.35, 0.25], 
                      cellLoc='center', 
                      loc='center', 
                      bbox=[0.1, 0.1, 0.8, 0.8]) 

    # 美化表格 
    table.auto_set_font_size(False) 
    table.set_fontsize(11) 
    table.scale(1, 1.8) 

    # 设置标题 
    plt.title('TCA + XGBoost: Target Domain Prediction Results (A-P)', 
              fontsize=16, fontweight='bold', pad=30) 
    plt.tight_layout() 
    plt.show() 

    return results_df 


def plot_feature_importance(xgb_model, feature_names=None, top_k=20): 
    """绘制特征重要性图""" 

    # 获取特征重要性 
    importance_scores = xgb_model.feature_importances_ 

    if feature_names is None: 
        feature_names = [f'Feature_{i}' for i in range(len(importance_scores))] 

    # 选择最重要的特征 
    indices = np.argsort(importance_scores)[::-1][:top_k] 

    plt.figure(figsize=(12, 8)) 
    bars = plt.barh(range(top_k), importance_scores[indices][::-1], 
                    color=plt.cm.viridis(np.linspace(0, 1, top_k))) 

    plt.yticks(range(top_k), [feature_names[i] for i in indices[::-1]]) 
    plt.xlabel('Feature Importance') 
    plt.title(f'Top {top_k} Feature Importance (XGBoost)', fontsize=14, fontweight='bold') 
    plt.grid(True, alpha=0.3) 

    # 添加数值标签 

    for i, bar in enumerate(bars): 
        width = bar.get_width() 
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, 
                 f'{width:.3f}', ha='left', va='center', fontsize=9) 

    plt.tight_layout() 
    plt.show() 


def plot_confidence_analysis_tca(target_filenames, confidence_scores, predicted_labels): 
    """绘制置信度分析图""" 

    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 

    # 置信度按文件排序 
    sorted_indices = np.argsort(target_filenames) 
    sorted_filenames = [target_filenames[i] for i in sorted_indices] 
    sorted_confidence = [confidence_scores[i] for i in sorted_indices] 
    sorted_labels = [predicted_labels[i] for i in sorted_indices] 

    # 置信度条形图 
    bars = axes[0, 0].bar(range(len(sorted_filenames)), sorted_confidence, 
                          color=plt.cm.viridis(np.linspace(0, 1, len(sorted_filenames)))) 
    axes[0, 0].set_title('Confidence Scores by File (A-P)', fontsize=14, fontweight='bold') 
    axes[0, 0].set_xlabel('File Name') 
    axes[0, 0].set_ylabel('Confidence Score') 
    axes[0, 0].set_xticks(range(len(sorted_filenames))) 
    axes[0, 0].set_xticklabels(sorted_filenames, rotation=45) 
    axes[0, 0].grid(True, alpha=0.3) 

    # 在条形图上添加数值标签 
    for i, (bar, confidence) in enumerate(zip(bars, sorted_confidence)): 
        height = bar.get_height() 
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.01, 
                        f'{confidence:.3f}', ha='center', va='bottom', fontsize=9) 

    # 按故障类型分组的置信度箱线图 
    confidence_by_fault = {} 
    for fault, confidence in zip(predicted_labels, confidence_scores): 
        if fault not in confidence_by_fault: 
            confidence_by_fault[fault] = [] 
        confidence_by_fault[fault].append(confidence) 

    fault_types = list(confidence_by_fault.keys()) 
    confidence_values = [confidence_by_fault[fault] for fault in fault_types] 

    if len(fault_types) > 0: 
        boxplot = axes[0, 1].boxplot(confidence_values, labels=fault_types, patch_artist=True) 
        axes[0, 1].set_title('Confidence Distribution by Fault Type', fontsize=14, fontweight='bold') 
        axes[0, 1].set_xlabel('Fault Type') 
        axes[0, 1].set_ylabel('Confidence Score') 
        axes[0, 1].grid(True, alpha=0.3) 

        # 美化箱线图 
        colors = plt.cm.Set3(np.linspace(0, 1, len(fault_types))) 
        for patch, color in zip(boxplot['boxes'], colors): 
            patch.set_facecolor(color) 
            patch.set_alpha(0.7) 

    # 置信度分布直方图 
    axes[1, 0].hist(confidence_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black') 
    axes[1, 0].axvline(np.mean(confidence_scores), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(confidence_scores):.3f}') 
    axes[1, 0].set_title('Confidence Score Distribution', fontsize=14, fontweight='bold') 
    axes[1, 0].set_xlabel('Confidence Score') 
    axes[1, 0].set_ylabel('Frequency') 
    axes[1, 0].legend() 
    axes[1, 0].grid(True, alpha=0.3) 

    # 故障类型分布饼图 
    unique_preds, pred_counts = np.unique(predicted_labels, return_counts=True) 
    if len(unique_preds) > 0: 
        wedges, texts, autotexts = axes[1, 1].pie(pred_counts, labels=unique_preds, autopct='%1.1f%%', 
                                                  startangle=90, 

                                                  colors=plt.cm.Set3(np.linspace(0, 1, len(unique_preds)))) 
        axes[1, 1].set_title('Predicted Fault Type Distribution', fontsize=14, fontweight='bold') 

        # 美化饼图文字 
        for autotext in autotexts: 
            autotext.set_color('white') 
            autotext.set_fontweight('bold') 

    plt.tight_layout() 
    plt.show() 


def plot_domain_distribution_comparison(X_src_train, X_tar_scaled, X_src_tca, X_tar_tca): 
    """绘制域分布对比图""" 

    fig, axes = plt.subplots(2, 2, figsize=(16, 12)) 

    # 原始特征均值分布 
    src_means_orig = X_src_train.mean(axis=1) 
    tar_means_orig = X_tar_scaled.mean(axis=1) 

    axes[0, 0].hist(src_means_orig, bins=30, alpha=0.7, label='Source Domain', color='blue', density=True) 
    axes[0, 0].hist(tar_means_orig, bins=30, alpha=0.7, label='Target Domain', color='red', density=True) 
    axes[0, 0].set_title('Original Feature Means Distribution', fontsize=12, fontweight='bold') 
    axes[0, 0].set_xlabel('Feature Mean') 
    axes[0, 0].set_ylabel('Density') 
    axes[0, 0].legend() 
    axes[0, 0].grid(True, alpha=0.3) 

    # TCA特征均值分布 
    src_means_tca = X_src_tca.mean(axis=1) 
    tar_means_tca = X_tar_tca.mean(axis=1) 

    axes[0, 1].hist(src_means_tca, bins=30, alpha=0.7, label='Source Domain', color='blue', density=True) 
    axes[0, 1].hist(tar_means_tca, bins=30, alpha=0.7, label='Target Domain', color='red', density=True) 
    axes[0, 1].set_title('TCA Feature Means Distribution', fontsize=12, fontweight='bold') 
    axes[0, 1].set_xlabel('Feature Mean') 
    axes[0, 1].set_ylabel('Density') 
    axes[0, 1].legend() 
    axes[0, 1].grid(True, alpha=0.3) 

    # 原始特征方差分布 
    src_vars_orig = X_src_train.var(axis=1) 
    tar_vars_orig = X_tar_scaled.var(axis=1) 

    axes[1, 0].hist(src_vars_orig, bins=30, alpha=0.7, label='Source Domain', color='blue', density=True) 
    axes[1, 0].hist(tar_vars_orig, bins=30, alpha=0.7, label='Target Domain', color='red', density=True) 
    axes[1, 0].set_title('Original Feature Variance Distribution', fontsize=12, fontweight='bold') 
    axes[1, 0].set_xlabel('Feature Variance') 
    axes[1, 0].set_ylabel('Density') 
    axes[1, 0].legend() 
    axes[1, 0].grid(True, alpha=0.3) 

    # TCA特征方差分布 
    src_vars_tca = X_src_tca.var(axis=1) 
    tar_vars_tca = X_tar_tca.var(axis=1) 

    axes[1, 1].hist(src_vars_tca, bins=30, alpha=0.7, label='Source Domain', color='blue', density=True) 
    axes[1, 1].hist(tar_vars_tca, bins=30, alpha=0.7, label='Target Domain', color='red', density=True) 
    axes[1, 1].set_title('TCA Feature Variance Distribution', fontsize=12, fontweight='bold') 
    axes[1, 1].set_xlabel('Feature Variance') 
    axes[1, 1].set_ylabel('Density') 
    axes[1, 1].legend() 
    axes[1, 1].grid(True, alpha=0.3) 

    plt.tight_layout() 
    plt.show() 


# 主程序 
if __name__ == "__main__": 
    # 1. 加载数据 
    print("=== TCA + XGBoost 模型 ===") 
    print("数据加载中...") 

    X_src, y_src, X_tar, target_filenames = load_and_prepare_data() 

    # 2. 标签编码 
    le = LabelEncoder() 
    y_src_encoded = le.fit_transform(y_src) 

    # 3. 训练TCA + XGBoost模型 
    print("\n=== 训练TCA + XGBoost模型 ===") 
    tca, xgb_model, X_src_tca, X_tar_tca, X_combined, domain_labels = train_tca_xgboost( 
        X_src, y_src, X_tar 
    ) 

    # 4. 预测目标域 
    print("\n=== 目标域预测 ===") 
    y_tar_pred, confidence_scores, y_tar_pred_proba = predict_target_domain_tca( 
        xgb_model, X_tar_tca, target_filenames 
    ) 

    # 5. 可视化分析 
    print("\n=== 可视化分析 ===") 

    # t-SNE对比可视化 
    original_tsne, tca_tsne = plot_tsne_comparison_tca( 
        X_src, X_tar, y_src, X_src_tca, X_tar_tca, target_filenames, y_tar_pred, le 
    ) 

    # 创建预测结果表格 
    results_df = create_prediction_table_tca(target_filenames, y_tar_pred, confidence_scores, le) 

    # 特征重要性可视化 
    plot_feature_importance(xgb_model, top_k=15) 

    # 置信度分析 
    plot_confidence_analysis_tca(target_filenames, confidence_scores, y_tar_pred) 

    # 域分布对比 
    plot_domain_distribution_comparison(X_src, X_tar, X_src_tca, X_tar_tca) 

    # 6. 打印详细结果 
    print("\n=== 详细预测结果 ===") 
    print("目标域样本预测结果:") 
    print("=" * 60) 
    for filename, pred_label, confidence in zip(target_filenames, y_tar_pred, confidence_scores): 
        print(f"文件: {filename:>3} | 预测故障类型: {pred_label:>15} | 置信度: {confidence:.4f}") 

    print("\n故障类型统计:") 
    print("=" * 30) 
    unique, counts = np.unique(y_tar_pred, return_counts=True) 
    for fault_type, count in zip(unique, counts): 
        print(f"{fault_type:>15}: {count:>2}个样本") 

    print(f"\n模型性能统计:") 
    print("=" * 30) 
    print(f"平均置信度: {np.mean(confidence_scores):.4f}") 
    print(f"置信度标准差: {np.std(confidence_scores):.4f}") 
    print(f"最低置信度: {np.min(confidence_scores):.4f}") 
    print(f"最高置信度: {np.max(confidence_scores):.4f}") 
    print(f"目标域样本数: {len(target_filenames)}") 

    # 保存预测结果到CSV文件 
    results_df.to_csv('/Users/slh/Downloads/数据集/tca_xgboost_predictions.csv', index=False) 
    print(f"\n预测结果已保存到: /Users/slh/Downloads/数据集/tca_xgboost_predictions.csv") 

    print("\n=== TCA + XGBoost 模型训练和预测完成 ===")