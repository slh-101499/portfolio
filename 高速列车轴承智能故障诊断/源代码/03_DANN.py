import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.manifold import TSNE 
from sklearn.metrics import confusion_matrix, classification_report 
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder, StandardScaler 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset 
import os 
import warnings 

warnings.filterwarnings('ignore') 

# 创建保存图片的目录 
save_dir = '/Users/slh/Downloads/planB/第三问可视化' 
os.makedirs(save_dir, exist_ok=True) 

# 设置中文字体和样式 
plt.rcParams['font.sans-serif'] = ['Arial'] 
plt.rcParams['axes.unicode_minus'] = False 
sns.set_style("whitegrid") 


def load_and_prepare_data(): 
    """加载和准备源域和目标域数据""" 

    # 加载源域数据（训练集和测试集） 
    train_src = pd.read_csv('/Users/slh/Downloads/planB/train_features.csv') 
    test_src = pd.read_csv('/Users/slh/Downloads/planB/test_features.csv') 

    # 合并源域数据 
    X_src = pd.concat([train_src, test_src], ignore_index=True) 

    # 移除不需要的列，保留特征 
    src_features_to_remove = ['file_path'] 
    X_src_features = X_src.drop(columns=src_features_to_remove, errors='ignore') 

    # 分离特征和标签 
    y_src = X_src_features['fault_class'] 
    X_src_final = X_src_features.drop('fault_class', axis=1) 

    # 加载目标域数据 
    target_data = pd.read_csv('/Users/slh/Downloads/planB/target_features.csv') 

    # 移除目标域不需要的列 
    tar_features_to_remove = ['file_path', 'fault_name', 'fault_class'] 
    X_tar = target_data.drop(columns=tar_features_to_remove, errors='ignore') 

    # 获取目标域文件名（A-P） 
    target_filenames = target_data['fault_name'].tolist() 

    # 数据标准化 
    scaler = StandardScaler() 
    X_src_scaled = scaler.fit_transform(X_src_final) 
    X_tar_scaled = scaler.transform(X_tar) 

    print(f"Source domain data shape: {X_src_scaled.shape}") 

    print(f"Source domain labels shape: {y_src.shape}") 
    print(f"Target domain data shape: {X_tar_scaled.shape}") 
    print(f"Number of target domain files: {len(target_filenames)}") 
    print(f"Target domain filenames: {target_filenames}") 

    return X_src_scaled, y_src, X_tar_scaled, target_filenames, scaler 


def encode_labels(y_src): 
    """对标签进行编码""" 
    le = LabelEncoder() 
    y_encoded = le.fit_transform(y_src) 
    class_names = le.classes_ 
    return y_encoded, le, class_names 


def calculate_mmd(X_src, X_tar): 
    """计算最大均值差异（MMD）""" 
    X_src_tensor = torch.FloatTensor(X_src) 
    X_tar_tensor = torch.FloatTensor(X_tar) 

    # 使用高斯核计算MMD 
    def gaussian_kernel(x, y, sigma=1.0): 
        x_size = x.size(0) 
        y_size = y.size(0) 
        dim = x.size(1) 

        x = x.unsqueeze(1)  # x_size x 1 x dim 
        y = y.unsqueeze(0)  # 1 x y_size x dim 

        tiled_x = x.expand(x_size, y_size, dim) 
        tiled_y = y.expand(x_size, y_size, dim) 

        squared_diff = (tiled_x - tiled_y).pow(2).sum(2) 
        return torch.exp(-squared_diff / (2 * sigma ** 2)) 

    K_XX = gaussian_kernel(X_src_tensor, X_src_tensor) 
    K_YY = gaussian_kernel(X_tar_tensor, X_tar_tensor) 
    K_XY = gaussian_kernel(X_src_tensor, X_tar_tensor) 

    mmd = K_XX.mean() + K_YY.mean() - 2 * K_XY.mean() 
    return mmd.item() 


def calculate_wasserstein(X_src, X_tar): 
    """计算Wasserstein距离的近似值""" 
    from scipy.stats import wasserstein_distance 

    # 由于Wasserstein距离计算高维数据较慢，我们使用主要成分 
    from sklearn.decomposition import PCA 
    pca = PCA(n_components=min(5, X_src.shape[1], X_tar.shape[1])) 
    X_combined = np.vstack([X_src, X_tar]) 
    X_combined_pca = pca.fit_transform(X_combined) 

    X_src_pca = X_combined_pca[:len(X_src)] 
    X_tar_pca = X_combined_pca[len(X_src):] 

    # 计算每个维度的Wasserstein距离并取平均 
    wasserstein_dists = [] 
    for i in range(X_src_pca.shape[1]): 
        dist = wasserstein_distance(X_src_pca[:, i], X_tar_pca[:, i]) 
        wasserstein_dists.append(dist) 

    return np.mean(wasserstein_dists) 


class FeatureExtractor(nn.Module): 
    """特征提取器""" 

    def __init__(self, input_dim, hidden_dims=[128, 64]): 
        super(FeatureExtractor, self).__init__() 
        layers = [] 
        prev_dim = input_dim 


        for hidden_dim in hidden_dims: 
            layers.append(nn.Linear(prev_dim, hidden_dim)) 
            layers.append(nn.BatchNorm1d(hidden_dim)) 
            layers.append(nn.ReLU()) 
            layers.append(nn.Dropout(0.3)) 
            prev_dim = hidden_dim 

        self.network = nn.Sequential(*layers) 

    def forward(self, x): 
        return self.network(x) 


class Classifier(nn.Module): 
    """分类器""" 

    def __init__(self, input_dim, num_classes): 
        super(Classifier, self).__init__() 
        self.network = nn.Sequential( 
            nn.Linear(input_dim, 32), 
            nn.BatchNorm1d(32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, num_classes) 
        ) 

    def forward(self, x): 
        return self.network(x) 


class DomainDiscriminator(nn.Module): 
    """领域判别器""" 

    def __init__(self, input_dim): 
        super(DomainDiscriminator, self).__init__() 
        self.network = nn.Sequential( 
            nn.Linear(input_dim, 32), 
            nn.ReLU(), 
            nn.Dropout(0.2), 
            nn.Linear(32, 1), 
            nn.Sigmoid() 
        ) 

    def forward(self, x): 
        return self.network(x) 


class DANNModel: 
    """领域对抗神经网络模型""" 

    def __init__(self, input_dim, num_classes, hidden_dims=[128, 64]): 
        self.feature_extractor = FeatureExtractor(input_dim, hidden_dims) 
        self.classifier = Classifier(hidden_dims[-1], num_classes) 
        self.domain_discriminator = DomainDiscriminator(hidden_dims[-1]) 

        self.optimizer = optim.Adam( 
            list(self.feature_extractor.parameters()) + 
            list(self.classifier.parameters()) + 
            list(self.domain_discriminator.parameters()), 
            lr=0.001 
        ) 

        self.criterion_class = nn.CrossEntropyLoss() 
        self.criterion_domain = nn.BCELoss() 

    def train(self, X_src, y_src, X_tar, epochs=100, batch_size=32): 
        """训练DANN模型""" 
        # 准备数据 
        X_src_tensor = torch.FloatTensor(X_src) 
        y_src_tensor = torch.LongTensor(y_src) 
        X_tar_tensor = torch.FloatTensor(X_tar) 

        # 创建数据加载器 
        src_dataset = TensorDataset(X_src_tensor, y_src_tensor) 

        src_loader = DataLoader(src_dataset, batch_size=batch_size, shuffle=True) 
        tar_loader = DataLoader(TensorDataset(X_tar_tensor), batch_size=batch_size, shuffle=True) 

        losses = [] 
        for epoch in range(epochs): 
            # 同时迭代源域和目标域数据 
            src_iter = iter(src_loader) 
            tar_iter = iter(tar_loader) 

            epoch_loss_class = 0 
            epoch_loss_domain = 0 
            for i in range(min(len(src_loader), len(tar_loader))): 
                # 获取批次数据 
                try: 
                    X_src_batch, y_src_batch = next(src_iter) 
                    X_tar_batch = next(tar_iter)[0] 
                except StopIteration: 
                    break 

                # 创建领域标签 
                batch_size_src = X_src_batch.size(0) 
                batch_size_tar = X_tar_batch.size(0) 

                domain_src = torch.ones(batch_size_src, 1) 
                domain_tar = torch.zeros(batch_size_tar, 1) 

                # 合并数据 
                X_batch = torch.cat([X_src_batch, X_tar_batch], dim=0) 
                domain_labels = torch.cat([domain_src, domain_tar], dim=0) 

                # 前向传播 
                features = self.feature_extractor(X_batch) 
                class_outputs = self.classifier(features[:batch_size_src]) 
                domain_outputs = self.domain_discriminator(features) 

                # 计算损失 
                class_loss = self.criterion_class(class_outputs, y_src_batch) 
                domain_loss = self.criterion_domain(domain_outputs, domain_labels) 

                # 总损失（包含梯度反转） 
                total_loss = class_loss + 0.1 * domain_loss 

                # 反向传播 
                self.optimizer.zero_grad() 
                total_loss.backward() 
                self.optimizer.step() 

                epoch_loss_class += class_loss.item() 
                epoch_loss_domain += domain_loss.item() 

            avg_class_loss = epoch_loss_class / len(src_loader) 
            avg_domain_loss = epoch_loss_domain / len(src_loader) 
            total_avg_loss = avg_class_loss + 0.1 * avg_domain_loss 
            losses.append(total_avg_loss) 

            if epoch % 20 == 0: 
                print( 
                    f"Epoch {epoch:3d} | Class Loss: {avg_class_loss:.4f} | Domain Loss: {avg_domain_loss:.4f} | Total Loss: {total_avg_loss:.4f}") 

        return losses 

    def predict(self, X): 
        """预测""" 
        self.feature_extractor.eval() 
        self.classifier.eval() 

        with torch.no_grad(): 
            X_tensor = torch.FloatTensor(X) 
            features = self.feature_extractor(X_tensor) 
            outputs = self.classifier(features) 
            probabilities = torch.softmax(outputs, dim=1) 
            _, predictions = torch.max(outputs, 1) 


        return predictions.numpy(), probabilities.numpy() 

    def extract_features(self, X): 
        """提取特征""" 
        self.feature_extractor.eval() 

        with torch.no_grad(): 
            X_tensor = torch.FloatTensor(X) 
            features = self.feature_extractor(X_tensor) 

        return features.numpy() 


def plot_tsne_comparison(X_src, X_tar, y_src, target_filenames, y_pred_tar=None, title="t-SNE Visualization", 
                         filename_suffix=""): 
    """绘制t-SNE对比图并保存""" 
    # 合并数据 
    X_combined = np.vstack([X_src, X_tar]) 

    # 创建标签 
    domain_labels = ['Source'] * len(X_src) + ['Target'] * len(X_tar) 

    if y_pred_tar is not None: 
        # 迁移后的情况 
        class_labels = list(y_src) + list(y_pred_tar) 
    else: 
        # 迁移前的情况 
        class_labels = list(y_src) + [-1] * len(X_tar)  # 使用-1表示未知 

    # t-SNE降维 
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X_combined) - 1)) 
    X_tsne = tsne.fit_transform(X_combined) 

    # 创建子图 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6)) 

    # 左图：按领域着色 
    domains = ['Source', 'Target'] 
    colors = ['blue', 'red'] 

    for i, domain in enumerate(domains): 
        mask = [d == domain for d in domain_labels] 
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    c=colors[i], label=domain, alpha=0.7, s=50) 

    ax1.set_title(f'{title} - By Domain', fontsize=14, fontweight='bold') 
    ax1.legend() 
    ax1.set_xlabel('t-SNE Component 1') 
    ax1.set_ylabel('t-SNE Component 2') 
    ax1.grid(True, alpha=0.3) 

    # 右图：按类别着色 
    unique_classes = np.unique(class_labels) 
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_classes))) 

    for i, cls in enumerate(unique_classes): 
        mask = [c == cls for c in class_labels] 
        if cls == -1:  # 未知类别（目标域） 
            color = 'gray' 
            label = 'Unknown' 
        else: 
            color = colors[i] 
            label = f'Class {cls}' 

        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                    c=[color], label=label, alpha=0.7, s=50) 

    ax2.set_title(f'{title} - By Class', fontsize=14, fontweight='bold') 
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
    ax2.set_xlabel('t-SNE Component 1') 
    ax2.set_ylabel('t-SNE Component 2') 
    ax2.grid(True, alpha=0.3) 

    plt.tight_layout() 


    # 保存图片 
    filename = f"tsne_comparison_{filename_suffix}.png" if filename_suffix else "tsne_comparison.png" 
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight') 
    plt.show() 


def create_prediction_table(target_filenames, y_pred, y_prob, class_names): 
    """创建预测结果表格""" 
    results = [] 
    for i, filename in enumerate(target_filenames): 
        pred_class = class_names[y_pred[i]] 
        confidence = y_prob[i].max() 

        # 获取前3个最高概率的类别 
        top3_indices = np.argsort(y_prob[i])[-3:][::-1] 
        top3_probs = [f"{class_names[idx]}: {y_prob[i][idx]:.3f}" for idx in top3_indices] 

        results.append({ 
            'Filename': filename, 
            'Predicted Class': pred_class, 
            'Confidence': f"{confidence:.3f}", 
            'Top 3 Probabilities': ', '.join(top3_probs) 
        }) 

    return pd.DataFrame(results) 


def plot_spectrum_analysis(X_tar, target_filenames, y_pred, y_prob, class_names): 
    """绘制频谱分析图并保存""" 
    n_samples = len(X_tar) 
    n_cols = 4 
    n_rows = (n_samples + n_cols - 1) // n_cols 

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4 * n_rows)) 
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes] 

    for i in range(n_samples): 
        ax = axes[i] 

        # 绘制频谱特征（选择前20个特征） 
        features_to_plot = min(20, X_tar.shape[1]) 
        values = X_tar[i] if isinstance(X_tar, np.ndarray) else X_tar.iloc[i].values 
        ax.bar(range(features_to_plot), values[:features_to_plot], alpha=0.7) 

        # 添加预测信息 
        pred_class = class_names[y_pred[i]] 
        confidence = y_prob[i].max() 

        ax.set_title(f'{target_filenames[i]}\nPred: {pred_class} (Conf: {confidence:.3f})', 
                     fontsize=10, fontweight='bold') 
        ax.set_xlabel('Feature Index') 
        ax.set_ylabel('Feature Value') 
        ax.grid(True, alpha=0.3) 

    # 隐藏多余的子图 
    for i in range(n_samples, len(axes)): 
        axes[i].set_visible(False) 

    plt.tight_layout() 

    # 保存图片 
    plt.savefig(os.path.join(save_dir, "spectrum_analysis.png"), dpi=300, bbox_inches='tight') 
    plt.show() 


def plot_domain_distance_comparison(mmd_before, mmd_after, wasserstein_before, wasserstein_after): 
    """绘制域距离对比图并保存""" 
    plt.figure(figsize=(10, 6)) 
    metrics = ['MMD', 'Wasserstein'] 
    before = [mmd_before, wasserstein_before] 
    after = [mmd_after, wasserstein_after] 

    x = np.arange(len(metrics)) 

    width = 0.35 

    bars1 = plt.bar(x - width / 2, before, width, label='Before Adaptation', alpha=0.8, color='lightcoral') 
    bars2 = plt.bar(x + width / 2, after, width, label='After Adaptation', alpha=0.8, color='lightseagreen') 

    plt.xlabel('Distance Metric', fontsize=12) 
    plt.ylabel('Distance Value', fontsize=12) 
    plt.title('Domain Distance Comparison Before and After Adaptation', 
              fontsize=14, fontweight='bold') 
    plt.xticks(x, metrics) 
    plt.legend(fontsize=10) 
    plt.grid(True, alpha=0.3) 

    # 添加数值标签 
    for bar, v in zip(bars1, before): 
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9) 

    for bar, v in zip(bars2, after): 
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, 
                 f'{v:.3f}', ha='center', va='bottom', fontsize=9) 

    plt.tight_layout() 

    # 保存图片 
    plt.savefig(os.path.join(save_dir, "domain_distance_comparison.png"), dpi=300, bbox_inches='tight') 
    plt.show() 


def plot_training_loss(losses): 
    """绘制训练损失曲线并保存""" 
    plt.figure(figsize=(10, 5)) 
    plt.plot(losses, linewidth=2) 
    plt.title('DANN Training Loss', fontsize=14, fontweight='bold') 
    plt.xlabel('Epoch') 
    plt.ylabel('Loss') 
    plt.grid(True, alpha=0.3) 
    plt.tight_layout() 

    # 保存图片 
    plt.savefig(os.path.join(save_dir, "training_loss.png"), dpi=300, bbox_inches='tight') 
    plt.show() 


def k_fold_validation(X_src, y_src, n_splits=5): 
    """K折交叉验证并保存结果图""" 
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42) 

    accuracies = [] 
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_src)): 
        # 使用numpy数组索引而不是iloc 
        X_train, X_val = X_src[train_idx], X_src[val_idx] 
        y_train, y_val = y_src[train_idx], y_src[val_idx] 

        # 创建简单分类器进行验证 
        from sklearn.ensemble import RandomForestClassifier 
        clf = RandomForestClassifier(n_estimators=100, random_state=42) 
        clf.fit(X_train, y_train) 

        y_pred = clf.predict(X_val) 
        accuracy = (y_pred == y_val).mean() 
        accuracies.append(accuracy) 

        print(f"Fold {fold + 1}: Accuracy = {accuracy:.4f}") 

    print(f"\nK-Fold Cross Validation Results:") 
    print(f"Mean Accuracy: {np.mean(accuracies):.4f} (+/- {np.std(accuracies):.4f})") 

    # 绘制K折验证结果 
    plt.figure(figsize=(8, 5)) 
    plt.bar(range(1, n_splits + 1), accuracies, color='skyblue', alpha=0.8) 
    plt.axhline(y=np.mean(accuracies), color='red', linestyle='--', 
                label=f'Mean Accuracy: {np.mean(accuracies):.4f}') 
    plt.xlabel('Fold Number') 

    plt.ylabel('Accuracy') 
    plt.title('K-Fold Cross Validation Results', fontsize=14, fontweight='bold') 
    plt.legend() 
    plt.grid(True, alpha=0.3) 
    plt.ylim(0, 1) 
    plt.tight_layout() 

    # 保存图片 
    plt.savefig(os.path.join(save_dir, "kfold_validation.png"), dpi=300, bbox_inches='tight') 
    plt.show() 

    return accuracies 


def main(): 
    """主函数""" 
    print("=== Domain Adaptation Fault Diagnosis System ===\n") 

    # 1. 加载数据 
    X_src, y_src, X_tar, target_filenames, scaler = load_and_prepare_data() 

    # 2. 对标签进行编码 
    y_src_encoded, label_encoder, class_names = encode_labels(y_src) 
    num_classes = len(class_names) 

    print(f"\nNumber of classes: {num_classes}") 
    print(f"Class names: {class_names}") 
    print(f"Encoded labels: {np.unique(y_src_encoded)}") 

    # 3. 计算迁移前的域差异 
    print("\n=== Pre-adaptation Domain Differences ===") 
    mmd_before = calculate_mmd(X_src, X_tar) 
    wasserstein_before = calculate_wasserstein(X_src, X_tar) 

    print(f"MMD before adaptation: {mmd_before:.4f}") 
    print(f"Wasserstein distance before adaptation: {wasserstein_before:.4f}") 

    # 4. 迁移前的可视化 
    print("\n=== Pre-adaptation Visualization ===") 
    plot_tsne_comparison(X_src, X_tar, y_src_encoded, 
                         target_filenames, title="Before Adaptation", filename_suffix="before") 

    # 5. 训练DANN模型 
    print("\n=== Training DANN Model ===") 
    input_dim = X_src.shape[1] 
    dann_model = DANNModel(input_dim, num_classes) 

    losses = dann_model.train(X_src, y_src_encoded, X_tar, epochs=100, batch_size=32) 

    # 绘制训练损失 
    plot_training_loss(losses) 

    # 6. 预测目标域 
    print("\n=== Predicting Target Domain ===") 
    y_pred_tar, y_prob_tar = dann_model.predict(X_tar) 

    # 7. 提取迁移后的特征 
    features_src_adapted = dann_model.extract_features(X_src) 
    features_tar_adapted = dann_model.extract_features(X_tar) 

    # 8. 计算迁移后的域差异 
    print("\n=== Post-adaptation Domain Differences ===") 
    mmd_after = calculate_mmd(features_src_adapted, features_tar_adapted) 
    wasserstein_after = calculate_wasserstein(features_src_adapted, features_tar_adapted) 

    print(f"MMD after adaptation: {mmd_after:.4f}") 
    print(f"Wasserstein distance after adaptation: {wasserstein_after:.4f}") 

    # 9. 迁移后的可视化 
    print("\n=== Post-adaptation Visualization ===") 
    plot_tsne_comparison(features_src_adapted, features_tar_adapted, 
                         y_src_encoded, target_filenames, y_pred_tar, 
                         title="After Adaptation", filename_suffix="after") 


    # 10. 创建预测结果表格 
    print("\n=== Prediction Results ===") 
    results_df = create_prediction_table(target_filenames, y_pred_tar, 
                                         y_prob_tar, class_names) 
    print(results_df.to_string(index=False)) 

    # 11. 绘制域差异对比 
    plot_domain_distance_comparison(mmd_before, mmd_after, wasserstein_before, wasserstein_after) 

    # 12. 绘制频谱分析 
    print("\n=== Spectrum Analysis ===") 
    plot_spectrum_analysis(X_tar, target_filenames, y_pred_tar, y_prob_tar, class_names) 

    # 13. K折交叉验证 
    print("\n=== K-Fold Cross Validation on Source Domain ===") 
    accuracies = k_fold_validation(X_src, y_src_encoded, n_splits=5) 

    # 14. 保存结果 
    results_df.to_csv(os.path.join(save_dir, 'target_domain_predictions.csv'), index=False) 
    print(f"\n=== Results saved to '{save_dir}/target_domain_predictions.csv' ===") 

    # 15. 保存域距离结果 
    distance_results = pd.DataFrame({ 
        'Metric': ['MMD', 'Wasserstein'], 
        'Before_Adaptation': [mmd_before, wasserstein_before], 
        'After_Adaptation': [mmd_after, wasserstein_after], 
        'Reduction_Percentage': [ 
            (mmd_before - mmd_after) / mmd_before * 100, 
            (wasserstein_before - wasserstein_after) / wasserstein_before * 100 
        ] 
    }) 
    distance_results.to_csv(os.path.join(save_dir, 'domain_distance_results.csv'), index=False) 

    print(f"\nAll visualizations have been saved to: {save_dir}") 

    return results_df, mmd_before, mmd_after, wasserstein_before, wasserstein_after 


if __name__ == "__main__": 
    # 运行主程序 
    try: 
        results, mmd_b, mmd_a, wass_b, wass_a = main() 

        # 打印总结 
        print("\n" + "=" * 50) 
        print("SUMMARY") 
        print("=" * 50) 
        print(f"Domain adaptation completed successfully!") 
        print(f"MMD reduction: {mmd_b:.4f} → {mmd_a:.4f} ({(mmd_b - mmd_a) / mmd_b * 100:.1f}% reduction)") 
        print(f"Wasserstein reduction: {wass_b:.4f} → {wass_a:.4f} ({(wass_b - wass_a) / wass_b * 100:.1f}% reduction)") 
        print(f"\nAll results and visualizations saved to: {save_dir}") 

    except Exception as e: 
        print(f"Error occurred: {e}") 
        import traceback 

        traceback.print_exc()