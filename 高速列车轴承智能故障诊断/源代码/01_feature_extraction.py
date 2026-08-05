import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from scipy import signal 
from scipy.fft import fft, fftfreq 
from scipy.signal import hilbert, welch 
from sklearn.preprocessing import StandardScaler 
import pywt 
import os 
import warnings 

warnings.filterwarnings('ignore') 

# 设置中文字体 
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False 


class BearingFeatureExtractor: 
    def __init__(self, fs=25600): 
        self.fs = fs  # 采样频率 
        # 轴承故障特征频率参数（需要根据实际轴承参数调整） 
        self.BPFO = 32000  # 外圈故障频率 
        self.BPFI = 32000  # 内圈故障频率 
        self.BSF = 32000  # 滚动体故障频率 
        self.FTF = 10  # 保持架故障频率 

    def extract_features(self, data): 
        """提取所有特征""" 
        features = [] 

        # 1. 时域特征 
        features.extend(self.time_domain_features(data)) 

        # 2. 频域特征 
        features.extend(self.frequency_domain_features(data)) 

        # 3. 包络谱特征 
        features.extend(self.envelope_spectrum_features(data)) 

        # 4. 小波包能量熵 
        features.extend(self.wavelet_packet_energy_entropy(data)) 

        return np.array(features) 

    def time_domain_features(self, data): 
        """时域特征""" 
        mean_val = np.mean(data) 
        rms_val = np.sqrt(np.mean(data ** 2)) 
        kurtosis_val = np.mean((data - mean_val) ** 4) / (np.std(data) ** 4) 

        # 峰值因子 
        peak_factor = np.max(np.abs(data)) / rms_val 

        # 脉冲因子 
        impulse_factor = np.max(np.abs(data)) / np.mean(np.abs(data)) 

        return [mean_val, rms_val, kurtosis_val, peak_factor, impulse_factor] 

    def frequency_domain_features(self, data): 
        """频域特征""" 
        n = len(data) 
        fft_vals = fft(data) 
        freqs = fftfreq(n, 1 / self.fs) 


        # 取正频率部分 
        positive_freq_idx = freqs > 0 
        freqs = freqs[positive_freq_idx] 
        fft_vals = np.abs(fft_vals[positive_freq_idx]) 

        # 频谱质心 
        spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals) 

        # 均方频率 
        mean_square_frequency = np.sum((freqs ** 2) * fft_vals) / np.sum(fft_vals) 

        return [spectral_centroid, mean_square_frequency] 

    def envelope_spectrum_features(self, data): 
        """包络谱特征""" 
        # Hilbert包络分析 
        analytic_signal = hilbert(data) 
        envelope = np.abs(analytic_signal) 

        # 包络谱 
        n = len(envelope) 
        envelope_fft = fft(envelope - np.mean(envelope)) 
        freqs = fftfreq(n, 1 / self.fs) 

        positive_freq_idx = (freqs > 0) & (freqs < self.fs / 2) 
        freqs = freqs[positive_freq_idx] 
        envelope_fft = np.abs(envelope_fft[positive_freq_idx]) 

        # 在故障频率附近的能量 
        def get_band_energy(center_freq, bandwidth=10): 
            idx = (freqs >= center_freq - bandwidth / 2) & (freqs <= center_freq + bandwidth / 2) 
            return np.sum(envelope_fft[idx] ** 2) if np.any(idx) else 0 

        bpo_energy = get_band_energy(self.BPFO) 
        bpi_energy = get_band_energy(self.BPFI) 
        bsf_energy = get_band_energy(self.BSF) 

        # Hilbert包络在BPFO处的幅值 
        bpo_idx = np.argmin(np.abs(freqs - self.BPFO)) 
        bpo_amplitude = envelope_fft[bpo_idx] if bpo_idx < len(envelope_fft) else 0 

        return [bpo_energy, bpi_energy, bsf_energy, bpo_amplitude] 

    def wavelet_packet_energy_entropy(self, data, wavelet='db4', level=3): 
        """小波包能量熵""" 
        # 小波包分解 
        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=level) 

        # 获取所有节点能量 
        energy_nodes = [] 
        for node in wp.get_level(level, order='freq'): 
            coeff = node.data 
            energy = np.sum(coeff ** 2) 
            energy_nodes.append(energy) 

        energy_nodes = np.array(energy_nodes) 
        total_energy = np.sum(energy_nodes) 

        # 计算能量熵 
        if total_energy > 0: 
            energy_ratio = energy_nodes / total_energy 
            energy_ratio = energy_ratio[energy_ratio > 0]  # 避免log(0) 
            energy_entropy = -np.sum(energy_ratio * np.log(energy_ratio)) 
        else: 
            energy_entropy = 0 

        return [energy_entropy] 

    def plot_time_domain(self, data, title, ax): 
        """绘制时域波形""" 
        t = np.arange(len(data)) / self.fs 
        ax.plot(t, data) 

        ax.set_xlabel('时间 (s)') 
        ax.set_ylabel('幅值') 
        ax.set_title(f'{title} - 时域波形') 
        ax.grid(True) 

    def plot_frequency_domain(self, data, title, ax): 
        """绘制频谱""" 
        n = len(data) 
        fft_vals = fft(data) 
        freqs = fftfreq(n, 1 / self.fs) 

        positive_freq_idx = freqs > 0 
        freqs = freqs[positive_freq_idx] 
        fft_vals = np.abs(fft_vals[positive_freq_idx]) 

        ax.semilogy(freqs, fft_vals) 
        ax.set_xlabel('频率 (Hz)') 
        ax.set_ylabel('幅值') 
        ax.set_title(f'{title} - 频谱') 
        ax.grid(True) 

        # 标记故障频率 
        ax.axvline(self.BPFO, color='r', linestyle='--', alpha=0.7, label=f'BPFO ({self.BPFO}Hz)') 
        ax.axvline(self.BPFI, color='g', linestyle='--', alpha=0.7, label=f'BPFI ({self.BPFI}Hz)') 
        ax.axvline(self.BSF, color='b', linestyle='--', alpha=0.7, label=f'BSF ({self.BSF}Hz)') 
        ax.legend() 

    def plot_wavelet_energy(self, data, title, ax): 
        """绘制小波能量分布""" 
        wavelet = 'db4' 
        level = 4 

        # 小波包分解 
        wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=level) 

        # 获取最后一层节点能量 
        energy_nodes = [] 
        node_labels = [] 
        for node in wp.get_level(level, order='freq'): 
            coeff = node.data 
            energy = np.sum(coeff ** 2) 
            energy_nodes.append(energy) 
            node_labels.append(node.path) 

        # 绘制能量分布 
        ax.bar(range(len(energy_nodes)), energy_nodes) 
        ax.set_xlabel('小波包节点') 
        ax.set_ylabel('能量') 
        ax.set_title(f'{title} - 小波包能量分布') 
        ax.set_xticks(range(len(energy_nodes))) 
        ax.set_xticklabels(node_labels, rotation=45) 
        ax.grid(True, alpha=0.3) 


def process_dataset(metadata_file, feature_extractor, scaler=None, save_path=None): 
    """处理数据集并提取特征""" 
    # 读取metadata 
    metadata = pd.read_csv(metadata_file) 

    all_features = [] 
    file_info = [] 

    print(f"处理 {metadata_file}...") 

    for idx, row in metadata.iterrows(): 
        file_path = row['file_path']  # 假设metadata中有file_path列 
        fault_type = row.get('fault_type', 'unknown')  # 假设有fault_type列 

        try: 
            # 读取数据文件（假设是CSV格式，单列数据） 
            data = pd.read_csv(file_path).iloc[:, 0].values 


            # 提取特征 
            features = feature_extractor.extract_features(data) 
            all_features.append(features) 
            file_info.append({'file_path': file_path, 'fault_type': fault_type}) 

            # 每处理10个文件打印进度 
            if (idx + 1) % 10 == 0: 
                print(f"已处理 {idx + 1}/{len(metadata)} 个文件") 

        except Exception as e: 
            print(f"处理文件 {file_path} 时出错: {e}") 
            continue 

    # 转换为DataFrame 
    feature_names = [ 
        'mean', 'rms', 'kurtosis', 'peak_factor', 'impulse_factor', 
        'spectral_centroid', 'mean_square_frequency', 
        'bpo_energy', 'bpi_energy', 'bsf_energy', 'bpo_amplitude', 
        'wavelet_energy_entropy' 
    ] 

    features_df = pd.DataFrame(all_features, columns=feature_names) 
    info_df = pd.DataFrame(file_info) 

    # 标准化 
    if scaler is None: 
        scaler = StandardScaler() 
        features_scaled = scaler.fit_transform(features_df) 
    else: 
        features_scaled = scaler.transform(features_df) 

    # 合并信息和特征 
    result_df = pd.concat([info_df, pd.DataFrame(features_scaled, columns=feature_names)], axis=1) 

    # 保存结果 
    if save_path: 
        result_df.to_csv(save_path, index=False) 
        print(f"特征已保存到: {save_path}") 

    return result_df, scaler 


def visualize_features(train_features, test_features, target_features, output_dir): 
    """特征可视化""" 

    # 1. 特征分布对比 
    feature_cols = [col for col in train_features.columns if col not in ['file_path', 'fault_type']] 

    fig, axes = plt.subplots(3, 4, figsize=(20, 15)) 
    axes = axes.ravel() 

    for i, feature in enumerate(feature_cols[:12]):  # 显示前12个特征 
        # 训练集特征分布 
        for fault_type in train_features['fault_type'].unique(): 
            data = train_features[train_features['fault_type'] == fault_type][feature] 
            axes[i].hist(data, alpha=0.6, label=f'{fault_type}', bins=20) 

        axes[i].set_title(f'{feature} 分布') 
        axes[i].set_xlabel(feature) 
        axes[i].set_ylabel('频数') 
        if i == 0: 
            axes[i].legend() 

    plt.tight_layout() 
    plt.savefig(f'{output_dir}/feature_distribution.png', dpi=300, bbox_inches='tight') 
    plt.show() 

    # 2. 相关性热力图 
    plt.figure(figsize=(12, 10)) 
    correlation_matrix = train_features[feature_cols].corr() 
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0) 
    plt.title('特征相关性热力图') 
    plt.tight_layout() 

    plt.savefig(f'{output_dir}/feature_correlation.png', dpi=300, bbox_inches='tight') 
    plt.show() 

    # 3. 故障类型对比 
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)) 

    # 选择几个关键特征进行对比 
    key_features = ['rms', 'kurtosis', 'bpo_energy', 'wavelet_energy_entropy'] 

    for i, feature in enumerate(key_features): 
        ax = axes[i // 2, i % 2] 
        fault_data = [] 
        fault_labels = [] 

        for fault_type in train_features['fault_type'].unique(): 
            data = train_features[train_features['fault_type'] == fault_type][feature] 
            fault_data.append(data) 
            fault_labels.append(fault_type) 

        ax.boxplot(fault_data, labels=fault_labels) 
        ax.set_title(f'{feature} - 不同故障类型对比') 
        ax.set_ylabel(feature) 
        ax.tick_params(axis='x', rotation=45) 

    plt.tight_layout() 
    plt.savefig(f'{output_dir}/fault_type_comparison.png', dpi=300, bbox_inches='tight') 
    plt.show() 


def plot_signal_analysis(sample_data, feature_extractor, fault_type, output_dir): 
    """绘制信号分析图""" 
    fig, axes = plt.subplots(2, 2, figsize=(15, 10)) 

    # 时域波形 
    feature_extractor.plot_time_domain(sample_data, fault_type, axes[0, 0]) 

    # 频谱 
    feature_extractor.plot_frequency_domain(sample_data, fault_type, axes[0, 1]) 

    # 小波能量分布 
    feature_extractor.plot_wavelet_energy(sample_data, fault_type, axes[1, 0]) 

    # 包络谱 
    analytic_signal = hilbert(sample_data) 
    envelope = np.abs(analytic_signal) 
    n = len(envelope) 
    envelope_fft = fft(envelope - np.mean(envelope)) 
    freqs = fftfreq(n, 1 / feature_extractor.fs) 
    positive_freq_idx = (freqs > 0) & (freqs < feature_extractor.fs / 2) 

    axes[1, 1].plot(freqs[positive_freq_idx], np.abs(envelope_fft[positive_freq_idx])) 
    axes[1, 1].set_xlabel('频率 (Hz)') 
    axes[1, 1].set_ylabel('幅值') 
    axes[1, 1].set_title(f'{fault_type} - 包络谱') 
    axes[1, 1].grid(True) 

    # 标记故障频率 
    axes[1, 1].axvline(feature_extractor.BPFO, color='r', linestyle='--', alpha=0.7, label=f'BPFO') 
    axes[1, 1].axvline(feature_extractor.BPFI, color='g', linestyle='--', alpha=0.7, label=f'BPFI') 
    axes[1, 1].axvline(feature_extractor.BSF, color='b', linestyle='--', alpha=0.7, label=f'BSF') 
    axes[1, 1].legend() 

    plt.tight_layout() 
    plt.savefig(f'{output_dir}/signal_analysis_{fault_type}.png', dpi=300, bbox_inches='tight') 
    plt.show() 


def main(): 
    # 设置路径 
    base_dir = '/Users/slh/Downloads/planB' 
    train_metadata = f'{base_dir}/train_metadata.csv' 
    test_metadata = f'{base_dir}/test_metadata.csv' 
    target_metadata = f'{base_dir}/target_metadata.csv' 


    # 创建特征提取器 
    feature_extractor = BearingFeatureExtractor(fs=25600) 

    # 处理训练集 
    print("=== 处理训练集 ===") 
    train_features, scaler = process_dataset( 
        train_metadata, 
        feature_extractor, 
        scaler=None, 
        save_path=f'{base_dir}/train_features.csv' 
    ) 

    # 保存标准化参数 
    scaler_params = { 
        'mean': scaler.mean_, 
        'scale': scaler.scale_ 
    } 
    np.savez(f'{base_dir}/scaler_params.npz', **scaler_params) 
    print(f"标准化参数已保存到: {base_dir}/scaler_params.npz") 

    # 处理测试集 
    print("\n=== 处理测试集 ===") 
    test_features, _ = process_dataset( 
        test_metadata, 
        feature_extractor, 
        scaler=scaler, 
        save_path=f'{base_dir}/test_features.csv' 
    ) 

    # 处理目标域数据 
    print("\n=== 处理目标域数据 ===") 
    target_features, _ = process_dataset( 
        target_metadata, 
        feature_extractor, 
        scaler=scaler, 
        save_path=f'{base_dir}/target_features.csv' 
    ) 

    # 特征可视化 
    print("\n=== 进行特征可视化 ===") 
    visualize_features(train_features, test_features, target_features, base_dir) 

    # 样本信号分析（选取每个故障类型的一个样本进行详细分析） 
    print("\n=== 进行样本信号分析 ===") 
    metadata = pd.read_csv(train_metadata) 
    for fault_type in metadata['fault_type'].unique(): 
        sample_file = metadata[metadata['fault_type'] == fault_type].iloc[0]['file_path'] 
        try: 
            sample_data = pd.read_csv(sample_file).iloc[:, 0].values 
            plot_signal_analysis(sample_data[:min(8192, len(sample_data))], 
                                 feature_extractor, fault_type, base_dir) 
        except Exception as e: 
            print(f"分析样本 {sample_file} 时出错: {e}") 

    print("\n=== 处理完成 ===") 
    print(f"训练集特征形状: {train_features.shape}") 
    print(f"测试集特征形状: {test_features.shape}") 
    print(f"目标域特征形状: {target_features.shape}") 

    # 显示特征统计信息 
    feature_cols = [col for col in train_features.columns if col not in ['file_path', 'fault_type']] 
    print(f"\n特征列表: {feature_cols}") 
    print(f"\n训练集特征统计:") 
    print(train_features[feature_cols].describe()) 


if __name__ == "__main__": 
    main()