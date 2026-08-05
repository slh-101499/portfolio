import os 
import pandas as pd 
import numpy as np 
import scipy.io as sio 
from scipy import signal 
from pathlib import Path 
import shutil 


def load_mat_data(file_path): 
    """加载.mat文件数据""" 

    try: 
        mat_data = sio.loadmat(file_path) 
        # .mat文件通常包含多个变量，我们需要找到包含振动数据的变量 
        # 通常变量名不是 '__header__', '__version__', '__globals__' 
        data_key = None 
        for key in mat_data.keys(): 
            if not key.startswith('__') and not key.startswith('readme'): 
                data_key = key 
                break 

        if data_key is None: 
            print(f"警告: 在文件 {file_path} 中未找到数据变量") 
            return None 

        data = mat_data[data_key].flatten()  # 展平为一维数组 
        return data 
    except Exception as e: 
        print(f"错误: 加载文件 {file_path} 时出错: {e}") 
        return None 


def antialias_lowpass_and_resample(data, original_fs, target_fs=32000): 
    """抗混叠低通滤波并降采样到目标频率""" 
    # 计算降采样比率 
    downsample_ratio = original_fs / target_fs 

    if downsample_ratio < 1: 
        print(f"错误: 原始采样率 {original_fs} 低于目标采样率 {target_fs}") 
        return None 

    # 设计抗混叠低通滤波器 (截止频率为目标奈奎斯特频率) 
    nyquist_ratio = 0.8  # 保守的截止频率比率 
    cutoff_freq = target_fs / 2 * nyquist_ratio 

    # 滤波器参数 
    filter_order = 8 
    b, a = signal.butter(filter_order, cutoff_freq / (original_fs / 2), 'low') 

    # 应用抗混叠滤波器 
    filtered_data = signal.filtfilt(b, a, data) 

    # 计算降采样因子（需要是整数） 
    downsample_factor = int(original_fs / target_fs) 

    # 降采样 
    resampled_data = signal.resample(filtered_data, 
                                     len(data) // downsample_factor) 

    return resampled_data 


def high_quality_interpolation(data, original_fs, target_fs=32000): 
    """高质量插值到目标频率""" 
    # 计算插值比率 
    interpolation_ratio = target_fs / original_fs 

    if interpolation_ratio < 1: 
        print(f"错误: 目标采样率 {target_fs} 低于原始采样率 {original_fs}") 
        return None 

    # 创建原始时间轴 
    original_time = np.arange(len(data)) / original_fs 

    # 创建目标时间轴 
    target_length = int(len(data) * interpolation_ratio) 
    target_time = np.arange(target_length) / target_fs 

    # 使用立方插值进行高质量插值 
    from scipy import interpolate 

    # 创建插值函数 
    interp_func = interpolate.interp1d(original_time, data, 
                                       kind='cubic', 

                                       bounds_error=False, 
                                       fill_value=(data[0], data[-1])) 

    # 执行插值 
    interpolated_data = interp_func(target_time) 

    return interpolated_data 


def process_source_data(metadata_file, output_base_dir): 
    """处理源域数据并进行采样率转换""" 

    # 读取元数据 
    df = pd.read_csv(metadata_file) 

    # 筛选源域数据 
    source_df = df[df['domain'] == 'source'] 

    # 创建输出目录 
    output_dir = os.path.join(output_base_dir, "source_32kHz") 
    os.makedirs(output_dir, exist_ok=True) 

    processed_files = [] 

    for idx, row in source_df.iterrows(): 
        try: 
            file_path = row['file_path'] 
            sampling_rate = row['sampling_rate'] 
            filename = row['filename'] 

            print(f"处理文件: {filename} (采样率: {sampling_rate})") 

            # 加载数据 
            data = load_mat_data(file_path) 
            if data is None: 
                continue 

            # 根据采样率选择处理方法 
            if sampling_rate == '48kHz': 
                original_fs = 48000 
                processed_data = antialias_lowpass_and_resample(data, original_fs, 32000) 

            elif sampling_rate == '12kHz': 
                original_fs = 12000 
                processed_data = high_quality_interpolation(data, original_fs, 32000) 

            else: 
                print(f"未知采样率: {sampling_rate}, 跳过文件: {filename}") 
                continue 

            if processed_data is None: 
                print(f"处理失败: {filename}") 
                continue 

            # 创建输出文件路径 
            # 保持原始文件夹结构但修改采样率信息 
            relative_path = os.path.relpath(file_path, "/Users/slh/Downloads/planB/data/source_data") 

            # 修改文件夹名中的采样率信息 
            path_parts = relative_path.split(os.sep) 
            if len(path_parts) > 0: 
                # 修改第一个文件夹名（包含采样率信息） 
                first_folder = path_parts[0] 
                if '12kHz' in first_folder: 
                    new_first_folder = first_folder.replace('12kHz', '32kHz') 
                elif '48kHz' in first_folder: 
                    new_first_folder = first_folder.replace('48kHz', '32kHz') 
                else: 
                    new_first_folder = '32kHz_' + first_folder 

                path_parts[0] = new_first_folder 

            new_relative_path = os.sep.join(path_parts) 

            output_file_path = os.path.join(output_dir, new_relative_path) 

            # 创建输出目录 
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True) 

            # 保存处理后的数据 
            sio.savemat(output_file_path, {'vibration_data': processed_data}) 

            # 更新元数据信息 
            new_metadata = row.copy() 
            new_metadata['file_path'] = output_file_path 
            new_metadata['sampling_rate'] = '32kHz' 
            new_metadata['original_sampling_rate'] = sampling_rate  # 保留原始采样率信息 

            processed_files.append(new_metadata) 

            print(f"成功处理: {filename} -> {output_file_path}") 

        except Exception as e: 
            print(f"处理文件 {row['filename']} 时出错: {e}") 
            continue 

    return processed_files 


def update_metadata(original_metadata_file, processed_files, output_dir): 
    """更新元数据文件""" 

    # 读取原始元数据 
    original_df = pd.read_csv(original_metadata_file) 

    # 创建处理后的数据DataFrame 
    processed_df = pd.DataFrame(processed_files) 

    # 从原始数据中移除已处理的源域文件 
    original_source_mask = original_df['domain'] == 'source' 
    original_df = original_df[~original_source_mask] 

    # 合并处理后的源域数据和其他数据 
    updated_df = pd.concat([original_df, processed_df], ignore_index=True) 

    # 保存更新后的元数据 
    updated_metadata_file = os.path.join(output_dir, "metadata_updated.csv") 
    updated_df.to_csv(updated_metadata_file, index=False) 

    # 同时覆盖原始文件（可选） 
    updated_df.to_csv(original_metadata_file, index=False) 

    print(f"元数据已更新并保存到: {updated_metadata_file}") 
    return updated_metadata_file 


def main(): 
    # 设置路径 
    metadata_file = "/Users/slh/Downloads/planB/metadata.csv" 
    output_base_dir = "/Users/slh/Downloads/planB" 

    # 检查元数据文件是否存在 
    if not os.path.exists(metadata_file): 
        print(f"元数据文件不存在: {metadata_file}") 
        return 

    # 处理源域数据 
    print("开始处理源域数据...") 
    processed_files = process_source_data(metadata_file, output_base_dir) 

    print(f"成功处理 {len(processed_files)} 个文件") 

    # 更新元数据 
    print("更新元数据...") 
    updated_metadata_file = update_metadata(metadata_file, processed_files, output_base_dir) 


    # 显示统计信息 
    print("\n处理完成!") 
    print(f"处理后的数据保存在: {os.path.join(output_base_dir, 'source_32kHz')}") 
    print(f"更新后的元数据保存在: {updated_metadata_file}") 

    # 显示处理统计 
    sampling_rates = {} 
    for file in processed_files: 
        original_rate = file['original_sampling_rate'] 
        if original_rate in sampling_rates: 
            sampling_rates[original_rate] += 1 
        else: 
            sampling_rates[original_rate] = 1 

    print("\n处理统计:") 
    for rate, count in sampling_rates.items(): 
        print(f"{rate} -> 32kHz: {count} 个文件") 


if __name__ == "__main__": 
    main()