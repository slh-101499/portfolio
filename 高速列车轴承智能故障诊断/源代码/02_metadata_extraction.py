import os 
import pandas as pd 
import re 


def extract_metadata_from_filename(filename, file_path, domain): 
    """从文件名中提取元信息""" 
    metadata = { 
        'domain': domain, 
        'filename': filename, 
        'file_path': file_path, 
        'sensor_channel': 'Unknown', 
        'sampling_rate': 'Unknown', 
        'fault_class': 'Unknown', 
        'load': 'Unknown', 
        'fault_size': 'Unknown' 
    } 

    # 处理目标域数据（简单的.mat文件） 
    if domain == 'target': 
        if filename.endswith('.mat'): 
            # 目标域数据只有基本文件名信息 
            metadata['filename'] = filename 
            # 其他字段保持为'Unknown' 
        return metadata 

    # 处理源域数据 
    try: 
        # 从文件路径中提取传感器通道和采样率 
        path_parts = file_path.split('/') 

        # 查找包含采样率和传感器通道信息的目录名 
        for part in path_parts: 
            if 'kHz' in part and ('DE' in part or 'FE' in part or 'BA' in part): 
                # 提取采样率（前5个字符，如"12kHz"） 
                if part.startswith('12kHz'): 
                    metadata['sampling_rate'] = '12kHz' 
                elif part.startswith('48kHz'): 
                    metadata['sampling_rate'] = '48kHz' 

                # 提取传感器通道 
                if '_DE_' in part: 
                    metadata['sensor_channel'] = 'DE' 
                elif '_FE_' in part: 
                    metadata['sensor_channel'] = 'FE' 
                elif '_BA_' in part: 
                    metadata['sensor_channel'] = 'BA' 
                break 

        # 从文件名中提取故障类别、载荷和故障尺寸 
        if filename.endswith('.mat'): 
            # 提取故障类别（文件名开头的字母组合） 
            fault_match = re.match(r'^([A-Z]+)', filename) 
            if fault_match: 
                metadata['fault_class'] = fault_match.group(1) 

            # 提取故障尺寸（文件名中的数字，通常是3位） 
            size_match = re.search(r'(\d{3,4})', filename) 
            if size_match: 
                size_str = size_match.group(1) 
                # 取后3位作为故障尺寸 
                if len(size_str) >= 3: 
                    metadata['fault_size'] = size_str[-3:] 

            # 提取载荷（文件名中的单个数字） 
            load_match = re.search(r'_(\d)[_.]', filename) 
            if load_match: 
                metadata['load'] = load_match.group(1) 
            else: 
                # 尝试其他模式，如IR028_2_(1750rpm).mat 
                load_match2 = re.search(r'_(\d)_\(', filename) 
                if load_match2: 
                    metadata['load'] = load_match2.group(1) 


        # 从目录路径中补充故障尺寸信息 
        for part in path_parts: 
            if part.isdigit() and len(part) == 4:  # 0007这样的目录 
                metadata['fault_size'] = part[-3:]  # 取后三位 
                break 

    except Exception as e: 
        print(f"处理文件 {filename} 时出错: {e}") 

    return metadata 


def collect_metadata(source_dir, target_dir): 
    """收集源域和目标域的元数据""" 
    all_metadata = [] 

    # 处理源域数据 
    print("正在处理源域数据...") 
    if os.path.exists(source_dir): 
        for root, dirs, files in os.walk(source_dir): 
            for file in files: 
                if file.endswith('.mat'): 
                    file_path = os.path.join(root, file) 
                    metadata = extract_metadata_from_filename(file, file_path, 'source') 
                    all_metadata.append(metadata) 
                    print(f"处理源域文件: {file}") 
    else: 
        print(f"警告: 源域目录不存在: {source_dir}") 

    # 处理目标域数据 
    print("正在处理目标域数据...") 
    if os.path.exists(target_dir): 
        for root, dirs, files in os.walk(target_dir): 
            for file in files: 
                if file.endswith('.mat'): 
                    file_path = os.path.join(root, file) 
                    metadata = extract_metadata_from_filename(file, file_path, 'target') 
                    all_metadata.append(metadata) 
                    print(f"处理目标域文件: {file}") 
    else: 
        print(f"警告: 目标域目录不存在: {target_dir}") 

    return all_metadata 


def main(): 
    # 设置路径 
    source_dir = "/Users/slh/Downloads/planB/data/source_data" 
    target_dir = "/Users/slh/Downloads/planB/data/target_data" 
    output_file = "/Users/slh/Downloads/planB/metadata.csv" 

    # 确保输出目录存在 
    os.makedirs(os.path.dirname(output_file), exist_ok=True) 

    # 收集元数据 
    print("开始收集元数据...") 
    metadata_list = collect_metadata(source_dir, target_dir) 

    if metadata_list: 
        # 创建DataFrame 
        df = pd.DataFrame(metadata_list) 

        # 保存为CSV文件（使用UTF-8编码避免中文乱码） 
        df.to_csv(output_file, index=False, encoding='utf-8') 
        print(f"元数据已保存到: {output_file}") 
        print(f"总共处理了 {len(metadata_list)} 个文件") 

        # 显示数据预览 
        print("\n数据预览:") 
        print(df.head()) 
        print(f"\n数据形状: {df.shape}") 

    else: 
        print("未找到任何.mat文件") 


if __name__ == "__main__": 
    main()