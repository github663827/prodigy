import os
import shutil

# 指定目录路径
directory_path = '/THL5/home/shyunie/xue_code/prodigy_artifacts/prodigy_ae_output/results'

# 获取目录下所有文件的名字
file_names = os.listdir(directory_path)

# 提取文件名中的前缀部分
prefixes = [os.path.splitext(file_name)[0] for file_name in file_names]

# 保存结果到指定文件
output_file_path = '/THL5/home/shyunie/xue_code/prodigy_artifacts/ai4hpc_deployment/src/prefixes.txt'

with open(output_file_path, 'w') as file:
    for prefix in prefixes:
        file.write(prefix + '\n')

print(f"Prefixes saved to {output_file_path}")



# 读取prefixes.txt文件中的前缀列表
prefixes_file_path = '/THL5/home/shyunie/xue_code/prodigy_artifacts/ai4hpc_deployment/src/prefixes.txt'

with open(prefixes_file_path, 'r') as file:
    prefixes = [line.strip() for line in file]

# 指定包含文件夹的目录路径
directory_path = '/THL5/home/shyunie/xue_code/prodigy_artifacts/ai4hpc_deployment/src/eclipse_small_prod_dataset_1'

# 删除对应的文件夹
for prefix in prefixes:
    folder_path = os.path.join(directory_path, prefix)
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path)
        print(f"Deleted folder: {folder_path}")
    else:
        print(f"Folder not found: {folder_path}")
