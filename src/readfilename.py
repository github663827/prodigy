import glob

# 根据节点文件的路径模式，获取所有节点文件路径
node_files_pattern = '/THL5/home/shyunie/new_data/zscore/cn*_*/final_metric.csv'
node_files = glob.glob(node_files_pattern)

# 输出文件路径
output_file = '/THL5/home/shyunie/xue_code/prodigy_artifacts/ai4hpc_deployment/src/eclipse_small_prod_dataset/node_files.csv'

# 将获取的路径格式化为所需的格式，并写入到文件中
with open(output_file, 'w') as f:
    formatted_paths = ',\n'.join([f"'{path}'" for path in node_files])
    f.write(formatted_paths)

print(f"节点文件路径已保存到 {output_file}")
