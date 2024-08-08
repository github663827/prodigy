import pandas as pd
import os
from datetime import datetime

# 节点文件路径列表
node_files = [
'/THL5/home/shyunie/new_data/zscore/cn4010_30019cd9cb0971cb498506d20a3d25b5158c4e07/final_metric.csv',
'/THL5/home/shyunie/new_data/zscore/cn4925_cb1a0db1a606c079f9c53282fe17ebf5af1df0c3/final_metric.csv',
'/THL5/home/shyunie/new_data/zscore/cn1081_08b1567692a5b18c8c85142bee7785c8fcd23c83/final_metric.csv',
'/THL5/home/shyunie/new_data/zscore/cn5098_308d0686db5936dab3e71b96032abc668642400f/final_metric.csv',
'/THL5/home/shyunie/new_data/zscore/cn2582_891dd183f976c38b79f01a22fabdd1bd98890c47/final_metric.csv',
'/THL5/home/shyunie/new_data/zscore/cn4461_9dbd1df49842506bd43c70ffd7667d4f53281a0d/final_metric.csv'
]

# label_job_2500文件路径
label_file = '/THL5/home/shyunie/xue_code/prodigy_artifacts/label_job_2500.csv'

# 读取label_job_2500文件
label_df = pd.read_csv(label_file)
label_df['job_start'] = pd.to_datetime(label_df['job_start'])
label_df['job_end'] = pd.to_datetime(label_df['job_end'])
label_df['job_node'] = label_df['job_node'].apply(lambda x: x.strip("[]").strip("'"))

# 目标目录
target_dir = '/THL5/home/shyunie/xue_code/prodigy_artifacts/ai4hpc_deployment/src/eclipse_small_prod_dataset'

for node_file in node_files:
    # 读取CSV文件
    df = pd.read_csv(node_file)

    # 插入空列 'uid', 'job_id', 'component_id'
    df.insert(0, 'uid', '')         
    df.insert(1, 'job_id', -1)  # 设置默认值为 -1     
    df.insert(2, 'component_id', '')

    # 确定分割点
    split_point = 25920

    # 获取前25920行数据(训练集)
    df_first_part = df.iloc[:split_point].copy()

    # 获取剩余的行数据(测试集)
    df_second_part = df.iloc[split_point:].copy()

    # 生成时间戳
    timestamp1 = list(range(1681660800, 1682049600, 15))
    timestamp2 = list(range(1682049600, 1682265615, 15))

    # 添加时间戳
    df_first_part['timestamp'] = timestamp1
    df_second_part['timestamp'] = timestamp2

    # 获取节点名称并提取component_id
    node_name = os.path.basename(os.path.dirname(node_file)).split('_')[0]
    component_id = int(node_name[2:])  # 提取数字部分作为component_id

    # 生成 uid 并设置 component_id
    df_first_part['component_id'] = component_id
    df_first_part['uid'] = range(25920)

    df_second_part['component_id'] = component_id
    df_second_part['uid'] = range(14401)
    df_second_part['job_id'] = df_second_part['uid']  # job_id 和 uid 相同

    # 创建对应的节点目录
    node_dir = os.path.join(target_dir, node_name)
    os.makedirs(node_dir, exist_ok=True)

    # 根据label_df更新训练集的job_id
    for _, row in label_df.iterrows():
        job_node = row['job_node']
        job_start = row['job_start']
        job_end = row['job_end']
        job_id = row.name  # 使用行号作为job_id

        if node_name in job_node:
            mask = (df_first_part['timestamp'] >= job_start.timestamp()) & (df_first_part['timestamp'] <= job_end.timestamp())
            df_first_part.loc[mask, 'job_id'] = job_id

    # # 构建新的CSV文件路径
    # train_csv_file = os.path.join(node_dir, f'{node_name}_train.csv')
    # test_csv_file = os.path.join(node_dir, f'{node_name}_test.csv')

    # # 保存更新后的数据到CSV文件
    # df_first_part.to_csv(train_csv_file, index=False)
    # df_second_part.to_csv(test_csv_file, index=False)

    # 构建新的HDF文件路径
    train_hdf_file = os.path.join(node_dir, f'{node_name}_train.hdf')
    test_hdf_file = os.path.join(node_dir, f'{node_name}_test.hdf')

    # 保存更新后的数据到HDF文件
    df_first_part.to_hdf(train_hdf_file, key='train', mode='w')
    df_second_part.to_hdf(test_hdf_file, key='test', mode='w')

    print(f'Processed {node_name} and saved to {node_dir}')
