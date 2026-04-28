import os
import pandas as pd
from Dataloader import dataloader
from Simulation_Experiments import  simulation_experiments




# 0、参数设定
N0 = [50, 20, 15, 150, 25, 30, 50, 40]
k0 = [3, 6, 9, 12, 15, 18]
R = 100
beta = 0.01
t = 10


# 1、获取路径
paths = os.listdir('./Datasets/')

# 2、依次读取
for i, path in enumerate(paths):
    if i % 2 == 0:
        print("\n--------------------------正在读取第{}个数据集--------------------------".format(int(i / 2 + 1)))

        # 2.1
        dataset_name = path[:-4]
        excel_path1 = f"./设定多个种子集的实验结果/{dataset_name}/目标规模N={N0[int(i/2)]}_种子集k={k0[0]}_.xlsx"
        excel_path2 = f"./设定多个种子集的实验结果/{dataset_name}/目标规模N={N0[int(i/2)]}_种子集k={k0[1]}_.xlsx"
        excel_path3 = f"./设定多个种子集的实验结果/{dataset_name}/目标规模N={N0[int(i/2)]}_种子集k={k0[2]}_.xlsx"
        excel_path4 = f"./设定多个种子集的实验结果/{dataset_name}/目标规模N={N0[int(i/2)]}_种子集k={k0[3]}_.xlsx"
        excel_path5 = f"./设定多个种子集的实验结果/{dataset_name}/目标规模N={N0[int(i/2)]}_种子集k={k0[4]}_.xlsx"
        excel_path6 = f"./设定多个种子集的实验结果/{dataset_name}/目标规模N={N0[int(i/2)]}_种子集k={k0[5]}_.xlsx"

        # 2.2
        path_data = './Datasets/' + path
        path_weight = './Datasets/' + path[:-4] + '_weights.txt'
        dl = dataloader(path_data, path_weight)
        dl.dataload()
        df_hyper_matrix = dl.hyper_matrix
        edge_weights = dl.edge_weights

        # 2.3
        all_sheets1 = pd.read_excel(excel_path1, sheet_name=None)
        all_sheets2 = pd.read_excel(excel_path2, sheet_name=None)
        all_sheets3 = pd.read_excel(excel_path3, sheet_name=None)
        all_sheets4 = pd.read_excel(excel_path4, sheet_name=None)
        all_sheets5 = pd.read_excel(excel_path5, sheet_name=None)
        all_sheets6 = pd.read_excel(excel_path6, sheet_name=None)

        # 2.4
        final_fitness_values1 = []
        final_fitness_values2 = []
        final_fitness_values3 = []
        final_fitness_values4 = []
        final_fitness_values5 = []
        final_fitness_values6 = []
        experiment_data1 = {}
        experiment_data2 = {}
        experiment_data3 = {}
        experiment_data4 = {}
        experiment_data5 = {}
        experiment_data6 = {}

        # 2.5
        for sheet_name, data in all_sheets1.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values1.append((exp_num, final_fitness))
                experiment_data1[exp_num] = data

        # 2.6
        best_exp_num1, best_fitness1 = min(final_fitness_values1, key=lambda x: x[1])
        best_data1 = experiment_data1[best_exp_num1]
        best_data1_seeds_ = best_data1['node_set'].iloc[-1]
        best_data1_seeds = [int(x) for x in best_data1_seeds_.strip('[]').split()]
        best_data1_seeds = str(best_data1_seeds)

        # 2.6.1
        for sheet_name, data in all_sheets2.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values2.append((exp_num, final_fitness))
                experiment_data2[exp_num] = data
        best_exp_num2, best_fitness2 = min(final_fitness_values2, key=lambda x: x[1])
        best_data2 = experiment_data2[best_exp_num2]
        best_data2_seeds_ = best_data2['node_set'].iloc[-1]
        best_data2_seeds = [int(x) for x in best_data2_seeds_.strip('[]').split()]
        best_data2_seeds = str(best_data2_seeds)

        # 2.6.2
        for sheet_name, data in all_sheets3.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values3.append((exp_num, final_fitness))
                experiment_data3[exp_num] = data
        best_exp_num3, best_fitness3 = min(final_fitness_values3, key=lambda x: x[1])
        best_data3 = experiment_data3[best_exp_num3]
        best_data3_seeds_ = best_data3['node_set'].iloc[-1]
        best_data3_seeds = [int(x) for x in best_data3_seeds_.strip('[]').split()]
        best_data3_seeds = str(best_data3_seeds)

        # 2.6.3
        for sheet_name, data in all_sheets4.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values4.append((exp_num, final_fitness))
                experiment_data4[exp_num] = data
        best_exp_num4, best_fitness4 = min(final_fitness_values4, key=lambda x: x[1])
        best_data4 = experiment_data4[best_exp_num4]
        best_data4_seeds_ = best_data4['node_set'].iloc[-1]
        best_data4_seeds = [int(x) for x in best_data4_seeds_.strip('[]').split()]
        best_data4_seeds = str(best_data4_seeds)

        # 2.6.4
        for sheet_name, data in all_sheets5.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values5.append((exp_num, final_fitness))
                experiment_data5[exp_num] = data
        best_exp_num5, best_fitness5 = min(final_fitness_values5, key=lambda x: x[1])
        best_data5 = experiment_data5[best_exp_num5]
        best_data5_seeds_ = best_data5['node_set'].iloc[-1]
        best_data5_seeds = [int(x) for x in best_data5_seeds_.strip('[]').split()]
        best_data5_seeds = str(best_data5_seeds)

        # 2.6.5
        for sheet_name, data in all_sheets6.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values6.append((exp_num, final_fitness))
                experiment_data6[exp_num] = data
        best_exp_num6, best_fitness6 = min(final_fitness_values6, key=lambda x: x[1])
        best_data6 = experiment_data6[best_exp_num6]
        best_data6_seeds_ = best_data6['node_set'].iloc[-1]
        best_data6_seeds = [int(x) for x in best_data6_seeds_.strip('[]').split()]
        best_data6_seeds = str(best_data6_seeds)


        # 3
        seeds_list_df = pd.DataFrame()

        seeds_list_df.loc[0, f'seed set {k0[0]}'] = best_data1_seeds
        seeds_list_df.loc[0, f'seed set {k0[1]}'] = best_data2_seeds
        seeds_list_df.loc[0, f'seed set {k0[2]}'] = best_data3_seeds
        seeds_list_df.loc[0, f'seed set {k0[3]}'] = best_data4_seeds
        seeds_list_df.loc[0, f'seed set {k0[4]}'] = best_data5_seeds
        seeds_list_df.loc[0, f'seed set {k0[5]}'] = best_data6_seeds

        # 4
        simulation_experiments.conduct_all_information(path, df_hyper_matrix, edge_weights, seeds_list_df, R, t, beta)
