import os
import pandas as pd
from Dataloader import dataloader
from Simulation_Experiments import  simulation_experiments



import matplotlib
matplotlib.use('TkAgg')




N1 = [40, 10, 8, 100, 10, 12, 30, 20]
N2 = [60, 12, 10, 200, 14, 16, 40, 30]
k0 = [5]
R = 100
beta = 0.01
t = 10



# 1
paths = os.listdir('./Datasets/')

# 2
for i, path in enumerate(paths):
    if i in [0]:
        print("\n--------------------------正在读取第{}个数据集--------------------------".format(int(i / 2 + 1)))

        # 2.1
        dataset_name = path[:-4]
        excel_path1 = f"./设定多个目标的实验结果/{dataset_name}/目标规模N={N1[int(i / 2)]}_种子集k={k0[0]}_.xlsx"
        excel_path2 = f"./设定多个目标的实验结果/{dataset_name}/目标规模N={N2[int(i / 2)]}_种子集k={k0[0]}_.xlsx"

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

        final_fitness_values1 = []
        final_fitness_values2 = []
        experiment_data1 = {}
        experiment_data2 = {}

        # 2.4
        for sheet_name, data in all_sheets1.items():
            if sheet_name.startswith('第') and sheet_name.endswith('次实验'):
                exp_num = int(sheet_name[1:-3])
                final_fitness = data['fitness'].iloc[-1]
                final_fitness_values1.append((exp_num, final_fitness))
                experiment_data1[exp_num] = data

        # 2.5
        best_exp_num1, best_fitness1 = min(final_fitness_values1, key=lambda x: x[1])
        best_data1 = experiment_data1[best_exp_num1]
        best_data1_seeds_ = best_data1['node_set'].iloc[-1]
        best_data1_seeds = [int(x) for x in best_data1_seeds_.strip('[]').split()]
        best_data1_seeds = str(best_data1_seeds)

        # 2.6
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

        # 3
        seeds_list_df = pd.DataFrame()

        seeds_list_df.loc[k0[0], 'N1_Config'] = best_data1_seeds
        seeds_list_df.loc[k0[0], 'N2_Config'] = best_data2_seeds

        # 4
        simulation_experiments.conduct_all_information(path, df_hyper_matrix, edge_weights, seeds_list_df, R, t, beta)
