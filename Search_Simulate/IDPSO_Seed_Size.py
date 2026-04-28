import numpy as np
import random
from EIMA_Fitness import EIMA, cal_infect_matrix
import os
from Dataloader import dataloader
from tqdm import tqdm
import pandas as pd



import matplotlib
matplotlib.use('TkAgg')


matplotlib.rcParams['font.family'] = 'Times New Roman'
matplotlib.rcParams['font.size'] = 12



class DPSO():

    # ----------------------DPSO参数设置-------------------------------
    def __init__(self, w, c1, c2, p, max_iter, df_hyper_matrix, edge_weights, k, beta, knum, N, infect_matrix):
        self.w = w  #惯性权重
        self.c1 = c1  #个体学习因子
        self.c2 = c2  #群体学习因子
        self.p = p  #粒子种群的数量
        self.max_iter = max_iter  #迭代次数
        self.X = np.zeros((self.p, k), dtype=int)  #所有粒子的位置
        self.V = np.zeros((self.p, k))  #所有粒子的速度
        self.pbest = np.zeros((self.p, k), dtype=int)  #个体历史最优位置
        self.p_fit = np.zeros(self.p)  #个体历史最优适应值
        self.gbest = np.zeros((1, k), dtype=int)   #全局最优位置
        self.fit = 1e5  #全局最优适应值
        self.gpf = 1e5  #全局最优个体对应的实际传播范围

        self.df_hyper_matrix = df_hyper_matrix
        self.edge_weights = edge_weights
        self.k = k #种子集的规模
        self.beta = beta #节点之间的感染概率
        self.knum = knum #EIMA评估的跳数
        self.N = N #目标传播规模
        self.infect_matrix = infect_matrix


    # ---------------------评估函数------------------------------------
    def function(self, XX):
        row_list = XX.tolist()
        fits = EIMA(self.df_hyper_matrix, self.edge_weights, row_list, self.beta, self.knum, self.infect_matrix)
        f = abs(fits.sum() - self.N)
        return f, fits.sum()


    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        num_nodes = self.df_hyper_matrix.shape[0]
        available_nodes = list(range(num_nodes))
        random.shuffle(available_nodes)

        for i in range(self.p):
            if len(available_nodes) < self.k:
                available_nodes = list(range(num_nodes))
                random.shuffle(available_nodes)

            self.X[i, :] = available_nodes[:self.k]
            available_nodes = available_nodes[self.k:]

            self.V[i, :] = np.zeros(self.k)
            self.pbest[i, :] = self.X[i, :].copy()
            tmp, pf = self.function(self.X[i, :])
            self.p_fit[i] = tmp

            if tmp < self.fit:
                self.gbest = self.X[i, :].copy()
                self.fit = tmp
                self.gpf = pf



    # 多点位替换
    def replace(self, XX, Vec):

        num_nodes = self.df_hyper_matrix.shape[0]
        XX_new = XX.copy()

        replace_indices = np.where(Vec == 1)[0]
        k_replace = len(replace_indices)

        f_old, _ = self.function(XX)

        def evaluate_batch(candidate_batch):
            XX_new[replace_indices] = candidate_batch
            return self.function(XX_new)[0], XX_new

        max_attempts = 50
        candidate_pool = np.setdiff1d(np.arange(num_nodes), XX)
        if len(candidate_pool) < k_replace:
            raise ValueError(f"候选池节点总数不足生成一个候选解，需要至少 {k_replace} 个候选节点，当前只有 {len(candidate_pool)} 个")

        remaining_indices = np.arange(len(candidate_pool))
        np.random.shuffle(remaining_indices)
        generated = 0

        while generated < max_attempts:
            if len(remaining_indices) < k_replace:
                remaining_indices = np.arange(len(candidate_pool))
                np.random.shuffle(remaining_indices)

            selected_indices = remaining_indices[:k_replace]
            remaining_indices = remaining_indices[k_replace:]

            f_new, XX_new = evaluate_batch(candidate_pool[selected_indices])
            if f_new < f_old:
                return XX_new
            generated += 1

        return XX_new


    # 定义Sigmoid函数
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))


    # ----------------------更新粒子位置--------------------------------
    def iterator(self):
        fitness = []
        gf = []
        gbest_history = []


        for t in tqdm(range(self.max_iter), desc='iter'):
            for i in range(self.p):
                temp, pf = self.function(self.X[i,:])
                if temp < self.p_fit[i]:
                    self.p_fit[i] = temp
                    self.pbest[i,:] = self.X[i,:].copy()
                    if self.p_fit[i] < self.fit:
                        self.gbest = self.X[i,:].copy()
                        self.fit = self.p_fit[i]
                        self.gpf = pf

            fitness.append(self.fit)
            gf.append(self.gpf)
            gbest_history.append(self.gbest.copy())

            for i in range(self.p):
                Vpbest = np.zeros_like(self.X[i, :], dtype=int)
                for j in range(len(self.X[i, :])):
                    if self.X[i, j] not in self.pbest[i, :]:
                        Vpbest[j] = 1
                    else:
                        Vpbest[j] = -1

                Vgbest = np.zeros_like(self.X[i, :], dtype=int)
                for j in range(len(self.X[i, :])):
                    if self.X[i, j] not in self.gbest:
                        Vgbest[j] = 1
                    else:
                        Vgbest[j] = -1

                r0 = np.random.rand(self.k)
                r1 = np.random.rand(self.k)
                r2 = np.random.rand(self.k)
                self.V[i,:] = self.w * self.V[i,:] + self.c1 * r1 * Vpbest + self.c2 * r2 * Vgbest

                prob_replace = self.sigmoid(self.V[i, :])
                Vec = (prob_replace >= r0).astype(int)

                if np.any(Vec == 1):
                    self.V[i, :][Vec == 1] = 0
                    self.X[i, :] = self.replace(self.X[i, :], Vec)


            print(f"\n最优个体(节点下标): {self.gbest}, 最优节点集: {self.gbest+1}, 最优传播范围: {self.gpf}, 最优适应度值: {self.fit}\n")

        return fitness, gf, gbest_history








# ----------------------程序执行-----------------------
if __name__ == '__main__':
    # 0、参数设定
    N0 = [50, 20, 15, 150, 25, 30, 50, 40]
    k0 = [3, 6, 9, 12, 15, 18]
    knum0 = 10
    R0 = 5
    pnum = 20
    iter = 101

    # 1、获取路径
    paths = os.listdir('./Datasets/')

    # 2、读取这些文件
    for i, path in enumerate(paths):
        if i % 2 == 0:
            print("\n--------------------------正在搜索第{}个数据集--------------------------".format(int(i / 2 + 1)))

            # 2.1
            path_data = './Datasets/' + path
            path_weight = './Datasets/' + path[:-4] + '_weights.txt'

            # 2.2
            dl = dataloader(path_data, path_weight)
            dl.dataload()
            matrix = dl.hyper_matrix
            weight = dl.edge_weights

            # 2.3
            for j, kk in enumerate(k0):
                print("\n------------------------正在搜索种子集的规模为{}------------------------".format(int(kk)))

                all_results_df = pd.DataFrame()

                # 2.4
                for r0 in tqdm( (range(R0)), desc='重复实验' ):
                    my_pso = DPSO(w=0.8, c1=2, c2=2, p=pnum, max_iter=iter, df_hyper_matrix=matrix, edge_weights=weight, k=kk, beta=0.01, knum=knum0, N=N0[int(i/2)], infect_matrix=cal_infect_matrix(matrix, weight, beta=0.01))
                    my_pso.init_Population()
                    fitness, gf, gbest_history = my_pso.iterator()

                    results = []
                    for gen_idx in range(len(fitness)):
                        results.append({
                            '重复实验': r0,
                            'generation': gen_idx,
                            'fitness': fitness[gen_idx],
                            'actual_spread': gf[gen_idx],
                            'node_set': str(gbest_history[gen_idx])
                        })
                    df = pd.DataFrame(results)

                    if all_results_df.empty:
                        all_results_df = df
                    else:
                        all_results_df = pd.concat([all_results_df, df], ignore_index=True)

                # 2.5
                dataset_name = path[:-4]
                excel_dir = f"./设定多个种子集的实验结果/{dataset_name}/"
                excel_path = f"{excel_dir}目标规模N={N0[int(i/2)]}_种子集k={kk}_.xlsx"

                os.makedirs(excel_dir, exist_ok=True)

                with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                    for r0 in range(R0):
                        data = all_results_df[all_results_df['重复实验'] == r0]
                        data.to_excel(writer, sheet_name=f'第{r0}次实验', index=False)
