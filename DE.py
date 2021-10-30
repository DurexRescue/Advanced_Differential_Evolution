import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



#'''
def fit_fun(X):  # 适应函数
    return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))
def fit_fun1(X, X1):  # 适应函数
    return -np.abs(np.sin(X) * np.cos(X1) * np.exp(np.abs(1 - np.sqrt(X ** 2 + X1 ** 2) / np.pi)))
#'''

'''
def fit_fun(X):  # 适应函数
    return X[0]*X[0]+X[1]*X[1]
def fit_fun1(X,X1):  # 适应函数
    return X**2+X1**2
#'''


class Unit:
    # 初始化
    def __init__(self, x_min, x_max, dim):
        self.__pos = np.array([x_min + random.random()*(x_max - x_min) for i in range(dim)])
        self.__mutation = np.array([0.0 for i in range(dim)])  # 个体突变后的向量
        self.__crossover = np.array([0.0 for i in range(dim)])  # 个体交叉后的向量
        self.__fitnessValue = fit_fun(self.__pos)  # 个体适应度

    def set_pos(self, i, value):
        self.__pos[i] = value

    def get_pos(self):
        return self.__pos

    def set_mutation(self, i, value):
        self.__mutation[i] = value

    def get_mutation(self):
        return self.__mutation

    def set_crossover(self, i, value):
        self.__crossover[i] = value

    def get_crossover(self):
        return self.__crossover

    def set_fitness_value(self, value):
        self.__fitnessValue = value

    def get_fitness_value(self):
        return self.__fitnessValue


class DE:
    def __init__(self, dim, size, iter_num, x_min, x_max, best_fitness_value=float('Inf'), worst_fitness_value=float('-Inf'), F=0.6, CR=0.2):
        self.F = F
        self.CR = CR
        self.dim = dim  # 维度
        self.size = size  # 总群个数
        self.iter_num = iter_num  # 迭代次数
        self.x_min = x_min
        self.x_max = x_max
        self.best_fitness_value = best_fitness_value
        self.best_position = [0.0 for i in range(dim)]  # 全局最优解求，最小值所以将最优值初始化为正无穷
        
        self.worst_fitness_value = worst_fitness_value
        self.worst_position = [0.0 for i in range(dim)]  # 全局最差解,求最小值所以将最差值初始化为负无穷
        self.fitness_val_list = []  # 每次迭代最优适应值

        # 对种群进行初始化
        self.unit_list = [Unit(self.x_min, self.x_max, self.dim) for i in range(self.size)]

    def get_kth_unit(self, k):
        return self.unit_list[k]

    def set_bestFitnessValue(self, value):
        self.best_fitness_value = value

    def get_bestFitnessValue(self):
        return self.best_fitness_value

    def set_bestPosition(self, i, value):
        self.best_position[i] = value

    def get_bestPosition(self):
        return self.best_position
    
    def set_worstFitnessValue(self, value):
        self.worst_fitness_value = value

    def get_worstFitnessValue(self):
        return self.worst_fitness_value

    def set_worstPosition(self, i, value):
        self.worst_position[i] = value

    def get_worstPosition(self):
        return self.worst_position
        
    

    # 普通变异算子
    def select_mutation(self):
        for i in range(self.size):
            r1 = r2 = r3 = 0
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2:
                r1 = random.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
            mutation = self.get_kth_unit(r1).get_pos() + \
                       self.F * (self.get_kth_unit(r2).get_pos() - self.get_kth_unit(r3).get_pos())
            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.x_min <= mutation[j] <= self.x_max:
                    self.get_kth_unit(i).set_mutation(j, mutation[j])
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.get_kth_unit(i).set_mutation(j, rand_value)
    
    # 采用下山单纯形法的变异算子
    def select_mutation_NMSim(self):
        #做Size次的下山单纯型（三点平均，不指定最好点来）
        for i in range(self.size):
            #设置最差点
            for k in range(self.size):
                worst_index = 0
                if self.get_kth_unit(i).get_fitness_value() > self.get_worstFitnessValue():
                    self.set_worstFitnessValue(self.get_kth_unit(i).get_fitness_value())
                    for j in range(self.dim):
                        self.set_worstPosition(j, self.get_kth_unit(i).get_pos()[j])
                    worst_index = k
            r1 = r2 = r3 = 0    #随机选三个点来做平均，保证勘探
            while r1 == i or r2 == i or r3 == i or r2 == r1 or r3 == r1 or r3 == r2 or r1 == worst_index or r2 == worst_index or r3 == worst_index:
                r1 = random.randint(0, self.size - 1)  # 随机数范围为[0,size-1]的整数
                r2 = random.randint(0, self.size - 1)
                r3 = random.randint(0, self.size - 1)
                
            mean = np.array([0.0 for i in range(self.dim)])
            for j in range(self.dim):
                mean[j] = (self.get_kth_unit(r1).get_pos()[j] + self.get_kth_unit(r2).get_pos()[j] + self.get_kth_unit(r3).get_pos()[j]) / 3
            
            Reflect_Rate = 1
            Expansion_Rate = 2
            Contraction_Rate = 0.5
            Shrinkage_Rate = 0.5
            mutation = np.array([(0.0, 0.0) for i in range(4)])
            mutation[0] = mean + Reflect_Rate * (mean - self.get_kth_unit(worst_index).get_pos())
            mutation[1] = mean + Expansion_Rate * (mutation[0] - mean)
            mutation[2] = mean + Contraction_Rate * (mutation[0] - mean)
            mutation[3] = mean + Shrinkage_Rate * (self.get_kth_unit(worst_index).get_pos() - mean)
            
            min = mutation[0]
            for r in mutation:
                if fit_fun(min) > fit_fun(r):
                    min = r
            
            for j in range(self.dim):
                #  判断变异后的值是否满足边界条件，不满足需重新生成
                if self.x_min <= min[j] <= self.x_max:
                    self.get_kth_unit(i).set_mutation(j, min[j])
                else:
                    rand_value = self.x_min + random.random()*(self.x_max - self.x_min)
                    self.get_kth_unit(i).set_mutation(j, rand_value)
                
    # 交叉
    def crossover(self):
        for unit in self.unit_list:
            for j in range(self.dim):
                rand_j = random.randint(0, self.dim - 1)
                rand_float = random.random()
                if rand_float <= self.CR or rand_j == j:
                    unit.set_crossover(j, unit.get_mutation()[j])
                else:
                    unit.set_crossover(j, unit.get_pos()[j])

    # 选择
    def selection(self):
        for unit in self.unit_list:
            new_fitness_value = fit_fun(unit.get_crossover())
            if new_fitness_value < unit.get_fitness_value():
                unit.set_fitness_value(new_fitness_value)
                for i in range(self.dim):
                    unit.set_pos(i, unit.get_crossover()[i])
            if new_fitness_value < self.get_bestFitnessValue():
                self.set_bestFitnessValue(new_fitness_value)
                for j in range(self.dim):
                    self.set_bestPosition(j, unit.get_crossover()[j])
            if new_fitness_value > self.get_worstFitnessValue():
                self.set_worstFitnessValue(new_fitness_value)
                for j in range(self.dim):
                    self.set_worstPosition(j, unit.get_crossover()[j])

    def update(self):
    
        fig = plt.figure()
        for i in range(self.iter_num):
#            self.select_mutation()
            self.select_mutation_NMSim()
            self.crossover()
            self.selection()
            self.fitness_val_list.append(self.get_bestFitnessValue())
            #画图，实时更新版本
            plt.clf()
            ax = plt.axes(projection="3d")
            x = [np.arange(self.x_min, self.x_max, 0.1), np.arange(self.x_min, self.x_max, 0.1)]
            X, Y = np.meshgrid(x[0], x[1])
            Z = fit_fun1(X,Y)
            for unit in self.unit_list:
                ax.scatter(unit.get_pos()[0], unit.get_pos()[1], unit.get_fitness_value(), c='r', marker='o', linewidths = 0.5, depthshade = False, s = 10)

            ax.plot_surface(X, Y, Z, alpha = 0.7)
            plt.title("Number of Generation:"+str(i))
            plt.pause(0.0001)
            #显示
            print("Generation"+str(i)+":", end=" ")
            print(self.get_bestFitnessValue())
            
        return self.fitness_val_list, self.get_bestPosition(), self.get_bestFitnessValue()
