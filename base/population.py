import random
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 设置随机种子
random.seed(42)

# 读取数据
df = pd.read_csv("data.csv", header=0)

# 将数据划分为特征和标签
X = df.iloc[:, :-3]
y = df.iloc[:, -1:]

# 数据归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=46)

# 定义超参数的搜索空间
space = {
    'hidden_layer_sizes': [(i, j) for i in range(1, 100) for j in range(1, 100)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs', 'sgd', 'adam'],
    'max_iter': range(1, 1000)
}

# 定义遗传算法相关参数
population_size = 50  # 种群大小
num_generations = 100  # 迭代次数
crossover_rate = 0.8  # 交叉概率
mutation_rate = 0.1  # 变异概率

# 定义适应度函数
def fitness_function(params):
    model = MLPClassifier(**params)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return score

# 定义遗传算法类
class GeneticAlgorithm:
    def __init__(self, population, fitness_function, crossover, mutation, selection):
        self.population = population
        self.fitness_function = fitness_function
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
    
    def evolve(self, crossover_rate, mutation_rate):
        # 计算每个候选解的适应度
        fitness_values = [self.fitness_function(individual) for individual in self.population]
        
        # 选择下一代的种群
        new_population = []
        for i in range(len(self.population)):
            parent1 = self.selection(fitness_values)
            parent2 = self.selection(fitness_values)
            child = self.crossover(parent1, parent2, crossover_rate)
            child = self.mutation(child, mutation_rate)
            new_population.append(child)
        self.population = new_population
    
    def best_individual(self):
        # 返回适应度最高的候选解及其适应度
        fitness_values = [self.fitness_function(individual) for individual in self.population]
        best_index = np.argmax(fitness_values)
        return self.population[best_index], fitness_values[best_index]

# 定义交叉算子
def crossover(parent1, parent2, crossover_rate):
    if random.random() < crossover_rate:
        # 随机选择一个交叉点
        index = random.randint(1, len(parent1) - 1)
        # 交叉
        child = parent1[:index] + parent2[index:]
        return child
    else:
        return parent1
#定义变异算子
def mutation(individual, mutation_rate):
    if random.random() < mutation_rate:
    # 随机选择一个基因进行变异
        index = random.randint(0, len(individual) - 1)
        # 变异
        individual[index] = random.choice(space[index])
    return individual

#定义选择算子
def selection(fitness_values):
    # 使用轮盘赌选择
    total_fitness = sum(fitness_values)
    r = random.uniform(0, total_fitness)
    s = 0
    for i, f in enumerate(fitness_values):
        s += f
        if s >= r:
            return population[i]

#初始化种群
population = [list(random.choice(list(space.values()))) for _ in range(population_size)]

#定义遗传算法对象
ga = GeneticAlgorithm(population, fitness_function, crossover, mutation, selection)

#进化
for i in range(num_generations):
    ga.evolve(crossover_rate, mutation_rate)
best_individual, best_fitness = ga.best_individual()
print(f"Generation {i+1}, Best Fitness: {best_fitness}, Best Individual: {best_individual}")

#训练模型并输出准确率
model = MLPClassifier(**dict(zip(list(space.keys()), best_individual)))
model.fit(X_train, y_train)
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Train Accuracy: {train_acc}, Test Accuracy: {test_acc}")
