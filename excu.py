from DE import DE
import matplotlib.pyplot as plt
import numpy as np


dim = 2
size = 40
iter_num = 30
x_max = 100

best_in_each = []
for i in range(30):
    de = DE(dim, size, iter_num, -x_max, x_max)
    fit_var_list2, best_pos2, best_value = de.update()
    print("DE最优位置:" + str(best_pos2))
#    print("DE最优解:" + str(fit_var_list2[-1]))
    print("DE最优解:" + str(best_value))
    best_in_each.append(best_value)

    fig = plt.figure()
    plt.plot(np.linspace(0, iter_num, iter_num), fit_var_list2, label="DE")
    plt.scatter(np.linspace(0, iter_num, iter_num), fit_var_list2)
    plt.title("Best Fitness in each generation")
    #
    #plt.legend()  # 显示lebel
    plt.show()
    
print(best_in_each)
print(np.mean(best_in_each))
