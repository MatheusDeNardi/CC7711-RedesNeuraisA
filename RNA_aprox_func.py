import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor


arquivo = np.load('teste5.npy')
bests_loss = []

x = arquivo[0]
y = arquivo[1]

plt.figure(figsize=(16,8))
plt.plot(x,y)
plt.show()

regr = MLPRegressor(hidden_layer_sizes=(45,45),
                    max_iter=10000,
                    solver='adam',
                    learning_rate = 'adaptive',
                    n_iter_no_change=1000)
for z in range(10):
    regr = regr.fit(x,y)  

    y_est = regr.predict(x)

    #print da curva de aprendizagem
    plt.figure(figsize=(16,8))
    plt.plot(regr.loss_curve_)
    plt.show()

    #preditor
    y_est = regr.predict(x)
    plt.figure(figsize=(16,8))
    plt.plot(x,y,x,y_est)
    plt.show()
    bests_loss.append(regr.best_loss_)
    plt.show()
mean = np.average(bests_loss)
desv_padrao = np.std(bests_loss)
print(f"Média: {mean}")
print(f"Desvio Padrão: {desv_padrao}")