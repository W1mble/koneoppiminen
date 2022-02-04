import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

'''Lineaariregressiomalli vs neuroverkko regressiotehtävässä: Bostonin alueiden asuntojen hintatasoa 
kuvaavan datan vastemuuttujan ennustamisessa'''


#Neuroverkko toimi paremmin kuin lineaarinen regressiomalli ja vielä paremmin kun parametrit optimoi. 
#Suuresta oppimiskapatiseetista oli näin ollen apua.
#Parhaimmat tulokset tuli (100,100) ja (100,100,100) eli malli hyötyi piilokerrosten lisäyksestä.

#Ladataan boston data 
boston = load_boston()
X = boston.data
y = boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)


#Luodaan minmax-skaalaus objekti
min_max_scaler = preprocessing.MinMaxScaler()
# Sovitetaan ja skaalataan opetusdata
X_train_minmax = min_max_scaler.fit_transform(X_train)
# Sovelletaan testidatalla opetettuja skaalauskertoimia testidataan
X_test_minmax = min_max_scaler.transform(X_test)

#Opetetaan lineaarinen regressiomalli skaalatulla datalla ja testataan se testidatalla
lr = LinearRegression()
lr.fit(X_train_minmax, y_train)
boston_y_test_pred = lr.predict(X_test_minmax)
boston_y_train_pred = lr.predict(X_train_minmax)

#2 neuroverkkoa, toisesta ei optimoida parametrejä
mlpdefault = MLPRegressor(max_iter=10000, random_state=0)
mlpdefault.fit(X_train_minmax, y_train)

mlp = MLPRegressor(max_iter=10000, random_state=0)


param_grid = { 'hidden_layer_sizes': [(1,), (100,), (200,), (1,1), (100, 100), (100, 10), (10, 100), (100, 100, 100)],
'activation': ["logistic", "relu", "tanh", "identity"], 
'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]}

print("Etsitään optimaalisia parametrejä...\nTämä vie muutamia minuutteja")

haku = GridSearchCV(mlp, param_grid, n_jobs = -1, cv=3, scoring='neg_mean_squared_error')
haku.fit(X_train_minmax, y_train)

boston_y_test_pred_neuro_default = mlpdefault.predict(X_test_minmax)
boston_y_train_pred_neuro_default = mlpdefault.predict(X_train_minmax)
train_mse_default = -mean_squared_error(y_train, boston_y_train_pred_neuro_default)
test_mse_default = -mean_squared_error(y_test, boston_y_test_pred_neuro_default)

boston_y_test_pred_neuro = haku.predict(X_test_minmax)
boston_y_train_pred_neuro = haku.predict(X_train_minmax)
train_mse = -mean_squared_error(y_train, boston_y_train_pred_neuro)
test_mse = -mean_squared_error(y_test, boston_y_test_pred_neuro)


print("Opetusdatan rivien (havaintojen) lukumäärä: ", len(X_train))
print("Testidatan rivien (havaintojen) lukumäärä: ", len(X_test), "\n")

#- Lineaarisen regressiomallin kertoimet
print('Lineaarisen regressiomallin kertoimet: \n', lr.coef_)
print('Lineaarisen regressiomallin testivirhe : %.2f'
      % mean_squared_error(y_test, boston_y_test_pred), "\n")

#Neuroverkko ilman parametrien optimointia
print("Opetuksen MSE default arvoilla:", np.round(train_mse_default,2))
print("Testin MSE default arvoilla:", np.round(test_mse_default,2), "\n")

#Optimoidut parametrit tulokset ja neg MSE
print("Parhaat parametrit: ", haku.best_params_)
print("Opetuksen MSE:", np.round(train_mse,2))
print("Testin MSE:", np.round(test_mse,2))


#Plottaus lineaariselle regressiomallille
plt.scatter(y_test, boston_y_test_pred)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Todellinen hinta ($1000s)')
plt.ylabel('Ennustettu hinta ($1000s)')
plt.title("Todellinen hinta vs ennustettu hinta : Lineaarinen regressiomalli")
plt.show()

#Plottaus neuroverkolle (default)
plt.scatter(y_test, boston_y_test_pred_neuro_default)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Todellinen hinta ($1000s)')
plt.ylabel('Ennustettu hinta ($1000s)')
plt.title("Todellinen hinta vs ennustettu hinta : Neuroverkko (default)")
plt.show()

#Plottaus neuroverkolle (optimoitu)
plt.scatter(y_test, boston_y_test_pred_neuro)
plt.plot([0, 50], [0, 50], '--k')
plt.xlabel('Todellinen hinta ($1000s)')
plt.ylabel('Ennustettu hinta ($1000s)')
plt.title("Todellinen hinta vs ennustettu hinta : Neuroverkko (optimoitu)")
plt.show()