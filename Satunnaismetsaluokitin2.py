from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')

#Yritetään kehittää paras mahdollinen ennustemalli käsin kirjoitettujen numeroiden luokitteluun satunnaismetsäluokittelijan avulla.
digits = datasets.load_digits()

#Luokittelun käyttämiseksi kuvat tasoitetaan kääntämällä kukin 2-D harmaasävyarvojen taulukko muodosta (8, 8) muotoon (64,). Sijoitetaan X syöte ja y vastemuuttuja.
n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target

#Jaetaan opetus ja testidata osiin ja asetetaan random_state = 0, jotta tulokset ovat vertailtavissa eli "seed" on sama.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)
    
#Skaalataan data, jotta tarkkuus paranee ja mallin oppiminen tapahtuu nopeammin. Käytetään StandardScaler, koska mielestäni datassa ei ollut poikkeavia havantoja.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#Valitaan 6 parametria, jolla oletetaan olevan suurin vaikutus (tieto etsitty netistä). Säädetään niille arvot ja sijoitetaan ne random_gridiin. 
#Luodaan clf ja clf_random -luokittelijat (jotta voidaan verrata myös). Skaalattu data opetetaan malleille.
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 2000, num = 20)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]


clf = RandomForestClassifier(n_jobs = -1, random_state=0)



random_grid = [{'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}]


clf_random = RandomizedSearchCV(estimator = clf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=0, n_jobs = -1, scoring='neg_mean_absolute_error')

clf.fit(X_train_scaled, y_train)
clf_random.fit(X_train_scaled, y_train)


#Asetetaan param_gridiin parhaimpia parametrien arvoja, joita on saatu useasta eri ajosta. Tehdään GridSearch ja opetetaan malli.
#Ajettu muutaman kerran ja lisätty param_grid arvoja, jotka ovat olleet random haun parhaita parametreja
param_grid = [{
    'n_estimators': [100, 200, 400, 800],
    'max_features': ['auto'],
    'max_depth': [10, 60, 80, 90, 100],
    'min_samples_split': [2, 5, 10,],
    'min_samples_leaf': [1, 3, 4],
    'bootstrap': [False],
}]

grid_search = GridSearchCV(estimator = clf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2, scoring='neg_mean_absolute_error')

grid_search.fit(X_train_scaled, y_train)


#RandomSearchista voi olla hyötyä silloin kun halutaan vähentää GridSearchin etsintäaikaa. 
#Eli jos data on valtavan kokoinen, iteraatiot suuria sekä parametrien määrät ja arvot myöskin isoja, niin sillä voi säästää huomattavan määrän aikaa, 
#kun löytää tietyt parametrit, joilla tulee suurin muutos. Näin opettamisajassa voi säästää paljonkin.
y_true, y_pred = y_test, clf.predict(X_test_scaled)
print()
print("Luokittelutarkkuus (Default)")
print(accuracy_score(y_test, y_pred))
print()
print("Luokittelutulos (Search)")
print(classification_report(y_true,y_pred))
print("Sekaannusmatriisi (Default)")
print()    
print(confusion_matrix(y_true, y_pred))


y_true2, y_pred2 = y_test, clf_random.predict(X_test_scaled)
print()
print("Luokittelutarkkuus (Random)")
print(accuracy_score(y_test, y_pred2))  
print()
print('Parhaat random parametrit:')
print(clf_random.best_params_)
print("Luokittelutulos (Random)")
print(classification_report(y_true2,y_pred2))
print("Sekaannusmatriisi (Random)")
print()    
print(confusion_matrix(y_true2, y_pred2))

grid_search.fit(X_train_scaled, y_train)
print()
y_true3, y_pred3 = y_test, grid_search.predict(X_test_scaled)
print()
print("Luokittelutarkkuus (Search)")
print(accuracy_score(y_test, y_pred3))  
print()
print('Parhaat search parametrit:')
print(grid_search.best_params_)
print()
print("Luokittelutulos (Search)")
print(classification_report(y_true3,y_pred3))
print("Sekaannusmatriisi (Search)")
print()    
print(confusion_matrix(y_true3, y_pred3))


#Plottasin testitulosten puiden määrät ja vertasin sitä neg_mean_squared erroriin (eli mitä lähempänä 0 niin sitä vähemmän virheitä). 
#Kun ajoin ohjelman niin se plottasi 800 puiden määräksi, kun tulos oli alhaisin (eli mihin tähdättiinkin).
param = 'n_estimators', 
param_name = 'param_%s' % param

test_scores = grid_search.cv_results_['mean_test_score']
param_values = list(grid_search.cv_results_[param_name])
plt.plot(param_values, test_scores, 'go-', label = 'test', linewidth=1.0)
plt.ylim(ymin = -0.2, ymax = 0)
plt.legend()
plt.xlabel('Number of Trees')
plt.ylabel('Neg Mean Absolute Error')
plt.title('Score vs Number of Trees')

plt.tight_layout()
plt.show()
