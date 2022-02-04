from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.neural_network import MLPClassifier
import time as time
from sklearn.svm import SVC

#Optimoidaan sekä k-lähimmän naapurin (kNN) menetelmä, MLP-neuroverkko ja SVC-luoktiin numeroiden tunnistamiseen.

#Tutkitaan kNN:ssä lähimpien naapurin lukumäärän ja etäisyysmitan painokertoimien vaikutusta.

#MLP:ssa tutkitaan piilokerrosten kokojen ja lukumäärän, aktivaatiofunktioiden, opetusalgoritmin ja regularisoinnin vaikutusta. 
#Hyperparametrien optimoinnissa käytetään 10-kertaista ristiinvalidointia.


""""SVC oli hieman tarkempi kuin MLP, mutta vain kun parametrien määrää kasvatettiin, mutta opetusajat alkoivat olemaan todella pitkiä <5min.
SVC:n tarkkuus oli paras, sitten MLP ja lopuksi KNN. """


digits = datasets.load_digits()


n_samples = len(digits.images)
X = digits.images.reshape((n_samples, -1))
y = digits.target


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=0)


pipe_knn = Pipeline([    
    ('scaling', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

pipe_mlp = Pipeline([    
    ('scaling', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=10000, random_state=0))
])

pipe_svc = Pipeline([    
    ('scaling', StandardScaler()),
    ('svc', SVC())
])


#------------------------------------------------------------------------#  


# Asetetaan ristiinvalidoitavat hyperparametrit
param_grid_svc = [{'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4],
                     'svc__C': [1, 10, 100, 1000]},
                    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000]}]

param_grid_mlp = [{
    'mlp__alpha': [0.0001, 0.01, 0.1],
    'mlp__activation': ['relu','logistic','tanh', 'identity'],
    'mlp__hidden_layer_sizes': [(1,), (100,), (100, 100,)],
    'mlp__solver': ['lbfgs', 'adam']
    }]

param_grid_knn = [{ 
    'knn__n_neighbors': [1, 2, 4, 8, 16, 32],
    'knn__weights': ['uniform', 'distance']}]


#------------------------------------------------------------------------#  


scores = ['precision', 'recall']

start_time = time.time()

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    grid_svc = GridSearchCV(
        pipe_svc, param_grid = param_grid_svc, scoring='%s_macro' % score, cv = 10, n_jobs = -1)
        
    
    grid_svc.fit(X_train, y_train)
    

    print("Parhaat opetusdatalla löydetyt parametrit ovat (SVC)")
    print()
    print(grid_svc.best_params_)
    print()
    print("score of: %f" % grid_svc.best_score_)
    print()
    print("GridSearch tulokset (SVC):")
    print()
    means = grid_svc.cv_results_['mean_test_score']
    stds = grid_svc.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_svc.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    total_time = time.time() - start_time
    
    print("Luokitteluraportti (SVC):")
    print()
    print("Opetusdatalla")
    print("Suorituskyky on arvioitu testidatalla")
    print()
    y_true, y_pred = y_test, grid_svc.predict(X_test)
    print(classification_report(y_true,y_pred))
    print("Sekaannusmatriisi (SVC)")
    print()    
    print(print(confusion_matrix(y_true, y_pred)))
    print()
    print("Laskenta-aika SVC:lle")
    print(total_time)
    accuracy = accuracy_score(y_test, y_pred)
    print("Luokittelutarkkuus %.2f" % accuracy)
    
#------------------------------------------------------------------------#  


start_time_2 = time.time()    
    
for score in scores:
    
    
    print("# Tuning hyper-parameters for %s" % score)
    print()

    
    grid_mlp = GridSearchCV(
        pipe_mlp, param_grid = param_grid_mlp, scoring='%s_macro' % score, cv = 10, n_jobs = -1)
        
    
    grid_mlp.fit(X_train, y_train)
    
    print("Parhaat opetusdatalla löydetyt parametrit ovat (MLP)")
    print()
    print(grid_mlp.best_params_)
    print()
    print("score of: %f" % grid_mlp.best_score_)
    print()
    print("GridSearch tulokset (MLP):")
    print()
    means = grid_mlp.cv_results_['mean_test_score']
    stds = grid_mlp.cv_results_['std_test_score']
    for mean2, std2, params2 in zip(means, stds, grid_mlp.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean2, std2 * 2, params2))
    print()

    total_time2 = time.time() - start_time_2
    
   
    
    print("Luokitteluraportti (MLP):")
    print()
    print("Opetusdatalla")
    print("Suorituskyky on arvioitu testidatalla")
    print()
    y_true2, y_pred2 = y_test, grid_mlp.predict(X_test)
    print(classification_report(y_true2, y_pred2))
    print("Sekaannusmatriisi (MLP)")
    print()    
    print(print(confusion_matrix(y_true2, y_pred2)))
    print()
    print("Laskenta-aika MLP:lle")
    print(total_time2)
    accuracy = accuracy_score(y_test, y_pred2)
    print("Luokittelutarkkuus %.2f" % accuracy)
    
    
#------------------------------------------------------------------------#    
   
    
start_time_3 = time.time()    
    
for score in scores:
    
    
    print("# Tuning hyper-parameters for %s" % score)
    print()

    
    grid_knn = GridSearchCV(
        pipe_knn, param_grid = param_grid_knn, scoring='%s_macro' % score, cv = 10, n_jobs = -1)
        
    
    grid_knn.fit(X_train, y_train)
    
    print("Parhaat opetusdatalla löydetyt parametrit ovat (KNN)")
    print()
    print(grid_knn.best_params_)
    print()
    print("score of: %f" % grid_knn.best_score_)
    print()
    print("GridSearch tulokset (KNN):")
    print()
    means = grid_knn.cv_results_['mean_test_score']
    stds = grid_knn.cv_results_['std_test_score']
    for mean3, std3, params3 in zip(means, stds, grid_knn.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean3, std3 * 2, params3))
    print()

    total_time3 = time.time() - start_time_3

    print("Luokitteluraportti (KNN):")
    print()
    print(" opetusdatalla")
    print("Suorituskyky on arvioitu testidatalla")
    print()
    y_true3, y_pred3 = y_test, grid_knn.predict(X_test)
    print(classification_report(y_true3, y_pred3))
    print("Sekaannusmatriisi (KNN)")
    print()    
    print(print(confusion_matrix(y_true3, y_pred3)))
    print()
    print("Laskenta-aika KNN:lle")
    print(total_time3)
    accuracy3 = accuracy_score(y_test, y_pred3)
    print("Luokittelutarkkuus %.2f" % accuracy3)