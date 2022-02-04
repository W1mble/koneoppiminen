from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#Tallennetaan .csv (eli data) tiedosto muuttujaan file ja luetaan sen käyttämällä pandas -kirjastoa. 
#Datassa on erottimena käytetty puolipistettä. Data on nyt df muuttujassa oikein luettuna.

file = "winequality-red.csv" #Tähän oikea tiedostopolku
df = pd.read_csv(file, delimiter = ";")

#y - muuttujaan asetetaan vastemuuttuja eli viinin laatu (quality) ja X:ään kaikki muut parametrit
y = df.quality
X = df.drop('quality', axis=1)

#Tarkastellaan kuinka data on jakautunut ja tutkitaan parametrien korrelaatioita.
print(df.head())

korrelaatio = df.corr()
s = sns.heatmap(korrelaatio)
s.set_yticklabels(s.get_yticklabels())
s.set_xticklabels(s.get_xticklabels())
plt.show() 

#Jaetaan data 80:20 suhteeseen (testi 20%), asetetaan random state, jotta ajot voi uusia (toistaa)
#ja käytetään stratify=y, jotta säilytetään kohteen (target) suhde/osuus niinkuin alkuperäisessä datassa, myös opetus ja testidatassa.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y)
    
#Data skaalataan, jotta oppiminen nopeutuu. (yleisesti tämä pätee, mutta ei tietysti aina). 
#Pipeline -objektille kerrotaan, että käytä skaalauksessa StandardScaleria.
pipe_svc = Pipeline([    
    ('scaling', StandardScaler()), #<------
    ('svc', SVC())
])


#K-lähimmän naapurinmenetelmä. Se on pikemminkin muistipohjainen, jossa valitaan optimaalinen määrä naapureita ja etäisyyksiä. 
#Ei vaadi oikeastaan laskentaa, mutta sovittamisessa voi mennä kauemmin jos dataa on paljon. Siksi se sopiikin hyvin pienille datoille.
pipe_knn = Pipeline([    
    ('scaling', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

#Tukivektorikone. Syötteet luokitellaan hypertason mukaan. 
#Optimaalisin hypertaso on se, jolla on suurin marginaali näiden kahden luokan välillä kun syötevektorit ovat lineaarisesti erotettavissa. 
#Tavoitteena on maksimoida luokkien väliin jäävä marginaali. Suuremmalla marginaanilla saavutetaan yleensä parempi yleistymiskyky. 
#Valmiin mallin parametrien valintaa on vaikeaa tulkita, mutta muistin käyttö on kohtuullista.
pipe_svc = Pipeline([    
    ('scaling', StandardScaler()),
    ('svc', SVC())
])

#Neuroverkkoluokitin. Simuloi, miten ihmisen aivot toimivat. Yleensä saa aika tarkkoja tuloksia. 
#Laskennallisesti aika raskasta, varsinkin jos joutuu käyttämään CPU:ta näytönohjaimen sijasta. 
#Hankala myös tulkita miten yksittäinen parametri vaikuttaa malliin. "Musta laatikko".
pipe_mlp = Pipeline([    
    ('scaling', StandardScaler()),
    ('mlp', MLPClassifier(max_iter=10000, random_state=0))
])


classifier_list=[pipe_svc, pipe_mlp, pipe_knn] 

#SVC: Testaillaan eri kerneleitä ja regularisointi parametrin (C) eri arvoja 
param_grid_svc = [{'svc__kernel': ['rbf'], 'svc__gamma': [1e-3, 1e-4],
                     'svc__C': [1, 10, 100, 1000]},
                    {'svc__kernel': ['linear'], 'svc__C': [1, 10, 100, 1000]}]
                    
#MLP: Testaillaan eri aktivaatiofunktioita, piilokerrosten määrää ja neuronien määrää, eri solvereita ja alpha arvoa (regularisointi)
param_grid_mlp = [{
    'mlp__alpha': [0.0001, 0.01, 0.1],
    'mlp__activation': ['relu','logistic','tanh', 'identity'],
    'mlp__hidden_layer_sizes': [(1,), (100,), (100, 100,)],
    'mlp__solver': ['lbfgs', 'adam']
    }]
    
#KNN: Testaillaan naapurien määrää ja painoarvojen vaikutusta.
param_grid_knn = [{ 
    'knn__n_neighbors': [1, 2, 4, 8, 16, 32],
    'knn__weights': ['uniform', 'distance']}]


parameters_list = [param_grid_svc, param_grid_mlp, param_grid_knn]

model_log=["_pipe_svc", "_pipe_mlp", "_pipe_knn"]

#Käydään silmukassa svc, mlp ja knn ja etsitään jokaiselle parhaat parametrit ja otetaan talteen paras tulos.
for i in range(len(classifier_list)):
    Grid=GridSearchCV(estimator=classifier_list[i], param_grid=parameters_list[i], 
                      n_jobs=-1, cv=10).fit(X_train, y_train)
    globals()['Grid%s' % model_log[i]]=pd.DataFrame(Grid.cv_results_)  


#Lasketaan testivirhe
y_test_pred = Grid.best_estimator_.predict(X_test)

print('Testivirhe : %.2f'
      % mean_squared_error(y_test, y_test_pred))
      
      
#Tulostetaan paras malli, sen parametrit, sen tulos ja kuvaaja, jonne sijoitetaan ennustetut ja todelliset arvot.      
print(Grid.best_params_)
print(Grid.best_score_)
print(Grid.best_estimator_)


plt.scatter(y_test, y_test_pred)
plt.plot([0, 12], [0, 12], '--k')
plt.xlabel('Todellinen arvo')
plt.ylabel('Ennustettu arvo')
plt.title("Todellinen vs ennustettu")
plt.show()