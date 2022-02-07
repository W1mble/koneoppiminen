from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.inspection import permutation_importance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Sloan Digital Sky Survey DR16 -dataa käytetty (https://www.kaggle.com/muhakabartay/sloan-digital-sky-survey-dr16)
#Luokitellaan havainnot kolmeen ryhmään; galaksit, tähdet ja kvasaarit.
#Käytetään satunnaismetsäluokitinta ja verrataan Feature importances vs Permutation importances eroja.


'''Permutation importances testiaineistolla asettaa järjestyksen: 
    redshift, u, g, i, z, r, plate, mjd, fiberid, ra ja dec.
    
Opetusdatalla järjestys oli: redshift, u, g, i, z, jonka jälkeen se muuttui plate, mdj, r, ra, dec, fiberid

Feature importances taas antoi tulokseksi 
ra, dec, u, g, r, i, z, redshift, plate, mjd ja fiberid 
eli tulos oli huomattavasti erinlainen.'''


sky = "Skyserver_12_30_2019 4_49_58 PM.csv"
data = pd.read_csv(sky, delimiter = ",")

#----------------------------------------------------------------------------------------------------#

y = data['class']
X = data.drop(['objid', 'run', 'rerun', 'camcol', 'field', 'specobjid', 'class'], axis=1) #objid, run, rerun, camcol, field, specobjid, kameroiden kulmia ja aikoja, tunnistetietoa
#eli ei vaikuta, joten jätetään pois

#----------------------------------------------------------------------------------------------------#
        
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

rfc = RandomForestClassifier(n_estimators=100, #The number of trees in the forest.
                             criterion="gini", #The function to measure the quality of a split.
                             max_depth=None, #The maximum depth of the tree.
                             min_samples_split=2, #The minimum number of samples required to split an internal node.
                             min_samples_leaf=1, #The minimum number of samples required to be at a leaf node.
                             max_features="auto", #The number of features to consider when looking for the best split
                                                  #When "auto", max_features=sqrt(n_features)
                             n_jobs = -1) 

rfc.fit(X_train, y_train)

#----------------------------------------------------------------------------------------------------#

print('Opetusaineisto:')
y_train_preds = rfc.predict(X_train)
print(classification_report(y_train, y_train_preds))
print(print(confusion_matrix(y_train, y_train_preds)))
print()

#----------------------------------------------------------------------------------------------------#

print('Testiaineisto:')
y_test_preds = rfc.predict(X_test)
print(classification_report(y_test, y_test_preds))
print(print(confusion_matrix(y_test, y_test_preds)))
print()

#----------------------------------------------------------------------------------------------------#


result_test = permutation_importance(rfc, X_test, y_test, n_repeats=10, random_state=0, n_jobs = -1)

#----------------------------------------------------------------------------------------------------#


sorted_idx_test = result_test.importances_mean.argsort()
fig, ax = plt.subplots()
ax.boxplot(result_test.importances[sorted_idx_test].T,
           vert=False, labels=X_test.columns[sorted_idx_test])
ax.set_title("Permutation Importances (test set)")
fig.tight_layout()
plt.show()

#----------------------------------------------------------------------------------------------------#

result_train = permutation_importance(rfc, X_train, y_train, n_repeats=10, random_state=0, n_jobs = -1)

#----------------------------------------------------------------------------------------------------#
sorted_idx_train = result_train.importances_mean.argsort()

fig2, ax2 = plt.subplots()
ax2.boxplot(result_train.importances[sorted_idx_train].T,
           vert=False, labels=X_train.columns[sorted_idx_train])
ax2.set_title("Permutation Importances (train set)")
fig2.tight_layout()
plt.show()


#----------------------------------------------------------------------------------------------------#


feature_names = list(X_train.columns.values)
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
        color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names, rotation='vertical')
plt.xlim([-1, X.shape[1]])
plt.show()
