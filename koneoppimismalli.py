import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

#Yksinkertaisen koneoppimismallin kouluttaminen.


#Ladataan diabetes data
diabetes = datasets.load_diabetes()

#Syötematriisi ja vastemuuttuja numpy-taulukkoihin X ja y
X = np.array(diabetes.data)
y = np.array(diabetes.target)

#Data jaetaan opetus ja testijoukkoon suhteessa 67%/33%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

print(X_train.shape,"Opetusmatriisin koko")
print(X_test.shape,"Testimatriisin koko")

#Skaalataan opetusdata vakiovälille minmaxscaler funktiolla
scaler = MinMaxScaler()
scaled_X_train = scaler.fit_transform(X_train)

#Opetusdatalla sovitettu skaalaus objekti, suoritetaan skaalaus myös testidatalle
scaled_X_test = scaler.transform(X_test)

print(scaled_X_train.max(), "Skaalatun opetusmatriisin max arvo")
print(scaled_X_train.min(), "Skaalatun opetusmatriisin min arvo")


#Opetaaan oletusparametreilla lineaarinen ennustemalli stokastista gradienttialgoritmia käyttäen
#Syötetaan mallille opetusdata ja laskeaan opetusvirhe ennusteiden ja todellisten arvojen välillä (mean_squared_error)
clf = linear_model.SGDRegressor()
clf.fit(scaled_X_train, y_train)
diabetes_y_test_pred = clf.predict(scaled_X_test)
diabetes_y_train_pred = clf.predict(scaled_X_train)

clf2 = linear_model.SGDRegressor(alpha = 0.001)
clf2.fit(scaled_X_train, y_train)
diabetes_y_test_pred2 = clf2.predict(scaled_X_test)
diabetes_y_train_pred2 = clf2.predict(scaled_X_train)


print('Mean squared error (testivirhe) alpha = default: %.2f'
      % mean_squared_error(y_test, diabetes_y_test_pred))
print('Mean squared error (mallin tarkkuus) alpha = default: %.2f'
      % mean_squared_error(y_train, diabetes_y_train_pred))

opetusv = mean_squared_error(y_test, diabetes_y_test_pred)
mallint = mean_squared_error(y_train, diabetes_y_train_pred)
print(opetusv - mallint)


print('Mean squared error (testivirhe2) alpha = 0.001: %.2f'
      % mean_squared_error(y_test, diabetes_y_test_pred2))
print('Mean squared error (mallin tarkkuus2) alpha = 0.001: %.2f'
      % mean_squared_error(y_train, diabetes_y_train_pred2))

opetusv2 = mean_squared_error(y_test, diabetes_y_test_pred2)
mallint2 = mean_squared_error(y_train, diabetes_y_train_pred2)
print(opetusv2 - mallint2)


'''Data pitäisi tarkistaa ettei siellä ole duplikaatteja tai virheellistä dataa.
#Datan jakamisessa voisi lisätä validointi setin, jotta parametrejä voitaisiin 
#hienosäätää.'''