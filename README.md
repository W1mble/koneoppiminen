## Koneoppimismallien kouluttamista, optimointia ja vertailua

[![Python 3.8.5](https://img.shields.io/badge/Python-3.8.5-blue.svg)](#)

Pythonilla toteutettuja koneoppimismalleja, joissa käytetty scikit-learn-kirjastoa. Malleja koulutetaan, optimoidaan ja/tai vertaillaan niitä muihin koneoppimismalleihin. 

Kaikki mallit eivät ole "valmiita" eli niitä ei välttämättä ole tehty kaikkia osa-alueita huomioon ottaen. Niiden tarkoitus on enemmänkin olla demonstraationa.

## Sisältö
* *Koneoppimismalli.py*

    - Opetetaan yksinkertainen koneoppimismalli käyttäen diabetes dataa scikit-learn-kirjastosta.
     
* *Koneoppimismalli-vertailu.py*

    - Lineaariregressiomalli vs neuroverkko regressiotehtävässä; Bostonin alueiden asuntojen hintatasoa 
kuvaavan datan vastemuuttujan ennustamisessa

* *Satunnaismetsaluokitin.py*

     - Luokitellaan havainnot kolmeen ryhmään; galaksit, tähdet ja kvasaarit käyttäen satunnaismetsäluokitinta. Verrataan feature importances vs permutation importances eroja.
     - https://www.kaggle.com/muhakabartay/sloan-digital-sky-survey-dr16

* *Satunnaismetsaluokitin2.py*

     - Yritetään kehittää paras mahdollinen ennustemalli käsin kirjoitettujen numeroiden luokitteluun satunnaismetsäluokittelijan avulla.

* *3-mallia-ja-optimointia.py*

     - Optimoidaan sekä k-lähimmän naapurin (kNN) menetelmä, MLP-neuroverkko ja SVC-luoktiin numeroiden tunnistamiseen.
     - Tutkitaan kNN:ssä lähimpien naapurin lukumäärän ja etäisyysmitan painokertoimien vaikutusta.
     - MLP:ssa tutkitaan piilokerrosten kokojen ja lukumäärän, aktivaatiofunktioiden, opetusalgoritmin ja regularisoinnin vaikutusta. 
     - Hyperparametrien optimoinnissa käytetään 10-kertaista ristiinvalidointia.

* *3-mallia-paras.py*

     - Samaa hommaa kuin aikaisemmassa tehtävässä, mutta eri datalla ja yritetty parantaa kommentointia.
     - https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009





### Vaatimukset
* **Python (testattu 3.8.5)**
* **Sklearn**
* **NumPy**
* **Matplotlib**
* **Pandas**
* **Seaborn**
