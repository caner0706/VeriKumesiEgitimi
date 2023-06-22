#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 15:40:52 2023

@author: caner
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")
print(veriler)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') #Eksik verileri tamamlamak için ortalama kullanılırken kullanılan kütüphane.
Yas = veriler.iloc[:, 1:4].values
print(Yas)

imputer = imputer.fit(Yas[:, 1:4])
Yas[:, 1:4] = imputer.transform(Yas[:, 1:4])
print(Yas)



# Kategorik Veriler

ulke = veriler.iloc[:,0:1].values  #Veri kaynağı ile alınan veriler filtrelenerek ülke kolonu seçilmiştir.
print(ulke) # Ülke sütunu ekrana yazdırılmıştır.

from sklearn import preprocessing 

le = preprocessing. LabelEncoder ()
ulke[:,0] = le.fit_transform(veriler.iloc[:,0]) # Ülke verileri sırası ile dönüştürme işlemi yapılmıştır.
print (ulke)  # Ekrana yazdırılmıştır.

ohe = preprocessing.OneHotEncoder () # Makine tarafından öğrenilen ülke verileri makine öğrenimi tarafından çevrilmek için kullanılır.
ulke = ohe.fit_transform(ulke). toarray ()
print (ulke) # Ekrana yazdırılır.

# Sonuç olarak sayısal olmayan veriler sayısala çevirilmiş ve bir şablon elde edilmiştir.


# Verilerin Birleştirilmesi

print(list(range(22)))
sonuc = pd. DataFrame (data=ulke, index = range (22), columns = ['fr', 'tr', 'us'])  #Verileri düzenlenmiş forma sokmak için kullanılır. Sütun haline getirir.
print(sonuc)


sonuc2 = pd.DataFrame (data=Yas, index = range (22), columns = ['boy', 'kilo', 'yas ']) #Yaş Boy Kilo verilerin düzenlemektedir.
print (sonuc2)


cinsiyet = veriler.iloc[:, -1].values #Veri kaynağı ile alınan veriler filtrelenerek cinsiyet kolonu seçilmiştir. Bir listeye atanmıştır.
print(cinsiyet   )# Cinsiyet sütunu ekrana yazdırılmıştır.


sonuc3 = pd.DataFrame (data = cinsiyet, index = range (22), columns = ['cinsiyet']) # Verileri düzenlenmiş forma sokmak için kullanılır. Sütun haline getirir
print(sonuc3)


# Oluşturulan veri tabloları tek bir şablon haline getirmek için kullanılmıştır.
s=pd.concat ([sonuc, sonuc2], axis=1) #Concat yapısı iki parametreden oluşur. axis ile düzenlenmesi gerçekleştirilir.Verilerin konumu belirlenir.
print(s)

s2=pd.concat([s,sonuc3], axis=1) #Concat yapısı iki parametreden oluşur. axis ile düzenlenmesi gerçekleştirilir.Verilerin konumu belirlenir.
print(s2)



# Veri kümesi eğitimleri için sütunların ayrıştırlması gerekmektedir. Bunun için veri kaynağı ayrıştırılır.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(s2, sonuc3, test_size=0.33, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X1_train = sc.fit_transform(X_train)
X1_test = sc.transform(X_test)










