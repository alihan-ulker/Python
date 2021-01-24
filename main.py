import pandas as pd
df=pd.read_csv('abalone.csv')
#butun ozniteliklerin min,max,ortalaması,standart sapması
#LENGTH
l1=df['Length'].min()
l2=df['Length'].max()
l3=df['Length'].mean()
l4=df['Length'].std()
print('--Length--')
print('-Minimum')
print(l1)
print('-Maksimum')
print(l2)
print('-Ortalama')
print(l3)
print('-Standart Sapma')
print(l4)
#DIAMETER
d1=df['Diameter'].min()
d2=df['Diameter'].max()
d3=df['Diameter'].mean()
d4=df['Diameter'].std()
print('--Diameter--')
print('-Minimum')
print(d1) 
print('-Maksimum')
print(d2)
print('-Ortalama')
print(d3)
print('-Standart Sapma')
print(d4)
#HEIGHT
h1=df['Height'].min()
h2=df['Height'].max()
h3=df['Height'].mean()
h4=df['Height'].std()
print('--Height--')
print('-Minimum')
print(h1) 
print('-Maksimum')
print(h2)
print('-Ortalama')
print(h3)
print('-Standart Sapma')
print(h4)
#WHOLE WEIGHT
w1=df['Whole weight'].min()
w2=df['Whole weight'].max()
w3=df['Whole weight'].mean()
w4=df['Whole weight'].std()
print('--Whole weight--')
print('-Minimum')
print(w1) 
print('-Maksimum')
print(w2)
print('-Ortalama')
print(w3)
print('-Standart Sapma')
print(w4)
#SHUCKED WEIGHT
s1=df['Shucked weight'].min()
s2=df['Shucked weight'].max()
s3=df['Shucked weight'].mean()
s4=df['Shucked weight'].std()
print('--Shucked weight--')
print('-Minimum')
print(s1) 
print('-Maksimum')
print(s2)
print('-Ortalama')
print(s3)
print('-Standart Sapma')
print(s4)
#VISCERA WEIGHT
v1=df['Viscera weight'].min()
v2=df['Viscera weight'].max()
v3=df['Viscera weight'].mean()
v4=df['Viscera weight'].std()
print('--Viscera weight--')
print('-Minimum')
print(v1) 
print('-Maksimum')
print(v2)
print('-Ortalama')
print(v3)
print('-Standart Sapma')
print(v4)
#SHELL WEIGHT
sh1=df['Shell weight'].min()
sh2=df['Shell weight'].max()
sh3=df['Shell weight'].mean()
sh4=df['Shell weight'].std()
print('--Shell weight--')
print('-Minimum')
print(sh1) 
print('-Maksimum')
print(sh2)
print('-Ortalama')
print(sh3)
print('-Standart Sapma')
print(sh4)
#RINGS
r1=df['Rings'].min()
r2=df['Rings'].max()
r3=df['Rings'].mean()
r4=df['Rings'].std()
print('--Rings--')
print('-Minimum')
print(r1) 
print('-Maksimum')
print(r2)
print('-Ortalama')
print(r3)
print('-Standart Sapma')
print(r4)

import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('abalone.csv')
#data olusturma
df.plot(kind='scatter',x='Diameter',y='Length')
#Grafik olusturma
plt.title('Length ve Diameter - Scatter Grafik')
plt.xlabel('Diameter')
plt.ylabel('Length')
#png olarak kaydetme
plt.savefig('Length and Diameter Scatter.png')
plt.show()

#data olusturma
df.plot(kind='scatter',x='Whole weight',y='Height')
#Grafik olusturma
plt.title('Height ve Whole Weight - Scatter Grafik')
plt.xlabel('Whole weight')
plt.ylabel('Height')
#png olarak kaydetme
plt.savefig('Height and Whole Weight Scatter.png')
plt.show()

#data olusturma
df.plot(kind='scatter',x='Shucked weight',y='Viscera weight')
#Grafik olusturma
plt.title('Shucked Weight ve Viscera Weight - Scatter Grafik')
plt.xlabel('Shucked weight')
plt.ylabel('Viscera weight')
#png olarak kaydetme
plt.savefig('Shucked Weight and Viscera Weight Scatter.png')
plt.show()

#data olusturma
df.plot(kind='scatter',x='Shell weight',y='Rings')
#Grafik olusturma
plt.title('Shell Weight ve Rings - Scatter Grafik')
plt.xlabel('Shell weight')
plt.ylabel('Rings')
#png olarak kaydetme
plt.savefig('Shell Weight and Rings Scatter.png')
plt.show()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#abalone verisetinin tum sutunlari secildi
column_names = ["Sex", "Length", "Diameter", "Height", "Whole weight", 
                "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
data = pd.read_csv("abalone.data", names=column_names)
print("Number of samples: %d" % len(data))
data.head()
#Sex sutununu string deger oldugu icin silindi
for label in "MFI":
    data[label] = data["Sex"] == label
del data["Sex"]
data.head()
#Ring sutunu degerleri 2 boyutlu dizi olusturmak icin silindi
y = data.Rings.values
del data["Rings"]
#Egitim sutununun %80 i rastgele secildi kalani test icin kullanildi
X = data.values.astype(np.float)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#Karar agaci olusturuldu
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
predicted_test_y = model.predict(X_test)
predicted_train_y = model.predict(X_train)
#Confusion Matrix olusturuldu
from sklearn.metrics import confusion_matrix
expected = [1, 1, 0, 1, 0, 0, 1, 0]
predicted = [1, 0, 0, 1, 0, 0, 1, 1]
results = confusion_matrix(expected, predicted)
print('Confusion Matrix')
print(results)
print('Alihan ÜLKER - 131001036')