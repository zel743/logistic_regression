#Oziel Misael Velazquez Carrizales 746441 ITC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import numpy as np

df = pd.read_csv("datos-logisticos.csv")
df.head(2)

x1 = 'Productos-Lote'
x2 = 'Tiempo-Entrega'
y = 'Defectuoso'

x = df[[x1,x2]]
y = df[y]

model = LogisticRegression(solver='liblinear')
model.fit(x,y)

x_min, x_max = x[x1].min() - 0.5, x[x1].max() + 0.5
y_min, y_max = x[x2].min() - 0.5, x[x2].max() + 0.5
h = (x_max - x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(10,6))
plt.scatter(x[y == 0][x1], x[y == 0][x2], color='blue', label='0')
plt.scatter(x[y == 1][x1], x[y == 1][x2], color='red', label='1')
plt.contour(xx, yy, z, cmap=plt.cm.Paired)

plt.xlabel(x1)
plt.ylabel(x2)
plt.legend()
plt.show()


#apartir d aqui es el codigo para crear el escalador

scaler = StandardScaler()

x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=0)
model = LogisticRegression(solver='liblinear')
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
x_min, x_max = x_scaled[:, 0].min() - 1, x_scaled[:, 0].max() + 1
y_min, y_max = x_scaled[:, 1].min() - 1, x_scaled[:, 1].max() + 1

h = (x_max - x_min)/100

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

z = model.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)

plt.figure(figsize=(10,6))
plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.3)
plt.scatter(x_scaled[y == 0, 0], x_scaled[y == 0,1], color = 'blue', label='0')
plt.scatter(x_scaled[y == 1, 0], x_scaled[y == 1,1], color = 'red', label='1')
plt.xlabel(x1)
plt.ylabel(x2)
plt.legend()
plt.show()


#predecir

Variable_x1 = int(input("ingresa el valor primero: "))
Variable_x2 = int(input("ingresa el valor segundo: "))

new_example = pd.DataFrame([[Variable_x1, Variable_x2]], columns=['Productos-Lote', 'Tiempo-Entrega'])
new_example_scaled = scaler.transform(new_example)
prediction = model.predict(new_example_scaled)
result_phrase = "defectuoso" if prediction[0] == 1 else "no defectuoso"

print("Un producto que este dentro de un lote de", Variable_x1, "unidades y dure",Variable_x2 ,"minutos en su entrega, es problable que estará en estado:", result_phrase)