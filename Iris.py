from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

iris_data=load_iris()

X=iris_data['data']
y=iris_data['target']

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)

knn = KNeighborsClassifier(n_neighbors=1)

knn.fit(X_train,y_train)

data = np.array([[4.4, 3, 1.3, 0.2]], dtype = float)
predictions = knn.predict(data)

print("Predicted Name:",iris_data['target_names'][predictions])
