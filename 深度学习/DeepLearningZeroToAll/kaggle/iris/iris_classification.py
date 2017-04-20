import numpy as np
#
# from zipfile import ZipFile
#
# ZipFile('iris-species.zip').extractall()
#
# # df = pd.read_csv('Iris.csv',sep=',')
# # df['Species'] = df['Species'].map(lambda v: 0 if v=='Iris-setosa' else 1 if v=='Iris-versicolor' else 2)
# # df.to_csv('iris_ok.csv',sep=',')
#
xy = np.loadtxt('iris_ok.csv', delimiter=',')

x_data = xy[:, 2:-1]*1000
y_data = xy[:, -1]*1000

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(n_jobs=-1,max_iter=100000000,tol=1e-30)
clf.fit(x_data,y_data)
print(clf.score(x_data,y_data))

