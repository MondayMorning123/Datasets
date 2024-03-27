import sys
print('Python: {}'.format(sys.version))
import scipy
print('scipy: {}'.format(scipy.__version__))
import numpy
print('numpy: {}'.format(numpy.__version__))
import matplotlib
print('matplotlib: {}'.format(matplotlib.__version__))
import pandas
print('pandas: {}'.format(pandas.__version__))
import sklearn
print('sklearn: {}'.format(sklearn.__version__))
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/MondayMorning123/Datasets/main/CarModels"
names = ['horsepower', 'doors', 'seats', 'cylinders', 'brand','models']
dataset = read_csv(url, names=names)


print(dataset.shape)
print(dataset.head())
print(dataset.describe())
print(dataset.groupby('brand').size())


scatter_matrix(dataset[['horsepower', 'cylinders', 'doors', 'seats']])
plt.show()
dataset.hist()
plt.show()

array = dataset.values
X = array[:,0:4]  
y = array[:,4]  
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)


models = [
    ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
    ('LDA', LinearDiscriminantAnalysis()),
    ('KNN', KNeighborsClassifier()),
    ('CART', DecisionTreeClassifier()),
    ('NB', GaussianNB()),
    ('SVM', SVC(gamma='auto'))
]


results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))


model = SVC(gamma='auto')
model.fit(X_train, Y_train)
predictions = model.predict(X_validation)


print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))

scatter_matrix(dataset[['doors', 'seats']])
plt.xlabel('Doors')
plt.ylabel('Seats')
plt.title('Relationship between Doors and Seats')
plt.show()

scatter_matrix(dataset[['cylinders', 'horsepower']])
plt.xlabel('Cylinders')
plt.ylabel('Horsepower')
plt.title('Relationship between Cylinders and Horsepower')
plt.show()

array = dataset.values
X = array[:,[0,2,3]]  
y = array[:,1]  
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

