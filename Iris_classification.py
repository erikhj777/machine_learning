from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'])

print(iris_dataset['target_names'])
print(iris_dataset['feature_names'])

#iris_dataset
iris_dataset['data'][:5] # preview of five, can remove to see all 150
iris_dataset['target']

import pandas as pd
# create dataframe from X_train and label the columns with the feature names from feature_names
iris_dataframe = pd.DataFrame(iris_dataset.data,
                              columns=iris_dataset.feature_names)

iris_dataframe = iris_dataframe.add_prefix('feature ') #this is easier than renaming columns!

iris_dataframe['species'] = [iris_dataset.target_names[x] 
                        for x 
                        in iris_dataset.target]

iris_feature_dataframe = iris_dataframe.loc[:,iris_dataframe.columns.str.startswith('feature')]                       
iris_dataframe.sample(5)

import seaborn as sns
sns.set_style("white")

with sns.hls_palette():
  sns.pairplot(iris_dataframe,
              hue = 'species');
  
  from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_feature_dataframe, #this is using simultaneorus, multivariate assignement
                                                    iris_dataframe.species,
                                                    random_state=0)

print(f'X_train shape: {X_train.shape}') #this is the subset of features held for training the algorithm
print(f'Y_train shape: {y_train.shape}')#and thsi is the subset of species in answer to the training features
print(f'X_test shape: {X_test.shape}') #the features data has four aspects
print(f'y_test shape: {y_test.shape}')#the species data is just the type of flower it is

X_train

#instantiate and train the model - this is the easy part!
from sklearn.tree import DecisionTreeClassifier

a_tree = DecisionTreeClassifier(random_state=0)
a_tree.fit(X_train, y_train);

#import numpy as np
X_new = np.array([[5, 2.9, 1, .2]]) #make up some random petal and sepal measurements
prediction = a_tree.predict(X_new) #pass thos made up measurements to the forest for a guess
prediction[0]

X_new = np.array([[5.2, 3.1, 1, .9]])
prediction = a_tree.predict(X_new)
prediction[0]

print(a_tree.score(X_train, y_train))
print(a_tree.score(X_test, y_test))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(a_tree, iris_feature_dataframe, iris_dataframe.species, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#can print the decision tree (when running in a notebook) to understand how the algorithm is making decisions
import pydot
from IPython.display import Image, display
from sklearn.tree import export_graphviz

dot_data = export_graphviz(a_tree,
                           out_file='tree.dot',
                           feature_names = iris_dataset['feature_names'],
                           class_names=iris_dataset['target_names'],
                           proportion=True,
                           filled=True,
                           rounded=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
Image('tree.png')

#use random forrest instead of just single decision tree to get a more accurate model
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, random_state=0)
forest.fit(X_train, y_train)
scores = cross_val_score(forest, iris_feature_dataframe, iris_dataframe.species, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#select a single tree to analyze
dot_data = export_graphviz(forest.estimators_[3],
                           out_file='tree.dot',
                           class_names=iris_dataset['target_names'],
                           proportion=True,
                           rounded=True,
                           filled=True)

(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
Image('tree.png')

#does a gradient-booted tree preform better?
from sklearn.ensemble import GradientBoostingClassifier
gbrt = GradientBoostingClassifier(random_state=0, n_estimators=5)
gbrt.fit(X_train, y_train)
scores = cross_val_score(gbrt, iris_feature_dataframe, iris_dataframe.species, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
