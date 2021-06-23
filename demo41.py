import seaborn as sb
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(sb.__version__)
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = np.array([iris.target_names[i] for i in iris.target])
#print(df)

sb.pairplot(df, hue='species')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df[iris.feature_names], iris.target,
                                                    test_size=0.5, stratify=iris.target)
forest1 = RandomForestClassifier(n_estimators=100, oob_score=True)
forest1.fit(X_train, y_train)
y_predict = forest1.predict(X_test)
accuracy = accuracy_score(y_test, y_predict)
print(f"oob score:{forest1.oob_score_}")
print(f"mean accuracy score:{accuracy}")

cm = pd.DataFrame(confusion_matrix(y_test, y_predict), columns=iris.target_names,
                  index=iris.target_names)
sb.heatmap(cm, annot=True)
plt.show()