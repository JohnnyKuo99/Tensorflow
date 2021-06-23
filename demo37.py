from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# prepare data
iris = datasets.load_iris()
features = iris.data
label = iris.target

regression1 = LogisticRegression()
svc1 = SVC(kernel='linear')
svc2 = SVC(kernel='poly')
svc3 = SVC(kernel='rbf')
tree1 = DecisionTreeClassifier()
knn2 = KNeighborsClassifier(n_neighbors=2)
knn3 = KNeighborsClassifier(n_neighbors=3)
knn4 = KNeighborsClassifier(n_neighbors=4)
classifiers = [regression1, svc1, svc2, svc3, tree1, knn2, knn3, knn4]
for c in classifiers:
    score = cross_val_score(c, features, label, cv=5)
    print(f"using {c}, \n ..score={score}, \n ..average={mean(score):.2f}")