from sklearn import datasets
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from numpy import mean

# prepare data
iris = datasets.load_iris()
features = iris.data
label = iris.target

regression1 = LogisticRegression()

score = cross_val_score(regression1, features, label, cv=5)
print(f"using {regression1}, score={score}, average={mean(score)}")