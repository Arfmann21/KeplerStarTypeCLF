import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

print("============ BENVENUTO ============")
print("Questo è un classificatore che predice il tipo delle stelle osservate dal telescopio Kepler")
df = pd.read_csv('data.csv')
le = LabelEncoder()

df['Star color'] = le.fit_transform(df['Star color'].values)
df['Spectral Class'] = le.fit_transform(df['Spectral Class'].values)


X = df.drop(['Star type'], axis = 1).values
y = df['Star type'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

precision = precision_score(y_test, y_pred, average = 'weighted')
cross = cross_val_score(clf, X_test, y_test, cv = 5)
recall = recall_score(y_test, y_pred, average = 'weighted')

y_pred_str = []

for value in y_pred:
  if y_pred[value] == 0: y_pred_str.append("Brown Dwarf")
  elif y_pred[value] == 1: y_pred_str.append("Red Dwarf")
  elif y_pred[value] == 2: y_pred_str.append("Red Dwarf")
  elif y_pred[value] == 3: y_pred_str.append("White Dwarf")
  elif y_pred[value] == 4: y_pred_str.append("Main Sequence")
  elif y_pred[value] == 5: y_pred_str.append("Supergiant")
  elif y_pred[value] == 6: y_pred_str.append("Hypergiant")

for star in y_pred_str:
  print(star)
