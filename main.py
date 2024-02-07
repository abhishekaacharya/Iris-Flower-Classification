import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
dataset = pd.read_csv("Iris.csv")
# print(dataset)
# a = sns.pairplot(dataset, hue="Species")
# plt.show()
inputs = dataset.iloc[:, 0:4]
outputs = dataset.iloc[:, 5]
(inputs_train, inputs_test, outputs_train, outputs_test) = train_test_split(inputs, outputs, test_size = 0.2)
# Model 1:Support Vector Machine Algorithm
model_svc = SVC()
model_svc.fit(inputs_train, outputs_train)

prediction1 = model_svc.predict(inputs_test)
print(accuracy_score(outputs_test, prediction1))

# Mo