import pandas as pd
import pickle as pkl
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
df = pd.read_csv('Naive-Bayes-Classification-Data.csv')
#print(df)
print(df.shape)
print(df.head(2))

X= df[['glucose','bloodpressure']]
y = df['diabetes']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state = 42)

model = GaussianNB()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
#print(y_pred)
print('Score :',model.score(X_test,y_test))
print(confusion_matrix(y_test,y_pred))


with open('Naive_Bayes.pkl', 'wb') as f:
    pkl.dump(model, f)


print(df['bloodpressure'].max())
print(df['bloodpressure'].min())