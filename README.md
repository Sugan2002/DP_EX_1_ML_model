### Exp No: 01
### Date: 03.09.2022
# <p align="center"> Developing a Neural Network Regression Model</p>

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

First we can take the dataset based on one input value and some mathematical calculaus output value.next define the neaural network model in three layers.first layer have four neaurons and second layer have three neaurons,third layer have two neaurons.the neural network model take inuput and produce actual output using regression.

## Neural Network Model

![187081945-b3f6b59b-40bd-4db0-9970-f5d27cf3599d](https://user-images.githubusercontent.com/75235455/187084916-9a89437f-8611-418e-83cd-c60a7dbdabdd.png)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
```python
from google.colab import auth
import gspread
from google.auth import default
import pandas as pd

auth.authenticate_user()
creds, _ = default()
gc = gspread.authorize(creds)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

worksheet = gc.open('DLDataset').sheet1
rows = worksheet.get_all_values()
df = pd.DataFrame(rows[1:], columns=rows[0])
df.head(n=9)
df.dtypes
df = df.astype({'X':'float'})
df = df.astype({'Y':'float'})
df.dtypes
x=df[['X']].values
x
y=df[['Y']].values
y
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.33,random_state=50)
X_train
#to scale the input from 0 to 1
scaler=MinMaxScaler()
scaler.fit(X_train)
X_train_scaled=scaler.transform(X_train)
X_train_scaled
ai_brain=Sequential([
    Dense(2,activation='relu'),
    Dense(1)
])
ai_brain.compile(optimizer='rmsprop',loss='mse')
ai_brain.fit(x=X_train_scaled,y=Y_train,epochs=20000)
loss_df=pd.DataFrame(ai_brain.history.history)
loss_df.plot()
X_test
X_test_scaled=scaler.transform(X_test)
X_test_scaled
ai_brain.evaluate(X_test_scaled,Y_test)
input=[[10]]
input_scaled=scaler.transform(input)
input_scaled.shape
ai_brain.predict(input_scaled)
```

## Dataset Information

![Screenshot_403](https://user-images.githubusercontent.com/77089743/187085346-758def9d-6edd-42dc-8ccd-ece63b9e37c4.png)


## OUTPUT

### Training Loss Vs Iteration Plot


![Screenshot_400](https://user-images.githubusercontent.com/77089743/187085355-d79cb99d-89a0-409f-8a23-20226b1031e0.png)

### Test Data Root Mean Squared Error


![Screenshot_402](https://user-images.githubusercontent.com/77089743/187085359-d4b95cd0-59bf-4015-9694-1007ea6df235.png)


### New Sample Data Prediction

![Screenshot_401](https://user-images.githubusercontent.com/77089743/187085368-425feb37-204f-4204-8f03-b529b1017754.png)


## RESULT
Thus a Neural network for Regression model is Implemented.

