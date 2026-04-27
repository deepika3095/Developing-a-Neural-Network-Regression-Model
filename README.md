# Developing a Neural Network Regression Model
## Date: 26-04-2025
## AIM
To develop a neural network regression model for the given dataset.

## THEORY
The objective of this experiment is to design, implement, and evaluate a Deep Learning–based Neural Network regression model to predict a continuous output variable from a given set of input features. The task is to preprocess the data, construct a neural network regression architecture, train the model using backpropagation and gradient descent, and evaluate its performance using appropriate regression metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R² score.

The experiment aims to understand the impact of different hyperparameters such as learning rate, number of hidden layers, and activation functions on model performance. It also focuses on improving the model’s generalization ability by applying techniques like normalization, regularization, and validation. Finally, the experiment helps in gaining practical knowledge of implementing deep learning models for real-world regression problems.

## Neural Network Model
<img width="1171" height="723" alt="image" src="https://github.com/user-attachments/assets/604fec6d-ccb9-486e-bdec-bd63746efdbb" />

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

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

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name: DEEPIKA R

### Register Number: 212223230038

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(1, 8)
        self.fc2=nn.Linear(8, 10)
        self.fc3=nn.Linear(10, 1)
        self.relu=nn.ReLU()
        self.history={'loss':[]}
  def forward(self, x):
        x=self.relu(self.fc1(x))
        x=self.relu(self.fc2(x))
        x=self.fc3(x)
        return x

lig=NeuralNet()
criterion=nn.MSELoss()
optimizer=optim.RMSprop(lig. parameters(), lr=0.001);

def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range (epochs):
        optimizer. zero_grad()
        loss=criterion(ai_brain(X_train), y_train)
        loss. backward()
        optimizer.step()
        lig .history['loss'].append(loss.item())
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
```

### Dataset Information
<img width="185" height="223" alt="image" src="https://github.com/user-attachments/assets/780900a2-90a9-4f8c-8a11-e5d533f23ae0" />
<img width="266" height="548" alt="image" src="https://github.com/user-attachments/assets/8aab50bc-9c31-493e-a0d1-8b378ad1a54f" />

## OUTPUT
<img width="400" height="265" alt="image" src="https://github.com/user-attachments/assets/6c3b487d-07b5-4380-a3b3-22d5179c6e0d" />

### Training Loss Vs Iteration Plot
<img width="657" height="517" alt="image" src="https://github.com/user-attachments/assets/4a8390d0-2e13-4118-8acc-d51a27369d00" />


### New Sample Data Prediction
<img width="292" height="42" alt="image" src="https://github.com/user-attachments/assets/a49188ef-d736-455c-8fb1-258c028bef4e" />


## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
