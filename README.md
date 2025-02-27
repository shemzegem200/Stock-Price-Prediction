# Multilinear Regression

## Overview
Multilinear regression is a statistical technique used to model the relationship between one dependent variable (Y) and multiple independent variables (X₁, X₂, ..., Xₙ). It extends simple linear regression, which has only one independent variable, to cases where multiple predictors influence the outcome.

### **Equation**
Multilinear regression follows the general form:

\[ Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon \]

where:
- \( Y \) is the dependent variable (output),
- \( X_1, X_2, \dots, X_n \) are independent variables (predictors),
- \( \beta_0 \) is the intercept,
- \( \beta_1, \beta_2, \dots, \beta_n \) are regression coefficients,
- \( \epsilon \) is the error term.

The goal is to estimate \( \beta \) values such that the error term \( \epsilon \) is minimized, typically using the **Ordinary Least Squares (OLS)** method.

---

## Mathematical Proof
### **Matrix Representation**
We represent the multilinear regression model in matrix form:

\[ \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\epsilon} \]

where:
- \( \mathbf{Y} \) is an \( m \times 1 \) vector of dependent variable values,
- \( \mathbf{X} \) is an \( m \times (n+1) \) matrix (including a column of ones for the intercept),
- \( \boldsymbol{\beta} \) is an \( (n+1) \times 1 \) vector of coefficients,
- \( \boldsymbol{\epsilon} \) is an \( m \times 1 \) vector of error terms.

### **Estimating Coefficients using OLS**
To minimize the residual sum of squares (RSS):

\[ RSS = (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta}) \]

Differentiating with respect to \( \boldsymbol{\beta} \) and setting it to zero:

\[ \frac{\partial RSS}{\partial \boldsymbol{\beta}} = -2\mathbf{X}^T (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta}) = 0 \]

Solving for \( \boldsymbol{\beta} \):

\[ \boldsymbol{\beta} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} \]

---

## Comparison with Simple Linear Regression
| Feature                | Simple Linear Regression | Multilinear Regression |
|------------------------|------------------------|------------------------|
| Number of predictors  | One (X)            | Multiple (X₁, X₂, ..., Xₙ) |
| Equation form         | \( Y = \beta_0 + \beta_1 X + \epsilon \) | \( Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + ... + \beta_n X_n + \epsilon \) |
| Model complexity      | Simple relationship     | More complex interactions |
| Visualization         | 2D line                 | Hyperplane in n-dimensional space |
| Overfitting risk      | Lower                   | Higher if too many predictors |

---

## Diagrams
### **1. Simple Linear Regression (2D)**
A simple linear regression model can be visualized as a straight line in a 2D space:

```
      Y
      |    * (Data points)
      |   /
      |  /
      | /
      |/________ X
```

### **2. Multilinear Regression (3D)**
A multilinear regression model with two independent variables forms a **plane** in 3D space:

```
        Y
       /|
      / |
     /  | * (Data points)
    /___|________ X2
   /
  / X1
```

For more than two independent variables, the model forms a hyperplane in n-dimensional space, which is hard to visualize.

---

## Performance Metrics
To evaluate a multilinear regression model, we use the following metrics:

### **1. Mean Squared Error (MSE)**
\[ MSE = \frac{1}{m} \sum_{i=1}^{m} (Y_i - \hat{Y}_i)^2 \]
- Measures the average squared difference between actual and predicted values.
- Lower MSE indicates a better fit.

### **2. Root Mean Squared Error (RMSE)**
\[ RMSE = \sqrt{MSE} \]
- Provides error in the same units as \( Y \).

### **3. R-Squared (\( R^2 \))**
\[ R^2 = 1 - \frac{\sum (Y_i - \hat{Y}_i)^2}{\sum (Y_i - \bar{Y})^2} \]
- Measures the proportion of variance explained by the model.
- \( R^2 \) ranges from 0 to 1, where 1 indicates a perfect fit.

### **4. Adjusted R-Squared (\( R^2_{adj} \))**
\[ R^2_{adj} = 1 - \left( \frac{(1 - R^2)(m-1)}{m-n-1} \right) \]
- Adjusts \( R^2 \) for the number of predictors.
- Prevents overfitting by penalizing excessive independent variables.

---


# Multi-Layer Perceptron (MLP)

## Abstract  
A **Multi-Layer Perceptron (MLP)** is a type of artificial neural network that consists of an input layer, one or more hidden layers, and an output layer. It is fully connected, meaning each neuron in a layer is connected to every neuron in the next layer. MLPs use activation functions like **ReLU, sigmoid, or tanh** to introduce non-linearity, enabling them to learn complex patterns. They are trained using backpropagation, where gradients of the loss function are computed using the chain rule and updated via optimization algorithms like **Stochastic Gradient Descent (SGD) or Adam**.

---

## Mathematical Formulation

### 1. Forward Propagation
Let:
- \( \mathbf{x} \) be the input vector of size \( n \)
- \( \mathbf{W}^{(l)} \) be the weight matrix for layer \( l \)
- \( \mathbf{b}^{(l)} \) be the bias vector for layer \( l \)
- \( \mathbf{a}^{(l)} \) be the activation at layer \( l \)
- \( f(\cdot) \) be the activation function

For each layer \( l \), the neuron outputs are computed as:

\[
\mathbf{z}^{(l)} = \mathbf{W}^{(l)} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
\]

\[
\mathbf{a}^{(l)} = f(\mathbf{z}^{(l)})
\]

For the output layer (if performing classification):

\[
\mathbf{\hat{y}} = \sigma(\mathbf{W}^{(L)} \mathbf{a}^{(L-1)} + \mathbf{b}^{(L)})
\]

where \( \sigma(\cdot) \) is the **softmax function** for multi-class classification:

\[
\sigma(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
\]

---

### 2. Backpropagation (Gradient Computation)
To update weights, we compute gradients of the loss function \( \mathcal{L} \) with respect to the weights and biases.

For a given loss function \( \mathcal{L} \), the error at the output layer is:

\[
\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{(L)}}
\]

For hidden layers:

\[
\delta^{(l)} = \left( \mathbf{W}^{(l+1)T} \delta^{(l+1)} \right) \odot f'(\mathbf{z}^{(l)})
\]

Weight updates (using gradient descent):

\[
\mathbf{W}^{(l)} = \mathbf{W}^{(l)} - \eta \cdot \delta^{(l)} \mathbf{a}^{(l-1)T}
\]

\[
\mathbf{b}^{(l)} = \mathbf{b}^{(l)} - \eta \cdot \delta^{(l)}
\]

where:
- \( \odot \) is element-wise multiplication,
- \( f'(\mathbf{z}) \) is the derivative of the activation function,
- \( \eta \) is the learning rate.

---

## Diagrams

### MLP Architecture
```
Input Layer      Hidden Layer(s)          Output Layer
  x1  ────▶(W,b)──▶ ○ ───▶ ○ ───▶ ○───▶ y1 (output)
  x2  ────▶(W,b)──▶ ○ ───▶ ○ ───▶ ○───▶ y2
  x3  ────▶(W,b)──▶ ○ ───▶ ○ ───▶ ○───▶ y3
      ...         ...
```

### Backpropagation Flow
```
         [Loss Computation]
               ▲
               │
       Error Signal (δ)
               │
      Weight Adjustments
               │
    Gradient Descent Update
               │
           [Forward Pass]
```

---

## Performance Metrics
Depending on the task (classification or regression), the following metrics are used:

### **For Classification:**
1. **Accuracy**:  
   \[
   \text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
   \]

2. **Precision, Recall, F1-score**:
   \[
   \text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}
   \]
   \[
   \text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}
   \]
   \[
   \text{F1-score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
   \]

3. **Cross-Entropy Loss** (for softmax classification):
   \[
   \mathcal{L} = -\sum_{i} y_i \log(\hat{y}_i)
   \]

### **For Regression:**
1. **Mean Squared Error (MSE)**:
   \[
   MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   \]

2. **Mean Absolute Error (MAE)**:
   \[
   MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
   \]

3. **R-squared (\(R^2\))**:
   \[
   R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}
   \]

---
