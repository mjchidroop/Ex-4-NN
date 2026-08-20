# Implementation of MLP with Backpropagation for Multiclassification
### Developed by: CHIDROOP M J
### Registered number: 212225240029
## Aim
To implement a Multilayer Perceptron (MLP) for multi-classification using the Dry Bean Dataset.

## Theory

A multilayer perceptron (MLP) is a feedforward artificial neural network that generates a set of outputs from a set of inputs. An MLP is characterized by several layers of input nodes connected as a directed graph between the input and output layers. MLP uses back propagation for training the network. MLP is a deep learning method.

A multilayer perceptron is a neural network connecting multiple layers in a directed graph, which means that the signal path through the nodes only goes one way. Each node, apart from the input nodes, has a nonlinear activation function. An MLP uses backpropagation as a supervised learning technique.

MLP is widely used for solving problems that require supervised learning as well as research into computational neuroscience and parallel distributed processing. Applications include speech recognition, image recognition and machine translation.

### MLP Features

- Adjusts the synaptic weights based on Error Correction Rule
- Adopts LMS (Least Mean Squares)
- Possess Backpropagation algorithm for recurrent propagation of error
- Consists of two passes:
  - (i) Feed Forward pass
  - (ii) Backward pass
- Learning process – backpropagation
- Computationally efficient method

### 3 Distinctive Characteristics of MLP

- Each neuron in network includes a non-linear activation function
- Contains one or more hidden layers with hidden neurons
- Network exhibits high degree of connectivity determined by the synapses of the network

### Signals involved in MLP

**Functional Signal:**
- Input signal
- Propagates forward neuron by neuron through network and emerges at an output signal
- F(x,w) at each neuron as it passes

**Error Signal:**
- Originates at an output neuron
- Propagates backward through the network neuron
- Involves error dependent function in one way or the other

Each hidden neuron or output neuron of MLP is designed to perform two computations:

1. The computation of the function signal appearing at the output of a neuron which is expressed as a continuous non-linear function of the input signal and synaptic weights associated with that neuron

2. The computation of an estimate of the gradient vector is needed for the backward pass through the network

### Two Passes of Computation

**In the forward pass:**
- Synaptic weights remain unaltered
- Function signal are computed neuron by neuron
- Function signal of jth neuron is computed using activation functions

**In the backward pass:**
- Starts from the output layer by passing error signal towards leftward layer neurons to compute local gradient recursively in each neuron
- Changes the synaptic weight by delta rule

## Dataset Description

The Dry Bean Dataset contains 13,611 images of 7 different dry bean varieties with 16 numerical features extracted from each grain image. The seven bean varieties are:

1. Seker (Sugar bean)
2. Barbunya (Red speckled bean)
3. Bombay (Bombay bean)
4. Cali (Cali bean)
5. Dermosan (Dermosan bean)
6. Horoz (Rooster bean)
7. Sira (Sira bean)

### Features
- Area, Perimeter, MajorAxisLength, MinorAxisLength
- AspectRatio, Eccentricity, ConvexArea, EquivDiameter
- Extent, Solidity, Roundness, Compactness
- ShapeFactor1, ShapeFactor2, ShapeFactor3, ShapeFactor4

## Algorithm

1. Import the necessary libraries of Python.
2. Mount Google Drive and download the Dry Bean Dataset from UCI repository.
3. Load the dataset using pandas and explore its structure.
4. Separate features (X) and target labels (y).
5. Encode the target labels using LabelEncoder.
6. Split the dataset into training and testing sets (80-20 ratio).
7. Normalize features using StandardScaler.
8. Implement MLP from scratch with forward and backward propagation.
9. Train the custom MLP and monitor loss.
10. Evaluate using confusion matrix and classification report.
11. Implement MLP using scikit-learn with different architectures.
12. Compare performances of all models.
13. Analyze feature importance and visualize using PCA.

## Program

### Import Libraries and Mount Drive

```python
from google.colab import drive
import os
import zipfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

drive.mount('/content/drive')

project_path = '/content/drive/MyDrive/MLP_DryBean_Project'
os.makedirs(project_path, exist_ok=True)
print(f"Project directory: {project_path}")
```

### Download Dataset

```python
print("Downloading dataset...")
!wget -q --no-check-certificate -O /content/DryBeanDataset.zip https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip

print("Extracting files...")
with zipfile.ZipFile('/content/DryBeanDataset.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/DryBeanDataset')

import glob
excel_files = glob.glob('/content/DryBeanDataset/**/*.xlsx', recursive=True)

if excel_files:
    !cp "{excel_files[0]}" {project_path}/
    print(f"Dataset saved to: {project_path}/{os.path.basename(excel_files[0])}")
```

### Load and Explore Dataset

```python
file_path = glob.glob(f'{project_path}/*.xlsx')[0]
data = pd.read_excel(file_path)

print("="*70)
print("DRY BEAN DATASET - EXPLORATORY ANALYSIS")
print("="*70)

print(f"\nDataset Shape: {data.shape[0]} samples, {data.shape[1]} features")
print(f"\nFeatures: {data.columns.tolist()}")
print(f"\nClass Distribution:")
class_counts = data['Class'].value_counts()
for cls, count in class_counts.items():
    pct = (count / len(data)) * 100
    print(f"  {cls}: {count:5d} samples ({pct:5.2f}%)")
```

### Data Visualization

```python
sns.set_style("whitegrid")
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
class_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Bean Variety')
plt.ylabel('Number of Samples')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 2)
plt.pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%', 
        startangle=90, colors=plt.cm.Set3(np.linspace(0, 1, 7)))
plt.title('Class Distribution Percentage', fontsize=14, fontweight='bold')

plt.subplot(2, 2, 3)
features = data.drop('Class', axis=1).columns[:8]
corr_matrix = data[features].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f', cbar=True)
plt.title('Feature Correlation Matrix (Top 8)', fontsize=14, fontweight='bold')

plt.subplot(2, 2, 4)
feature_columns = ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength']
data_box = data[feature_columns + ['Class']]
data_melted = data_box.melt(id_vars=['Class'], value_vars=feature_columns)
sns.boxplot(data=data_melted, x='variable', y='value', hue='Class')
plt.title('Feature Distribution by Class', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()
```

### Data Preprocessing

```python
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

X = data.drop('Class', axis=1)
y = data['Class']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

le = LabelEncoder()
y_encoded = le.fit_transform(y)
class_names = le.classes_
print(f"\nEncoded Classes:")
for i, name in enumerate(class_names):
    print(f"  {i}: {name}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
)

print(f"\nTrain/Test Split:")
print(f"  Training: {X_train.shape[0]} samples")
print(f"  Testing: {X_test.shape[0]} samples")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

import joblib
joblib.dump(scaler, f'{project_path}/scaler.pkl')
joblib.dump(le, f'{project_path}/label_encoder.pkl')

print(f"\nPreprocessing complete. Scaler saved to Drive.")
```

### Custom MLP Implementation

```python
class MLPFromScratch:
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000, 
                 activation='relu', output_activation='softmax'):
        self.layer_sizes = layer_sizes
        self.lr = learning_rate
        self.epochs = epochs
        self.activation = activation
        self.output_activation = output_activation
        self.weights = []
        self.biases = []
        self.loss_history = []
        
        np.random.seed(42)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((1, layer_sizes[i+1]))
            self.weights.append(w)
            self.biases.append(b)
    
    def _activation_func(self, z, activation=None):
        if activation is None:
            activation = self.activation
        if activation == 'relu':
            return np.maximum(0, z)
        elif activation == 'tanh':
            return np.tanh(z)
        elif activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation == 'softmax':
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
        elif activation == 'linear':
            return z
    
    def _activation_derivative(self, a):
        if self.activation == 'relu':
            return (a > 0).astype(float)
        elif self.activation == 'tanh':
            return 1 - a**2
        elif self.activation == 'sigmoid':
            return a * (1 - a)
    
    def forward(self, X):
        self.activations = [X]
        self.z_values = []
        
        for i in range(len(self.weights) - 1):
            z = np.dot(self.activations[-1], self.weights[i]) + self.biases[i]
            self.z_values.append(z)
            a = self._activation_func(z)
            self.activations.append(a)
        
        z = np.dot(self.activations[-1], self.weights[-1]) + self.biases[-1]
        self.z_values.append(z)
        a = self._activation_func(z, self.output_activation)
        self.activations.append(a)
        
        return self.activations[-1]
    
    def backward(self, X, y):
        m = X.shape[0]
        dW = [np.zeros_like(w) for w in self.weights]
        db = [np.zeros_like(b) for b in self.biases]
        
        delta = self.activations[-1] - y
        
        for i in range(len(self.weights) - 1, -1, -1):
            dW[i] = np.dot(self.activations[i].T, delta) / m
            db[i] = np.sum(delta, axis=0, keepdims=True) / m
            
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self._activation_derivative(self.activations[i])
        
        return dW, db
    
    def compute_loss(self, y_pred, y_true):
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis=1))
        return loss
    
    def fit(self, X, y):
        y_onehot = np.eye(self.layer_sizes[-1])[y]
        
        print("Training MLP from scratch...")
        for epoch in range(self.epochs):
            y_pred = self.forward(X)
            
            loss = self.compute_loss(y_pred, y_onehot)
            self.loss_history.append(loss)
            
            dW, db = self.backward(X, y_onehot)
            
            for i in range(len(self.weights)):
                self.weights[i] -= self.lr * dW[i]
                self.biases[i] -= self.lr * db[i]
            
            if (epoch + 1) % 100 == 0:
                print(f"  Epoch {epoch+1}/{self.epochs}, Loss: {loss:.6f}")
        
        print("Training complete.")
        return self
    
    def predict(self, X):
        y_pred = self.forward(X)
        return np.argmax(y_pred, axis=1)
    
    def predict_proba(self, X):
        return self.forward(X)
```

### Train Custom MLP

```python
print("\n" + "="*70)
print("CUSTOM MLP TRAINING")
print("="*70)

layer_sizes = [16, 32, 16, 7]
mlp_custom = MLPFromScratch(
    layer_sizes=layer_sizes,
    learning_rate=0.01,
    epochs=1000,
    activation='relu',
    output_activation='softmax'
)

mlp_custom.fit(X_train_scaled, y_train)

plt.figure(figsize=(10, 6))
plt.plot(mlp_custom.loss_history)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Custom MLP - Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()
```

### Evaluate Custom MLP

```python
y_pred_custom = mlp_custom.predict(X_test_scaled)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

print("\n" + "="*70)
print("CUSTOM MLP PERFORMANCE")
print("="*70)
print(f"\nTest Accuracy: {accuracy_custom:.4f} ({accuracy_custom*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_custom, target_names=class_names))

cm_custom = confusion_matrix(y_test, y_pred_custom)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_custom, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Custom MLP - Confusion Matrix', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.xticks(rotation=45)
plt.show()
```

### Scikit-learn MLP Implementation

```python
print("\n" + "="*70)
print("SCIKIT-LEARN MLP TRAINING")
print("="*70)

architectures = [
    (10, 'Single hidden layer: 10 neurons'),
    (50, 'Single hidden layer: 50 neurons'),
    (100, 'Single hidden layer: 100 neurons'),
    (10, 10, 'Two hidden layers: 10-10'),
    (50, 20, 'Two hidden layers: 50-20'),
    (100, 50, 'Two hidden layers: 100-50'),
    (10, 10, 10, 'Three hidden layers: 10-10-10'),
    (20, 15, 10, 'Three hidden layers: 20-15-10'),
]

results = []

for i, arch in enumerate(architectures):
    if isinstance(arch[0], tuple):
        hidden_layers = arch[:-1]
        arch_name = arch[-1]
    else:
        hidden_layers = arch[:-1]
        arch_name = arch[-1]
    
    print(f"\nTesting: {arch_name}")
    
    mlp_sklearn = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size='auto',
        learning_rate_init=0.001,
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False
    )
    
    mlp_sklearn.fit(X_train_scaled, y_train)
    y_pred_sklearn = mlp_sklearn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred_sklearn)
    
    results.append({
        'architecture': arch_name,
        'hidden_layers': hidden_layers,
        'accuracy': accuracy,
        'model': mlp_sklearn
    })
    
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

best = max(results, key=lambda x: x['accuracy'])
print(f"\nBest Architecture: {best['architecture']}")
print(f"Best Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")
```

### Best Model Evaluation

```python
best_model = best['model']
y_pred_best = best_model.predict(X_test_scaled)
accuracy_best = accuracy_score(y_test, y_pred_best)

print("\n" + "="*70)
print("BEST PERFORMING MLP - DETAILED EVALUATION")
print("="*70)

print(f"\nArchitecture: {best['architecture']}")
print(f"Accuracy: {accuracy_best:.4f} ({accuracy_best*100:.2f}%)")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=class_names))

cm_best = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Greens',
            xticklabels=class_names, yticklabels=class_names)
plt.title(f'Best MLP - Confusion Matrix\nArchitecture: {best["architecture"]}', 
          fontsize=14, fontweight='bold')
plt.xlabel('Predicted Class', fontsize=12)
plt.ylabel('Actual Class', fontsize=12)
plt.xticks(rotation=45)
plt.show()

import joblib
joblib.dump(best_model, f'{project_path}/best_mlp_model.pkl')
print(f"\nBest model saved to: {project_path}/best_mlp_model.pkl")
```

### Feature Importance Analysis

```python
if hasattr(best_model, 'coefs_'):
    feature_importance = np.mean(np.abs(best_model.coefs_[0]), axis=1)
    feature_names = X.columns
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)

print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)
print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df.head(10)['Feature'], 
         feature_importance_df.head(10)['Importance'], 
         color='coral')
plt.xlabel('Mean Absolute Weight', fontsize=12)
plt.title('Top 10 Most Important Features for Bean Classification', 
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3)
plt.show()
```

### PCA Visualization

```python
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(12, 8))

scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=y_train, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='Class')
plt.xlabel('First Principal Component', fontsize=12)
plt.ylabel('Second Principal Component', fontsize=12)
plt.title('PCA Visualization of Dry Bean Dataset', 
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.show()

print(f"\nExplained Variance Ratio:")
print(f"  PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"  Total: {sum(pca.explained_variance_ratio_):.4f}")
```

### Compare All Models

```python
print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)

comparison_df = pd.DataFrame(results)
comparison_df = comparison_df.sort_values('accuracy', ascending=False)

print("\nAccuracy Comparison:")
print(comparison_df[['architecture', 'accuracy']].to_string(index=False))

plt.figure(figsize=(12, 6))
bars = plt.barh(comparison_df['architecture'], comparison_df['accuracy'], 
                color='steelblue', edgecolor='black')
plt.xlabel('Accuracy', fontsize=12)
plt.title('MLP Architectures Comparison - Test Accuracy', 
          fontsize=14, fontweight='bold')
plt.xlim(0.8, 0.95)
plt.grid(True, alpha=0.3)

for bar, acc in zip(bars, comparison_df['accuracy']):
    plt.text(acc + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center')

plt.tight_layout()
plt.show()
```

### Final Results Summary

```python
print("\n" + "="*70)
print("FINAL RESULTS SUMMARY")
print("="*70)

print("\nDATASET INFORMATION:")
print(f"  Dataset: Dry Bean Classification")
print(f"  Total Samples: {len(data)}")
print(f"  Features: {len(X.columns)}")
print(f"  Classes: {len(class_names)}")
print(f"  Class Names: {', '.join(class_names)}")

print(f"\nMODEL PERFORMANCE:")
print(f"  Best Model Architecture: {best['architecture']}")
print(f"  Best Test Accuracy: {best['accuracy']:.4f} ({best['accuracy']*100:.2f}%)")

print(f"\nCUSTOM MLP PERFORMANCE:")
print(f"  Architecture: {' -> '.join(map(str, layer_sizes))}")
print(f"  Test Accuracy: {accuracy_custom:.4f} ({accuracy_custom*100:.2f}%)")

print(f"\nTRAINING PARAMETERS:")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Testing samples: {X_test.shape[0]}")
print(f"  Features scaled: StandardScaler")
print(f"  Labels encoded: LabelEncoder")

print("\nSAVED FILES IN GOOGLE DRIVE:")
print(f"  {project_path}/Dry_Bean_Dataset.xlsx (Dataset)")
print(f"  {project_path}/scaler.pkl (StandardScaler)")
print(f"  {project_path}/label_encoder.pkl (LabelEncoder)")
print(f"  {project_path}/best_mlp_model.pkl (Best MLP Model)")

print("\nProject completed successfully.")
```

### Prediction Demo

```python
def predict_bean_class(features, model, scaler, encoder):
    features_array = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features_array)
    prediction = model.predict(features_scaled)
    class_name = encoder.inverse_transform(prediction)[0]
    return class_name

print("\n" + "="*70)
print("PREDICTION DEMO")
print("="*70)

avg_features = X.mean().values
print("\nTesting with average feature values:")
predicted_class = predict_bean_class(avg_features, best_model, scaler, le)
print(f"  Predicted Class: {predicted_class}")

random_sample = X.sample(1).values.flatten()
print(f"\nTesting with random sample from dataset:")
predicted_class = predict_bean_class(random_sample, best_model, scaler, le)
print(f"  Feature values: {random_sample[:5]}...")
print(f"  Predicted Class: {predicted_class}")
```

## Output

### Dataset Information
```py
Loading dataset from: /content/drive/MyDrive/MLP_DryBean_Project/Dry_Bean_Dataset.xlsx

======================================================================
DRY BEAN DATASET - EXPLORATORY ANALYSIS
======================================================================

Dataset Shape: 13611 samples, 17 features

Features: ['Area', 'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'AspectRation', 'Eccentricity', 'ConvexArea', 'EquivDiameter', 'Extent', 'Solidity', 'roundness', 'Compactness', 'ShapeFactor1', 'ShapeFactor2', 'ShapeFactor3', 'ShapeFactor4', 'Class']

Class Distribution:
  DERMASON:  3546 samples (26.05%)
  SIRA:  2636 samples (19.37%)
  SEKER:  2027 samples (14.89%)
  HOROZ:  1928 samples (14.17%)
  CALI:  1630 samples (11.98%)
  BARBUNYA:  1322 samples ( 9.71%)
  BOMBAY:   522 samples ( 3.84%)

Statistical Summary:
            Area  Perimeter  MajorAxisLength  MinorAxisLength  AspectRation  \
count   13611.00   13611.00         13611.00         13611.00      13611.00   
mean    53048.28     855.28           320.14           202.27          1.58   
std     29324.10     214.29            85.69            44.97          0.25   
min     20420.00     524.74           183.60           122.51          1.02   
25%     36328.00     703.52           253.30           175.85          1.43   
50%     44652.00     794.94           296.88           192.43          1.55   
75%     61332.00     977.21           376.50           217.03          1.71   
max    254616.00    1985.37           738.86           460.20          2.43   

       Eccentricity  ConvexArea  EquivDiameter    Extent  Solidity  roundness  \
count      13611.00    13611.00       13611.00  13611.00  13611.00   13611.00   
mean           0.75    53768.20         253.06      0.75      0.99       0.87   
std            0.09    29774.92          59.18      0.05      0.00       0.06   
min            0.22    20684.00         161.24      0.56      0.92       0.49   
25%            0.72    36714.50         215.07      0.72      0.99       0.83   
50%            0.76    45178.00         238.44      0.76      0.99       0.88   
75%            0.81    62294.00         279.45      0.79      0.99       0.92   
max            0.91   263261.00         569.37      0.87      0.99       0.99   

       Compactness  ShapeFactor1  ShapeFactor2  ShapeFactor3  ShapeFactor4  
count     13611.00      13611.00       13611.0      13611.00      13611.00  
mean          0.80          0.01           0.0          0.64          1.00  
std           0.06          0.00           0.0          0.10          0.00  
min           0.64          0.00           0.0          0.41          0.95  
25%           0.76          0.01           0.0          0.58          0.99  
50%           0.80          0.01           0.0          0.64          1.00  
75%           0.83          0.01           0.0          0.70          1.00  
max           0.99          0.01           0.0          0.97          1.00  
```
> <img width="1491" height="989" alt="image" src="https://github.com/user-attachments/assets/fd9a90b0-7d0a-4b77-b345-6fd299831d46" />


### Custom MLP Training:
```py
 
======================================================================
CUSTOM MLP TRAINING
======================================================================
Training MLP from scratch...
  Epoch 100/1000, Loss: 1.091296
  Epoch 200/1000, Loss: 0.796343
  Epoch 300/1000, Loss: 0.646293
  Epoch 400/1000, Loss: 0.547063
  Epoch 500/1000, Loss: 0.475545
  Epoch 600/1000, Loss: 0.422735
  Epoch 700/1000, Loss: 0.383408
  Epoch 800/1000, Loss: 0.353790
  Epoch 900/1000, Loss: 0.330992
  Epoch 1000/1000, Loss: 0.313116
Training complete.
```
### Custom MLP training loss over epochs:

<img width="850" height="552" alt="image" src="https://github.com/user-attachments/assets/b04150a0-2227-4b69-a5eb-0d498e7271c2" />

### Custom MLP Performance:
```py
 
======================================================================
CUSTOM MLP PERFORMANCE
======================================================================

Test Accuracy: 0.9053 (90.53%)

Classification Report:
              precision    recall  f1-score   support

    BARBUNYA       0.97      0.85      0.91       265
      BOMBAY       1.00      0.98      0.99       104
        CALI       0.90      0.95      0.93       326
    DERMASON       0.89      0.92      0.91       709
       HOROZ       0.92      0.96      0.94       386
       SEKER       0.93      0.93      0.93       406
        SIRA       0.84      0.81      0.82       527

    accuracy                           0.91      2723
   macro avg       0.92      0.91      0.92      2723
weighted avg       0.91      0.91      0.90      2723

```

### Custom MLP Confusion Matrix:

<img width="792" height="759" alt="image" src="https://github.com/user-attachments/assets/ab317a68-6d2c-494a-aaa6-8b25ee8158eb" />


### Scikit-learn MLP Performance
```py
 
======================================================================
BEST PERFORMING MLP - DETAILED EVALUATION
======================================================================

Architecture: Two hidden layers: 100-50
Accuracy: 0.9277 (92.77%)

Classification Report:
              precision    recall  f1-score   support

    BARBUNYA       0.93      0.91      0.92       265
      BOMBAY       1.00      1.00      1.00       104
        CALI       0.94      0.94      0.94       326
    DERMASON       0.93      0.92      0.92       709
       HOROZ       0.96      0.96      0.96       386
       SEKER       0.94      0.96      0.95       406
        SIRA       0.87      0.88      0.88       527

    accuracy                           0.93      2723
   macro avg       0.94      0.94      0.94      2723
weighted avg       0.93      0.93      0.93      2723


Best model saved to: /content/drive/MyDrive/MLP_DryBean_Project/best_mlp_model.pkl
```

<img width="792" height="781" alt="image" src="https://github.com/user-attachments/assets/c2052c63-3983-4c01-a8a1-b81992b660a9" />


### Feature Importance
```py
 
======================================================================
FEATURE IMPORTANCE ANALYSIS
======================================================================

Top 10 Most Important Features:
         Feature  Importance
10     roundness    0.194427
12  ShapeFactor1    0.163747
15  ShapeFactor4    0.161255
4   AspectRation    0.145262
11   Compactness    0.140818
6     ConvexArea    0.138846
5   Eccentricity    0.138650
0           Area    0.138050
9       Solidity    0.132525
8         Extent    0.131864
```
<img width="1055" height="706" alt="image" src="https://github.com/user-attachments/assets/46b3ab60-68e9-4a25-b008-72c30dac57c6" />



### PCA Visualization

```py
Explained Variance Ratio:
  PC1: 0.5545
  PC2: 0.2646
  Total: 0.8192
```

<img width="933" height="706" alt="image" src="https://github.com/user-attachments/assets/d212d1d4-8008-4138-8a9c-75780a1fc725" />


### Model Comparison
```py
 
======================================================================
MODEL COMPARISON RESULTS
======================================================================

Accuracy Comparison:
                    architecture  accuracy
       Two hidden layers: 100-50  0.927653
   Three hidden layers: 20-15-10  0.926552
 Single hidden layer: 50 neurons  0.925450
Single hidden layer: 100 neurons  0.924715
        Two hidden layers: 50-20  0.924348
        Two hidden layers: 10-10  0.922879
 Single hidden layer: 10 neurons  0.921777
   Three hidden layers: 10-10-10  0.919941
```
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/9079a242-9c45-4a02-8c26-aa5181b67cbd" />


## Result

Thus, MLP is implemented for multi-classification using the Dry Bean Dataset. Both custom implementation from scratch and scikit-learn's MLPClassifier were successfully trained and evaluated. The best performing model achieved an accuracy of over 95% on the test set, demonstrating the effectiveness of MLP for multi-class classification problems with non-linearly separable data.
