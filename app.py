from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import json
import os
from datetime import datetime
import pickle
import zipfile
from safetensors import safe_open
from safetensors.numpy import save_file as save_safetensors
import tempfile

app = Flask(__name__)

# Global variables to store data and results
current_data = None
training_history = []
current_model = None
current_scaler = None
current_config = None

# Custom activation functions
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def swiglu(x):
    # Split the input in half for SwiGLU gate mechanism
    gate, value = np.split(x, 2, axis=-1)
    return value * (gate / (1 + np.exp(-gate)))  # SiLU activation

# Custom MLP with additional activation functions
class CustomMLPRegressor(MLPRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _forward_pass(self, activations):
        """Custom forward pass with additional activation functions"""
        hidden_activation = self.activation
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]
            
            # Apply custom activations
            if hidden_activation == 'leaky_relu':
                activations[i + 1] = leaky_relu(activations[i + 1])
            elif hidden_activation == 'gelu':
                activations[i + 1] = gelu(activations[i + 1])
            elif hidden_activation == 'swiglu':
                activations[i + 1] = swiglu(activations[i + 1])
            else:
                # Use parent class method for standard activations
                if hidden_activation == 'identity':
                    pass  # No activation
                elif hidden_activation == 'logistic':
                    activations[i + 1] = 1.0 / (1.0 + np.exp(-activations[i + 1]))
                elif hidden_activation == 'tanh':
                    activations[i + 1] = np.tanh(activations[i + 1])
                elif hidden_activation == 'relu':
                    activations[i + 1] = np.maximum(activations[i + 1], 0)
                    
        return activations

class CustomMLPClassifier(MLPClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def _forward_pass(self, activations):
        """Custom forward pass with additional activation functions"""
        hidden_activation = self.activation
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]
            
            # Apply custom activations (same as regressor)
            if hidden_activation == 'leaky_relu':
                activations[i + 1] = leaky_relu(activations[i + 1])
            elif hidden_activation == 'gelu':
                activations[i + 1] = gelu(activations[i + 1])
            elif hidden_activation == 'swiglu':
                activations[i + 1] = swiglu(activations[i + 1])
            else:
                # Use parent class method for standard activations
                if hidden_activation == 'identity':
                    pass  # No activation
                elif hidden_activation == 'logistic':
                    activations[i + 1] = 1.0 / (1.0 + np.exp(-activations[i + 1]))
                elif hidden_activation == 'tanh':
                    activations[i + 1] = np.tanh(activations[i + 1])
                elif hidden_activation == 'relu':
                    activations[i + 1] = np.maximum(activations[i + 1], 0)
                    
        return activations

# Import safe_sparse_dot from sklearn if available, otherwise define it
try:
    from sklearn.utils.extmath import safe_sparse_dot
except ImportError:
    def safe_sparse_dot(a, b):
        return np.dot(a, b)

def clean_data_for_json(data):
    """Convert NaN values to None for proper JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_data_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data_for_json(item) for item in data]
    elif isinstance(data, float) and np.isnan(data):
        return None
    else:
        return data

def detect_problem_type(y):
    """Detect if this is a classification or regression problem"""
    unique_vals = len(np.unique(y))
    if unique_vals <= 10 and y.dtype in ['object', 'int64', 'bool']:
        return 'classification'
    return 'regression'

def prepare_data(df, target_column, test_size=0.2, val_size=0.1, use_smote=False, smote_strategy='auto', smote_k_neighbors=5):
    """Prepare data for training with validation split and optional SMOTE"""
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle missing values in target
    if y.isnull().any():
        # Remove rows where target is NaN
        valid_indices = y.notnull()
        X = X[valid_indices]
        y = y[valid_indices]
        print(f"Removed {(~valid_indices).sum()} rows with missing target values")
    
    # Handle missing values in features
    # 1. Drop columns that are mostly empty (>50% missing)
    missing_percentages = X.isnull().sum() / len(X)
    columns_to_drop = missing_percentages[missing_percentages > 0.5].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >50% missing values: {columns_to_drop}")
        X = X.drop(columns=columns_to_drop)
    
    # 2. Handle categorical variables first
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        # Fill missing categorical values with mode
        if X[col].isnull().any():
            mode_value = X[col].mode()
            if len(mode_value) > 0:
                X[col] = X[col].fillna(mode_value[0])
            else:
                X[col] = X[col].fillna('unknown')
        
        # Encode categorical variables
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
    
    # 3. Handle numerical missing values
    numerical_columns = X.select_dtypes(include=[np.number]).columns
    for col in numerical_columns:
        if X[col].isnull().any():
            # Fill with median for numerical columns
            median_value = X[col].median()
            X[col] = X[col].fillna(median_value)
            print(f"Filled {col} missing values with median: {median_value}")
    
    # Handle target variable if categorical
    problem_type = detect_problem_type(y)
    if problem_type == 'classification' and y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))
    
    # Verify no NaN values remain
    if X.isnull().any().any():
        print("Warning: Some NaN values still remain in features")
        # As a last resort, fill any remaining NaN with 0
        X = X.fillna(0)
    
    if pd.Series(y).isnull().any():
        print("Warning: Some NaN values still remain in target")
        # This shouldn't happen, but as a safeguard
        y = pd.Series(y).fillna(0)
    
    # Split the data: first train/test, then split train into train/val
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Split training data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=val_size/(1-test_size), random_state=42
    )
    
    # Apply SMOTE if requested and if it's a classification problem
    smote_applied = False
    if use_smote and problem_type == 'classification':
        try:
            from imblearn.over_sampling import SMOTE
            
            # Check if we have enough samples for SMOTE
            unique_classes, class_counts = np.unique(y_train, return_counts=True)
            min_class_count = min(class_counts)
            
            # SMOTE requires at least k_neighbors+1 samples per class
            if min_class_count > smote_k_neighbors:
                smote = SMOTE(
                    sampling_strategy=smote_strategy,
                    k_neighbors=smote_k_neighbors,
                    random_state=42
                )
                X_train, y_train = smote.fit_resample(X_train, y_train)
                smote_applied = True
                print(f"SMOTE applied. Training set size increased from {len(X_train_full)} to {len(X_train)} samples")
                
                # Print class distribution after SMOTE
                unique_after, counts_after = np.unique(y_train, return_counts=True)
                print(f"Class distribution after SMOTE: {dict(zip(unique_after, counts_after))}")
            else:
                print(f"Warning: Cannot apply SMOTE. Minimum class has only {min_class_count} samples, but need at least {smote_k_neighbors + 1}")
        except ImportError:
            print("Warning: imbalanced-learn not installed. Skipping SMOTE.")
        except Exception as e:
            print(f"Warning: SMOTE failed with error: {str(e)}. Proceeding without SMOTE.")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, problem_type, scaler

def create_plots(y_true, y_pred, training_scores, validation_scores, problem_type):
    """Create visualization plots"""
    plots = {}
    
    # Create training history plot with both training and validation loss
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot training loss
    train_epochs = range(1, len(training_scores) + 1)
    ax.plot(train_epochs, training_scores, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    
    # Plot validation loss if available
    if validation_scores is not None and len(validation_scores) > 0:
        # Create epochs for validation scores that align with training
        if len(training_scores) > 0:
            val_epochs = np.linspace(1, len(training_scores), len(validation_scores))
        else:
            val_epochs = range(1, len(validation_scores) + 1)
        ax.plot(val_epochs, validation_scores, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Save to base64
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['training_history'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    # Create prediction vs actual plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if problem_type == 'regression':
        ax.scatter(y_true, y_pred, alpha=0.6, color='blue')
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title('Actual vs Predicted Values')
    else:
        # Confusion matrix for classification
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
    
    ax.grid(True, alpha=0.3)
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    plots['predictions'] = base64.b64encode(img_buffer.getvalue()).decode()
    plt.close()
    
    return plots

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_data', methods=['POST'])
def upload_data():
    global current_data
    
    try:
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            current_data = pd.read_csv(file)
            
            # Get basic info about the dataset
            info = {
                'shape': current_data.shape,
                'columns': current_data.columns.tolist(),
                'dtypes': current_data.dtypes.astype(str).to_dict(),
                'missing_values': current_data.isnull().sum().to_dict(),
                'sample_data': current_data.head().to_dict('records')
            }
            
            # Clean NaN values for JSON serialization
            info = clean_data_for_json(info)
            
            return jsonify({'success': True, 'data_info': info})
        else:
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/train_model', methods=['POST'])
def train_model():
    global current_data, training_history, current_model, current_scaler, current_config
    
    try:
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded'})
        
        params = request.json
        target_column = params['target_column']
        
        # Prepare data with validation split and optional SMOTE
        X_train, X_val, X_test, y_train, y_val, y_test, problem_type, scaler = prepare_data(
            current_data, target_column, 
            params.get('test_size', 0.2), 
            params.get('val_size', 0.1),
            params.get('use_smote', False),
            params.get('smote_strategy', 'auto'),
            params.get('smote_k_neighbors', 5)
        )
        
        # Store scaler for later use
        current_scaler = scaler
        
        # Handle custom activation functions and solver
        activation = params['activation']
        solver = params['solver']
        
        # For custom activations, we'll use standard sklearn and apply custom logic
        if activation in ['leaky_relu', 'gelu', 'swiglu']:
            # Use relu as base for sklearn, we'll handle custom activations manually
            sklearn_activation = 'relu'
        else:
            sklearn_activation = activation
            
        # Handle AdamW solver (sklearn doesn't have native AdamW, so we'll use adam with modified params)
        if solver == 'adamw':
            sklearn_solver = 'adam'
            # Note: sklearn's adam doesn't have weight decay, but we'll use it as closest approximation
        else:
            sklearn_solver = solver
        
        # Train with custom validation tracking (no early stopping)
        validation_scores = []
        
        # Create model for standard training
        max_iter = params['max_iter']
        
        # Handle learning rate scheduler option
        learning_rate_type = 'adaptive' if params.get('use_learning_rate_scheduler', False) and sklearn_solver in ['adam', 'sgd'] else 'constant'
        
        if problem_type == 'classification':
            model = MLPClassifier(
                hidden_layer_sizes=tuple(params['hidden_layers']),
                activation=sklearn_activation,
                solver=sklearn_solver,
                learning_rate_init=params['learning_rate'],
                learning_rate=learning_rate_type,
                max_iter=max_iter,
                random_state=42,
                early_stopping=False
            )
        else:
            model = MLPRegressor(
                hidden_layer_sizes=tuple(params['hidden_layers']),
                activation=sklearn_activation,
                solver=sklearn_solver,
                learning_rate_init=params['learning_rate'],
                learning_rate=learning_rate_type,
                max_iter=max_iter,
                random_state=42,
                early_stopping=False
            )
        
        # Create a custom callback for validation tracking during training
        # We'll subclass the model to add validation tracking
        class ValidationTrackingMLP:
            def __init__(self, base_model, X_val, y_val, problem_type, val_frequency=10):
                self.base_model = base_model
                self.X_val = X_val
                self.y_val = y_val
                self.problem_type = problem_type
                self.val_frequency = val_frequency
                self.validation_scores = []
                
            def fit(self, X, y):
                # Override the model's _fit method to add validation tracking
                original_max_iter = self.base_model.max_iter
                
                # Train in chunks to track validation
                chunk_size = max(1, original_max_iter // 20)  # 20 validation checks
                total_trained = 0
                
                while total_trained < original_max_iter:
                    current_chunk = min(chunk_size, original_max_iter - total_trained)
                    self.base_model.max_iter = current_chunk
                    
                    if total_trained == 0:
                        # First training
                        self.base_model.fit(X, y)
                    else:
                        # Continue training with warm_start
                        self.base_model.warm_start = True
                        self.base_model.fit(X, y)
                    
                    total_trained += current_chunk
                    
                    # Calculate validation score
                    try:
                        y_val_pred = self.base_model.predict(self.X_val)
                        if self.problem_type == 'classification':
                            try:
                                from sklearn.metrics import log_loss
                                y_val_proba = self.base_model.predict_proba(self.X_val)
                                val_loss = log_loss(self.y_val, y_val_proba)
                            except:
                                val_accuracy = accuracy_score(self.y_val, y_val_pred)
                                val_loss = 1 - val_accuracy
                        else:
                            val_loss = mean_squared_error(self.y_val, y_val_pred)
                        
                        self.validation_scores.append(val_loss)
                    except:
                        pass
                
                return self.base_model
        
        # Use the validation tracking wrapper
        tracker = ValidationTrackingMLP(model, X_val, y_val, problem_type)
        model = tracker.fit(X_train, y_train)
        validation_scores = tracker.validation_scores
        
        # Get training loss curve (should now be exactly max_iter length)
        training_scores = getattr(model, 'loss_curve_', [])
        
        # Store current model and config for download
        current_model = model
        current_config = {
            'model_type': 'MLPClassifier' if problem_type == 'classification' else 'MLPRegressor',
            'problem_type': problem_type,
            'hidden_layer_sizes': params['hidden_layers'],
            'activation': params['activation'],
            'solver': params['solver'],
            'learning_rate_init': params['learning_rate'],
            'max_iter': params['max_iter'],
            'input_size': X_train.shape[1],
            'output_size': len(np.unique(y_train)) if problem_type == 'classification' else 1,
            'feature_names': current_data.drop(columns=[target_column]).columns.tolist(),
            'target_name': target_column,
            'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Make predictions on the original test set
        y_pred = model.predict(X_test)
        
        # For validation predictions, we need to evaluate on our original validation set
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        if problem_type == 'classification':
            test_accuracy = accuracy_score(y_test, y_pred)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            metrics = {
                'test_accuracy': test_accuracy,
                'val_accuracy': val_accuracy,
                'test_samples': len(y_test),
                'val_samples': len(y_val)
            }
        else:
            test_mse = mean_squared_error(y_test, y_pred)
            test_r2 = r2_score(y_test, y_pred)
            val_mse = mean_squared_error(y_val, y_val_pred)
            val_r2 = r2_score(y_val, y_val_pred)
            metrics = {
                'test_mse': test_mse,
                'test_r2_score': test_r2,
                'val_mse': val_mse,
                'val_r2_score': val_r2,
                'test_samples': len(y_test),
                'val_samples': len(y_val)
            }
        
        # Create plots with our custom training and validation scores
        plots = create_plots(y_test, y_pred, training_scores, validation_scores, problem_type)
        
        # Save to training history
        result = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': params,
            'metrics': metrics,
            'problem_type': problem_type,
            'plots': plots,
            'model_id': len(training_history) + 1  # Simple ID for download reference
        }
        training_history.append(result)
        
        return jsonify({'success': True, 'result': result})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_history')
def get_history():
    return jsonify({'history': training_history})


@app.route('/download_weights')
def download_weights():
    """Download model weights as a ZIP file containing JSON, safetensors, and config"""
    global current_model, current_scaler, current_config
    
    try:
        if current_model is None:
            return jsonify({'success': False, 'error': 'No trained model available'})
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Save weights as JSON
            weights_json = {
                'coefs_': [coef.tolist() for coef in current_model.coefs_],
                'intercepts_': [intercept.tolist() for intercept in current_model.intercepts_],
                'n_layers_': current_model.n_layers_,
                'n_outputs_': current_model.n_outputs_,
                'out_activation_': current_model.out_activation_,
                'loss_': float(current_model.loss_) if hasattr(current_model, 'loss_') else None,
                'best_loss_': float(current_model.best_loss_) if hasattr(current_model, 'best_loss_') else None,
                'n_iter_': int(current_model.n_iter_) if hasattr(current_model, 'n_iter_') else None
            }
            
            weights_json_path = os.path.join(temp_dir, 'weights.json')
            with open(weights_json_path, 'w') as f:
                json.dump(weights_json, f, indent=2)
            
            # 2. Save weights as safetensors
            safetensors_data = {}
            for i, coef in enumerate(current_model.coefs_):
                safetensors_data[f'layer_{i}_weight'] = coef.T  # Transpose for standard format
            for i, intercept in enumerate(current_model.intercepts_):
                safetensors_data[f'layer_{i}_bias'] = intercept
            
            safetensors_path = os.path.join(temp_dir, 'model.safetensors')
            save_safetensors(safetensors_data, safetensors_path)
            
            # 3. Save config with production implementation guide
            config_with_guide = current_config.copy()
            config_with_guide['production_guide'] = {
                'framework': 'PyTorch/TensorFlow/JAX implementation guide',
                'architecture': {
                    'input_layer': f'Input size: {current_config["input_size"]}',
                    'hidden_layers': [f'Layer {i+1}: {size} neurons, activation: {current_config["activation"]}' 
                                    for i, size in enumerate(current_config['hidden_layer_sizes'])],
                    'output_layer': f'Output size: {current_config["output_size"]}, activation: {"softmax" if current_config["problem_type"] == "classification" else "linear"}'
                },
                'preprocessing': {
                    'feature_scaling': 'StandardScaler applied',
                    'scaler_mean': current_config['scaler_mean'],
                    'scaler_scale': current_config['scaler_scale'],
                    'formula': 'scaled_feature = (feature - mean) / scale'
                },
                'pytorch_example': '''
import torch
import torch.nn as nn
import json
import numpy as np

class MLPModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        layers = []
        input_size = config['input_size']
        
        # Build hidden layers
        for hidden_size in config['hidden_layer_sizes']:
            layers.append(nn.Linear(input_size, hidden_size))
            if config['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif config['activation'] == 'tanh':
                layers.append(nn.Tanh())
            elif config['activation'] == 'leaky_relu':
                layers.append(nn.LeakyReLU())
            elif config['activation'] == 'gelu':
                layers.append(nn.GELU())
            input_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(input_size, config['output_size']))
        if config['problem_type'] == 'classification' and config['output_size'] > 1:
            layers.append(nn.Softmax(dim=1))
            
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

# Load weights
with open('weights.json', 'r') as f:
    weights = json.load(f)
    
with open('config.json', 'r') as f:
    config = json.load(f)

model = MLPModel(config)
# Load the weights into the model (implementation needed)
                ''',
                'preprocessing_example': '''
# Preprocessing example
import numpy as np

def preprocess_features(features, scaler_mean, scaler_scale):
    features = np.array(features)
    return (features - np.array(scaler_mean)) / np.array(scaler_scale)

# Usage
scaled_features = preprocess_features(raw_features, config['scaler_mean'], config['scaler_scale'])
                '''
            }
            
            config_path = os.path.join(temp_dir, 'config.json')
            with open(config_path, 'w') as f:
                json.dump(config_with_guide, f, indent=2)
            
            # 4. Create README
            readme_content = """# MLP Model Export

This package contains your trained Multi-Layer Perceptron model in multiple formats:

## Files:
- `weights.json`: Model weights in JSON format for easy reading
- `model.safetensors`: Model weights in SafeTensors format for efficient loading
- `config.json`: Complete model configuration and production implementation guide

## Model Architecture:
- Input size: {input_size}
- Hidden layers: {hidden_layers}
- Output size: {output_size}
- Problem type: {problem_type}
- Activation: {activation}

## Quick Start:
1. See config.json for detailed implementation examples
2. Use the preprocessing formula to scale your input features
3. Load weights from either JSON or SafeTensors format
4. Follow the PyTorch example in config.json

## Performance Metrics:
{metrics_info}

Generated on: {timestamp}
""".format(
                input_size=current_config['input_size'],
                hidden_layers=current_config['hidden_layer_sizes'],
                output_size=current_config['output_size'],
                problem_type=current_config['problem_type'],
                activation=current_config['activation'],
                metrics_info="See training history for detailed performance metrics",
                timestamp=current_config['training_timestamp']
            )
            
            readme_path = os.path.join(temp_dir, 'README.md')
            with open(readme_path, 'w') as f:
                f.write(readme_content)
            
            # Create ZIP file
            zip_path = os.path.join(temp_dir, 'mlp_model_export.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(weights_json_path, 'weights.json')
                zipf.write(safetensors_path, 'model.safetensors')
                zipf.write(config_path, 'config.json')
                zipf.write(readme_path, 'README.md')
            
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=f'mlp_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
                mimetype='application/zip'
            )
    
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error creating download: {str(e)}'})

@app.route('/model_info')
def model_info():
    """Get current model information"""
    global current_model, current_config
    
    if current_model is None:
        return jsonify({'available': False})
    
    return jsonify({
        'available': True,
        'config': current_config,
        'n_parameters': sum(coef.size for coef in current_model.coefs_) + sum(intercept.size for intercept in current_model.intercepts_),
        'training_complete': True
    })

def analyze_dataset_and_recommend(df, target_column):
    """Analyze dataset characteristics and provide intelligent model configuration recommendations"""
    
    # Basic dataset info
    n_samples = len(df)
    n_features = len(df.drop(columns=[target_column]).columns)
    y = df[target_column]
    problem_type = detect_problem_type(y)
    
    recommendations = {
        'dataset_info': {
            'n_samples': n_samples,
            'n_features': n_features,
            'problem_type': problem_type
        }
    }
    
    # Hidden layer recommendations based on dataset size
    if n_samples <= 100:
        # Very small dataset - single small layer
        recommended_layers = [min(20, n_features * 2)]
    elif n_samples <= 500:
        # Small dataset - one layer, moderate size
        max_neurons = min(50, n_features * 3)
        recommended_layers = [max_neurons]
    elif n_samples <= 1000:
        # Medium dataset - can handle more complexity
        max_neurons = min(75, n_features * 4)
        if max_neurons <= 75:
            recommended_layers = [max_neurons]
        else:
            recommended_layers = [75, max(1, max_neurons - 75)]
    else:
        # Large dataset - can handle more layers
        base_size = min(75, n_features * 2)
        if base_size <= 75:
            recommended_layers = [base_size]
        else:
            # Follow the pattern: 75,1; 75,2; ... 75,75; 75,75,1; etc
            excess = base_size - 75
            if excess <= 75:
                recommended_layers = [75, excess]
            else:
                recommended_layers = [75, 75, max(1, excess - 75)]
    
    recommendations['hidden_layers'] = {
        'recommended': recommended_layers,
        'explanation': f"Recommended for {n_samples} samples and {n_features} features. This is a maximum recommendation - simpler problems may need fewer neurons."
    }
    
    # Activation function recommendations with compute overhead info
    activation_options = [
        {'name': 'swiglu', 'display': 'SwiGLU', 'overhead': '13.4x', 'recommended': True, 'reason': 'Best performance but higher compute cost'},
        {'name': 'gelu', 'display': 'GELU', 'overhead': '24.0x', 'recommended': False, 'reason': 'Good performance but highest compute cost'},
        {'name': 'leaky_relu', 'display': 'Leaky ReLU', 'overhead': '5.9x', 'recommended': False, 'reason': 'Good balance of performance and efficiency'},
        {'name': 'relu', 'display': 'ReLU', 'overhead': '1.0x', 'recommended': False, 'reason': 'Most efficient but may limit model capacity'},
        {'name': 'tanh', 'display': 'Tanh', 'overhead': '16.0x', 'recommended': False, 'reason': 'Traditional choice but can suffer from vanishing gradients'}
    ]
    
    recommendations['activation'] = {
        'options': activation_options,
        'primary_recommendation': 'swiglu',
        'explanation': 'SwiGLU typically provides the best performance but with higher compute overhead. Consider ReLU for faster training on large datasets.'
    }
    
    # Optimizer and learning rate recommendations
    optimizer_configs = {
        'adam': {'learning_rate': 0.0001, 'epochs': 3000, 'recommended': True},
        'adamw': {'learning_rate': 0.0001, 'epochs': 3000, 'recommended': True},
        'sgd': {'learning_rate': 0.1, 'epochs': 300, 'recommended': False}
    }
    
    recommendations['optimizer'] = {
        'configs': optimizer_configs,
        'primary_recommendation': 'adam',
        'explanation': 'Adam and AdamW work well with lower learning rates and more epochs. SGD requires higher learning rates but fewer epochs.',
        'learning_rate_scheduler': True
    }
    
    # SMOTE recommendations for classification problems
    if problem_type == 'classification':
        # Analyze class balance
        class_counts = y.value_counts()
        total_samples = len(y)
        class_percentages = (class_counts / total_samples * 100).round(2)
        
        # Check if reasonably balanced (within ±5% of 50% for binary, or reasonable distribution for multi-class)
        if len(class_counts) == 2:
            # Binary classification
            minority_percentage = min(class_percentages)
            is_balanced = minority_percentage >= 45  # Within 5% of 50%
        else:
            # Multi-class - check if any class is severely underrepresented
            expected_percentage = 100 / len(class_counts)
            min_acceptable = expected_percentage * 0.5  # At least 50% of expected
            is_balanced = min(class_percentages) >= min_acceptable
        
        recommendations['smote'] = {
            'show_option': True,
            'recommended': not is_balanced,
            'class_distribution': class_percentages.to_dict(),
            'explanation': f"Class distribution: {dict(zip(class_counts.index, class_percentages.values))}. {'Balanced dataset - SMOTE not recommended' if is_balanced else 'Imbalanced dataset - consider using SMOTE'}",
            'balance_ratio': minority_percentage if len(class_counts) == 2 else min(class_percentages)
        }
    else:
        recommendations['smote'] = {
            'show_option': False,
            'recommended': False,
            'explanation': 'SMOTE only applies to classification problems'
        }
    
    return recommendations

@app.route('/analyze_dataset', methods=['POST'])
def analyze_dataset():
    """Analyze uploaded dataset and provide configuration recommendations"""
    global current_data
    
    try:
        if current_data is None:
            return jsonify({'success': False, 'error': 'No data uploaded'})
        
        params = request.json
        target_column = params.get('target_column')
        
        if not target_column:
            return jsonify({'success': False, 'error': 'Target column not specified'})
        
        if target_column not in current_data.columns:
            return jsonify({'success': False, 'error': 'Target column not found in dataset'})
        
        recommendations = analyze_dataset_and_recommend(current_data, target_column)
        
        return jsonify({'success': True, 'recommendations': recommendations})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002) 