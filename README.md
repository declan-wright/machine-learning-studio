# Machine Learning Studio


## Supported Activation Functions

- **SwiGLU** (Recommended) - State-of-the-art activation from modern transformers
- **GELU** - Gaussian Error Linear Unit, popular in modern ML
- **Leaky ReLU** - Improved version of ReLU with small negative slope
- **ReLU** - Classic Rectified Linear Unit
- **Tanh** - Hyperbolic tangent activation

## Supported Data Types

- **Regression**: Predicting continuous numerical values
- **Classification**: Predicting categories or classes
- **Mixed Data**: Handles both numerical and categorical features automatically

## Model Configuration Options

- **Hidden Layers**: Customize architecture (e.g., "100,50,25")
- **Learning Rate**: Fine-tune training speed
- **Solvers**: Adam, AdamW, L-BFGS, SGD
- **SMOTE**: Automatic class balancing for imbalanced datasets
- **Learning Rate Scheduling**: Adaptive learning rate reduction

## Project Structure

```
ml-studio/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/
│   ├── css/
│   │   └── style.css  # Styling
│   └── js/
│       └── app.js     # Frontend logic
└── templates/
    └── index.html     # Main UI template
```

## License

MIT License
