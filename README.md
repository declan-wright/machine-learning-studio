# Machine Learning Studio


## Supported Activation Functions

- **SwiGLU** 
- **GELU** 
- **Leaky ReLU**
- **ReLU**
- **Tanh** 

## Supported Data Types

- **Regression**
- **Classification**
- **Mixed Data**

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
