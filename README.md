# Machine Learning Studio 🧠

A powerful web-based application for training multilayer perceptrons (neural networks) on your own datasets with an intuitive interface and advanced features.

## Features 🚀

- **Easy Data Upload**: Drag and drop CSV files or browse to upload
- **Intelligent Preprocessing**: Automatic handling of missing values, categorical encoding, and data cleaning
- **Advanced Neural Networks**: Custom activation functions including SwiGLU, GELU, and Leaky ReLU
- **Smart Recommendations**: AI-powered suggestions for optimal model configuration
- **Class Balancing**: SMOTE integration for handling imbalanced datasets
- **Real-time Visualization**: Training progress plots and prediction analysis
- **Model Export**: Download trained models in multiple formats (JSON, SafeTensors, Config)
- **Beautiful UI**: Modern, responsive interface with dark mode support

## Supported Activation Functions

- **SwiGLU** (Recommended) - State-of-the-art activation from modern transformers
- **GELU** - Gaussian Error Linear Unit, popular in modern ML
- **Leaky ReLU** - Improved version of ReLU with small negative slope
- **ReLU** - Classic Rectified Linear Unit
- **Tanh** - Hyperbolic tangent activation

## Installation 💻

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ml-studio.git
   cd ml-studio
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python app.py
   ```

5. **Open in browser**:
   Navigate to `http://localhost:5000`

## Usage 📊

1. **Upload Data**: Drag and drop or select a CSV file
2. **Select Target**: Choose which column you want to predict
3. **Configure Model**: Adjust neural network parameters or use AI recommendations
4. **Train**: Click train and watch real-time progress
5. **Analyze**: View performance metrics and prediction plots
6. **Export**: Download your trained model for later use

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

## Requirements

- Python 3.8+
- Flask 2.3.0+
- scikit-learn 1.3.0+
- pandas 2.0.0+
- matplotlib 3.7.0+
- See `requirements.txt` for complete list

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

## Contributing 🤝

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License 📝

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments 🙏

- Built with Flask and scikit-learn
- UI inspired by modern machine learning platforms
- Activation functions based on latest research in neural networks

---

**Happy Machine Learning!** 🎯 