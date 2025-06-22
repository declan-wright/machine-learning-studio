// Global variables
let currentData = null;
let trainingHistory = [];

// Utility functions
function showElement(id) {
    document.getElementById(id).style.display = 'block';
}

function hideElement(id) {
    document.getElementById(id).style.display = 'none';
}

function showAlert(message, type = 'info') {
    // Create a modern notification that fits the new design
    const alertDiv = document.createElement('div');
    alertDiv.className = `modern-alert alert-${type}`;
    alertDiv.innerHTML = `
        <div class="alert-content">
            <i class="fas ${getAlertIcon(type)}"></i>
            <span>${message}</span>
        </div>
        <button type="button" class="alert-close" onclick="this.parentElement.remove()">
            <i class="fas fa-times"></i>
        </button>
    `;
    
    // Style the alert
    alertDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: var(--box-bg);
        border: 1px solid var(--border-color);
        border-radius: var(--radius-lg);
        padding: var(--spacing-md);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
        z-index: 2000;
        min-width: 300px;
        max-width: 500px;
        opacity: 0;
        transform: translateX(100%);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--spacing-sm);
    `;
    
    document.body.appendChild(alertDiv);
    
    // Animate in
    setTimeout(() => {
        alertDiv.style.opacity = '1';
        alertDiv.style.transform = 'translateX(0)';
    }, 10);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.style.opacity = '0';
            alertDiv.style.transform = 'translateX(100%)';
            setTimeout(() => alertDiv.remove(), 300);
        }
    }, 5000);
}

function getAlertIcon(type) {
    switch(type) {
        case 'success': return 'fa-check-circle';
        case 'danger': return 'fa-exclamation-triangle';
        case 'warning': return 'fa-exclamation-circle';
        default: return 'fa-info-circle';
    }
}

// Data management functions
function uploadData(file) {
    if (!file) {
        showAlert('Please select a CSV file first', 'warning');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show uploading state
    const uploadArea = document.getElementById('uploadArea');
    uploadArea.classList.add('uploading');
    
    fetch('/upload_data', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        uploadArea.classList.remove('uploading');
        if (data.success) {
            currentData = data.data_info;
            displayDataInfo(data.data_info);
            populateTargetColumns(data.data_info.columns);
            uploadArea.classList.add('uploaded');
            showAlert('Data uploaded successfully! 🎉', 'success');
        } else {
            showAlert('Error uploading data: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        uploadArea.classList.remove('uploading');
        showAlert('Error: ' + error.message, 'danger');
    });
}

// Set up drag and drop and file input functionality
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('dataFile');
    
    // Prevent default drag behaviors
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
        document.body.addEventListener(eventName, preventDefaults, false);
    });
    
    // Highlight drop area when item is dragged over it
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    // Handle dropped files
    uploadArea.addEventListener('drop', handleDrop, false);
    
    // Handle file input change
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            uploadData(file);
        }
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight(e) {
        uploadArea.classList.add('dragover');
    }
    
    function unhighlight(e) {
        uploadArea.classList.remove('dragover');
    }
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            const file = files[0];
            if (file.type === 'text/csv' || file.name.endsWith('.csv')) {
                uploadData(file);
            } else {
                showAlert('Please select a CSV file', 'warning');
            }
        }
    }
});

function displayDataInfo(dataInfo) {
    const dataDetails = document.getElementById('dataDetails');
    
    dataDetails.innerHTML = `
        <div class="data-info-item">
            <span class="data-info-label">Rows:</span>
            <span class="data-info-value">${dataInfo.shape[0].toLocaleString()}</span>
        </div>
        <div class="data-info-item">
            <span class="data-info-label">Columns:</span>
            <span class="data-info-value">${dataInfo.shape[1]}</span>
        </div>
        <div class="data-info-item">
            <span class="data-info-label">Missing Values:</span>
            <span class="data-info-value">${Object.values(dataInfo.missing_values).reduce((a, b) => a + b, 0).toLocaleString()}</span>
        </div>
    `;
    
    const dataInfoElement = document.getElementById('dataInfo');
    dataInfoElement.style.display = 'block';
    // Add smooth show animation
    setTimeout(() => {
        dataInfoElement.classList.add('show');
    }, 100);
}

function populateTargetColumns(columns) {
    const targetSelect = document.getElementById('targetColumn');
    targetSelect.innerHTML = '<option value="">Select target column...</option>';
    
    columns.forEach(column => {
        const option = document.createElement('option');
        option.value = column;
        
        // Truncate very long column names for display
        const maxLength = 50;
        const displayText = column.length > maxLength 
            ? column.substring(0, maxLength) + '...' 
            : column;
        
        option.textContent = displayText;
        option.title = column; // Full name as tooltip
        targetSelect.appendChild(option);
    });
    
    // Add event listener for target column change to get recommendations
    targetSelect.addEventListener('change', function() {
        if (this.value) {
            getRecommendations(this.value);
        } else {
            hideRecommendations();
        }
    });
}

let currentRecommendations = null;

function getRecommendations(targetColumn) {
    if (!currentData) return;
    
    fetch('/analyze_dataset', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ target_column: targetColumn })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            currentRecommendations = data.recommendations;
            displayRecommendations(data.recommendations);
        } else {
            showAlert('Error getting recommendations: ' + data.error, 'warning');
        }
    })
    .catch(error => {
        console.error('Error getting recommendations:', error);
    });
}

function displayRecommendations(recommendations) {
    const section = document.getElementById('recommendationsSection');
    
    // Just show the recommendations section with the apply button, no detailed content
    section.style.display = 'block';
    setTimeout(() => section.classList.add('fade-in'), 100);
    
    // Update individual recommendation hints
    updateRecommendationHints(recommendations);
    
    // Show/hide SMOTE section based on problem type
    const smoteSection = document.getElementById('smoteSection');
    if (recommendations.smote.show_option) {
        smoteSection.style.display = 'block';
        updateSmoteRecommendation(recommendations.smote);
    } else {
        smoteSection.style.display = 'none';
    }
}

function updateRecommendationHints(recommendations) {
    // Hidden layers hint
    const hiddenLayersHint = document.getElementById('hiddenLayersRec');
    hiddenLayersHint.innerHTML = `<i class="fa-solid fa-lightbulb"></i> Recommended: [${recommendations.hidden_layers.recommended.join(', ')}] for your dataset size`;
    hiddenLayersHint.className = 'recommendation-hint recommended';
    hiddenLayersHint.style.display = 'block';
    
    // Activation hint
    const activationHint = document.getElementById('activationRec');
    const primaryActivation = recommendations.activation.options.find(opt => opt.name === recommendations.activation.primary_recommendation);
    activationHint.innerHTML = `<i class="fa-solid fa-lightbulb"></i> Recommended: ${primaryActivation.display} - ${primaryActivation.reason}`;
    activationHint.className = 'recommendation-hint recommended';
    activationHint.style.display = 'block';
    
    // Learning rate and epochs hints
    const learningRateHint = document.getElementById('learningRateRec');
    const epochsHint = document.getElementById('epochsRec');
    
    // Update based on currently selected solver
    const currentSolver = document.getElementById('solver').value;
    const config = recommendations.optimizer.configs[currentSolver];
    
    if (config) {
        learningRateHint.innerHTML = `<i class="fa-solid fa-lightbulb"></i> Recommended for ${currentSolver.toUpperCase()}: ${config.learning_rate}`;
        learningRateHint.className = 'recommendation-hint recommended';
        learningRateHint.style.display = 'block';
        
        epochsHint.innerHTML = `<i class="fa-solid fa-lightbulb"></i> Recommended for ${currentSolver.toUpperCase()}: ${config.epochs} epochs`;
        epochsHint.className = 'recommendation-hint recommended';
        epochsHint.style.display = 'block';
    }
}

function updateSmoteRecommendation(smoteRec) {
    const smoteHint = document.getElementById('smoteRec');
    if (smoteRec.recommended) {
        smoteHint.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Dataset appears imbalanced (${smoteRec.balance_ratio.toFixed(1)}% minority class) - SMOTE recommended`;
        smoteHint.className = 'recommendation-hint warning';
    } else {
        smoteHint.innerHTML = `<i class="fa-solid fa-check-circle"></i> Dataset appears balanced (${smoteRec.balance_ratio.toFixed(1)}% minority class) - SMOTE not needed`;
        smoteHint.className = 'recommendation-hint recommended';
    }
    smoteHint.style.display = 'block';
}

function hideRecommendations() {
    const section = document.getElementById('recommendationsSection');
    section.style.display = 'none';
    section.classList.remove('fade-in');
    
    // Hide all recommendation hints
    const hints = document.querySelectorAll('.recommendation-hint');
    hints.forEach(hint => hint.style.display = 'none');
    
    // Hide SMOTE section
    document.getElementById('smoteSection').style.display = 'none';
}

function applyRecommendations() {
    if (!currentRecommendations) return;
    
    // Apply hidden layers
    document.getElementById('hiddenLayers').value = currentRecommendations.hidden_layers.recommended.join(',');
    
    // Apply activation function
    document.getElementById('activation').value = currentRecommendations.activation.primary_recommendation;
    
    // Apply optimizer settings based on current solver
    const solver = document.getElementById('solver').value;
    const config = currentRecommendations.optimizer.configs[solver];
    if (config) {
        document.getElementById('learningRate').value = config.learning_rate;
        document.getElementById('maxIter').value = config.epochs;
    }
    
    // Apply SMOTE recommendation
    if (currentRecommendations.smote.show_option) {
        document.getElementById('useSmote').checked = currentRecommendations.smote.recommended;
        
        // Trigger SMOTE options display if recommended
        if (currentRecommendations.smote.recommended) {
            const smoteOptions = document.getElementById('smoteOptions');
            smoteOptions.style.display = 'block';
            smoteOptions.classList.add('fade-in');
        }
    }
    
    // Enable learning rate scheduler for Adam/AdamW
    if (['adam', 'adamw'].includes(solver)) {
        document.getElementById('useLearningRateScheduler').checked = true;
    }
    
    showAlert('Recommendations applied! 🎯', 'success');
}

// Model training functions
function trainModel() {
    if (!currentData) {
        showAlert('Please upload data first', 'warning');
        return;
    }
    
    const targetColumn = document.getElementById('targetColumn').value;
    if (!targetColumn) {
        showAlert('Please select a target column', 'warning');
        return;
    }
    
    // Get parameters from form
    const hiddenLayersText = document.getElementById('hiddenLayers').value;
    const hiddenLayers = hiddenLayersText.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
    
    if (hiddenLayers.length === 0) {
        showAlert('Please enter valid hidden layer sizes', 'warning');
        return;
    }
    
    const params = {
        target_column: targetColumn,
        hidden_layers: hiddenLayers,
        activation: document.getElementById('activation').value,
        solver: document.getElementById('solver').value,
        learning_rate: parseFloat(document.getElementById('learningRate').value),
        max_iter: parseInt(document.getElementById('maxIter').value),
        test_size: parseFloat(document.getElementById('testSize').value),
        use_smote: document.getElementById('useSmote').checked,
        smote_strategy: document.getElementById('smoteStrategy').value,
        smote_k_neighbors: parseInt(document.getElementById('smoteKNeighbors').value),
        use_learning_rate_scheduler: document.getElementById('useLearningRateScheduler').checked
    };
    
    // Show modern loading overlay with animations
    hideElement('noResults');
    hideElement('resultsContainer');
    showElement('loadingSpinner');
    
    // Animate loading text
    setTimeout(() => {
        const loadingTexts = document.querySelectorAll('.loading-text');
        loadingTexts.forEach((text, index) => {
            setTimeout(() => {
                text.classList.add('show');
            }, index * 1000);
        });
    }, 500);
    
    fetch('/train_model', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(params)
    })
    .then(response => response.json())
    .then(data => {
        hideElement('loadingSpinner');
        
        if (data.success) {
            displayResults(data.result);
            addToHistory(data.result);
            showAlert('Model trained successfully! 🎉', 'success');
            
            // Show results with smooth animation
            const resultsContainer = document.getElementById('resultsContainer');
            resultsContainer.style.display = 'block';
            setTimeout(() => {
                resultsContainer.classList.add('show');
            }, 100);
        } else {
            showElement('noResults');
            showAlert('Error training model: ' + data.error, 'danger');
        }
    })
    .catch(error => {
        hideElement('loadingSpinner');
        showElement('noResults');
        showAlert('Error: ' + error.message, 'danger');
    });
}

function displayResults(result) {
    // Display metrics in combined format
    const metricsDisplay = document.getElementById('metricsDisplay');
    let metricsHtml = '';
    
    if (result.problem_type === 'classification') {
        metricsHtml = `
            <div class="metric-item combined">
                <span class="metric-value">${(result.metrics.test_accuracy * 100).toFixed(2)}%</span>
                <span class="metric-label">Test Accuracy</span>
            </div>
            <div class="metric-item combined">
                <span class="metric-value">${(result.metrics.val_accuracy * 100).toFixed(2)}%</span>
                <span class="metric-label">Validation Accuracy</span>
            </div>
        `;
    } else {
        metricsHtml = `
            <div class="metric-item combined">
                <div class="metric-line">
                    <span class="metric-value">${result.metrics.test_mse.toFixed(4)} MSE</span>
                </div>
                <div class="metric-line">
                    <span class="metric-value">${(result.metrics.test_r2_score * 100).toFixed(2)}% R² Score</span>
                </div>
                <span class="metric-label">Test Results</span>
            </div>
            <div class="metric-item combined">
                <div class="metric-line">
                    <span class="metric-value">${result.metrics.val_mse.toFixed(4)} MSE</span>
                </div>
                <div class="metric-line">
                    <span class="metric-value">${(result.metrics.val_r2_score * 100).toFixed(2)}% R² Score</span>
                </div>
                <span class="metric-label">Validation Results</span>
            </div>
        `;
    }
    
    metricsDisplay.innerHTML = metricsHtml;
    
    // Display configuration
    const configDisplay = document.getElementById('configDisplay');
    let smoteInfo = '';
    if (result.parameters.use_smote && result.problem_type === 'classification') {
        smoteInfo = `
            <div class="config-item">
                <strong>SMOTE:</strong> ${result.parameters.smote_strategy} strategy, ${result.parameters.smote_k_neighbors} neighbors
            </div>
        `;
    }
    
    configDisplay.innerHTML = `
        <div class="config-item">
            <strong>Hidden Layers:</strong> [${result.parameters.hidden_layers.join(', ')}]
        </div>
        <div class="config-item">
            <strong>Activation:</strong> ${result.parameters.activation}
        </div>
        <div class="config-item">
            <strong>Solver:</strong> ${result.parameters.solver}
        </div>
        <div class="config-item">
            <strong>Learning Rate:</strong> ${result.parameters.learning_rate}${result.parameters.use_learning_rate_scheduler ? ' (with scheduler)' : ''}
        </div>
        <div class="config-item">
            <strong>Max Iterations:</strong> ${result.parameters.max_iter}
        </div>
        <div class="config-item">
            <strong>Problem Type:</strong> ${result.problem_type}
        </div>
        ${smoteInfo}
    `;
    
    // Display plots
    document.getElementById('trainingPlot').src = 'data:image/png;base64,' + result.plots.training_history;
    document.getElementById('predictionsPlot').src = 'data:image/png;base64,' + result.plots.predictions;
    
    showElement('resultsContainer');
}

function addToHistory(result) {
    trainingHistory.unshift(result); // Add to beginning
    updateHistoryDisplay();
}

function updateHistoryDisplay() {
    const historyContainer = document.getElementById('historyContainer');
    
    if (trainingHistory.length === 0) {
        historyContainer.innerHTML = `
            <div class="text-center text-muted">
                <i class="fas fa-history fa-2x mb-3"></i>
                <p>No training history yet.</p>
            </div>
        `;
        return;
    }
    
    let historyHtml = '';
    trainingHistory.forEach((result, index) => {
        let metricsHtml = '';
        if (result.problem_type === 'classification') {
            metricsHtml = `
                <span class="history-metric">Test: ${(result.metrics.test_accuracy * 100).toFixed(2)}%</span>
                <span class="history-metric">Val: ${(result.metrics.val_accuracy * 100).toFixed(2)}%</span>
            `;
        } else {
            metricsHtml = `
                <span class="history-metric">Test: ${result.metrics.test_mse.toFixed(4)} MSE, ${(result.metrics.test_r2_score * 100).toFixed(2)}% R²</span>
                <span class="history-metric">Val: ${result.metrics.val_mse.toFixed(4)} MSE, ${(result.metrics.val_r2_score * 100).toFixed(2)}% R²</span>
            `;
        }
        
        historyHtml += `
            <div class="history-item fade-in">
                <div class="history-timestamp">
                    <i class="fas fa-clock me-1"></i>
                    ${result.timestamp}
                </div>
                <div class="row">
                    <div class="col-md-8">
                        <strong>Configuration:</strong> 
                        [${result.parameters.hidden_layers.join(', ')}] layers, 
                        ${result.parameters.activation} activation, 
                        ${result.parameters.solver} solver
                    </div>
                    <div class="col-md-4">
                        <div class="history-metrics">
                            ${metricsHtml}
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    historyContainer.innerHTML = historyHtml;
}

function clearResults() {
    hideElement('resultsContainer');
    showElement('noResults');
    trainingHistory = [];
    updateHistoryDisplay();
    showAlert('Results cleared', 'info');
}

function downloadWeights() {
    // Check if model is available
    fetch('/model_info')
        .then(response => response.json())
        .then(data => {
            if (data.available) {
                // Create a temporary link element and trigger download
                const link = document.createElement('a');
                link.href = '/download_weights';
                link.download = `mlp_model_${new Date().toISOString().slice(0,19).replace(/:/g, '-')}.zip`;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                
                showAlert('Model weights download started! 📦', 'success');
            } else {
                showAlert('No trained model available for download. Please train a model first.', 'warning');
            }
        })
        .catch(error => {
            showAlert('Error checking model availability: ' + error.message, 'danger');
        });
}

// Event listeners
document.addEventListener('DOMContentLoaded', function() {
    // Form submission
    document.getElementById('trainingForm').addEventListener('submit', function(e) {
        e.preventDefault();
        trainModel();
    });
    
    // SMOTE checkbox toggle
    document.getElementById('useSmote').addEventListener('change', function() {
        const smoteOptions = document.getElementById('smoteOptions');
        if (this.checked) {
            smoteOptions.style.display = 'block';
            smoteOptions.classList.add('fade-in');
        } else {
            smoteOptions.style.display = 'none';
            smoteOptions.classList.remove('fade-in');
        }
    });
    
    // Apply recommendations button
    document.getElementById('applyRecommendations').addEventListener('click', applyRecommendations);
    
    // Solver change event to update learning rate and epochs recommendations
    document.getElementById('solver').addEventListener('change', function() {
        if (currentRecommendations) {
            updateRecommendationHints(currentRecommendations);
        }
    });
    
    // Load training history on page load
    fetch('/get_history')
        .then(response => response.json())
        .then(data => {
            trainingHistory = data.history || [];
            updateHistoryDisplay();
        })
        .catch(error => {
            console.error('Error loading history:', error);
        });
    
    // Add animation classes to cards
    document.querySelectorAll('.card').forEach(card => {
        card.classList.add('fade-in');
    });
});

// Auto-load default data on page load
window.addEventListener('load', function() {
    setTimeout(() => {
        useDefaultData();
    }, 500);
}); 