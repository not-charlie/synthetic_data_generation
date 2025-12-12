# Synthetic Data Modeling & Simulation

A comprehensive Streamlit application for generating synthetic datasets, performing exploratory data analysis, applying machine learning models, and evaluating simulation outcomes.

## Project Overview

This project explores modeling and simulation concepts using Python. It provides hands-on experience with popular Python libraries (NumPy, Pandas, Scikit-learn, Matplotlib, etc.) for modeling and simulation tasks.

## Features

### 1. Synthetic Data Generation
- **Regression Datasets**: Generate regression datasets with configurable features, samples, and noise levels
- **Classification Datasets**: Create classification datasets with multiple classes and informative features
- **Time Series Data**: Generate time series with trend, seasonal, and noise components

### 2. Exploratory Data Analysis (EDA)
- Dataset overview and statistics
- Distribution visualizations
- Correlation matrices
- Feature-target relationships
- Time series plots

### 3. Machine Learning Modeling
- **Regression Models**: Linear Regression, Random Forest Regressor
- **Classification Models**: Logistic Regression, Random Forest Classifier
- Model performance metrics (MSE, R², Accuracy, Confusion Matrix)
- Prediction vs Actual visualizations

### 4. Simulation & Evaluation
- Generate simulated outcomes using trained models
- Compare simulated vs actual data distributions
- Statistical comparison metrics
- Model evaluation against known data properties

## Installation

1. Clone the repository:
```bash
git clone https://github.com/not-charlie/synthetic_data_generation.git
cd synthetic_data_generation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

## How to Use

1. **Configure Dataset**: Use the sidebar to select dataset type (Regression/Classification/Time Series) and adjust parameters
2. **Generate Data**: Click "Generate Synthetic Data" to create your dataset
3. **Explore Data**: Navigate to the "Exploratory Data Analysis" tab to discover data characteristics
4. **Train Model**: Go to the "Modeling" tab, select a model, and train it
5. **Simulate & Evaluate**: Use the "Simulation & Evaluation" tab to generate simulations and compare them with actual data

## Project Structure

```
synthetic_data_generation/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── project_details.txt    # Project requirements and goals
└── README.md             # This file
```

## Technologies Used

- **Streamlit**: Interactive web application framework
- **NumPy**: Numerical computing
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models and utilities
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization

## Project Goals

This project implements the following tasks:
1. ✅ Generate a synthetic dataset with known properties
2. ✅ Perform EDA to discover data characteristics
3. ✅ Apply suitable modeling techniques
4. ✅ Use fitted models to generate simulated outcomes
5. ✅ Evaluate model performance by comparing simulations with original data

## License

This project is open source and available for educational purposes.

