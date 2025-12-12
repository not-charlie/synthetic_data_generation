import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, 
    classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Synthetic Data Modeling & Simulation",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("📊 Synthetic Data Modeling & Simulation")
st.markdown("---")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# Dataset type selection
dataset_type = st.sidebar.selectbox(
    "Select Dataset Type",
    ["Regression", "Classification", "Time Series"]
)

# Parameters based on dataset type
if dataset_type == "Regression":
    n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 5)
    noise = st.sidebar.slider("Noise Level", 0.0, 50.0, 10.0)
    random_state = st.sidebar.slider("Random State", 0, 100, 42)
    
elif dataset_type == "Classification":
    n_samples = st.sidebar.slider("Number of Samples", 100, 5000, 1000)
    n_features = st.sidebar.slider("Number of Features", 2, 10, 5)
    n_classes = st.sidebar.slider("Number of Classes", 2, 5, 2)
    n_informative = st.sidebar.slider("Informative Features", 2, n_features, min(3, n_features))
    random_state = st.sidebar.slider("Random State", 0, 100, 42)
    
else:  # Time Series
    n_samples = st.sidebar.slider("Number of Time Points", 100, 1000, 200)
    trend_strength = st.sidebar.slider("Trend Strength", 0.0, 2.0, 0.5)
    seasonal_strength = st.sidebar.slider("Seasonal Strength", 0.0, 2.0, 1.0)
    noise_level = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.1)
    random_state = st.sidebar.slider("Random State", 0, 100, 42)

# Initialize session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'model_type' not in st.session_state:
    st.session_state.model_type = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None

# Function to generate synthetic data
def generate_synthetic_data(dataset_type, **kwargs):
    if dataset_type == "Regression":
        X, y = make_regression(
            n_samples=kwargs['n_samples'],
            n_features=kwargs['n_features'],
            noise=kwargs['noise'],
            random_state=kwargs['random_state']
        )
        # Create DataFrame
        feature_names = [f'Feature_{i+1}' for i in range(kwargs['n_features'])]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
    elif dataset_type == "Classification":
        X, y = make_classification(
            n_samples=kwargs['n_samples'],
            n_features=kwargs['n_features'],
            n_classes=kwargs['n_classes'],
            n_informative=kwargs['n_informative'],
            n_redundant=kwargs['n_features'] - kwargs['n_informative'],
            random_state=kwargs['random_state']
        )
        # Create DataFrame
        feature_names = [f'Feature_{i+1}' for i in range(kwargs['n_features'])]
        df = pd.DataFrame(X, columns=feature_names)
        df['Target'] = y
        
    else:  # Time Series
        t = np.arange(kwargs['n_samples'])
        # Generate trend
        trend = kwargs['trend_strength'] * t
        # Generate seasonal component
        seasonal = kwargs['seasonal_strength'] * np.sin(2 * np.pi * t / 12)
        # Generate noise
        noise = np.random.RandomState(kwargs['random_state']).normal(0, kwargs['noise_level'], kwargs['n_samples'])
        # Combine components
        y = trend + seasonal + noise
        
        # Create features (lagged values)
        df = pd.DataFrame({
            'Time': t,
            'Lag_1': np.concatenate([[y[0]], y[:-1]]),
            'Lag_2': np.concatenate([[y[0], y[1]], y[:-2]]),
            'Lag_3': np.concatenate([[y[0], y[1], y[2]], y[:-3]]),
            'Trend': trend,
            'Seasonal': seasonal,
            'Target': y
        })
    
    return df, X, y

# Custom Random Forest wrapper with progress tracking
class RandomForestWithProgress(BaseEstimator):
    def __init__(self, base_model, progress_container):
        self.base_model = base_model
        self.progress_container = progress_container
        self.n_estimators = base_model.n_estimators
        
    def fit(self, X, y):
        progress_bar = self.progress_container.progress(0)
        status_text = self.progress_container.empty()
        
        # Use warm_start for efficient incremental training
        batch_size = max(1, self.n_estimators // 20)  # Update progress ~20 times
        
        # Create model with warm_start enabled
        model_class = type(self.base_model)
        temp_model = model_class(
            n_estimators=batch_size,
            random_state=self.base_model.random_state,
            max_depth=getattr(self.base_model, 'max_depth', None),
            max_features=getattr(self.base_model, 'max_features', 'sqrt'),
            warm_start=True  # Enable incremental training
        )
        
        # Train incrementally in batches
        for batch_end in range(batch_size, self.n_estimators + 1, batch_size):
            batch_end = min(batch_end, self.n_estimators)
            temp_model.n_estimators = batch_end
            temp_model.fit(X, y)
            
            # Update progress
            progress = batch_end / self.n_estimators
            progress_bar.progress(progress)
            status_text.text(f"Training trees: {batch_end}/{self.n_estimators} ({progress*100:.1f}%)")
        
        self.base_model = temp_model
        status_text.text(f"✅ Training complete! ({self.n_estimators} trees)")
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def __getattr__(self, name):
        # Delegate all other attributes to base_model
        return getattr(self.base_model, name)

# Generate Data Button
if st.sidebar.button("🔄 Generate Synthetic Data", type="primary"):
    with st.spinner("Generating synthetic data..."):
        if dataset_type == "Regression":
            st.session_state.df, X, y = generate_synthetic_data(
                dataset_type,
                n_samples=n_samples,
                n_features=n_features,
                noise=noise,
                random_state=random_state
            )
        elif dataset_type == "Classification":
            st.session_state.df, X, y = generate_synthetic_data(
                dataset_type,
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=n_informative,
                random_state=random_state
            )
        else:  # Time Series
            st.session_state.df, X, y = generate_synthetic_data(
                dataset_type,
                n_samples=n_samples,
                trend_strength=trend_strength,
                seasonal_strength=seasonal_strength,
                noise_level=noise_level,
                random_state=random_state
            )
        
        # Split data
        if dataset_type == "Time Series":
            # For time series, use first 80% for training
            split_idx = int(0.8 * len(X))
            st.session_state.X_train = X[:split_idx]
            st.session_state.X_test = X[split_idx:]
            st.session_state.y_train = y[:split_idx]
            st.session_state.y_test = y[split_idx:]
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
        
        st.session_state.data_generated = True
        st.session_state.model = None
    st.success("✅ Data generated successfully!")

# Main content
if st.session_state.data_generated:
    # Tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Data Overview", "🔍 Exploratory Data Analysis", "🤖 Modeling", "🎯 Simulation & Evaluation"])
    
    with tab1:
        st.header("📈 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(st.session_state.df))
        with col2:
            st.metric("Features", len(st.session_state.df.columns) - 1)
        with col3:
            st.metric("Training Samples", len(st.session_state.X_train))
        with col4:
            st.metric("Test Samples", len(st.session_state.X_test))
        
        st.subheader("Dataset Preview")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)
        
        st.subheader("Dataset Statistics")
        st.dataframe(st.session_state.df.describe(), use_container_width=True)
        
        st.subheader("Data Types & Missing Values")
        info_df = pd.DataFrame({
            'Column': st.session_state.df.columns,
            'Data Type': st.session_state.df.dtypes,
            'Non-Null Count': st.session_state.df.count(),
            'Null Count': st.session_state.df.isnull().sum()
        })
        st.dataframe(info_df, use_container_width=True)
        
        # Download CSV button
        st.subheader("📥 Download Dataset")
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Dataset as CSV",
            data=csv,
            file_name=f"synthetic_{dataset_type.lower()}_dataset.csv",
            mime="text/csv",
            type="primary"
        )
        
        # Also provide download for train/test splits
        col1, col2 = st.columns(2)
        with col1:
            # Get feature names based on dataset type
            if dataset_type == "Time Series":
                feature_names = ['Lag_1', 'Lag_2', 'Lag_3', 'Trend', 'Seasonal']
            else:
                feature_names = [f'Feature_{i+1}' for i in range(st.session_state.X_train.shape[1])]
            
            train_df = pd.DataFrame(st.session_state.X_train, columns=feature_names)
            train_df['Target'] = st.session_state.y_train
            train_csv = train_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Training Set (CSV)",
                data=train_csv,
                file_name=f"train_{dataset_type.lower().replace(' ', '_')}_dataset.csv",
                mime="text/csv"
            )
        with col2:
            test_df = pd.DataFrame(st.session_state.X_test, columns=feature_names)
            test_df['Target'] = st.session_state.y_test
            test_csv = test_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Test Set (CSV)",
                data=test_csv,
                file_name=f"test_{dataset_type.lower().replace(' ', '_')}_dataset.csv",
                mime="text/csv"
            )
    
    with tab2:
        st.header("🔍 Exploratory Data Analysis")
        
        # Distribution plots
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        if dataset_type == "Classification":
            st.session_state.df['Target'].value_counts().plot(kind='bar', ax=ax)
            ax.set_xlabel('Class')
            ax.set_ylabel('Count')
            ax.set_title('Class Distribution')
        else:
            st.session_state.df['Target'].hist(bins=30, ax=ax, edgecolor='black')
            ax.set_xlabel('Target Value')
            ax.set_ylabel('Frequency')
            ax.set_title('Target Variable Distribution')
        st.pyplot(fig)
        
        # Correlation matrix
        st.subheader("Correlation Matrix")
        numeric_cols = st.session_state.df.select_dtypes(include=[np.number]).columns
        corr_matrix = st.session_state.df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature_cols = [col for col in st.session_state.df.columns if col != 'Target']
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for i, col in enumerate(feature_cols):
            axes[i].hist(st.session_state.df[col], bins=30, edgecolor='black', alpha=0.7)
            axes[i].set_title(f'{col} Distribution')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
        
        # Hide extra subplots
        for i in range(n_features, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Scatter plots for regression
        if dataset_type == "Regression":
            st.subheader("Feature vs Target Relationships")
            n_cols = 3
            n_rows = (n_features + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
            axes = axes.flatten() if n_features > 1 else [axes]
            
            for i, col in enumerate(feature_cols):
                axes[i].scatter(st.session_state.df[col], st.session_state.df['Target'], alpha=0.5)
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Target')
                axes[i].set_title(f'{col} vs Target')
            
            for i in range(n_features, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        # Time series plot
        if dataset_type == "Time Series":
            st.subheader("Time Series Plot")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(st.session_state.df['Time'], st.session_state.df['Target'], linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Target Value')
            ax.set_title('Time Series Data')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab3:
        st.header("🤖 Model Training")
        
        # Model selection
        if dataset_type == "Regression":
            model_choice = st.selectbox(
                "Select Model",
                ["Linear Regression", "Random Forest Regressor"]
            )
        elif dataset_type == "Classification":
            model_choice = st.selectbox(
                "Select Model",
                ["Logistic Regression", "Random Forest Classifier"]
            )
        else:  # Time Series
            model_choice = st.selectbox(
                "Select Model",
                ["Linear Regression", "Random Forest Regressor"]
            )
        
        # Model hyperparameters
        if "Random Forest" in model_choice:
            n_estimators = st.slider("Number of Trees", 10, 500, 100, help="More trees = better performance but slower training")
            max_depth = st.slider("Max Depth", 2, 20, 10, help="Maximum depth of the trees")
        
        if st.button("🚀 Train Model", type="primary"):
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(st.session_state.X_train)
            X_test_scaled = scaler.transform(st.session_state.X_test)
            
            # Train model with progress tracking for Random Forest
            if model_choice == "Linear Regression":
                with st.spinner("Training Linear Regression..."):
                    model = LinearRegression()
                    st.session_state.model_type = "regression"
                    model.fit(X_train_scaled, st.session_state.y_train)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    
            elif model_choice == "Random Forest Regressor":
                st.session_state.model_type = "regression"
                progress_container = st.container()
                base_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model = RandomForestWithProgress(base_model, progress_container)
                model.fit(X_train_scaled, st.session_state.y_train)
                st.session_state.model = model.base_model  # Store the actual trained model
                st.session_state.scaler = scaler
                
            elif model_choice == "Logistic Regression":
                with st.spinner("Training Logistic Regression..."):
                    model = LogisticRegression(random_state=42, max_iter=1000)
                    st.session_state.model_type = "classification"
                    model.fit(X_train_scaled, st.session_state.y_train)
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    
            else:  # Random Forest Classifier
                st.session_state.model_type = "classification"
                progress_container = st.container()
                base_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                model = RandomForestWithProgress(base_model, progress_container)
                model.fit(X_train_scaled, st.session_state.y_train)
                st.session_state.model = model.base_model  # Store the actual trained model
                st.session_state.scaler = scaler
            
            # Make predictions
            y_train_pred = st.session_state.model.predict(X_train_scaled)
            y_test_pred = st.session_state.model.predict(X_test_scaled)
            
            st.session_state.y_train_pred = y_train_pred
            st.session_state.y_test_pred = y_test_pred
            
            st.success("✅ Model trained successfully!")
        
        if st.session_state.model is not None:
            st.subheader("Model Performance Metrics")
            
            if st.session_state.model_type == "regression":
                train_mse = mean_squared_error(st.session_state.y_train, st.session_state.y_train_pred)
                test_mse = mean_squared_error(st.session_state.y_test, st.session_state.y_test_pred)
                train_r2 = r2_score(st.session_state.y_train, st.session_state.y_train_pred)
                test_r2 = r2_score(st.session_state.y_test, st.session_state.y_test_pred)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train MSE", f"{train_mse:.4f}")
                with col2:
                    st.metric("Test MSE", f"{test_mse:.4f}")
                with col3:
                    st.metric("Train R²", f"{train_r2:.4f}")
                with col4:
                    st.metric("Test R²", f"{test_r2:.4f}")
                
                # Prediction vs Actual plots
                st.subheader("Prediction vs Actual")
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                axes[0].scatter(st.session_state.y_train, st.session_state.y_train_pred, alpha=0.5)
                axes[0].plot([st.session_state.y_train.min(), st.session_state.y_train.max()],
                           [st.session_state.y_train.min(), st.session_state.y_train.max()], 'r--', lw=2)
                axes[0].set_xlabel('Actual')
                axes[0].set_ylabel('Predicted')
                axes[0].set_title('Training Set')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].scatter(st.session_state.y_test, st.session_state.y_test_pred, alpha=0.5)
                axes[1].plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                           [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--', lw=2)
                axes[1].set_xlabel('Actual')
                axes[1].set_ylabel('Predicted')
                axes[1].set_title('Test Set')
                axes[1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:  # Classification
                train_acc = accuracy_score(st.session_state.y_train, st.session_state.y_train_pred)
                test_acc = accuracy_score(st.session_state.y_test, st.session_state.y_test_pred)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Accuracy", f"{train_acc:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{test_acc:.4f}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_test_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix (Test Set)')
                st.pyplot(fig)
                
                # Classification report
                st.subheader("Classification Report")
                report = classification_report(st.session_state.y_test, st.session_state.y_test_pred, output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df, use_container_width=True)
    
    with tab4:
        st.header("🎯 Simulation & Evaluation")
        
        if st.session_state.model is None:
            st.warning("⚠️ Please train a model first in the Modeling tab.")
        else:
            st.subheader("Generate Simulated Outcomes")
            
            n_simulations = st.slider("Number of Simulations", 10, 1000, 100)
            
            if st.button("🎲 Generate Simulations", type="primary"):
                with st.spinner("Generating simulations..."):
                    # Generate new synthetic data with same properties
                    if dataset_type == "Regression":
                        X_sim, y_sim = make_regression(
                            n_samples=n_simulations,
                            n_features=st.session_state.X_train.shape[1],
                            noise=noise,
                            random_state=random_state + 1
                        )
                    elif dataset_type == "Classification":
                        X_sim, y_sim = make_classification(
                            n_samples=n_simulations,
                            n_features=st.session_state.X_train.shape[1],
                            n_classes=n_classes,
                            n_informative=n_informative,
                            random_state=random_state + 1
                        )
                    else:  # Time Series
                        t_sim = np.arange(n_simulations)
                        trend_sim = trend_strength * t_sim
                        seasonal_sim = seasonal_strength * np.sin(2 * np.pi * t_sim / 12)
                        noise_sim = np.random.RandomState(random_state + 1).normal(0, noise_level, n_simulations)
                        y_sim = trend_sim + seasonal_sim + noise_sim
                        
                        X_sim = np.column_stack([
                            np.concatenate([[y_sim[0]], y_sim[:-1]]),
                            np.concatenate([[y_sim[0], y_sim[1]], y_sim[:-2]]),
                            np.concatenate([[y_sim[0], y_sim[1], y_sim[2]], y_sim[:-3]]),
                            trend_sim,
                            seasonal_sim
                        ])
                    
                    # Scale and predict
                    X_sim_scaled = st.session_state.scaler.transform(X_sim)
                    y_sim_pred = st.session_state.model.predict(X_sim_scaled)
                    
                    # Store simulations
                    st.session_state.X_sim = X_sim
                    st.session_state.y_sim = y_sim
                    st.session_state.y_sim_pred = y_sim_pred
                
                st.success(f"✅ Generated {n_simulations} simulations!")
            
            if 'y_sim_pred' in st.session_state:
                st.subheader("Simulation Results")
                
                # Compare distributions
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                if st.session_state.model_type == "regression":
                    axes[0].hist(st.session_state.y_sim, bins=30, alpha=0.7, label='Actual', edgecolor='black')
                    axes[0].hist(st.session_state.y_sim_pred, bins=30, alpha=0.7, label='Simulated', edgecolor='black')
                    axes[0].set_xlabel('Value')
                    axes[0].set_ylabel('Frequency')
                    axes[0].set_title('Distribution Comparison')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    axes[1].scatter(st.session_state.y_sim, st.session_state.y_sim_pred, alpha=0.5)
                    axes[1].plot([st.session_state.y_sim.min(), st.session_state.y_sim.max()],
                               [st.session_state.y_sim.min(), st.session_state.y_sim.max()], 'r--', lw=2)
                    axes[1].set_xlabel('Actual')
                    axes[1].set_ylabel('Simulated')
                    axes[1].set_title('Simulated vs Actual')
                    axes[1].grid(True, alpha=0.3)
                    
                    # Evaluation metrics
                    sim_mse = mean_squared_error(st.session_state.y_sim, st.session_state.y_sim_pred)
                    sim_r2 = r2_score(st.session_state.y_sim, st.session_state.y_sim_pred)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Simulation MSE", f"{sim_mse:.4f}")
                    with col2:
                        st.metric("Simulation R²", f"{sim_r2:.4f}")
                    
                else:  # Classification
                    axes[0].bar(['Actual', 'Simulated'], 
                               [len(np.unique(st.session_state.y_sim)), 
                                len(np.unique(st.session_state.y_sim_pred))])
                    axes[0].set_ylabel('Number of Classes')
                    axes[0].set_title('Class Count Comparison')
                    axes[0].grid(True, alpha=0.3)
                    
                    sim_acc = accuracy_score(st.session_state.y_sim, st.session_state.y_sim_pred)
                    axes[1].bar(['Accuracy'], [sim_acc])
                    axes[1].set_ylabel('Accuracy')
                    axes[1].set_title('Simulation Accuracy')
                    axes[1].set_ylim([0, 1])
                    axes[1].grid(True, alpha=0.3)
                    
                    st.metric("Simulation Accuracy", f"{sim_acc:.4f}")
                
                plt.tight_layout()
                st.pyplot(fig)
                
                # Summary statistics comparison
                st.subheader("Statistical Comparison")
                if st.session_state.model_type == "regression":
                    comparison_df = pd.DataFrame({
                        'Metric': ['Mean', 'Std', 'Min', 'Max'],
                        'Actual': [
                            np.mean(st.session_state.y_sim),
                            np.std(st.session_state.y_sim),
                            np.min(st.session_state.y_sim),
                            np.max(st.session_state.y_sim)
                        ],
                        'Simulated': [
                            np.mean(st.session_state.y_sim_pred),
                            np.std(st.session_state.y_sim_pred),
                            np.min(st.session_state.y_sim_pred),
                            np.max(st.session_state.y_sim_pred)
                        ]
                    })
                    comparison_df['Difference'] = comparison_df['Actual'] - comparison_df['Simulated']
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    comparison_df = pd.DataFrame({
                        'Class': np.unique(st.session_state.y_sim),
                        'Actual Count': [np.sum(st.session_state.y_sim == c) for c in np.unique(st.session_state.y_sim)],
                        'Simulated Count': [np.sum(st.session_state.y_sim_pred == c) for c in np.unique(st.session_state.y_sim)]
                    })
                    st.dataframe(comparison_df, use_container_width=True)

else:
    st.info("👈 Please configure the parameters in the sidebar and click 'Generate Synthetic Data' to get started!")

