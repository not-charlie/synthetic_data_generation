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
    classification_report, confusion_matrix,
    mean_absolute_error, precision_score, recall_score, f1_score
)
from sklearn.model_selection import cross_val_score, KFold
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

# Data source selection
data_source = st.sidebar.radio(
    "Data Source",
    ["Generate Synthetic Data", "Upload CSV File"]
)

# Dataset type selection
if data_source == "Generate Synthetic Data":
    dataset_type = st.sidebar.selectbox(
        "Select Dataset Type",
        ["Regression", "Classification", "Time Series"]
    )
else:
    dataset_type = st.sidebar.selectbox(
        "Select Dataset Type (for uploaded data)",
        ["Regression", "Classification", "Time Series"],
        help="Select the type of problem your uploaded data represents"
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
if 'plot_width' not in st.session_state:
    st.session_state.plot_width = 10
if 'plot_height' not in st.session_state:
    st.session_state.plot_height = 6
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'models' not in st.session_state:
    st.session_state.models = {}  # Store multiple models for comparison
if 'use_cross_validation' not in st.session_state:
    st.session_state.use_cross_validation = False
if 'cv_scores' not in st.session_state:
    st.session_state.cv_scores = None

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
        
        # Extract X (features) from DataFrame (exclude 'Target' and 'Time')
        X = df.drop(['Target', 'Time'], axis=1).values
    
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

# CSV Upload Section
if data_source == "Upload CSV File":
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload CSV File")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with features and a target column"
    )
    
    if uploaded_file is not None:
        try:
            df_uploaded = pd.read_csv(uploaded_file)
            st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
            st.sidebar.info(f"Shape: {df_uploaded.shape[0]} rows × {df_uploaded.shape[1]} columns")
            
            # Let user select target column
            target_column = st.sidebar.selectbox(
                "Select Target Column",
                df_uploaded.columns.tolist(),
                help="Select the column that contains your target variable"
            )
            
            # Get feature columns (all except target)
            feature_columns = [col for col in df_uploaded.columns if col != target_column]
            
            if len(feature_columns) == 0:
                st.sidebar.error("❌ No feature columns found. Please ensure your CSV has at least one feature column.")
            else:
                if st.sidebar.button("📊 Load Uploaded Data", type="primary"):
                    # Prepare data
                    X = df_uploaded[feature_columns].values
                    y = df_uploaded[target_column].values
                    
                    # Create DataFrame with consistent structure
                    df_processed = df_uploaded.copy()
                    # Rename target column to 'Target' for consistency
                    df_processed = df_processed.rename(columns={target_column: 'Target'})
                    
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
                    
                    st.session_state.df = df_processed
                    st.session_state.data_generated = True
                    st.session_state.model = None
                    st.session_state.uploaded_file = uploaded_file.name
                    st.sidebar.success("✅ Data loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            st.sidebar.info("Please ensure your CSV file is properly formatted.")

# Generate Data Button
if data_source == "Generate Synthetic Data" and st.sidebar.button("🔄 Generate Synthetic Data", type="primary"):
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
        st.session_state.uploaded_file = None
    st.success("✅ Data generated successfully!")

# Clear button in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("🔄 Delete Option")
if st.sidebar.button("🗑️ Clear All Data", help="Clear all data and models"):
    st.session_state.data_generated = False
    st.session_state.df = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.model = None
    st.session_state.models = {}
    st.session_state.model_type = None
    st.session_state.scaler = None
    st.session_state.uploaded_file = None
    st.session_state.cv_scores = None
    st.session_state.use_cross_validation = False
    st.sidebar.success("✅ All data cleared!")
    st.rerun()


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
            # Get feature names based on dataset type and data source
            if data_source == "Upload CSV File":
                # Use original feature column names from uploaded data
                feature_names = [col for col in st.session_state.df.columns if col != 'Target']
            elif dataset_type == "Time Series":
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
        
        # Plot size controls
        st.subheader("📐 Plot Size Controls")
        col1, col2 = st.columns(2)
        with col1:
            plot_width = st.slider("Plot Width", 5, 20, st.session_state.plot_width, help="Adjust the width of plots")
            st.session_state.plot_width = plot_width
        with col2:
            plot_height = st.slider("Plot Height", 3, 15, st.session_state.plot_height, help="Adjust the height of plots")
            st.session_state.plot_height = plot_height
        
        st.markdown("---")
        
        # Distribution plots
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_height))
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
        fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_width))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)
        
        # Feature distributions
        st.subheader("Feature Distributions")
        feature_cols = [col for col in st.session_state.df.columns if col != 'Target']
        n_features = len(feature_cols)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(st.session_state.plot_width * n_cols, st.session_state.plot_height * n_rows))
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
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(st.session_state.plot_width * n_cols, st.session_state.plot_height * n_rows))
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
            fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_height))
            ax.plot(st.session_state.df['Time'], st.session_state.df['Target'], linewidth=2)
            ax.set_xlabel('Time')
            ax.set_ylabel('Target Value')
            ax.set_title('Time Series Data')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
    
    with tab3:
        st.header("🤖 Model Training")
        
        # Cross-validation option
        st.subheader("⚙️ Training Options")
        use_cv = st.checkbox("Use Cross-Validation", value=st.session_state.use_cross_validation, 
                            help="Enable cross-validation for more robust model evaluation")
        st.session_state.use_cross_validation = use_cv
        
        if use_cv:
            cv_folds = st.slider("Number of CV Folds", 3, 10, 5, help="Number of folds for cross-validation")
        
        # Model comparison option
        compare_models = st.checkbox("Compare Multiple Models", help="Train and compare multiple models side-by-side")
        
        if compare_models:
            st.subheader("Select Models to Compare")
            if dataset_type == "Regression":
                selected_models = st.multiselect(
                    "Choose models",
                    ["Linear Regression", "Random Forest Regressor"],
                    default=["Linear Regression", "Random Forest Regressor"]
                )
            elif dataset_type == "Classification":
                selected_models = st.multiselect(
                    "Choose models",
                    ["Logistic Regression", "Random Forest Classifier"],
                    default=["Logistic Regression", "Random Forest Classifier"]
                )
            else:  # Time Series
                selected_models = st.multiselect(
                    "Choose models",
                    ["Linear Regression", "Random Forest Regressor"],
                    default=["Linear Regression", "Random Forest Regressor"]
                )
        else:
            # Single model selection
            if dataset_type == "Regression":
                model_choice = st.selectbox(
                    "Select Model",
                    ["Linear Regression", "Random Forest Regressor"]
                )
                selected_models = [model_choice]
            elif dataset_type == "Classification":
                model_choice = st.selectbox(
                    "Select Model",
                    ["Logistic Regression", "Random Forest Classifier"]
                )
                selected_models = [model_choice]
            else:  # Time Series
                model_choice = st.selectbox(
                    "Select Model",
                    ["Linear Regression", "Random Forest Regressor"]
                )
                selected_models = [model_choice]
        
        # Model hyperparameters
        if "Random Forest" in str(selected_models):
            st.subheader("Random Forest Hyperparameters")
            n_estimators = st.slider("Number of Trees", 10, 500, 100, help="More trees = better performance but slower training")
            max_depth = st.slider("Max Depth", 2, 20, 10, help="Maximum depth of the trees")
        
        if st.button("🚀 Train Model(s)", type="primary"):
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(st.session_state.X_train)
            X_test_scaled = scaler.transform(st.session_state.X_test)
            
            # Clear previous models if comparing
            if compare_models:
                st.session_state.models = {}
            
            # Train each selected model
            for model_name in selected_models:
                with st.spinner(f"Training {model_name}..."):
                    if model_name == "Linear Regression":
                        model = LinearRegression()
                        model_type = "regression"
                        model.fit(X_train_scaled, st.session_state.y_train)
                        
                    elif model_name == "Random Forest Regressor":
                        model_type = "regression"
                        progress_container = st.container()
                        base_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        rf_model = RandomForestWithProgress(base_model, progress_container)
                        rf_model.fit(X_train_scaled, st.session_state.y_train)
                        model = rf_model.base_model
                        
                    elif model_name == "Logistic Regression":
                        model = LogisticRegression(random_state=42, max_iter=1000)
                        model_type = "classification"
                        model.fit(X_train_scaled, st.session_state.y_train)
                        
                    else:  # Random Forest Classifier
                        model_type = "classification"
                        progress_container = st.container()
                        base_model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
                        rf_model = RandomForestWithProgress(base_model, progress_container)
                        rf_model.fit(X_train_scaled, st.session_state.y_train)
                        model = rf_model.base_model
                    
                    # Make predictions
                    y_train_pred = model.predict(X_train_scaled)
                    y_test_pred = model.predict(X_test_scaled)
                    
                    # Cross-validation if enabled
                    cv_scores = None
                    if use_cv:
                        if model_type == "regression":
                            scoring = 'neg_mean_squared_error'
                        else:
                            scoring = 'accuracy'
                        cv_scores = cross_val_score(model, X_train_scaled, st.session_state.y_train, 
                                                   cv=cv_folds, scoring=scoring)
                    
                    # Store model info
                    model_info = {
                        'model': model,
                        'model_type': model_type,
                        'y_train_pred': y_train_pred,
                        'y_test_pred': y_test_pred,
                        'cv_scores': cv_scores
                    }
                    st.session_state.models[model_name] = model_info
                    
                    # Set primary model (first one or single model)
                    if len(st.session_state.models) == 1:
                        st.session_state.model = model
                        st.session_state.model_type = model_type
                        st.session_state.scaler = scaler
                        st.session_state.y_train_pred = y_train_pred
                        st.session_state.y_test_pred = y_test_pred
                        st.session_state.cv_scores = cv_scores
            
            st.success(f"✅ {len(selected_models)} model(s) trained successfully!")
        
        # Display results
        if len(st.session_state.models) > 0:
            # Model comparison table if multiple models
            if compare_models and len(st.session_state.models) > 1:
                st.subheader("📊 Model Comparison")
                comparison_data = []
                
                for model_name, model_info in st.session_state.models.items():
                    if model_info['model_type'] == "regression":
                        train_mse = mean_squared_error(st.session_state.y_train, model_info['y_train_pred'])
                        test_mse = mean_squared_error(st.session_state.y_test, model_info['y_test_pred'])
                        train_r2 = r2_score(st.session_state.y_train, model_info['y_train_pred'])
                        test_r2 = r2_score(st.session_state.y_test, model_info['y_test_pred'])
                        train_mae = mean_absolute_error(st.session_state.y_train, model_info['y_train_pred'])
                        test_mae = mean_absolute_error(st.session_state.y_test, model_info['y_test_pred'])
                        
                        comparison_data.append({
                            'Model': model_name,
                            'Train MSE': f"{train_mse:.4f}",
                            'Test MSE': f"{test_mse:.4f}",
                            'Train R²': f"{train_r2:.4f}",
                            'Test R²': f"{test_r2:.4f}",
                            'Train MAE': f"{train_mae:.4f}",
                            'Test MAE': f"{test_mae:.4f}"
                        })
                        
                        if model_info['cv_scores'] is not None:
                            comparison_data[-1]['CV MSE (mean)'] = f"{-model_info['cv_scores'].mean():.4f}"
                            comparison_data[-1]['CV MSE (std)'] = f"{model_info['cv_scores'].std():.4f}"
                    else:  # Classification
                        train_acc = accuracy_score(st.session_state.y_train, model_info['y_train_pred'])
                        test_acc = accuracy_score(st.session_state.y_test, model_info['y_test_pred'])
                        train_precision = precision_score(st.session_state.y_train, model_info['y_train_pred'], average='weighted', zero_division=0)
                        test_precision = precision_score(st.session_state.y_test, model_info['y_test_pred'], average='weighted', zero_division=0)
                        train_f1 = f1_score(st.session_state.y_train, model_info['y_train_pred'], average='weighted', zero_division=0)
                        test_f1 = f1_score(st.session_state.y_test, model_info['y_test_pred'], average='weighted', zero_division=0)
                        
                        comparison_data.append({
                            'Model': model_name,
                            'Train Accuracy': f"{train_acc:.4f}",
                            'Test Accuracy': f"{test_acc:.4f}",
                            'Train Precision': f"{train_precision:.4f}",
                            'Test Precision': f"{test_precision:.4f}",
                            'Train F1': f"{train_f1:.4f}",
                            'Test F1': f"{test_f1:.4f}"
                        })
                        
                        if model_info['cv_scores'] is not None:
                            comparison_data[-1]['CV Accuracy (mean)'] = f"{model_info['cv_scores'].mean():.4f}"
                            comparison_data[-1]['CV Accuracy (std)'] = f"{model_info['cv_scores'].std():.4f}"
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
                st.markdown("---")
            
            # Display primary model (first model or single model)
            primary_model_name = list(st.session_state.models.keys())[0]
            model_info = st.session_state.models[primary_model_name]
            st.session_state.model = model_info['model']
            st.session_state.model_type = model_info['model_type']
            st.session_state.y_train_pred = model_info['y_train_pred']
            st.session_state.y_test_pred = model_info['y_test_pred']
            st.session_state.cv_scores = model_info['cv_scores']
            
            st.subheader(f"Model Performance Metrics - {primary_model_name}")
            
            # Cross-validation results
            if use_cv and st.session_state.cv_scores is not None:
                st.subheader("📈 Cross-Validation Results")
                cv_mean = st.session_state.cv_scores.mean()
                cv_std = st.session_state.cv_scores.std()
                
                if st.session_state.model_type == "regression":
                    st.metric("CV MSE (mean)", f"{-cv_mean:.4f}", help="Mean squared error across CV folds")
                    st.metric("CV MSE (std)", f"{cv_std:.4f}", help="Standard deviation of MSE across CV folds")
                    st.metric("CV RMSE (mean)", f"{np.sqrt(-cv_mean):.4f}", help="Root mean squared error")
                else:
                    st.metric("CV Accuracy (mean)", f"{cv_mean:.4f}", help="Mean accuracy across CV folds")
                    st.metric("CV Accuracy (std)", f"{cv_std:.4f}", help="Standard deviation of accuracy across CV folds")
                
                # CV scores visualization
                fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_height))
                fold_numbers = range(1, len(st.session_state.cv_scores) + 1)
                if st.session_state.model_type == "regression":
                    scores_to_plot = -st.session_state.cv_scores  # Convert to positive MSE
                    ax.plot(fold_numbers, scores_to_plot, 'o-', linewidth=2, markersize=8)
                    ax.axhline(y=-cv_mean, color='r', linestyle='--', label=f'Mean: {-cv_mean:.4f}')
                    ax.set_ylabel('MSE')
                    ax.set_title('Cross-Validation MSE by Fold')
                else:
                    ax.plot(fold_numbers, st.session_state.cv_scores, 'o-', linewidth=2, markersize=8)
                    ax.axhline(y=cv_mean, color='r', linestyle='--', label=f'Mean: {cv_mean:.4f}')
                    ax.set_ylabel('Accuracy')
                    ax.set_title('Cross-Validation Accuracy by Fold')
                ax.set_xlabel('Fold')
                ax.set_xticks(fold_numbers)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                st.markdown("---")
            
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
                
                # Additional metrics
                train_mae = mean_absolute_error(st.session_state.y_train, st.session_state.y_train_pred)
                test_mae = mean_absolute_error(st.session_state.y_test, st.session_state.y_test_pred)
                train_rmse = np.sqrt(train_mse)
                test_rmse = np.sqrt(test_mse)
                
                st.subheader("Additional Regression Metrics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Train MAE", f"{train_mae:.4f}")
                with col2:
                    st.metric("Test MAE", f"{test_mae:.4f}")
                with col3:
                    st.metric("Train RMSE", f"{train_rmse:.4f}")
                with col4:
                    st.metric("Test RMSE", f"{test_rmse:.4f}")
                
                # Prediction vs Actual plots
                st.subheader("Prediction vs Actual")
                fig, axes = plt.subplots(1, 2, figsize=(st.session_state.plot_width * 1.5, st.session_state.plot_height))
                
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
                
                # Feature importance for tree-based models
                if "Random Forest" in primary_model_name:
                    st.subheader("📊 Feature Importance")
                    try:
                        feature_importance = st.session_state.model.feature_importances_
                        # Get feature names
                        if data_source == "Upload CSV File":
                            feature_names_list = [col for col in st.session_state.df.columns if col != 'Target']
                        elif dataset_type == "Time Series":
                            feature_names_list = ['Lag_1', 'Lag_2', 'Lag_3', 'Trend', 'Seasonal']
                        else:
                            feature_names_list = [f'Feature_{i+1}' for i in range(len(feature_importance))]
                        
                        # Create feature importance DataFrame
                        importance_df = pd.DataFrame({
                            'Feature': feature_names_list,
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_height))
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_title('Feature Importance')
                        ax.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display as table
                        st.dataframe(importance_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {str(e)}")
                
            else:  # Classification
                train_acc = accuracy_score(st.session_state.y_train, st.session_state.y_train_pred)
                test_acc = accuracy_score(st.session_state.y_test, st.session_state.y_test_pred)
                
                # Calculate precision, recall, F1 (handle multi-class with average='weighted')
                try:
                    train_precision = precision_score(st.session_state.y_train, st.session_state.y_train_pred, average='weighted', zero_division=0)
                    test_precision = precision_score(st.session_state.y_test, st.session_state.y_test_pred, average='weighted', zero_division=0)
                    train_recall = recall_score(st.session_state.y_train, st.session_state.y_train_pred, average='weighted', zero_division=0)
                    test_recall = recall_score(st.session_state.y_test, st.session_state.y_test_pred, average='weighted', zero_division=0)
                    train_f1 = f1_score(st.session_state.y_train, st.session_state.y_train_pred, average='weighted', zero_division=0)
                    test_f1 = f1_score(st.session_state.y_test, st.session_state.y_test_pred, average='weighted', zero_division=0)
                except:
                    train_precision = test_precision = train_recall = test_recall = train_f1 = test_f1 = 0.0
                
                st.subheader("Classification Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Train Accuracy", f"{train_acc:.4f}")
                    st.metric("Train Precision", f"{train_precision:.4f}")
                    st.metric("Train Recall", f"{train_recall:.4f}")
                    st.metric("Train F1-Score", f"{train_f1:.4f}")
                with col2:
                    st.metric("Test Accuracy", f"{test_acc:.4f}")
                    st.metric("Test Precision", f"{test_precision:.4f}")
                    st.metric("Test Recall", f"{test_recall:.4f}")
                    st.metric("Test F1-Score", f"{test_f1:.4f}")
                
                # Confusion matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(st.session_state.y_test, st.session_state.y_test_pred)
                fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_height))
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
                
                # Feature importance for tree-based models
                if "Random Forest" in primary_model_name:
                    st.subheader("📊 Feature Importance")
                    try:
                        feature_importance = st.session_state.model.feature_importances_
                        # Get feature names
                        if data_source == "Upload CSV File":
                            feature_names_list = [col for col in st.session_state.df.columns if col != 'Target']
                        elif dataset_type == "Time Series":
                            feature_names_list = ['Lag_1', 'Lag_2', 'Lag_3', 'Trend', 'Seasonal']
                        else:
                            feature_names_list = [f'Feature_{i+1}' for i in range(len(feature_importance))]
                        
                        # Create feature importance DataFrame
                        importance_df = pd.DataFrame({
                            'Feature': feature_names_list,
                            'Importance': feature_importance
                        }).sort_values('Importance', ascending=False)
                        
                        # Plot feature importance
                        fig, ax = plt.subplots(figsize=(st.session_state.plot_width, st.session_state.plot_height))
                        ax.barh(importance_df['Feature'], importance_df['Importance'])
                        ax.set_xlabel('Importance')
                        ax.set_title('Feature Importance')
                        ax.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Display as table
                        st.dataframe(importance_df, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Could not display feature importance: {str(e)}")
    
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
                    if data_source == "Upload CSV File":
                        # For uploaded data, use bootstrap sampling from training data
                        np.random.seed(42)
                        indices = np.random.choice(len(st.session_state.X_train), size=n_simulations, replace=True)
                        X_sim = st.session_state.X_train[indices]
                        y_sim = st.session_state.y_train[indices]
                    elif dataset_type == "Regression":
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
                fig, axes = plt.subplots(1, 2, figsize=(st.session_state.plot_width * 1.5, st.session_state.plot_height))
                
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
                    sim_mae = mean_absolute_error(st.session_state.y_sim, st.session_state.y_sim_pred)
                    sim_rmse = np.sqrt(sim_mse)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Simulation MSE", f"{sim_mse:.4f}")
                    with col2:
                        st.metric("Simulation R²", f"{sim_r2:.4f}")
                    with col3:
                        st.metric("Simulation MAE", f"{sim_mae:.4f}")
                    with col4:
                        st.metric("Simulation RMSE", f"{sim_rmse:.4f}")
                    
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
    if data_source == "Upload CSV File":
        st.info("👈 Please upload a CSV file in the sidebar and select your target column to get started!")
    else:
        st.info("👈 Please configure the parameters in the sidebar and click 'Generate Synthetic Data' to get started!")

