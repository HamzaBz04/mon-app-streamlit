import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Gradient Descent ML App",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GradientDescentRegressor:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.costs = []
        
    def fit(self, X, y):
        # Initialize parameters
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        self.costs = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            self.update_weights()
            cost = self.compute_cost()
            self.costs.append(cost)
            
    def update_weights(self):
        y_pred = self.predict(self.X)
        
        # Calculate gradients
        dw = (1/self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1/self.m) * np.sum(y_pred - self.y)
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
    def compute_cost(self):
        y_pred = self.predict(self.X)
        cost = (1/(2*self.m)) * np.sum((y_pred - self.y)**2)
        return cost
        
    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

class GradientDescentClassifier:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.costs = []
        
    def sigmoid(self, z):
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
        
    def fit(self, X, y):
        # Initialize parameters
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        self.costs = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            self.update_weights()
            cost = self.compute_cost()
            self.costs.append(cost)
            
    def update_weights(self):
        z = np.dot(self.X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        
        # Calculate gradients
        dw = (1/self.m) * np.dot(self.X.T, (y_pred - self.y))
        db = (1/self.m) * np.sum(y_pred - self.y)
        
        # Update parameters
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
        
    def compute_cost(self):
        z = np.dot(self.X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        
        # Avoid log(0) by adding small epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        cost = -(1/self.m) * np.sum(self.y * np.log(y_pred) + (1 - self.y) * np.log(1 - y_pred))
        return cost
        
    def predict(self, X):
        z = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(z)
        return (y_pred >= 0.5).astype(int)
    
    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

def main():
    st.title("🎯 Gradient Descent Machine Learning Application")
   
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Choose a page", [
        "📊 Dataset Upload & Overview",
        "🔧 Data Preprocessing", 
        "🤖 Model Training",
        "📈 Results & Visualization"
    ])
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'model' not in st.session_state:
        st.session_state.model = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    
    if page == "📊 Dataset Upload & Overview":
        dataset_upload_page()
    elif page == "🔧 Data Preprocessing":
        preprocessing_page()
    elif page == "🤖 Model Training":
        model_training_page()
    elif page == "📈 Results & Visualization":
        results_page()

def dataset_upload_page():
    st.header("📊 Dataset Upload & Overview")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your dataset in CSV format"
    )
    
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            
            st.success(f"✅ Dataset uploaded successfully! Shape: {data.shape}")
            
            # Display basic information
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Dataset Info")
                st.write(f"**Rows:** {data.shape[0]}")
                st.write(f"**Columns:** {data.shape[1]}")
                st.write(f"**Memory Usage:** {data.memory_usage(deep=True).sum() / 1024:.2f} KB")
            
            with col2:
                st.subheader("🔍 Data Types")
                st.write(data.dtypes.value_counts())
            
            # Display first few rows
            st.subheader("👀 Data Preview")
            st.dataframe(data.head(10))
            
            # Statistical summary
            st.subheader("📊 Statistical Summary")
            st.dataframe(data.describe())
            
            # Missing values
            st.subheader("❓ Missing Values")
            missing_data = data.isnull().sum()
            if missing_data.sum() > 0:
                fig = px.bar(
                    x=missing_data.index, 
                    y=missing_data.values,
                    title="Missing Values by Column"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
            
            # Correlation heatmap for numerical columns
            numerical_cols = data.select_dtypes(include=[np.number]).columns
            if len(numerical_cols) > 1:
                st.subheader("🔥 Correlation Heatmap")
                fig = px.imshow(
                    data[numerical_cols].corr(),
                    text_auto=True,
                    aspect="auto",
                    title="Feature Correlation Matrix"
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error loading dataset: {str(e)}")
    else:
        st.info("👆 Please upload a CSV file to get started")
        
        # Sample datasets
        st.subheader("🎯 Try with Sample Data")
        if st.button("Generate Sample Regression Data"):
            np.random.seed(42)
            n_samples = 100
            X = np.random.randn(n_samples, 2)
            y = 3*X[:, 0] + 2*X[:, 1] + np.random.randn(n_samples) * 0.1
            
            sample_data = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
            sample_data['target'] = y
            st.session_state.data = sample_data
            st.rerun()
            
        if st.button("Generate Sample Classification Data"):
            np.random.seed(42)
            n_samples = 100
            X = np.random.randn(n_samples, 2)
            y = (X[:, 0] + X[:, 1] > 0).astype(int)
            
            sample_data = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
            sample_data['target'] = y
            st.session_state.data = sample_data
            st.rerun()

def preprocessing_page():
    st.header("🔧 Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload a dataset first!")
        return
    
    data = st.session_state.data.copy()
    
    st.subheader("🎯 Target Variable Selection")
    target_column = st.selectbox(
        "Select target variable:",
        options=data.columns.tolist(),
        index=len(data.columns)-1
    )
    
    # Determine problem type
    if data[target_column].dtype == 'object' or data[target_column].nunique() <= 10:
        problem_type = st.selectbox(
            "Problem Type:",
            ["Classification", "Regression"],
            index=0
        )
    else:
        problem_type = st.selectbox(
            "Problem Type:",
            ["Regression", "Classification"],
            index=0
        )
    
    st.info(f"🎯 Problem Type: **{problem_type}**")
    
    # Feature selection
    st.subheader("📊 Feature Selection")
    feature_columns = [col for col in data.columns if col != target_column]
    selected_features = st.multiselect(
        "Select features for training:",
        options=feature_columns,
        default=feature_columns
    )
    
    if not selected_features:
        st.warning("⚠️ Please select at least one feature!")
        return
    
    # Handle missing values
    st.subheader("🔧 Handle Missing Values")
    missing_strategy = st.selectbox(
        "Missing value strategy:",
        ["Drop rows with missing values", "Fill with mean", "Fill with median", "Fill with mode"]
    )
    
    # Apply missing value strategy
    if missing_strategy == "Drop rows with missing values":
        data = data.dropna()
    elif missing_strategy == "Fill with mean":
        for col in selected_features:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].mean(), inplace=True)
    elif missing_strategy == "Fill with median":
        for col in selected_features:
            if data[col].dtype in ['int64', 'float64']:
                data[col].fillna(data[col].median(), inplace=True)
    elif missing_strategy == "Fill with mode":
        for col in selected_features:
            data[col].fillna(data[col].mode()[0], inplace=True)
    
    # Encode categorical variables
    st.subheader("🏷️ Categorical Encoding")
    categorical_cols = data[selected_features].select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write("Categorical columns found:", categorical_cols)
        encoding_strategy = st.selectbox(
            "Encoding strategy:",
            ["Label Encoding", "One-Hot Encoding"]
        )
        
        if encoding_strategy == "Label Encoding":
            le = LabelEncoder()
            for col in categorical_cols:
                data[col] = le.fit_transform(data[col].astype(str))
        else:  # One-Hot Encoding
            data = pd.get_dummies(data, columns=categorical_cols, prefix=categorical_cols)
            # Update selected features
            new_features = [col for col in data.columns if col != target_column]
            selected_features = new_features
    
    # Feature scaling
    st.subheader("⚖️ Feature Scaling")
    scaling_option = st.selectbox(
        "Scaling method:",
        ["None", "Standardization (Z-score)", "Min-Max Scaling"]
    )
    
    X = data[selected_features].values
    y = data[target_column].values
    
    # Handle target variable for classification
    if problem_type == "Classification":
        if y.dtype == 'object' or len(np.unique(y)) > 2:
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            st.info(f"Target classes: {le_target.classes_}")
    
    # Apply scaling
    if scaling_option == "Standardization (Z-score)":
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    elif scaling_option == "Min-Max Scaling":
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)
    
    # Train-test split
    st.subheader("🔄 Train-Test Split")
    test_size = st.slider("Test size ratio:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.number_input("Random state:", value=42, min_value=0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Store processed data
    processed_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': selected_features,
        'problem_type': problem_type,
        'target_name': target_column
    }
    
    st.session_state.processed_data = processed_data
    
    # Display preprocessing summary
    st.subheader("✅ Preprocessing Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Training samples", X_train.shape[0])
        st.metric("Features", X_train.shape[1])
        
    with col2:
        st.metric("Test samples", X_test.shape[0])
        st.metric("Problem type", problem_type)
    
    st.success("✅ Data preprocessing completed successfully!")

def model_training_page():
    st.header("🤖 Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please complete data preprocessing first!")
        return
    
    data = st.session_state.processed_data
    problem_type = data['problem_type']
    
    st.subheader("⚙️ Hyperparameter Tuning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        learning_rate = st.slider(
            "Learning Rate (α):",
            min_value=0.001,
            max_value=1.0,
            value=0.01,
            step=0.001,
            format="%.3f"
        )
        
    with col2:
        n_iterations = st.slider(
            "Number of Iterations:",
            min_value=100,
            max_value=5000,
            value=1000,
            step=100
        )
    
    if st.button("🚀 Train Model", type="primary"):
        with st.spinner("Training model..."):
            # Initialize model based on problem type
            if problem_type == "Regression":
                model = GradientDescentRegressor(
                    learning_rate=learning_rate,
                    n_iterations=n_iterations
                )
            else:
                model = GradientDescentClassifier(
                    learning_rate=learning_rate,
                    n_iterations=n_iterations
                )
            
            # Train model
            model.fit(data['X_train'], data['y_train'])
            
            # Make predictions
            y_train_pred = model.predict(data['X_train'])
            y_test_pred = model.predict(data['X_test'])
            
            # Calculate metrics
            if problem_type == "Regression":
                from sklearn.metrics import r2_score, mean_absolute_error
                
                train_mse = mean_squared_error(data['y_train'], y_train_pred)
                test_mse = mean_squared_error(data['y_test'], y_test_pred)
                train_rmse = np.sqrt(train_mse)
                test_rmse = np.sqrt(test_mse)
                
                # Add R² score (coefficient of determination) - regression accuracy measure
                train_r2 = r2_score(data['y_train'], y_train_pred)
                test_r2 = r2_score(data['y_test'], y_test_pred)
                
                # Add Mean Absolute Error
                train_mae = mean_absolute_error(data['y_train'], y_train_pred)
                test_mae = mean_absolute_error(data['y_test'], y_test_pred)
                
                metrics = {
                    'train_mse': train_mse,
                    'test_mse': test_mse,
                    'train_rmse': train_rmse,
                    'test_rmse': test_rmse,
                    'train_r2': train_r2,
                    'test_r2': test_r2,
                    'train_mae': train_mae,
                    'test_mae': test_mae
                }
            else:
                train_acc = accuracy_score(data['y_train'], y_train_pred)
                test_acc = accuracy_score(data['y_test'], y_test_pred)
                
                metrics = {
                    'train_accuracy': train_acc,
                    'test_accuracy': test_acc
                }
            
            # Store results
            results = {
                'model': model,
                'y_train_pred': y_train_pred,
                'y_test_pred': y_test_pred,
                'metrics': metrics,
                'learning_rate': learning_rate,
                'n_iterations': n_iterations
            }
            
            st.session_state.results = results
            
        st.success("✅ Model training completed!")
        
        # Display metrics
        st.subheader("📊 Training Results")

        if problem_type == "Regression":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Train RMSE", f"{metrics['train_rmse']:.4f}")
                st.metric("Train MSE", f"{metrics['train_mse']:.4f}")
            with col2:
                st.metric("Test RMSE", f"{metrics['test_rmse']:.4f}")
                st.metric("Test MSE", f"{metrics['test_mse']:.4f}")
            with col3:
                st.metric("Train R² Score", f"{metrics['train_r2']:.4f}")
                st.metric("Test R² Score", f"{metrics['test_r2']:.4f}")
            
            # Additional row for MAE
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train MAE", f"{metrics['train_mae']:.4f}")
            with col2:
                st.metric("Test MAE", f"{metrics['test_mae']:.4f}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Train Accuracy", f"{metrics['train_accuracy']:.4f}")
            with col2:
                st.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}")
        
        # Learning curve
        st.subheader("📈 Learning Curve")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(model.costs))),
            y=model.costs,
            mode='lines',
            name='Cost',
            line=dict(color='blue', width=2)
        ))
        fig.update_layout(
            title="Cost Function Over Iterations",
            xaxis_title="Iterations",
            yaxis_title="Cost",
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)

def results_page():
    st.header("📈 Results & Visualization")
    
    if st.session_state.results is None:
        st.warning("⚠️ Please train a model first!")
        return
    
    results = st.session_state.results
    data = st.session_state.processed_data
    problem_type = data['problem_type']
    
    # Model parameters
    st.subheader("⚙️ Model Parameters")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Learning Rate", results['learning_rate'])
    with col2:
        st.metric("Iterations", results['n_iterations'])
    
    # Performance metrics
    st.subheader("🎯 Performance Metrics")
    metrics = results['metrics']

    if problem_type == "Regression":
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Train RMSE", f"{metrics['train_rmse']:.4f}")
            st.metric("Train MAE", f"{metrics['train_mae']:.4f}")
        with col2:
            st.metric("Test RMSE", f"{metrics['test_rmse']:.4f}")
            st.metric("Test MAE", f"{metrics['test_mae']:.4f}")
        with col3:
            st.metric("Train MSE", f"{metrics['train_mse']:.4f}")
            st.metric("Train R²", f"{metrics['train_r2']:.4f}")
        with col4:
            st.metric("Test MSE", f"{metrics['test_mse']:.4f}")
            st.metric("Test R²", f"{metrics['test_r2']:.4f}")
            
        # Add interpretation of R² score
        st.info(f"""
        **R² Score Interpretation:**
        - R² = 1.0: Perfect predictions
        - R² = 0.0: Model performs as well as predicting the mean
        - R² < 0.0: Model performs worse than predicting the mean
        
        **Current Test R²:** {metrics['test_r2']:.4f} 
        ({metrics['test_r2']*100:.2f}% of variance explained)
        """)
            
        # Prediction vs Actual plot
        st.subheader("🎯 Predictions vs Actual Values")
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Training Set', 'Test Set')
        )
        
        # Training set
        fig.add_trace(
            go.Scatter(
                x=data['y_train'],
                y=results['y_train_pred'],
                mode='markers',
                name='Train',
                marker=dict(color='blue', opacity=0.6)
            ),
            row=1, col=1
        )
        
        # Perfect prediction line for training
        min_val = min(data['y_train'].min(), results['y_train_pred'].min())
        max_val = max(data['y_train'].max(), results['y_train_pred'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=1
        )
        
        # Test set
        fig.add_trace(
            go.Scatter(
                x=data['y_test'],
                y=results['y_test_pred'],
                mode='markers',
                name='Test',
                marker=dict(color='green', opacity=0.6)
            ),
            row=1, col=2
        )
        
        # Perfect prediction line for test
        min_val = min(data['y_test'].min(), results['y_test_pred'].min())
        max_val = max(data['y_test'].max(), results['y_test_pred'].max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Perfect Prediction',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ),
            row=1, col=2
        )
        
        fig.update_xaxes(title_text="Actual Values")
        fig.update_yaxes(title_text="Predicted Values")
        fig.update_layout(height=500, template="plotly_white")
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:  # Classification
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Train Accuracy", f"{metrics['train_accuracy']:.4f}")
        with col2:
            st.metric("Test Accuracy", f"{metrics['test_accuracy']:.4f}")
        
        # Confusion Matrix
        st.subheader("🔍 Confusion Matrix")
        cm = confusion_matrix(data['y_test'], results['y_test_pred'])
        
        fig = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            title="Confusion Matrix (Test Set)",
            labels=dict(x="Predicted", y="Actual")
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification Report
        st.subheader("📊 Classification Report")
        report = classification_report(
            data['y_test'], 
            results['y_test_pred'], 
            output_dict=True
        )
        
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4))
    
    # Learning Curve
    st.subheader("📈 Learning Curve")
    model = results['model']
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(model.costs))),
        y=model.costs,
        mode='lines',
        name='Cost',
        line=dict(color='blue', width=2)
    ))
    fig.update_layout(
        title="Cost Function Over Iterations",
        xaxis_title="Iterations",
        yaxis_title="Cost",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Model weights
    st.subheader("⚖️ Model Weights")
    weights_df = pd.DataFrame({
        'Feature': data['feature_names'],
        'Weight': model.weights
    })
    weights_df['Abs_Weight'] = np.abs(weights_df['Weight'])
    weights_df = weights_df.sort_values('Abs_Weight', ascending=False)
    
    fig = px.bar(
        weights_df,
        x='Feature',
        y='Weight',
        title="Feature Weights",
        color='Weight',
        color_continuous_scale='RdBu'
    )
    fig.update_layout(template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.write(f"**Bias term:** {model.bias:.4f}")

if __name__ == "__main__":
    main()
