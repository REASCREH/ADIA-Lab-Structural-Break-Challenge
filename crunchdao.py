# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


#%pip install crunch-cli --upgrade --quiet --progress-bar off
#!crunch setup-notebook structural-break TI6EEHIjjshOZbB6k4GqBSaW


import os
import typing

# Import your dependencies
import joblib
import pandas as pd
import scipy
import sklearn.metrics


import crunch

# Load the Crunch Toolings
#crunch = crunch.load_notebook()


# Load the data simply
#X_train, y_train, X_test = crunch.load_data()


import os
import typing
import joblib
import pandas as pd
import sklearn.metrics
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
import crunch

# Load the Crunch Toolings
#crunch = crunch.load_notebook()

# ----------------------------------------------------------------------------------------------------------------------
# 1. THE TRAIN FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_directory_path: str,
):
    """
    Trains a LightGBM model with enhanced feature engineering
    """

    def enhanced_preprocess(df: pd.DataFrame):
        """
        Enhanced preprocessing: group by ID and calculate various statistics and transformations
        """
        # Basic statistics
        grouped = df.groupby(level='id').agg({
            'value': ['mean', 'std', 'min', 'max', 'median', 'skew', pd.Series.kurt,
                      lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
            'period': ['mean', 'std']
        })
        
        # Flatten the multi-index columns and rename new quantile columns
        grouped.columns = [f'{col[0]}_{col[1]}' for col in grouped.columns]
        grouped = grouped.rename(columns={'value_<lambda_0>': 'value_q05', 'value_<lambda_1>': 'value_q95'})
        
        # Additional grouped calculations
        additional_features = df.groupby(level='id').apply(lambda x: pd.Series({
            'value_range': x['value'].max() - x['value'].min(),
            'value_iqr': x['value'].quantile(0.75) - x['value'].quantile(0.25),
            'value_mad': (x['value'] - x['value'].mean()).abs().mean(),
            'value_abs_mean': np.abs(x['value']).mean(),
            'value_abs_std': np.abs(x['value']).std(),
            
            # Logarithmic transformations (with handling for zero/negative values)
            'value_log_mean': np.log(np.abs(x['value']) + 1e-10).mean(),
            'value_log_std': np.log(np.abs(x['value']) + 1e-10).std(),
            
            # Exponential transformations
            'value_exp_mean': np.exp(x['value']).mean(),
            'value_exp_std': np.exp(x['value']).std(),
            
            # Square transformations
            'value_sq_mean': (x['value'] ** 2).mean(),
            'value_sq_std': (x['value'] ** 2).std(),
            
            # Square root transformations (with handling for negative values)
            'value_sqrt_mean': np.sqrt(np.abs(x['value'])).mean(),
            'value_sqrt_std': np.sqrt(np.abs(x['value'])).std(),
            
            # Multiplication features
            'value_period_product_mean': (x['value'] * x['period']).mean(),
            'value_period_product_std': (x['value'] * x['period']).std(),
            
            # Division features (with handling for division by zero)
            'value_period_ratio_mean': (x['value'] / (x['period'] + 1e-10)).mean(),
            'value_period_ratio_std': (x['value'] / (x['period'] + 1e-10)).std(),
            
            # Interaction features
            'value_mean_times_period_mean': x['value'].mean() * x['period'].mean(),
            'value_std_times_period_std': x['value'].std() * x['period'].std(),
            
            # Count features
            'total_observations': len(x),
            'period_0_count': (x['period'] == 0).sum(),
            'period_1_count': (x['period'] == 1).sum(),
            'period_ratio': (x['period'] == 1).sum() / len(x) if len(x) > 0 else 0,
            
            # Time-based features
            'time_correlation': x.reset_index().corr()['time']['value'] if len(x) > 1 else 0,
            'value_time_slope': np.polyfit(x.reset_index()['time'], x['value'], 1)[0] if len(x) > 1 else 0,
            
            # ADVANCED MATH FEATURES
            'value_entropy': entropy(x['value'].value_counts(normalize=True)) if len(x['value'].unique()) > 1 else 0,
            'period_entropy': entropy(x['period'].value_counts(normalize=True)) if len(x['period'].unique()) > 1 else 0,
            
            # Fourier Transform Features: Capture periodicity
            'fft_max_freq_amp': np.abs(np.fft.fft(x['value'].values)[1:len(x['value'])//2]).max() if len(x) > 1 else 0,
            'fft_mean_freq_amp': np.abs(np.fft.fft(x['value'].values)[1:len(x['value'])//2]).mean() if len(x) > 1 else 0,
            
            # Difference features: Capture change rates
            'value_diff_mean': x['value'].diff().mean() if len(x) > 1 else 0,
            'value_diff_std': x['value'].diff().std() if len(x) > 1 else 0,
            
            # Lagged features
            'value_lag_1_corr': x['value'].corr(x['value'].shift(1)) if len(x) > 2 else 0,

            # NEW FEATURES
            # Coefficient of Variation (with handling for mean == 0)
            'value_cv': x['value'].std() / (x['value'].mean() + 1e-10),
            
            # Signal-to-Noise Ratio (with handling for std == 0)
            'value_snr': x['value'].mean() / (x['value'].std() + 1e-10),

            # Time-based transformations
            'period_sin': np.sin(2 * np.pi * x['period']).mean(),
            'period_cos': np.cos(2 * np.pi * x['period']).mean(),
            
            # Rolling statistics (using a 3-period window)
            'value_rolling_mean_3': x['value'].rolling(window=3, min_periods=1).mean().iloc[-1] if len(x) >= 1 else 0,
            'value_rolling_std_3': x['value'].rolling(window=3, min_periods=1).std().iloc[-1] if len(x) >= 1 else 0,
            
        }))
        
        # Combine all features
        result = pd.concat([grouped, additional_features], axis=1)
        
        # Replace infinite values with large finite numbers
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        return result

    # Print class distribution before data preprocessing
    print("Class Distribution Before Preprocessing:")
    print(y_train.value_counts())
    print("\n" + "="*50 + "\n")

    # Process training data - enhanced preprocessing
    X_train_processed = enhanced_preprocess(X_train)
    
    # Ensure y_train is aligned with processed X_train
    y_train_aligned = y_train.loc[X_train_processed.index].astype(int)

    # Print class distribution after data preprocessing
    print("Class Distribution After Preprocessing:")
    print(y_train_aligned.value_counts())
    print("\n" + "="*50 + "\n")
    
    # Print shape information
    print(f"X_train shape: {X_train.shape}")
    print(f"X_train_processed shape: {X_train_processed.shape}")
    print(f"y_train_aligned shape: {y_train_aligned.shape}")
    print(f"\nNumber of features: {len(X_train_processed.columns)}")
    print("Features:", X_train_processed.columns.tolist())
    print("\n" + "="*50 + "\n")

    # Split into train and validation sets
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train_aligned, test_size=0.2, random_state=42, stratify=y_train_aligned
    )

    # Scale features
    scaler = StandardScaler()
    X_train_split_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)

    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train_split_scaled, label=y_train_split)
    val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

    # Define LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 160,
        'learning_rate': 0.09,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42,
        'max_depth': 20,
        'lambda_l1': 5,  # L1 regularization
        'lambda_l2': 4,  # L2 regularization

    }

    # Train the model with early stopping
    model = lgb.train(
        params,
        train_data,
        num_boost_round=1500,
        valid_sets=[val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=True),
        ]
    )

    # Save model and scaler
    os.makedirs(model_directory_path, exist_ok=True)
    model_path = os.path.join(model_directory_path, 'lgb_model.txt')
    model.save_model(model_path)
    
    # Save scaler
    scaler_path = os.path.join(model_directory_path, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    
    # Save feature names for inference
    feature_names_path = os.path.join(model_directory_path, 'feature_names.pkl')
    joblib.dump(X_train_processed.columns.tolist(), feature_names_path)

    # Training metrics
    X_train_scaled = scaler.transform(X_train_processed)
    y_train_pred = model.predict(X_train_scaled)
    roc_auc = sklearn.metrics.roc_auc_score(y_train_aligned, y_train_pred)
    print(f"Training ROC AUC: {roc_auc:.4f}")
    print("Classification Report (Train):")
    print(sklearn.metrics.classification_report(y_train_aligned, (y_train_pred > 0.5).astype(int)))
    
    return model, scaler

# ----------------------------------------------------------------------------------------------------------------------
# 2. THE INFER FUNCTION
# ----------------------------------------------------------------------------------------------------------------------

def infer(
    X_test: typing.Iterable[pd.DataFrame],
    model_directory_path: str,
):
    """
    Makes predictions using the trained LightGBM model with enhanced preprocessing
    """
    
    def enhanced_preprocess_infer(df: pd.DataFrame):
        """
        Enhanced preprocessing for inference
        """
        # Basic statistics
        grouped = df.groupby(level='id').agg({
            'value': ['mean', 'std', 'min', 'max', 'median', 'skew', pd.Series.kurt,
                      lambda x: x.quantile(0.05), lambda x: x.quantile(0.95)],
            'period': ['mean', 'std']
        })
        
        # Flatten the multi-index columns and rename new quantile columns
        grouped.columns = [f'{col[0]}_{col[1]}' for col in grouped.columns]
        grouped = grouped.rename(columns={'value_<lambda_0>': 'value_q05', 'value_<lambda_1>': 'value_q95'})
        
        # Additional grouped calculations
        additional_features = df.groupby(level='id').apply(lambda x: pd.Series({
            'value_range': x['value'].max() - x['value'].min(),
            'value_iqr': x['value'].quantile(0.75) - x['value'].quantile(0.25),
            'value_mad': (x['value'] - x['value'].mean()).abs().mean(),
            'value_abs_mean': np.abs(x['value']).mean(),
            'value_abs_std': np.abs(x['value']).std(),
            'value_log_mean': np.log(np.abs(x['value']) + 1e-10).mean(),
            'value_log_std': np.log(np.abs(x['value']) + 1e-10).std(),
            'value_exp_mean': np.exp(x['value']).mean(),
            'value_exp_std': np.exp(x['value']).std(),
            'value_sq_mean': (x['value'] ** 2).mean(),
            'value_sq_std': (x['value'] ** 2).std(),
            'value_sqrt_mean': np.sqrt(np.abs(x['value'])).mean(),
            'value_sqrt_std': np.sqrt(np.abs(x['value'])).std(),
            'value_period_product_mean': (x['value'] * x['period']).mean(),
            'value_period_product_std': (x['value'] * x['period']).std(),
            'value_period_ratio_mean': (x['value'] / (x['period'] + 1e-10)).mean(),
            'value_period_ratio_std': (x['value'] / (x['period'] + 1e-10)).std(),
            'value_mean_times_period_mean': x['value'].mean() * x['period'].mean(),
            'value_std_times_period_std': x['value'].std() * x['period'].std(),
            'total_observations': len(x),
            'period_0_count': (x['period'] == 0).sum(),
            'period_1_count': (x['period'] == 1).sum(),
            'period_ratio': (x['period'] == 1).sum() / len(x) if len(x) > 0 else 0,
            'time_correlation': x.reset_index().corr()['time']['value'] if len(x) > 1 else 0,
            'value_time_slope': np.polyfit(x.reset_index()['time'], x['value'], 1)[0] if len(x) > 1 else 0,
            
            # ADVANCED MATH FEATURES
            'value_entropy': entropy(x['value'].value_counts(normalize=True)) if len(x['value'].unique()) > 1 else 0,
            'period_entropy': entropy(x['period'].value_counts(normalize=True)) if len(x['period'].unique()) > 1 else 0,
            'fft_max_freq_amp': np.abs(np.fft.fft(x['value'].values)[1:len(x['value'])//2]).max() if len(x) > 1 else 0,
            'fft_mean_freq_amp': np.abs(np.fft.fft(x['value'].values)[1:len(x['value'])//2]).mean() if len(x) > 1 else 0,
            'value_diff_mean': x['value'].diff().mean() if len(x) > 1 else 0,
            'value_diff_std': x['value'].diff().std() if len(x) > 1 else 0,
            'value_lag_1_corr': x['value'].corr(x['value'].shift(1)) if len(x) > 2 else 0,

            # NEW FEATURES
            'value_cv': x['value'].std() / (x['value'].mean() + 1e-10),
            'value_snr': x['value'].mean() / (x['value'].std() + 1e-10),
            'period_sin': np.sin(2 * np.pi * x['period']).mean(),
            'period_cos': np.cos(2 * np.pi * x['period']).mean(),
            'value_rolling_mean_3': x['value'].rolling(window=3, min_periods=1).mean().iloc[-1] if len(x) >= 1 else 0,
            'value_rolling_std_3': x['value'].rolling(window=3, min_periods=1).std().iloc[-1] if len(x) >= 1 else 0,
            
        }))
        
        # Combine all features
        result = pd.concat([grouped, additional_features], axis=1)
        
        # Replace infinite values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.fillna(0)
        
        return result

    # Load the trained model
    model_path = os.path.join(model_directory_path, 'lgb_model.txt')
    model = lgb.Booster(model_file=model_path)
    
    # Load scaler
    scaler_path = os.path.join(model_directory_path, 'scaler.pkl')
    scaler = joblib.load(scaler_path)
    
    # Load feature names
    feature_names_path = os.path.join(model_directory_path, 'feature_names.pkl')
    feature_names = joblib.load(feature_names_path)

    # Yield once before the loop (required by Crunch framework)
    yield

    for dataset in X_test:
        # Preprocess the test data
        X_test_processed = enhanced_preprocess_infer(dataset)
        
        # Ensure the test data has the same columns as training data
        X_test_processed = X_test_processed.reindex(columns=feature_names, fill_value=0)
        
        # Scale features
        X_test_scaled = scaler.transform(X_test_processed)
        
        # Make prediction
        prediction = model.predict(X_test_scaled)[0]
        
        yield prediction

# ----------------------------------------------------------------------------------------------------------------------
# 3. LOCAL TESTING
# ----------------------------------------------------------------------------------------------------------------------

#if __name__ == "__main__":
#    # Load data
#    X_train, y_train, X_test = crunch.load_data()
#    
#    # Train the model
#    model_directory_path = "models"
#    model, scaler = train(X_train, y_train, model_directory_path)
#    
#    # Test using crunch
#    crunch.test()
#    
#    # Load saved predictions and evaluate
#    try:
#        prediction = pd.read_parquet("data/prediction.parquet")
#        target = pd.read_parquet("data/y_test.reduced.parquet")["structural_breakpoint"]
#
#        roc_auc = sklearn.metrics.roc_auc_score(target, prediction)
#        print(f"Local Test ROC AUC Score: {roc_auc:.4f}")
#        print("Classification Report (Test):")
#        print(sklearn.metrics.classification_report(target, (prediction > 0.5).astype(int)))
#    except Exception as e:
#        print(f"Error during evaluation: {e}")
