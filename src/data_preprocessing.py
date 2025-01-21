import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    # You can add more preprocessing steps as needed
    df = df.dropna()  # Drop missing values
    return df

def split_data(df):
    X = df.drop(['failure', 'timestamp'], axis=1)
    y = df['failure']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test
