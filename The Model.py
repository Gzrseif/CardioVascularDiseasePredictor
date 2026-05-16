import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def run_honest_regression():
    print("1. Loading dataset...")
    df = pd.read_csv('cardio_train.csv', sep=';')

    print("2. Cleaning data...")
    # Dropping columns that aren't useful for this specific prediction
    df = df.drop(['id', 'cardio'], axis=1, errors='ignore')
    
    # age from days to years
    df['age'] = df['age'] / 365.25

    # Filtering out impossible physiological numbers (data entry errors)
    # We do this BEFORE dropping ap_lo, because a bad ap_lo reading usually means the whole row is corrupted data.
    
    df = df[(df['ap_hi'] >= 80) & (df['ap_hi'] <= 250)]
    df = df[(df['ap_lo'] >= 50) & (df['ap_lo'] <= 150)]

    print(f"Valid rows remaining: {len(df)}")

    # 3Features (X) and Target (y)
    # dropping the blood pressure data so it learns only from lifestyle and habits:)
    X = df.drop(['ap_hi', 'ap_lo'], axis=1)
    y = df['ap_hi']

    # Split and Trainnn
    print("3. Training the model (this might take a few seconds If you need a bathroom break*_*)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # standard Random Forest
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Evaluating
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("\n=== REGRESSION REPORT ===")
    print(f"Mean Absolute Error (MAE): {mae:.2f} mmHg")
    print(f"R-squared (R2) Score:      {r2:.4f}")
    
    print("\n=== FEATURE IMPORTANCE ===")
    importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    for _, row in importances.iterrows():
        print(f"{row['Feature']:>12}: {row['Importance']:.4f}")

if __name__ == "__main__":
    run_honest_regression()