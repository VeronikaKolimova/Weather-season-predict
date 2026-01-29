# data_loader.py
import pandas as pd


def preprocess_dataframe(df_raw):
    n_original_rows = len(df_raw)

    df = df_raw.copy()
    df.columns = [
        'Date/Time', 'Temp_C', 'Dew Point Temp_C', 'Rel Hum_%',
        'Wind Speed_km/h', 'Visibility_km', 'Press_kPa', 'Weather'
    ]

    df['Date/Time_dt'] = pd.to_datetime(df['Date/Time'], format='%m/%d/%Y %H:%M')
    df['Month'] = df['Date/Time_dt'].dt.month

    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Fall', 10: 'Fall', 11: 'Fall'}
    df['Season'] = df['Month'].map(season_map)

    missing_before = df.isnull().sum().sum()
    df_clean = df.dropna()
    dropped_rows = n_original_rows - len(df_clean)

    columns_for_report = [
        'Temp_C', 'Dew Point Temp_C', 'Rel Hum_%',
        'Wind Speed_km/h', 'Visibility_km', 'Press_kPa', 'Weather', 'Season'
    ]
    df_for_report = df_clean[columns_for_report].copy()

    numeric_cols = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%',
                    'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    categorical_cols = ['Weather', 'Season']

    report = {
        "original_shape": (n_original_rows, df_raw.shape[1]),
        "final_shape": df_for_report.shape,
        "dropped_rows": dropped_rows,
        "missing_before": missing_before,
        "missing_after": 0,
        "columns": df_for_report.columns.tolist(),
        "dtypes": {col: str(df_for_report[col].dtype) for col in df_for_report.columns},
        "numeric_describe": df_for_report[numeric_cols].describe().to_dict(),
        "categorical_describe": {col: df_for_report[col].value_counts().to_dict() for col in categorical_cols},
        "season_mapping": {v: k for k, v in season_map.items()}
    }

    return df_clean, df_for_report, report