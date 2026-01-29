# predictor.py
import pandas as pd


def predict_season(model, dew_point, rel_hum, wind_speed, visibility, press_kpa, weather):
    input_df = pd.DataFrame([{
        'Dew Point Temp_C': dew_point,
        'Rel Hum_%': rel_hum,
        'Wind Speed_km/h': wind_speed,
        'Visibility_km': visibility,
        'Press_kPa': press_kpa,
        'Weather': weather
    }])
    prediction = model.predict(input_df)[0]
    return prediction