# app.py
import streamlit as st
import pandas as pd
from src.data_loader import preprocess_dataframe
from src.model_trainer import train_models
from src.visualizer import plot_confusion_matrix, plot_feature_importance_dt, plot_accuracy_comparison
from src.predictor import predict_season
import os

st.set_page_config(page_title="Предсказание СЕЗОНА", layout="wide")
st.title("🌦️ Прогнозирование сезона по погодным данным")

uploaded_file = st.file_uploader("Загрузите файл weather.csv", type="csv")

if uploaded_file is not None:
    with st.spinner("Загружаем датасет..."):
        df_raw = pd.read_csv(uploaded_file)

    st.subheader("Исходные данные (до предобработки)")

    st.markdown("Первые 5 строк датасета")
    st.dataframe(df_raw.head())

    st.markdown("#### Размер датасета (`df.shape`)")
    st.write(f"Датасет содержит **{df_raw.shape[0]:,} строк** и **{df_raw.shape[1]} столбцов**.")

    st.markdown("#### Типы данных")
    dtypes_raw_df = pd.DataFrame(df_raw.dtypes.astype(str), columns=["Тип"]).reset_index()
    dtypes_raw_df.columns = ["Столбец", "Тип"]
    st.dataframe(dtypes_raw_df, width='stretch')

    st.markdown("#### Пропущенные значения")
    missing_raw = df_raw.isnull().sum()
    missing_raw_df = pd.DataFrame(missing_raw, columns=["Пропусков"]).reset_index()
    missing_raw_df.columns = ["Столбец", "Пропусков"]
    st.dataframe(missing_raw_df, width='stretch')

    st.markdown("---")

    with st.spinner("Выполняем предобработку данных..."):
        df_clean, df_for_report, preprocessing_report = preprocess_dataframe(df_raw)

    st.subheader("Данные после предобработки")

    st.markdown("#### Сводка по очистке")
    st.write(f"- **Исходный размер:** {preprocessing_report['original_shape'][0]:,} строк × {preprocessing_report['original_shape'][1]} столбцов")
    st.write(f"- **Финальный размер:** {preprocessing_report['final_shape'][0]:,} строк × {preprocessing_report['final_shape'][1]} столбцов")
    st.write(f"- **Удалено строк с пропусками:** {preprocessing_report['dropped_rows']:,}")

    st.markdown("#### Типы данных после предобработки")
    dtypes_df = pd.DataFrame(list(preprocessing_report['dtypes'].items()), columns=["Столбец", "Тип"])
    st.dataframe(dtypes_df, width='stretch')

    st.markdown("#### Описательная статистика")

    st.markdown("##### Числовые признаки")
    numeric_desc_df = pd.DataFrame(preprocessing_report['numeric_describe'])
    st.dataframe(numeric_desc_df, width='stretch')

    st.markdown("##### Категориальные признаки")
    for col, counts in preprocessing_report['categorical_describe'].items():
        st.markdown(f"**{col}**")
        counts_df = pd.DataFrame(list(counts.items()), columns=[col, "Количество"])
        st.dataframe(counts_df, width='stretch')

    st.markdown("---")

    with st.spinner("Обучение моделей..."):
        results, best_model_name, best_model, X_test, y_test, y_pred_best = train_models(df_clean)

    st.subheader("МЕТРИКИ МОДЕЛЕЙ")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### KNN")
        st.write(f"**Accuracy:** {results['KNN']['accuracy']:.4f}")
        st.write(f"**F1 (macro):** {results['KNN']['f1_macro']:.4f}")
        st.write(f"**F1 (weighted):** {results['KNN']['f1_weighted']:.4f}")
        st.write("**Classification Report:**")
        knn_report_df = pd.DataFrame(results['KNN']['report']).transpose()
        st.dataframe(knn_report_df.style.format("{:.4f}"))

    with col2:
        st.markdown("### Decision Tree")
        st.write(f"**Accuracy:** {results['DecisionTree']['accuracy']:.4f}")
        st.write(f"**F1 (macro):** {results['DecisionTree']['f1_macro']:.4f}")
        st.write(f"**F1 (weighted):** {results['DecisionTree']['f1_weighted']:.4f}")
        st.write("**Classification Report:**")
        dt_report_df = pd.DataFrame(results['DecisionTree']['report']).transpose()
        st.dataframe(dt_report_df.style.format("{:.4f}"))

    st.success(f"Лучшая модель: **{best_model_name}** (Accuracy: {results[best_model_name]['accuracy']:.4f})")

    st.subheader("Визуализация")

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    cm_img = plot_confusion_matrix(
        y_test, y_pred_best, best_model.classes_,
        f"Матрица ошибок ({best_model_name})",
        save_path=os.path.join(output_dir, "confusion_matrix.png")
    )
    st.image(cm_img, caption="Матрица ошибок")

    fi_img = None
    if best_model_name == "DecisionTree":
        fi_img = plot_feature_importance_dt(
            best_model.best_estimator_,
            ['Точка росы Temp_C', 'Отн.влажнсть_%', 'Скорость ветра_km/h', 'Видимость_km', 'Атмосф.Давление_kPa'],
            "Важность признаков (Дерево решений)",
            save_path=os.path.join(output_dir, "feature_importance.png")
        )
        if fi_img:
            st.image(fi_img, caption="Важность признаков")

    acc_img = plot_accuracy_comparison(
        results['KNN']['accuracy'],
        results['DecisionTree']['accuracy'],
        save_path=os.path.join(output_dir, "model_comparison.png")
    )
    st.image(acc_img, caption="Сравнение точности моделей")

    st.subheader("Предсказание СЕЗОНА")
    with st.form("prediction_form"):
        dew = st.number_input("Dew Point Temp (°C)", value=5.0)
        hum = st.number_input("Relative Humidity (%)", value=80)
        wind = st.number_input("Wind Speed (km/h)", value=15)
        vis = st.number_input("Visibility (km)", value=10.0)
        press = st.number_input("Pressure (kPa)", value=101.3)
        weather = st.selectbox("Weather", options=df_clean['Weather'].unique())
        submitted = st.form_submit_button("Predict Season")  # ← кнопка есть!

    if submitted:
        pred = predict_season(
            best_model.best_estimator_,
            dew, hum, wind, vis, press, weather
        )
        st.success(f"Предсказанный СЕЗОН: **{pred}**")

    import json
    from datetime import datetime

    metrics = {
        "KNN": {
            "accuracy": float(results["KNN"]["accuracy"]),
            "f1_macro": float(results["KNN"]["f1_macro"]),
            "f1_weighted": float(results["KNN"]["f1_weighted"]),
            "best_params": results["KNN"]["model"].best_params_
        },
        "DecisionTree": {
            "accuracy": float(results["DecisionTree"]["accuracy"]),
            "f1_macro": float(results["DecisionTree"]["f1_macro"]),
            "f1_weighted": float(results["DecisionTree"]["f1_weighted"]),
            "best_params": results["DecisionTree"]["model"].best_params_
        },
        "best_model": best_model_name,
        "timestamp": datetime.now().isoformat()
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)

    knn_report_df.to_csv(os.path.join(output_dir, "classification_report_knn.csv"))
    dt_report_df.to_csv(os.path.join(output_dir, "classification_report_dt.csv"))

    st.subheader("Скачать результаты")

    with open(metrics_path, "r", encoding="utf-8") as f:
        st.download_button("Скачать метрики (JSON)", f.read(), "metrics.json", "application/json")

    with open(os.path.join(output_dir, "classification_report_knn.csv"), "rb") as f:
        st.download_button("Скачать отчёт KNN (CSV)", f.read(), "classification_report_knn.csv", "text/csv")

    with open(os.path.join(output_dir, "classification_report_dt.csv"), "rb") as f:
        st.download_button("Скачать отчёт Decision Tree (CSV)", f.read(), "classification_report_dt.csv", "text/csv")