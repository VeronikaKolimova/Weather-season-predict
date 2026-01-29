# model_trainer.py
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report


def train_models(df):
    feature_columns = [
        'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h',
        'Visibility_km', 'Press_kPa', 'Weather'
    ]
    X = df[feature_columns]
    y = df['Season']

    numeric_features = ['Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    categorical_features = ['Weather']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


    knn_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', KNeighborsClassifier())])
    knn_params = {'classifier__n_neighbors': [3, 5, 7], 'classifier__weights': ['uniform', 'distance']}
    knn_search = GridSearchCV(
        knn_pipe,
        knn_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1)
    knn_search.fit(X_train, y_train)

    dt_pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(random_state=42))
    ])

    dt_params = {
        'classifier__max_depth': [3, 5, 7, None],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }

    dt_search = GridSearchCV(
        dt_pipe,
        dt_params,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    dt_search.fit(X_train, y_train)

    y_pred_knn = knn_search.predict(X_test)
    y_pred_dt = dt_search.predict(X_test)

    acc_knn = accuracy_score(y_test, y_pred_knn)
    acc_dt = accuracy_score(y_test, y_pred_dt)

    results = {
        'KNN': {
            'model': knn_search,
            'accuracy': acc_knn,
            'f1_macro': f1_score(y_test, y_pred_knn, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred_knn, average='weighted'),
            'report': classification_report(y_test, y_pred_knn, output_dict=True),
            'y_pred': y_pred_knn
        },
        'DecisionTree': {
            'model': dt_search,
            'accuracy': acc_dt,
            'f1_macro': f1_score(y_test, y_pred_dt, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred_dt, average='weighted'),
            'report': classification_report(y_test, y_pred_dt, output_dict=True),
            'y_pred': y_pred_dt
        }
    }

    best_model_name = 'KNN' if acc_knn >= acc_dt else 'DecisionTree'
    best_model = results[best_model_name]['model']
    y_pred_best = results[best_model_name]['y_pred']

    return results, best_model_name, best_model, X_test, y_test, y_pred_best