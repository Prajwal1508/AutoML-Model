import streamlit as st
import plotly.express as px
import pandas as pd
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso, Ridge
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier, MLPRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from tpot import TPOTClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
import joblib  # Import joblib

# Set the title of the web app and page icon
st.set_page_config(page_title="AutoML App", page_icon="🧠", layout="wide")

# Header and description
st.header("AutoML web Application")
st.subheader("Upload your dataset and explore various machine learning models.")

# Local image file path
image_path = 'https://www.inteliment.com/wp-content/uploads/2021/05/Automated-Machine-Learning.png'

# Define the desired width for the image (in pixels)
desired_width = 280

# Display the image in the app with the specified width
st.image(image_path, width=desired_width)

def load_data(file, nrows=None):
    return pd.read_csv(file, nrows=nrows)

def train_models(models, X, y):
    best_model_name = None
    best_model_accuracy = 0.0
    best_model_instance = None
    accuracy_scores = []

    for model_name, model in models.items():
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        accuracy = model.score(X_test, y_test)
        accuracy_scores.append(accuracy)

        if accuracy > best_model_accuracy:
            best_model_name = model_name
            best_model_accuracy = accuracy
            best_model_instance = model  # Store the best model instance

        st.write(f"{model_name} Accuracy: {accuracy:.2f}")

    # Save the best model as a .pkl file
    if best_model_instance is not None:
        joblib.dump(best_model_instance, 'best_model.pkl')

    return best_model_name, best_model_accuracy, accuracy_scores

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.csv'):
        try:
            df = load_data(uploaded_file, nrows=1000)
            st.dataframe(df.head())

            if st.button("Generate Data Summary"):
                st.write("### Data Summary")
                profile = ProfileReport(df, explorative=True)
                profile_report = st_profile_report(profile)
                # Save the profile report to a file
                profile_report_file_path = "data_summary_report.html"
                profile.to_file(profile_report_file_path)
                st.success("Data Summary Report Generated Successfully!")

                # Add a download link for the saved data summary report
                download_link = f'<a href="./{profile_report_file_path}" download>Click here to download the Data Summary Report</a>'
                st.markdown(download_link, unsafe_allow_html=True)

            if st.button("Train Models"):
                with st.spinner("Training Models..."):
                    st.write("### Training Machine Learning Models")
                    target_column = st.selectbox("Select the target column", df.columns)
                    if target_column:
                        X = df.drop(columns=[target_column])
                        y = df[target_column]

                        categorical_columns = X.select_dtypes(include=['object']).columns
                        X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

                        imputer = SimpleImputer(strategy='mean')
                        X_imputed = imputer.fit_transform(X_encoded)

                        scaler = MinMaxScaler(feature_range=(0, 1))
                        X_scaled = scaler.fit_transform(X_imputed)

                        label_encoder = LabelEncoder()
                        y_encoded = label_encoder.fit_transform(y)

                        models = {
                            "Decision Tree Classifier": DecisionTreeClassifier(),
                            "Decision Tree Regressor": DecisionTreeRegressor(),
                            "Random Forest Classifier": RandomForestClassifier(),
                            "Random Forest Regressor": RandomForestRegressor(),
                            "Extra Trees Classifier": ExtraTreesClassifier(),
                            "Extra Trees Regressor": ExtraTreesRegressor(),
                            "SVM Classifier": SVC(),
                            "SVM Regressor": SVR(),
                            "K-Neighbors Classifier": KNeighborsClassifier(),
                            "K-Neighbors Regressor": KNeighborsRegressor(),
                            "Logistic Regression": LogisticRegression(),
                            "Linear Regression": LinearRegression(),
                            "Lasso": Lasso(),
                            "Ridge": Ridge(),
                            "Gaussian Naive Bayes": GaussianNB(),
                            "Multinomial Naive Bayes": MultinomialNB(),
                            "MLP Classifier": MLPClassifier(),
                            "MLP Regressor": MLPRegressor(),
                            "XGBoost Classifier": XGBClassifier(),
                            "XGBoost Regressor": XGBRegressor(),
                            "LightGBM Classifier": LGBMClassifier(),
                            "LightGBM Regressor": LGBMRegressor(),
                            "CatBoost Classifier": CatBoostClassifier(),
                            "CatBoost Regressor": CatBoostRegressor(),
                            "TPOT Classifier": TPOTClassifier(verbosity=2, generations=5, population_size=20, random_state=42)
                        }
                        best_model_name, best_model_accuracy, accuracy_scores = train_models(models, X_scaled, y_encoded)
                        st.success("All Models Trained Successfully!")

                        st.write("### Best Model")
                        st.write(f"Model: {best_model_name}")
                        st.write(f"Accuracy: {best_model_accuracy:.2f}")

                        # Create an interactive Plotly bar chart
                        df_plot = pd.DataFrame({'Model': list(models.keys()), 'Accuracy': accuracy_scores})

                        fig = px.bar(df_plot, x='Model', y='Accuracy', text='Accuracy',
                                     labels={'Accuracy': 'Accuracy Score'}, color='Model')

                        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                        fig.update_layout(xaxis_tickangle=-45, xaxis_title='Model', yaxis_title='Accuracy Score',
                                          title='Model Performance')

                        st.plotly_chart(fig, use_container_width=True)  # Adjust chart width to fit the screen
                    else:
                        st.warning("Please select the target column.")
        except Exception as e:
            st.error(f"Error occurred while processing the data: {e}")
    else:
        st.error("Please upload a CSV file.")

