import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Function to load and preprocess data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    data = data.drop(['id'], axis=1)

    # Fill missing values
    data['bmi'].fillna(data['bmi'].mean(), inplace=True)
    data['smoking_status'].fillna('Missing', inplace=True)
    
    # Handle outliers for BMI
    Q1 = data['bmi'].quantile(0.25)
    Q3 = data['bmi'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['bmi'] = np.where(data['bmi'] > upper_bound, upper_bound, data['bmi'])
    data['bmi'] = np.where(data['bmi'] < lower_bound, lower_bound, data['bmi'])
    
    # Handle outliers for avg_glucose_level
    Q1 = data['avg_glucose_level'].quantile(0.25)
    Q3 = data['avg_glucose_level'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data['avg_glucose_level'] = np.where(data['avg_glucose_level'] < lower_bound, lower_bound, data['avg_glucose_level'])
    data['avg_glucose_level'] = np.where(data['avg_glucose_level'] > upper_bound, upper_bound, data['avg_glucose_level'])
    
    # Encode categorical variables
    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    label_encoders = {col: LabelEncoder() for col in categorical_columns}
    
    for col in categorical_columns:
        data[col] = label_encoders[col].fit_transform(data[col].astype(str))
    
    return data

# Function to add sidebar
def add_sidebar():
    st.sidebar.header("Cerebral Health Measurements")
    data = get_clean_data()
    
    slider_labels = [
        ("Age", "age"),
        ("Average Glucose Level", "avg_glucose_level"),
        ("BMI", "bmi")
    ]

    dropdown_labels = [
        ("Hypertension", "hypertension", [0, 1]),
        ("Heart Disease", "heart_disease", [0, 1]),
        ("Gender", "gender", ['Male', 'Female']),
        ("Ever Married", "ever_married", ['No', 'Yes']),
        ("Work Type", "work_type", ['Children', 'Government Job', 'Never Worked', 'Private', 'Self-employed']),
        ("Residence Type", "Residence_type", ['Rural', 'Urban']),
        ("Smoking Status", "smoking_status", ['formerly smoked', 'never smoked', 'smokes', 'Missing'])
    ]

    input_dict = {}

    for index, (label, key) in enumerate(slider_labels):
        if index == 0:  
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=1,
                max_value=100,
                value=85
            )
        elif index == 1:  
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(50),
                max_value=float(250),
                value=float(90)
            )
        else:  
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=float(10),
                max_value=float(100),
                value=float(45)
            )
    
    for label, key, options in dropdown_labels:
        input_dict[key] = st.sidebar.selectbox(
            label,
            options,
            index=0
        )
    
    return input_dict

# Function to scale values
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['stroke'], axis=1)
    
    scaled_dict = {}
    
    for key, value in input_dict.items():
        if key in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
            scaled_dict[key] = value
        else:
            max_val = X[key].max()
            min_val = X[key].min()
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value
    
    return scaled_dict

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    feature_names = pickle.load(open("model/feature_names.pkl", "rb"))

    # Convert categorical inputs to numerical using the saved encoders
    encoders = {
        "gender": LabelEncoder().fit(['Male', 'Female']),
        "ever_married": LabelEncoder().fit(['No', 'Yes']),
        "work_type": LabelEncoder().fit(['Children', 'Government Job', 'Never Worked', 'Private', 'Self-employed']),
        "Residence_type": LabelEncoder().fit(['Rural', 'Urban']),
        "smoking_status": LabelEncoder().fit(['formerly smoked', 'never smoked', 'smokes', 'Missing'])
    }
    
    for key in encoders.keys():
        input_data[key] = encoders[key].transform([input_data[key]])[0]

    input_df = pd.DataFrame([input_data])
    
    # One-hot encoding for categorical variables to match the training data structure
    input_df = pd.get_dummies(input_df, columns=["gender", "ever_married", "work_type", "Residence_type", "smoking_status"], drop_first=True)
    
    # Ensure all expected columns are present
    for col in feature_names:
        if col not in input_df.columns:
            input_df[col] = 0
    
    input_df = input_df[feature_names]
    
    # Scale numerical features
    numerical_cols = ['age', 'bmi', 'avg_glucose_level']
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    input_array = input_df.values
    
    prediction_prob = model.predict_proba(input_array)[0]
    prediction = (prediction_prob[1] > 0.45).astype(int) 
    
    st.subheader("Stroke Prediction")
    st.markdown(
        f"""
        <div class="prediction-output">
        {"<span class='diagnosis benign'>No Stroke</span>" if prediction == 0 else "<span class='diagnosis malicious'>Stroke</span>"}
        <p class='assist'>This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

# Main function
def main():
    st.set_page_config(
        page_title="Cerebral Health Predictor",
        page_icon=":brain:",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
    
    input_data = add_sidebar()
    
    with st.container():
        st.title("Cerebral Health Predictor")
        st.write("This app predicts the likelihood of a stroke based on various health measurements.")
    
    add_predictions(input_data)

if __name__ == '__main__':
    main()