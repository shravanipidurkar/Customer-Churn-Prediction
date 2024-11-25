import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Function to Load Data
@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv('churn-bigml-20.csv')  # Replace with your dataset file

# Function to Train and Save the Model
def train_and_evaluate_model(df):
    # Preprocessing and splitting
    X = df.drop(['Churn', 'State'], axis=1)
    y = df['Churn']

    categorical_features = ['International plan', 'Voice mail plan']
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features),
                      ('num', 'passthrough', numerical_features)]
    )

    X_preprocessed = preprocessor.fit_transform(X)

    smote = SMOTE(random_state=42)
    X_balanced, y_balanced = smote.fit_resample(X_preprocessed, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced, y_balanced, test_size=0.2, random_state=42, stratify=y_balanced
    )

    # Train the model
    classifier = RandomForestClassifier(random_state=42)
    classifier.fit(X_train, y_train)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
    joblib.dump(pipeline, 'churn_model.pkl')

    # Predictions and evaluation
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=False)  # Fixed this line
    confusion = confusion_matrix(y_test, y_pred)

    return accuracy, report, confusion

# Function to Load the Trained Model
@st.cache(allow_output_mutation=True)
def load_model():
    return joblib.load('churn_model.pkl')

# Streamlit Web Interface
st.title("Customer Churn Prediction")

# Step 1: Load Data
df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())

# Step 2: Visualize Data
st.write("### Churn Distribution")
fig, ax = plt.subplots()
sns.countplot(x='Churn', data=df, ax=ax)
ax.set_title("Churn vs Non-Churn Customers")
st.pyplot(fig)

# Step 3: Train the Model
st.subheader("Model Training")
if st.button("Train Model"):
    try:
        with st.spinner("Training model..."):
            accuracy, report, confusion = train_and_evaluate_model(df)
        st.success(f"Model Trained Successfully! Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)
        st.text("Confusion Matrix:")
        st.write(confusion)
    except ValueError as e:
        st.error(str(e))  # Display error if there's a class imbalance

# Step 4: Load the Trained Model
model = None
try:
    model = load_model()
except:
    st.warning("Please train the model first by clicking the 'Train Model' button.")

# Step 5: Make Predictions
st.subheader("Make a Prediction")
if model:
    # Collect user input for prediction
    account_length = st.number_input('Account Length', min_value=0, value=100)
    area_code = st.selectbox('Area Code', df['Area code'].unique())
    intl_plan = st.selectbox('International Plan', ['No', 'Yes'])
    voice_mail_plan = st.selectbox('Voice Mail Plan', ['No', 'Yes'])
    num_vmail_messages = st.number_input('Number of Voicemail Messages', min_value=0, value=0)
    total_day_minutes = st.number_input('Total Day Minutes', min_value=0.0, value=180.0)
    total_day_calls = st.number_input('Total Day Calls', min_value=0, value=100)
    total_day_charge = st.number_input('Total Day Charge', min_value=0.0, value=30.0)
    total_eve_minutes = st.number_input('Total Evening Minutes', min_value=0.0, value=200.0)
    total_eve_calls = st.number_input('Total Evening Calls', min_value=0, value=100)
    total_eve_charge = st.number_input('Total Evening Charge', min_value=0.0, value=20.0)
    total_night_minutes = st.number_input('Total Night Minutes', min_value=0.0, value=200.0)
    total_night_calls = st.number_input('Total Night Calls', min_value=0, value=100)
    total_night_charge = st.number_input('Total Night Charge', min_value=0.0, value=10.0)
    total_intl_minutes = st.number_input('Total International Minutes', min_value=0.0, value=10.0)
    total_intl_calls = st.number_input('Total International Calls', min_value=0, value=5)
    total_intl_charge = st.number_input('Total International Charge', min_value=0.0, value=2.0)
    customer_service_calls = st.number_input('Customer Service Calls', min_value=0, value=1)

    if st.button("Predict Churn"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Account length': [account_length],
            'Area code': [area_code],
            'International plan': [intl_plan],
            'Voice mail plan': [voice_mail_plan],
            'Number vmail messages': [num_vmail_messages],
            'Total day minutes': [total_day_minutes],
            'Total day calls': [total_day_calls],
            'Total day charge': [total_day_charge],
            'Total eve minutes': [total_eve_minutes],
            'Total eve calls': [total_eve_calls],
            'Total eve charge': [total_eve_charge],
            'Total night minutes': [total_night_minutes],
            'Total night calls': [total_night_calls],
            'Total night charge': [total_night_charge],
            'Total intl minutes': [total_intl_minutes],
            'Total intl calls': [total_intl_calls],
            'Total intl charge': [total_intl_charge],
            'Customer service calls': [customer_service_calls]
        })

        try:
            # Use the trained pipeline to transform and predict
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1]

            # Display result
            if prediction == 1:
                st.error(f"**Customer is likely to churn!** (Probability: {prediction_proba:.2f})")
            else:
                st.success(f"**Customer is not likely to churn.** (Probability: {1 - prediction_proba:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
