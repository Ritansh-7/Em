import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv("Salary.csv")

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['Overtime', 'Education level', 'Location', 'Job role', 'Specialization'],
                    drop_first=True)

# Define the features used during training
features_used = [
    'Years of Experience', 'Hours/week', 'Overtime_Yes',
    'Education level_BCA', 'Specialization_Data Science', 'Location_Hyderabad',
    'Job role_Software Dev'
]

# Filter the dataset based on the defined features
df = df[['Salary'] + features_used]

# Split features and target variable
X = df.drop(columns=['Salary'])
y = df['Salary']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor = LinearRegression()
regressor.fit(X_train, y_train)


# Predict function
def predict_salary(years_exp, hours_week, overtime, specialization, education, location, job_role):
    # One-hot encode user inputs
    user_data = {
        'Years of Experience': [years_exp],
        'Hours/week': [hours_week],
        'Overtime_Yes': [1 if overtime.lower() == 'yes' else 0],
        'Education level_BCA': [1 if education == 'BCA' else 0],
        'Specialization_Data Science': [1 if specialization == 'Data Science' else 0],
        'Location_Hyderabad': [1 if location == 'Hyderabad' else 0],
        'Job role_Software Dev': [1 if job_role == 'Software Dev' else 0]
    }

    # Convert user input to DataFrame
    user_df = pd.DataFrame(user_data)

    # Predict salary
    predicted_salary = regressor.predict(user_df)[0]
    return predicted_salary


# Streamlit UI
st.title('Salary Predictor')

# Sidebar for user inputs
st.sidebar.header('Enter Job Details')
years_exp = st.sidebar.slider('Years of Experience', min_value=1, max_value=20, step=1, value=5)
hours_week = st.sidebar.slider('Hours per Week', min_value=1, max_value=100, step=1, value=40)
overtime = st.sidebar.selectbox('Overtime', ['Yes', 'No'])
specialization = st.sidebar.selectbox('Specialization', ['Data Science', 'Cyber Security', 'Development'])
education = st.sidebar.selectbox('Education Level', ['BCA', 'B.TECH'])
location = st.sidebar.selectbox('Location', ['Hyderabad', 'Bengaluru', 'Gurgaon'])
job_role = st.sidebar.selectbox('Job Role', ['Software Dev', 'Data Scientist', 'Risk analyst'])

# Predict salary based on user inputs
predicted_salary = predict_salary(years_exp, hours_week, overtime, specialization, education, location, job_role)

# Display predicted salary
st.subheader('Predicted Salary')
st.write(f'The predicted salary is ${predicted_salary:,.2f}')
image_path = 'Designer.png'
st.image(image_path, caption='Salary Predictor', use_column_width=True)









