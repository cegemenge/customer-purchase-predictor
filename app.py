import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Set page config
st.set_page_config(page_title="Customer Purchase Predictor", page_icon="🛒", layout="centered")

# Apply custom background color
page_bg_img = '''
<style>
body {
background-color: #f0f2f6;
}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)

# Load data
df = pd.read_csv('customer_data.csv')

# Feature Engineering
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Device_Type'] = df['Device_Type'].map({'Mobile': 0, 'Desktop': 1, 'Tablet': 2})
df['High_Time_Spender'] = df['Time_Spent_Minutes'].apply(lambda x: 1 if x > 30 else 0)
df['Pages_Per_Minute'] = df['Pages_Viewed'] / (df['Time_Spent_Minutes'] + 0.01)
df['Is_Young_Buyer'] = df['Age'].apply(lambda x: 1 if x < 30 else 0)
df.drop('CustomerID', axis=1, inplace=True)

# Train Model
X = df.drop('Bought', axis=1)
y = df['Bought']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Streamlit App
st.title("🛒 Customer Purchase Predictor")

st.subheader("Enter Customer Information:")

# Input fields
age = st.slider("Age", 18, 60, 30)
gender = st.selectbox("Gender", ["Male", "Female"])
time_spent = st.slider("Time Spent on Website (minutes)", 1, 60, 10)
pages_viewed = st.slider("Number of Pages Viewed", 1, 20, 5)
device_type = st.selectbox("Device Type", ["Mobile", "Desktop", "Tablet"])

# Processing input
gender = 0 if gender == "Male" else 1
device_type = {"Mobile": 0, "Desktop": 1, "Tablet": 2}[device_type]
high_time_spender = 1 if time_spent > 30 else 0
pages_per_minute = pages_viewed / (time_spent + 0.01)
is_young_buyer = 1 if age < 30 else 0

# Create input array
input_data = np.array([[age, gender, time_spent, pages_viewed, device_type, high_time_spender, pages_per_minute, is_young_buyer]])

# Prediction
if st.button("Predict Purchase"):
    prediction = model.predict(input_data)
    if prediction[0] == 1:
        st.success("✅ The customer is likely to BUY!")
    else:
        st.warning("❌ The customer is NOT likely to buy.")

# Footer
st.markdown("---")
st.markdown("Built by **Odunayo Data Science 🚀**")
