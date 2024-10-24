import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI
import utils as ut

# Set page config
st.set_page_config(
    layout="wide",
    page_title="Customer Churn Prediction",
    # initial_sidebar_state="expanded",
    # menu_items={
    #     'Get Help': 'https://www.extremelycoolapp.com/help',
    #     'Report a bug': "https://www.extremelycoolapp.com/bug",
    #     'About': "# This is a header. This is an *extremely* cool app!"
    # }
)

# Custom CSS
st.markdown("""
<style>
    .element-container {
        margin-bottom: 0.3rem !important;
    }
    .row-widget.stCheckbox {
        margin-bottom: -1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Load environment variables from .env file
load_dotenv()

def load_model(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)
    
client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.getenv("GROQ_API_KEY")
)
    
# Instantiate the models.
xgboost_model = load_model("xgb_model.pkl")
naive_bayes_model = load_model("nb_model.pkl")
random_forest_model = load_model("rf_model.pkl")
decision_tree_model = load_model("dt_model.pkl")
svm_model = load_model("svm_model.pkl")
knn_model = load_model("knn_model.pkl")
voting_clf_model = load_model("voting_clf_hard.pkl")
xgboost_smote_model = load_model("xgb_model_fe_smote.pkl")
xgboost_feature_engineered_model = load_model("xgb_model_feature_engineered.pkl")

# Prepare the input data for the models.
# Takes in the customer attributes and returns a dataframe and a dictionary to make
# predictions with our models.
def prepare_input(credit_score, location, gender, age, tenure, balance, 
                  num_products, has_credit_card, is_active_member, estimated_salary):
    
    input_dict = {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active_member),
        "EstimatedSalary": estimated_salary,
        "Geography_France": 1 if location == "France" else 0,
        "Geography_Germany": 1 if location == "Germany" else 0,
        "Geography_Spain": 1 if location == "Spain" else 0,
        "Gender_Male": 1 if gender == "Male" else 0,
        "Gender_Female": 1 if gender == "Female" else 0
    }
    
    input_df = pd.DataFrame([input_dict])
    
    return input_df, input_dict

# Define a function to make predictions using the ML models we trained.
# Takes in the input dataframe and input dictionary from prepare_input().
def make_predictions(input_df, input_dict, customer_percentiles):
    
    # Dictionary representing the predictions of the probabilities for the models.  
    # predict_proba() returns an array of predicted probabilities for each
    # class.  This scenario has two classes, 0 and 1, also called the negative
    # class and the positive class, where negative is "retained" and positive is
    # "churned".
    # We only care about the positive value (the second value, corresponding to
    # predicting who will churn), so we store the value at index 1, and not the 
    # one at index 0.
    # We're using the models we trained, so it knows what the target value is
    # (whether the customer exited or not), and that is binary (either 0 or 1.)
    probabilities = {
        "XGBoost": xgboost_model.predict_proba(input_df)[0][1],
        "Random Forest": random_forest_model.predict_proba(input_df)[0][1],
        "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1]
    }
    
    # Average the predictions together.
    avg_probability = np.mean(list(probabilities.values()))
    
    col1, col2 = st.columns(2)

    with col1:
        fig = ut.create_gauge_chart(avg_probability)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig_probs = ut.create_model_probability_chart(probabilities)
        st.plotly_chart(fig_probs, use_container_width=True)
    
    fig_percentile = ut.create_customer_percentiles_chart(customer_percentiles)
    st.plotly_chart(fig_percentile, use_container_width=True)
    
    return avg_probability

def explain_prediction(probability, input_dict, surname):

    
    chance_of_churning = "low" if round(probability * 100, 1) < 40 else "high"
    
    prompt = f"""
    A customer named {surname} has 
    a {chance_of_churning} chance of churning, based on their data as 
    a customer, here: {input_dict}
    
    
    - The number of products the customer has (NumOfProducts)
    - If the customer is an active member (IsActiveMember)
    - The customer's age (Age)
    - Their account balance (Balance)


    Your job:
    Explain why {surname} has a {chance_of_churning} chance of churning, based on their data, following
    this format:
    
    **Number of products**\n
    
    ...

    **Customer's age**\n
    
    ...
    
    **Account balance**\n
    
    ...

    **Tenure with the bank**\n

    ...

    **Active member status**\n

    ...

    **Conclusion**\n

    Reiterate that the customer aligns with a **____** likelihood of churn in a short, single sentence.

    """

    print("EXPLANATION PROMPT", prompt)

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content

def generate_email(probability, input_dict, explanation, surname):
    print("Probability of churn in generate_email:", round(probability * 
            100, 1))
    
    prompt = f"""You are a manager at a bank. You are responsible for 
            ensuring customers stay with the bank.

            You noticed a customer named {surname} has a {round(probability * 
            100, 1)}% risk of churning.

            Here is the customer's information:
            {input_dict}

            Here is an explanation as to why the customer might be at risk 
            of churning:
            {explanation}

            If their risk of churning is greater than 40%, generate an email to the customer 
            based on their information, asking them to stay and offering them
            incentives so that they become more loyal to the bank. You want to make 
            the email as enticing as possible to the customer.
            
            Make sure to list out a set of incentives to stay based on their 
            information, in bullet point format. 
            
            If their risk of churning is less than 40%, simply Mention their probability of churning 
            and whether it is greater than 40%.  Do not write an email.
            """

    raw_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    print("\n\nEMAIL PROMPT", prompt)

    return raw_response.choices[0].message.content
    

st.title("Customer Churn Prediction")

# Read in the csv file.
churn_df = pd.read_csv("churn.csv")

# Create a list of customers for the dropdown menu.
customers = [
    # Iterates through each row of the dataframe and creates a string for each
    # customer, in the form "Customer ID - Surname".
    f"{row['Surname']} [{row['CustomerId']}]" for _, row in churn_df.iterrows()
]

# Display the list of customers in a dropdown menu.
selected_customer_option = st.selectbox("Customer [ID] (Select one)", customers)

# When a user selects a customer, we want to store the ID and surname in separate
# variables.
if selected_customer_option:
    selected_customer_id = int(selected_customer_option.split("[")[1].strip("]"))
    
    selected_customer_surname = selected_customer_option.split(" [")[0]
    
    # Filter the dataframe to get all the data associated with the selected 
    # customer, using the loc accessor from the dataframe.
    selected_customer = churn_df.loc[churn_df["CustomerId"] == selected_customer_id].iloc[0]
    print("Selected Customer:\n", selected_customer)
    
    # Create four columns for us to add UI elements to.
    col1, col2, col3, col4 = st.columns(4)

    # Display the customers attributes in four columns
    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=850,
            value=int(selected_customer["CreditScore"])
        )
        
        age = st.number_input(
            "Age",
            min_value=10,
            max_value=100,
            value=int(selected_customer["Age"])
        )
        

    with col2:
        location = st.selectbox(
            "Location", 
            ["Spain", "France", "Germany"],
            index=["Spain", "France", "Germany"].index(
                selected_customer["Geography"]
            )
        )
        
        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=50,
            value=int(selected_customer["Tenure"])
        )
        

    with col3:
        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=float(selected_customer["Balance"])
        )
        
        num_products = st.number_input(
            "Number of Products",
            min_value=1,
            max_value=10,
            value=int(selected_customer["NumOfProducts"])
        )

    with col4:
        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=float(selected_customer["EstimatedSalary"])
        )
        
        # Create two sub-columns within col4
        subcol1, subcol2 = st.columns(2)
        
        with subcol1:
            gender = st.radio(
                "Gender",
                ["Male", "Female"],
                index=0 if selected_customer["Gender"] == "Male" else 1
            )
        
        with subcol2:
            st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Customer Status</p>", unsafe_allow_html=True) 
            has_credit_card = st.checkbox(
                "Credit Card",
                value=bool(selected_customer["HasCrCard"])
            )
            
            is_active_member = st.checkbox(
                "Active Member",
                value=bool(selected_customer["IsActiveMember"])
            )
    
    input_df, input_dict = prepare_input(
        credit_score, 
        location, 
        gender, 
        age, 
        tenure, 
        balance, 
        num_products, 
        has_credit_card, 
        is_active_member, 
        estimated_salary
    )
    
    customer_percentiles = ut.calculate_percentiles(selected_customer_id, churn_df)
    
    avg_probability = make_predictions(input_df, input_dict, customer_percentiles)
    
    explanation = explain_prediction(
            avg_probability, input_dict, selected_customer['Surname']
        )

    st.markdown("---")

    st.subheader("Explanation of Prediction")

    st.markdown(explanation)
    
    email = generate_email(
            avg_probability, input_dict, explanation, selected_customer["Surname"]
        )
    
    st.markdown("---")
    
    st.subheader("Personalized Email")

    st.markdown(email)