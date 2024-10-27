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
xgboost_model = load_model("models/xgb_model.pkl")
naive_bayes_model = load_model("models/nb_model.pkl")
random_forest_model = load_model("models/rf_model.pkl")
decision_tree_model = load_model("models/dt_model.pkl")
svm_model = load_model("models/svm_model.pkl")
knn_model = load_model("models/knn_model.pkl")
voting_clf_model = load_model("models/voting_clf_hard.pkl") #
xgboost_smote_model = load_model("models/xgb_model_fe_smote.pkl")
xgboost_feature_engineered_model = load_model("models/xgb_model_fe.pkl")

# Instantiate additional models.
gradient_boosting_model = load_model("models/gradient_boosting_clf.pkl")
bagging_model = load_model("models/bagging_clf.pkl")
stacking_model = load_model("models/stacking_clf.pkl")

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
        "K-Nearest Neighbors": knn_model.predict_proba(input_df)[0][1],
        "Gradient Boosting": gradient_boosting_model.predict_proba(input_df)[0][1],
        "Bagging": bagging_model.predict_proba(input_df)[0][1],
        "Stacking": stacking_model.predict_proba(input_df)[0][1],
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

    # Extracting values from dictionary of customer data.
    credit_score = input_dict['CreditScore']
    age = input_dict['Age']
    tenure = input_dict['Tenure']
    balance = f"{input_dict['Balance']:,.2f}"
    number_of_products = f"{input_dict['NumOfProducts']} product" if input_dict['NumOfProducts'] == 1 else f"{input_dict['NumOfProducts']} products"
    estimated_salary = f"{input_dict['EstimatedSalary']:,.2f}"

    # Converting binary values for HasCrCard and IsActiveMember
    credit_card = "have a credit card" if input_dict['HasCrCard'] == 1 else "do not have a credit card"
    active_member = "active member" if input_dict['IsActiveMember'] == 1 else "inactive member"

    # Mapping Gender based on dictionary values
    if input_dict['Gender_Male'] == 1:
        gender = "Male"
    else:
        gender = "Female"

    # Mapping Geography based on dictionary values
    if input_dict['Geography_France'] == 1:
        location = "France"
    elif input_dict['Geography_Germany'] == 1:
        location = "Germany"
    else:
        location = "Spain"
    
    if round(probability * 100, 1) > 35:
        churn_likelihood = "high"
    else:
        churn_likelihood = "low"
    
    
    prompt = f"""You are an expert data scientist at a bank, where you specialize 
        in interpreting and explaining predictions about customers data.
        
        It is predicted that a customer named {surname} 
        has a relatively {churn_likelihood} chance of churning.  They are a {age} 
        {gender} from {location}, who has been with the bank for {tenure} years.  
        They have a credit score of {credit_score}, ${balance} in their account, 
        and an estimated salary of ${estimated_salary}.  They also {credit_card},  
        have {number_of_products} with the bank, and are an {active_member} 
        of the bank.  
        
        Explain which characteristics of the customer support the prediction.  
        Do not try to prove otherwise.
        
        IMPORTANT - Do not mention a model, and do not use any special text formatting 
        like italics.
        
        At the end, reiterate that the customer aligns with a **[high/low]** 
        likelihood of churn in a short, single sentence.\n
        
        """


    raw_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

    return raw_response.choices[0].message.content

def generate_email(input_dict, surname):
    print("Input_dict:", input_dict)
    
    prompt = f"""You are a manager at a bank. You are responsible for 
            ensuring customers stay with the bank.
            
            Write an email to a customer named {surname}, who you are afraid
            may leave the bank. Let them know they are a valued member, 
            and extend to them a number of incentives. 
            You want to make the email as enticing as possible to the customer.
            
            """

    raw_response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{
            "role": "user", 
            "content": prompt
        }],
    )

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
    

    # Initialize session states
    if 'explanation_generated' not in st.session_state:
        st.session_state.explanation_generated = False
    if 'explanation_text' not in st.session_state:
        st.session_state.explanation_text = ""
    if 'previous_customer' not in st.session_state:
        st.session_state.previous_customer = selected_customer['CustomerId']
    if 'email_generated' not in st.session_state:
        st.session_state.email_generated = False
    if 'email_text' not in st.session_state:
        st.session_state.email_text = ""

    # Reset states when customer changes
    if st.session_state.previous_customer != selected_customer['CustomerId']:
        st.session_state.explanation_generated = False
        st.session_state.explanation_text = ""
        st.session_state.email_generated = False
        st.session_state.email_text = ""
        st.session_state.previous_customer = selected_customer['CustomerId']

    st.markdown("---")
    st.subheader("Explanation of Prediction")

    explanation_placeholder = st.empty()
    generate_explanation_button_placeholder = st.empty()

    if not st.session_state.explanation_generated:
        if generate_explanation_button_placeholder.button("Generate Explanation"):
            explanation = explain_prediction(
                avg_probability, input_dict, selected_customer['Surname']
            )
            st.session_state.explanation_text = explanation
            st.session_state.explanation_generated = True
            generate_explanation_button_placeholder.empty()  # Hide button
            st.markdown(explanation)
    else:
        st.markdown(st.session_state.explanation_text)

    st.markdown("---")

    st.subheader("Personalized Email")

    if avg_probability > .35:
        email = generate_email(input_dict, selected_customer["Surname"])
        st.session_state.email_text = email
        st.session_state.email_generated = True
        st.markdown(email)
    else:
        if not st.session_state.email_generated:
            text = f"""{selected_customer['Surname']} has a relatively low likelihood 
            of churning.  If you'd like to generate an email to send to them anyway, 
            click "Generate Email"."""
            
            text_placeholder = st.empty()
            text_placeholder.markdown(text)
            
            generate_email_button_placeholder = st.empty()
            
            if generate_email_button_placeholder.button("Generate Email"):
                email = generate_email(input_dict, selected_customer["Surname"])
                st.session_state.email_text = email
                st.session_state.email_generated = True
                text_placeholder.empty()    # Hide the original text
                generate_email_button_placeholder.empty()  # Hide button
                st.markdown(email)
        else:
            st.markdown(st.session_state.email_text)