Working with a dataset from a bank to see how likely a customer is to churn
(stop being a customer of the bank).

This project has two parts:

Part 1:
Downloading the dataset from kaggle
Understanding it with visualizations
Preprocessing it
Training ML models to make predicitions of how likely a customer is to churn
Evaluating the accuracy of those models.

Part 2:
Creating a web app the use the ML models to make predictions for new and unseen
customers to see how likely they are the churn.
Taking the models predictions to generate a personalized email to send to the
customer so that they are more likely to stay with the bank using Llama 3.1 and Groq.

Data Set Acknowledgements from Kaggle:

it is much more expensive to sign in a new client than keeping an existing one.
It is advantageous for banks to know what leads a client towards the decision to leave the company.

Churn prevention allows companies to develop loyalty programs and retention campaigns to keep as many customers as possible.

Data Set Properties:

Contains 10,000 customers, each with 10 different features

age, gender, credit score, account balance, tenure with bank
country, number of products, has credit card, is active member, estimated salary
surname, customer id, whether they churned or not

CustomerId Surname CreditScore Geography Gender Age Tenure Balance NumOfProducts HasCrCard IsActiveMember EstimatedSalary Exited

Motivation for Using Machine-Learning:

Because it contains 10,000 data points, it would be very difficult for a human to come up with a rule or equation that determines whether or not a customer will churn. And consider the nuances of the data and understand exactly how the ten different features interact with each other which is crucial for making accurate predictions in a real-world scenario such as this.

Hence, why we're using a ML model to make this prediction. We feed the model the data and it learns the patterns and relationships in the data to make predictions of whether or not the customers will churn. We're giving the computer a large set of data and asking it to learn the patterns in the data so it can make accurate predictions based on new and unseen data.

This is the premise of Machine Learning.
