Working with a dataset from a bank to see how likely a customer is to churn
(stop being a customer of the bank).

Link to dataset:
https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download

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

Contains 10,000 customers, each with 13 different properties.

CustomerId—contains random values and has no effect on customer leaving the bank.
Surname—the surname of a customer has no impact on their decision to leave the bank.
CreditScore—can have an effect on customer churn, since a customer with a higher credit score is less likely to leave the bank.
Geography—a customer’s location can affect their decision to leave the bank.
Gender—it’s interesting to explore whether gender plays a role in a customer leaving the bank.
Age—this is certainly relevant, since older customers are less likely to leave their bank than younger ones.
Tenure—refers to the number of years that the customer has been a client of the bank. Normally, older clients are more loyal and less likely to leave a bank.
Balance—also a very good indicator of customer churn, as people with a higher balance in their accounts are less likely to leave the bank compared to those with lower balances.
NumOfProducts—refers to the number of products that a customer has purchased through the bank.
HasCrCard—denotes whether or not a customer has a credit card. This column is also relevant, since people with a credit card are less likely to leave the bank.
IsActiveMember—active customers are less likely to leave the bank.
EstimatedSalary—as with balance, people with lower salaries are more likely to leave the bank compared to those with higher salaries.
Exited—whether or not the customer left the bank.

Motivation for Use of Machine-Learning:

Because it contains 10,000 data points, it would be very difficult for a human to come up with a rule or equation that determines whether or not a customer will churn. And consider the nuances of the data and understand exactly how the different features of a customer interact with each other which is crucial for making accurate predictions in a real-world scenario such as this.

Hence, why we're using a ML model to make this prediction. We feed the model the data and it learns the patterns and relationships in the data to make predictions of whether or not the customers will churn. We're giving the computer a large set of data and asking it to learn the patterns in the data so it can make accurate predictions based on new and unseen data.

This is the premise of Machine Learning.
