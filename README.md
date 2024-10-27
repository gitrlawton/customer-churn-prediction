# Customer Churn Prediction

## Overview

This data science project predicts customer churn for a bank using various machine learning models. It utilizes classical machine-learning and generative AI.

This project consists of two parts:

1. A Jupyter Notebook where the models were trained and analyzed.

2. A web application which displays the model predictions and the customer data in a graphical user interface. The application allows the user to input new data and see how that influences their likelihood of churn. The application also provides a detailed explanation of the prediction and visualizations to help the user understand it.

The dataset used in this project was downloaded from:
https://www.kaggle.com/datasets/mathchi/churn-for-bank-customers?resource=download

## Features

- **Customer Data Input**: Users can update customer attributes such as credit score, age, tenure, balance, and more.
- **Churn Prediction**: The application predicts the likelihood of a customer churning based on the input data using multiple machine learning models.
- **Detailed Explanation**: Users receive explanation of the prediction, helping them understand the factors influencing the churn likelihood.
- **Email Generation**: Users can generate personalized email to send to customers identified as likely to churn, offering incentives to remain with the bank.
- **Visualizations**: The application provides visual representations of the churn probabilities and customer percentiles for better insights.

## Installation

To set up the project, ensure you have Python installed on your machine. Then, follow these steps:

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:

   ```bash
   python -m venv .venv
   ```

3. Activate the virtual environment:

   - On Windows (using Command Prompt):
     ```bash
     .venv\Scripts\activate
     ```
   - On Windows (using Git Bash):
     ```bash
     source .venv/Scripts/activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

5. Create a `.env` file in the root directory and add your API keys:

   ```plaintext
   GROQ_API_KEY=your_groq_api_key
   ```

6. Create a folder in the root directory called `models`, if it does not already exist.

## Usage

1. Run all of the cells in notebook.ipynb, in order. This will instantiate, train and save the models to the project directory.

2. Run the Streamlit application:

   ```bash
   streamlit run main.py
   ```

3. Open the provided local URL in your web browser.

4. Select a customer from the dropdown to display their data and predict churn likelihood.

5. Update individual values to see how likelihood of churn changes.

6. Click "Generate Explanation" to explain the prediction.

7. Click "Generate Email" to generate a retention email to send to the customer.

## File Descriptions

- **main.py**: The main application file containing the Streamlit code for user interaction and predictions.
- **utils.py**: Utility functions for creating visualizations and calculating customer percentiles.
- **churn.csv**: Sample dataset containing customer information used for training and testing the models.
- **notebook.ipynb**: Jupyter notebook containing the model training and evaluation code.
- **requirements.txt**: List of required Python packages for the project.

## Dependencies

- **Streamlit**: For building the web application interface.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Scikit-learn**: For machine learning algorithms and model evaluation.
- **Plotly**: For creating interactive visualizations.
- **Jupyter**: For running the Jupyter Notebook used to train and evaluate models.
- **OpenAI**: For providing the API used to prompt a model in Groq's suite of LLMs.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.
