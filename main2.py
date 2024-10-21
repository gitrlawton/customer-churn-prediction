import streamlit as st

# Set page config for wider layout
st.set_page_config(layout="wide", page_title="Customer Profile")

# Sample customer data
customer_data = {
    "name": "Sarah Johnson",
    "credit_score": 750,
    "location": "New York, NY",
    "gender": "Female",
    "age": 34,
    "tenure": 5,
    "num_products": 2,
    "estimated_salary": 75000,
    "has_credit_card": True,
    "is_active": True
}

# Custom CSS to improve the design
st.markdown("""
    <style>
    .card {
        background-color: white;
        border-radius: 10px;
        border: 2px solid black;
        padding: 20px;
        max-width: 600px;
        margin: 0 auto;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .customer-header {
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 0;
        padding-bottom: 0;
    }
    .customer-subheader {
        color: #666;
        font-size: 16px;
        margin-top: 0;
        padding-top: 0;
    }
    .metric-row {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        margin: 10px 0;
        border-bottom: 1px solid #eee;
    }
    .metric-row:last-child {
        border-bottom: none;
    }
    .metric-item {
        flex: 0 0 45%;
        min-width: 150px;
        padding: 1px 1px;
    }
    .metric-label {
        color: #666;
        font-size: 14px;
        margin-bottom: 0px;
    }
    .metric-value {
        font-size: 16px;
        font-weight: 500;
        word-wrap: break-word;
    }
    .status-active {
        color: #28a745;
    }
    .status-inactive {
        color: #666;
    }
    .icon-container {
        background-color: #F0F8FF;
        padding: 10px;
        border-radius: 10%;
        margin-right: 10px;
        display: inline-block;
    }
    @media (max-width: 600px) {
        .customer-header {
            font-size: 24px;
        }
        .customer-subheader {
            font-size: 14px;
        }
        .metric-item {
            flex: 1 1 100%;
        }
    }
    </style>
""", unsafe_allow_html=True)


st.markdown(f"""
    <div style="display: flex; align-items: center; margin-bottom: 5px;">
        <div class="icon-container">👤</div>
        <div style="margin: 5px; padding: 0; margin: 0;">
            <h1 class="customer-header" style="font-size: 22px; padding-top: 5px; margin: 0;">{customer_data['name']}</h1>
            <p class="customer-subheader" style="font-size: 14px; padding-bottom: 5px; margin: 0;">Customer Profile</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Metrics rows
metrics = [
    # Row 1
    [
        ("✔️", "Credit Score", str(customer_data["credit_score"])),
        ("📌", "Location", customer_data["location"])
    ],
    # Row 2
    [
        ("👤", "Demographics", f"{customer_data['age']} years old | {customer_data['gender']}"),
        ("⏱️", "Bank Tenure", f"{customer_data['tenure']} years")
    ],
    # Row 3
    [
        ("📦", "Number of Products", str(customer_data["num_products"])),
        ("💰", "Estimated Salary", f"${customer_data['estimated_salary']:,.2f}")
    ],
    # Row 4
    [
        ("💳", "Credit Card Status", 
            f"""<span class="status-{'active' if customer_data['has_credit_card'] else 'inactive'}">
                {'Has Credit Card' if customer_data['has_credit_card'] else 'No Credit Card'}</span>"""),
        ("⭐", "Member Status", 
            f"""<span class="status-{'active' if customer_data['is_active'] else 'inactive'}">
                {'Active Member' if customer_data['is_active'] else 'Inactive Member'}</span>""")
    ]
]

for row in metrics:
    st.markdown("""
        <div class="metric-row">
            <div class="metric-item">
                <div style="display: flex; align-items: center;">
                    <div class="icon-container">{}</div>
                    <div>
                        <div class="metric-label">{}</div>
                        <div class="metric-value">{}</div>
                    </div>
                </div>
            </div>
            <div class="metric-item">
                <div style="display: flex; align-items: center;">
                    <div class="icon-container">{}</div>
                    <div>
                        <div class="metric-label">{}</div>
                        <div class="metric-value">{}</div>
                    </div>
                </div>
            </div>
        </div>
    """.format(
        row[0][0], row[0][1], row[0][2],
        row[1][0], row[1][1], row[1][2]
    ), unsafe_allow_html=True)

# Close the card div
st.markdown("</div>", unsafe_allow_html=True)
