import plotly.graph_objects as go
import pandas as pd

def create_gauge_chart(probability):
    # Determine color based on churn probability
    if probability < 0.3:
        color = "green"
    elif probability < 0.6:
        color = "yellow"
    else:
        color = "red"

    # Create a gauge chart
    fig = go.Figure(
        go.Indicator(mode="gauge",
            value=probability * 100,
            domain={
                'x': [0, 1],
                'y': [0, 1]
            },
            title={
                'font': {
                    'size': 24,
                    'color': 'white'
                }
            },
            number={'font': {
                'size': 40,
                'color': 'white'
            }},
            gauge={
                'axis': {
                    'range': [0, 100],
                    'tickwidth': 1,
                    'tickcolor': "white"
                },
                'bar': {
                    'color': color
                },
                'bgcolor': 'rgba(0,0,0,0)',
                'borderwidth': 2,
                'bordercolor': 'white',
                'steps': [{
                    'range': [0, 30],
                    'color': 'rgba(0, 255, 0, 0.3)'
                }, {
                    'range': [30, 60],
                    'color': 'rgba(255, 255, 0, 0.3)'
                }, {
                    'range': [60, 100],
                    'color': 'rgba(255, 0, 0, 0.3)'
                }],
                'threshold': {
                    'line': {
                        'color': "white",
                        'width': 4
                    },
                    'thickness': 0.75,
                    'value': 100
                }
            }
    ))

    # Update chart layout
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "white"},
        width=400,
        height=415,
        margin=dict(l=20, r=40, t=50, b=20))
    
    # Add a percentage sign to the displayed value
    fig.add_annotation(
        text=f"{probability * 100:.2f}%",  # Display the value with a % sign
        font=dict(size=40, color="white"),
        showarrow=False,
        x=0.5,
        y=0,
        xref="paper",
        yref="paper"
    )
    
    # Add a title
    fig.add_annotation(
        text="Churn Probability",  # Display the value with a % sign
        font=dict(size=30, color="white"),
        showarrow=False,
        x=0.5,
        y=1.15,
        xref="paper",
        yref="paper"
    )
    
    return fig

def create_model_probability_chart(probabilities):
    models = list(probabilities.keys())
    probs = list(probabilities.values())
    
    fig = go.Figure(data=[
        go.Bar(y=models,
              x=probs,
              orientation='h',
              text=[f'{p:.2%}' for p in probs],
              textposition='auto',
        )
    ])
    
    fig.update_layout(title='Churn Probability by Model',
                     yaxis_title='Models',
                     xaxis_title='Probability',
                     xaxis=dict(tickformat='.0%', range=[0, 1]),
                     height=400,
                     margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig
    
def calculate_percentiles(customer_id, df):
    
    customer_features = ['CreditScore', 'Tenure', 'EstimatedSalary', 'Balance', 'NumOfProducts']
    
    customer_percentiles = dict()
    
    for feature in customer_features:
        rank = df[feature].rank().loc[df['CustomerId'] == customer_id].values[0]
        customer_percentiles[feature] = (rank - 1) / (10000 - 1) * 100
    
    return customer_percentiles
    

#
def create_customer_percentiles_chart(customer_percentiles):
    percentiles = [
        int(customer_percentiles['CreditScore']),
        int(customer_percentiles['Tenure']),
        int(customer_percentiles['EstimatedSalary']),
        int(customer_percentiles['Balance']),
        int(customer_percentiles['NumOfProducts'])
    ]
    print("\nPercentiles in create_chart2():", percentiles, "\n")
    
    # Print debug information
    metrics = ['Credit Score', 'Tenure', 'Estimated Salary', 'Balance', 'Number of Products']
    
    data = {
        'Metric': metrics,
        'Percentile': percentiles
    }
    df = pd.DataFrame(data)
    
    # Function to determine the correct suffix
    def get_suffix(x):
        if x % 10 == 1 and x // 10 != 1:
            return "st"
        elif x % 10 == 2 and x // 10 != 1:
            return "nd"
        elif x % 10 == 3 and x // 10 != 1:
            return "rd"
        else:
            return "th"

    # Create hover templates
    hover_templates = [f"{p}{get_suffix(p)} percentile<br>{m}<extra></extra>" for p, m in zip(percentiles, metrics)]
    
    # Create the horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df['Percentile'],
        y=df['Metric'],
        orientation='h',
        marker=dict(
            color='rgba(135, 206, 235, 0.7)',
            line=dict(color='rgba(135, 206, 235, 1)', width=1)
        ),
        hovertemplate=hover_templates
    ))
    
    # Update layout to match the dark theme
    fig.update_layout(
        title=dict(
            text="Customer Percentiles",
            font=dict(size=24),
            x=0
        ),
        xaxis=dict(
            title='Percentile',
            titlefont=dict(color='#F5F5F5'),
            tickfont=dict(color='#F5F5F5'),
            range=[0, 101],  
            tickformat=',',  # Format ticks as ints
            tickvals=list(range(0, 101, 20)),  # Set tick values from 0 to 100 in steps of 10
            ticktext=[f"{i}th" for i in range(0, 101, 20)],  # Custom tick text
            showgrid=True,
            zeroline=False
        ),
        yaxis=dict(
            title='Metric',
            titlefont=dict(color='#F5F5F5'),
            tickfont=dict(color='#F5F5F5'),
            showgrid=False
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
        paper_bgcolor='rgba(0,0,0,0)'  # Transparent background
    )
    
    return fig
    