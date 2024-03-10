import pandas as pd
import matplotlib.pyplot as plt

# Specify the path to the BERT.XLSX file
file_path = 'BERT_Unlabelled.XLSX'

# Read the Excel file into a DataFrame
df = pd.read_excel(file_path)
df = df.drop(['ZH','EN','PT'], axis=1)



# Extract the relevant data from the dictionary
zhbn = df['ZHBN'].to_list()
zhbn = [str(i) for i in zhbn]
en_score = df['ENScore'].to_list()
zh_score = df['ZHScore'].to_list()
pt_score = df['PTScore'].to_list()


"""
# Create a scatter plot
plt.scatter(zhbn, en_score, c=en_label.map({'LABEL_1': 'red', 'LABEL_0': 'blue'}))

# Set the labels and title
plt.xlabel('ZHBN')
plt.ylabel('EN Score')
plt.title('EN Sentiment Analysis')

# Show the plot
plt.show()
"""

import plotly.express as px


# Create a DataFrame for the plot
plot_df = pd.DataFrame({
    'ZHBN': zhbn,
    'EN Score': en_score,
})

# Create an interactive scatter plot
fig = px.scatter(plot_df, x='ZHBN', y='EN Score', hover_data=['ZHBN', 'EN Score'])

# Show the plot
fig.write_html('EN_Sentiment_Analysis.html')


import plotly.graph_objects as go

# Create a DataFrame for the plot
plot_df = pd.DataFrame({
    'ZHBN': zhbn,
    'EN Score': en_score,
    'ZH Score': zh_score,
    'PT Score': pt_score
})

# Create an interactive scatter plot for each language
fig = go.Figure()

fig.add_trace(go.Scatter(x=plot_df['ZHBN'], y=plot_df['EN Score'], mode='markers', name='EN Score'))
fig.add_trace(go.Scatter(x=plot_df['ZHBN'], y=plot_df['ZH Score'], mode='markers', name='ZH Score'))
fig.add_trace(go.Scatter(x=plot_df['ZHBN'], y=plot_df['PT Score'], mode='markers', name='PT Score'))

# Set the labels and title
fig.update_layout(title='Sentiment Analysis', xaxis_title='ZHBN', yaxis_title='Score')

# Show the plot
fig.write_html('Sentiment_Analysis.html')