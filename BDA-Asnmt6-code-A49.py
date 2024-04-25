import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pandas.plotting import table

data = {
    'Date': pd.date_range('2022-01-01', '2022-01-10'),
    'Category': ['Grocery', 'Electronics', 'Clothing', 'Grocery', 'Electronics', 'Clothing', 'Grocery', 'Electronics', 'Clothing', 'Grocery'],
    'Amount': [50, 120, 30, 40, 100, 25, 35, 80, 20, 45],
    'Transaction Type': ['Purchase', 'Purchase', 'Refund', 'Purchase', 'Purchase', 'Refund', 'Purchase', 'Purchase', 'Refund', 'Purchase']
}

df = pd.DataFrame(data)

print("Data Overview:")
print(df.head())

summary_stats = df.describe()

crosstab_data = pd.crosstab(df['Category'], df['Transaction Type'])

grouped_data = df.groupby('Category').agg({'Amount': 'sum'}).sort_values(by='Amount', ascending=False)

filtered_data = df[df['Amount'] > 0]

plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Amount', data=grouped_data.reset_index())
plt.title('Total Amount by Category')
plt.show()
plt.savefig('bar_chart.png')

plt.figure(figsize=(8, 8))
plt.pie(grouped_data['Amount'], labels=grouped_data.index, autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Total Amount by Category')
plt.show()

scatter_plot = px.scatter(df, x='Date', y='Amount', color='Category', title='Amount Over Time')
scatter_plot.show()

plt.figure(figsize=(10, 6))
ax = sns.heatmap(crosstab_data, annot=True, cmap='Blues', fmt='d')
plt.title('Transaction Type vs Category')
plt.show()

summary_stats.to_csv('summary_stats.csv')

crosstab_data.to_excel('crosstab_report.xlsx')