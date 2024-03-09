import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

num=5000
path= 'Output/'
# Read the data from CSV file
df = pd.read_csv(f'{path}prediction_result_{num}.csv')

path = 'Output/images/'


# Box plot of MAE
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='MAE')
plt.title('Distribution of Mean Absolute Error (MAE)')
plt.xlabel('Mean Absolute Error (MAE)')
plt.tight_layout()
plt.savefig(f'{path}_mae_boxplot_{num}.png')  # Save the plot as a PNG file

# Box plot of R-squared
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='R-squared')
plt.title('Distribution of R-squared')
plt.xlabel('R-squared')
plt.tight_layout()
plt.savefig(f'{path}_r_squared_boxplot.png')  # Save the plot as a PNG file

# Pairplot
sns.pairplot(df[['MAE', 'R-squared', 'Final Portfolio Value', 'Profit/Loss']])
plt.suptitle('Pairwise Relationships')
plt.tight_layout()
plt.savefig(f'{path}_pairplot_{num}.png')  # Save the plot as a PNG file

# Histogram of MAE
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='MAE', bins=10, kde=True)
plt.title('Distribution of Mean Absolute Error (MAE)')
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{path}_mae_histogram_{num}.png')  # Save the plot as a PNG file

# Histogram of R-squared
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='R-squared', bins=10, kde=True)
plt.title('Distribution of R-squared')
plt.xlabel('R-squared')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{path}_r_squared_histogram_{num}.png')  # Save the plot as a PNG file

# Scatter plot of MAE vs. R-squared
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MAE', y='R-squared')
plt.title('MAE vs. R-squared')
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('R-squared')
plt.tight_layout()
plt.savefig(f'{path}_mae_vs_r_squared_{num}.png')  # Save the plot as a PNG file



# Scatter direction accuracy vs accuracy including the name of the ticker as a label
def labelpoints(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'], point['y'], str(point['val']))

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Direction Accuracy', y='1%Accuracy', hue='Ticker')
plt.title('Direction Accuracy vs. 1%Accuracy')
plt.xlabel('Direction Accuracy')
plt.ylabel('1%Accuracy')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.tight_layout()
labelpoints(df['Direction Accuracy'], df['1%Accuracy'], df['Ticker'], plt.gca())
plt.savefig(f'{path}_direction_accuracy_vs_accuracy_{num}.png')  # Save the plot as a PNG file


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Direction Accuracy', y='Percent Profit/Loss', hue='Ticker')
plt.title('Direction Accuracy vs. Percent Profit/Loss')
plt.xlabel('Direction Accuracy')
plt.ylabel('Percent Profit/Loss')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)
plt.tight_layout()
labelpoints(df['Direction Accuracy'], df['Percent Profit/Loss'], df['Ticker'], plt.gca())
plt.savefig(f'{path}_direction_accuracy_vs_percent_pl_{num}.png')  # Save the plot as a PNG file


