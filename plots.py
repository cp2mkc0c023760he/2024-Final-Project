import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the data from CSV file
df = pd.read_csv('prediction_results.csv')
path= 'Output/images/'

# Box plot of MAE
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='MAE')
plt.title('Distribution of Mean Absolute Error (MAE)')
plt.xlabel('Mean Absolute Error (MAE)')
plt.tight_layout()
plt.savefig(f'{path}mae_boxplot.png')  # Save the plot as a PNG file

# Box plot of R-squared
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='R-squared')
plt.title('Distribution of R-squared')
plt.xlabel('R-squared')
plt.tight_layout()
plt.savefig(f'{path}r_squared_boxplot.png')  # Save the plot as a PNG file

# Pairplot
sns.pairplot(df[['MAE', 'R-squared', 'Final Portfolio Value', 'Profit/Loss']])
plt.suptitle('Pairwise Relationships')
plt.tight_layout()
plt.savefig(f'{path}pairplot.png')  # Save the plot as a PNG file

# Histogram of MAE
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='MAE', bins=10, kde=True)
plt.title('Distribution of Mean Absolute Error (MAE)')
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{path}mae_histogram.png')  # Save the plot as a PNG file

# Histogram of R-squared
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='R-squared', bins=10, kde=True)
plt.title('Distribution of R-squared')
plt.xlabel('R-squared')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(f'{path}r_squared_histogram.png')  # Save the plot as a PNG file

# Scatter plot of MAE vs. R-squared
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='MAE', y='R-squared')
plt.title('MAE vs. R-squared')
plt.xlabel('Mean Absolute Error (MAE)')
plt.ylabel('R-squared')
plt.tight_layout()
plt.savefig(f'{path}mae_vs_r_squared.png')  # Save the plot as a PNG file



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
plt.savefig(f'{path}direction_accuracy_vs_accuracy.png')  # Save the plot as a PNG file


