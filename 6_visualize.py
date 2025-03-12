import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load classified data
df = pd.read_csv("classified_results.csv")

# Plot modulation distribution
sns.countplot(x="Predicted Modulation", data=df)
plt.title("Modulation Classification Results")
plt.xlabel("Modulation Type")
plt.ylabel("Count")
plt.show()
