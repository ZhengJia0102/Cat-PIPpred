import matplotlib.pyplot as plt
import numpy as np

models = ['CNN', 'BiLSTM', 'Cat-PIPpred']
data = {
    'CNN': [0.681, 0.556, 0.807, 0.703, 0.375],
    'BiLSTM': [0.565, 0.674, 0.456, 0.608, 0.135],
    'Cat-PIPpred': [0.691, 0.520, 0.821, 0.738, 0.361]
}

x = np.arange(len(models))  
width = 0.15  

fig, ax = plt.subplots(figsize=(12, 6))

x_points = range(1, 6)
for model in models:
    ax.plot(x_points, data[model], marker='o', label=model, linewidth=2)

ax.set_xlabel('Data Point')
ax.set_ylabel('Score')
ax.set_title('Model Performance Across Different Data Points')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(x_points)
plt.legend(
           fontsize=16,  
           frameon=True,  
           framealpha=0,  
           edgecolor='black')  
plt.grid(False)  
plt.tight_layout()
plt.show()