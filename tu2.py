import matplotlib.pyplot as plt
import numpy as np


architectures = ['CNN', 'BiLSTM', 'Cat-PIPpred']
metrics = ['Acc', 'Sn', 'Sp', 'AUC', 'MCC']

data = {
    'CNN': [0.681, 0.556, 0.807, 0.703, 0.375],
    'BiLSTM': [0.565, 0.674, 0.456, 0.608, 0.135],
    'Cat-PIPpred': [0.691, 0.520, 0.821, 0.738, 0.361]
}


plt.figure(figsize=(9, 7))


x = np.arange(len(metrics)) * 0.5  


colors = ['#EFCA72', '#93CC82', '#88C4E8']  
markers = ['o', '^', 's']  


for idx, arch in enumerate(architectures):
    plt.plot(x, data[arch], 
             marker=markers[idx], 
             color=colors[idx],
             label=arch, 
             linewidth=2,
             markersize=8)


for idx, arch in enumerate(architectures):
    for i, val in enumerate(data[arch]):
        plt.text(x[i], val + 0.01, f'{val:.3f}', 
                 ha='center', 
                 va='bottom', 
                 fontsize=9,
                 )


#plt.title('Performance Comparison of Deep Learning Architectures', fontsize=14)
plt.xlabel('Metrics', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.xticks(x, metrics, rotation=45, ha='right')  
plt.ylim(0, 1.0)  
plt.grid(False)  
plt.legend(loc='best', fontsize=10)  


plt.tight_layout()


plt.show()