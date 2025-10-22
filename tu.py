import numpy as np
import matplotlib.pyplot as plt


labels = ['Sn', 'Sp', 'MCC', 'AUC', 'Acc']
#MultiFeatVote = [0.508,0.830,0.360,0.718,0.691]
#CatPIPpred = [0.520,0.821,0.361,0.738,0.691]
MultiFeatVote = [0.521,0.790,0.322,0.686,0.655]
CatPIPpred = [0.556,0.813,0.381,0.705,0.684]

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()


MultiFeatVote += MultiFeatVote[:1]
#my_rfe += my_rfe[:1]
CatPIPpred += CatPIPpred[:1]
angles += angles[:1]


fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
ax.fill(angles, MultiFeatVote, color="#F5BE8F", alpha=0.6, label='MultiFeatVote')
#ax.fill(angles, my_rfe, color='blue', alpha=0.25, label='my_rfe')
ax.fill(angles, CatPIPpred, color='#FDDED7', alpha=0.5, label='CatPIPpred')

ax.set_yticklabels([])
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)


plt.legend(loc='upper right', bbox_to_anchor=(0.5, 0.1), 
           fontsize=16,  
           frameon=True,  
           framealpha=0,  
           edgecolor='black')  

plt.show()
