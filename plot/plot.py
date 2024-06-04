from matplotlib import pyplot as plt
import numpy as np

TIC = np.array([
    [0.1513422818791946, 51.00000],
    [0.2238255033557047, 58.03125],
    [0.3318791946308726, 63.87500],
    [0.4748322147651008, 68.31250],
])

TransTIC = np.array([
    [0.1043624161073825, 58.9375000000000071],
    [0.1761744966442954, 66.7812500000000000],
    [0.2731543624161074, 70.1250000000000000],
    [0.4016778523489933, 72.6875000000000000],
])

full_finetuning = np.array([
    [0.0936241610738255, 73.34375],
    [0.1560402684563759, 74.56250],
    [0.2577181208053692, 75.71875],
    [0.3983221476510068, 75.90625],
])

TransTIC_WhiteVIP = np.array([
    [0.17704, 55.926],  # 1, 26.11172
    [0.25873, 61.860],  # 2, 27.81601
    [0.37220, 66.164],  # 3, 28.43561
    [0.52946, 69.594],  # 4, 28.88352
])


TransTIC_BlackVIP = np.array([
    [0.17966, 52.528],    
    [0.26256, 60.322],    
    [0.37783, 65.176], 
    [0.53359, 68.942]
])


plt.figure(figsize=(12,8))
plt.title("TransTIC + BlackVIP", fontsize=30)
plt.xlabel("Bit-rate (bpp)", fontsize=28)
plt.ylabel('Top-1 Accuracy (%)', fontsize=28)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid()

plt.axhline(76.723998, color='gray', linestyle='--', label='Uncompressed')
plt.plot(full_finetuning[:,0], full_finetuning[:,1], linestyle=(0, (4, 2)), color='black', label='Full fine-tuning', linewidth=3)
plt.plot(TIC[:,0], TIC[:,1], color='blue', linestyle=(0, (10, 3)), marker='o', label='TIC', linewidth=2)
plt.plot(TransTIC[:,0], TransTIC[:,1], color='red', marker='o', label='TransTIC', linewidth=3)
plt.plot(TransTIC_WhiteVIP[:,0], TransTIC_WhiteVIP[:,1], color='#FFCA00', marker='o', label='TIC + White-Box Setting', linewidth=3)
plt.plot(TransTIC_BlackVIP[:,0], TransTIC_BlackVIP[:,1], color='green', marker='o', label='TIC + Black-Box Setting', linewidth=3)

plt.legend(loc='lower right', prop={'size': 18})
plt.savefig('plot/TransTIC_WhiteVIP.png')
