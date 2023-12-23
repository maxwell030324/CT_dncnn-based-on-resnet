import dataset_create
import numpy as np
import math
import matplotlib.pyplot as plt
import random

test_raw_dir = './test_raw'
test_tgt_dir = './test_tgt'
res_dir = './res'

raw_np = dataset_create.get_dataset(test_raw_dir)
tgt_np = dataset_create.get_dataset(test_tgt_dir)
res_np = dataset_create.get_dataset(res_dir)

# 计算峰值信噪比

def cal_PSNR(mtx1,mtx2):
    mse = np.mean((mtx1 * 1.0 - mtx2 * 1.0) ** 2)
    return 10.0 * math.log10(255.0 *255.0 / mse)


PSNR_list_1 = [] # raw & tgt
PSNR_list_2 = [] # res & tgt

for i in range(raw_np.shape[0]):
    PSNR_list_1.append(cal_PSNR(raw_np[i],tgt_np[i]))
    PSNR_list_2.append(cal_PSNR(res_np[i],tgt_np[i]))
    
mean1 = 0
mean2 = 0 
    
for i in range(len(PSNR_list_1)):
    mean1 += PSNR_list_1[i]
    mean2 += PSNR_list_2[i]

mean1 = mean1 / len(PSNR_list_1)
mean2 = mean2 / len(PSNR_list_2)

plt.figure()

ax1 = plt.subplot(1,2,1)
plt.scatter([random.uniform(0, 1) for i in range(len(PSNR_list_1))],PSNR_list_1,c='red', s=15)
plt.axhline(y=mean1, color='k', linestyle='-',linewidth=6)
plt.gca().get_xaxis().set_visible(False)
plt.ylabel('PSNR')
plt.title("PSNR before denoise")
ax2 = plt.subplot(1,2,2)
plt.scatter([random.uniform(0, 1) for i in range(len(PSNR_list_2))],PSNR_list_2,c='red', s=15)
plt.axhline(y=mean2, color='k', linestyle='-',linewidth=6)
plt.gca().get_xaxis().set_visible(False)
plt.ylabel('PSNR')
plt.subplots_adjust(wspace=0.45, hspace=0)
plt.title("PSNR after denoise")
plt.show()
    



