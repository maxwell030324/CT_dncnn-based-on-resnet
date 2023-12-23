import matplotlib.pyplot as plt
import pandas as pd
import math

file_path = './loss_log/运行结果.xlsx'
data = pd.read_excel(file_path)
# print(data)
loss_data = list(data['loss'])
# print(loss_data)
val_loss_data = list(data['val_loss'])
loss_data_log = [math.log(i) for i in loss_data]
val_loss_data_log = [math.log(i) for i in val_loss_data]
epoch_list = [i+1 for i in range(len(val_loss_data))]

# plt绘图


plt.figure()
plt.title("loss_epoch")
plt.xlabel("epoch")
plt.ylabel(("loss_log"))
plt.plot(epoch_list,loss_data_log,label='train_loss')
plt.plot(epoch_list,val_loss_data_log,label='val_loss')
plt.legend()
plt.show()



