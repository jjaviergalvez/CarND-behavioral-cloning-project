import matplotlib.pyplot as plt
import pickle
import numpy as np

with open('history_train.p', mode='rb') as f:
    history = pickle.load(f)

# list all data in history
print(history.keys())

fig = plt.figure()
ax = fig.gca()
#ax.set_xticks(np.arange(0,6,0.1))
ax.set_yticks(np.arange(0,1,0.002))

plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss (mse)')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.grid()
plt.show()

#credits to http://machinelearningmastery.com/display-deep-learning-model-training-history-in-keras/