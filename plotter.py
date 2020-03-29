import matplotlib
matplotlib.use("MacOSX")
import matplotlib.pyplot as plt


x = [.25, .5, .75, .8, 1.00]
# F1
y1 = [59.902, 63.056, 65.939, 64.687, 69.549]
# Precision
y2 = [61.179, 65.542, 67.464, 68.517, 63.128]
# Recall
y3 = [58.677, 60.752, 64.481, 61.263, 66.183]

plt.plot(x, y1, color='blue', label='F1')
plt.plot(x, y2, color='red', label='Precision')
plt.plot(x, y3, color='orange', label='Recall')
plt.xlabel('Training Set Proportions')
plt.ylabel('Scores (%s)')
plt.title('GCN Model performance by training set size')
plt.legend()
plt.show()