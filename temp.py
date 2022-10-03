import json
import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt




#["positive", "negative", "positive", "negative", "positive", "positive", "positive", "negative", "positive", "negative"]


y_true1 = np.asarray([1, 0, 1, 0, 1, 1, 1, 0, 1, 0])
y_scores1 = np.asarray([0.7, 0.3, 0.5, 0.6, 0.55, 0.9, 0.75, 0.2, 0.8, 0.3])
#map1 = round(average_precision_score(y_true1, y_scores1),3)
map1 = average_precision_score(y_true1, y_scores1)
print(map1)


#y_true = ["negative", "positive", "positive", "negative", "negative", "positive", "positive", "positive", "negative", "positive"]
y_true2 = np.asarray([0, 1, 1, 0, 0, 1, 1, 1, 0, 1])
y_scores2 = np.asarray([0.32, 0.9, 0.5, 0.1, 0.25, 0.9, 0.55, 0.3, 0.35, 0.85])
map2 = round(average_precision_score(y_true2, y_scores2),3)

print(map2)
print((map1+map2)/2)
