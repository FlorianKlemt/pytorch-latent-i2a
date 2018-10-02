import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

def get_image_i(i):
    obs = cv2.imread(folder + 'obs_' + str(i) + '.png')
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    pred = cv2.imread(folder + 'pred_' + str(i) + '.png')
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    return obs, pred


BLUE = [255,0,0]
folder = '/home/meins/Studium/GuidedResearch/refactoring_guided_research/images/Game4/'
#plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0.05, hspace=0.05)

plt.figure(figsize = (6,2), dpi= 320)
gs1 = gridspec.GridSpec(2, 6)
gs1.update(wspace=0.025, hspace=0.0)

for i in range(6):
    obs, pred = get_image_i(i)
    plt.subplot(gs1[i])
    plt.axis('off')
    plt.imshow(obs)

for i in range(6):
    obs, pred = get_image_i(i)
    plt.subplot(gs1[6+i])
    plt.axis('off')
    plt.imshow(pred)

plt.show()
print("wait")
