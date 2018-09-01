import cv2
import numpy as np
from matplotlib import pyplot as plt

BLUE = [255,0,0]
folder = '/home/flo/Dokumente/I2A_Presentation/EnvModel_dSSM_prediction/Game2/'
plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                wspace=0.05, hspace=0.05)
for i in range(3):
    obs = cv2.imread(folder+'obs_'+str(i)+'.png')
    obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
    pred = cv2.imread(folder+'pred_' + str(i) + '.png')
    pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.subplot(25*10+(i+1)),plt.imshow(obs)
    plt.axis('off')
    plt.subplot(25*10+(5+(i+1))), plt.imshow(obs)

plt.axis('off')
plt.show()
