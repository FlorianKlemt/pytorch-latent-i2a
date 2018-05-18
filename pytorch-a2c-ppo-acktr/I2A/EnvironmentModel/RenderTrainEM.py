import cv2
import numpy as np

class RenderTrainEM():
    def __init__(self):
        render_window_sizes = (400, 400)
        cv2.namedWindow('target', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('target', render_window_sizes)
        cv2.namedWindow('prediction', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('prediction', render_window_sizes)

    def render_observation_in_window(self, window_name, observation):
        drawable_state = observation.permute(1,2,0)

        drawable_state = drawable_state.data.cpu().numpy()

        frame_data = (drawable_state * 255.0)

        frame_data[frame_data < 0] = 0
        frame_data[frame_data > 255] = 255
        frame_data = frame_data.astype(np.uint8)

        cv2.imshow(window_name, frame_data)
        cv2.waitKey(1)


    def render_observation(self, target, prediction):
        self.render_observation_in_window('target', target)
        self.render_observation_in_window('prediction', prediction)
