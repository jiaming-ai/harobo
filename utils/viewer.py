import cv2
import numpy as np
from home_robot.core.interfaces import DiscreteNavigationAction


class OpenCVViewer:
    def __init__(self, name="OpenCVViewer", exit_on_escape=True):
        self.name = name
        cv2.namedWindow(name, cv2.WINDOW_NORMAL) # use cv2.WINDOW_NORMAL to allow window resizing for large images
        self.exit_on_escape = exit_on_escape

    def parse_key(self, key):
 
        c = chr(key).lower()
        if c == "a":
            # Left
            # base_action = [0, 1]
            action = DiscreteNavigationAction.TURN_LEFT
        elif c == "d":
            # Right
            action = DiscreteNavigationAction.TURN_RIGHT
        elif c == "s":
            # Back
            action = DiscreteNavigationAction.STOP
        elif c == "w":
            # Forward
            action = DiscreteNavigationAction.MOVE_FORWARD
        else:
            action = None

        return action
    
    def imshow(self, image: np.ndarray, rgb=True, non_blocking=False, delay=0):
        if image.ndim == 2:
            image = np.tile(image[..., np.newaxis], (1, 1, 3))
        elif image.ndim == 3 and image.shape[-1] == 1:
            image = np.tile(image, (1, 1, 3))
        assert image.ndim == 3, image.shape

        if rgb:
            image = image[..., ::-1]
        cv2.imshow(self.name, image)

        if non_blocking:
            return
        else:
            action = DiscreteNavigationAction.STOP
            info = ""
            
            key = cv2.waitKey(delay)
            if key == 27:  # escape
                exit(0)
            elif key == -1:  # timeout
                pass
            elif key == 32:  # space
                return {'info':'done','action':action}
            elif key == 13:  # enter
                info = 'plan_high'
            else:
                action = self.parse_key(key)
                if action is not None:
                    return {'action':action, 'info':info}
                

    def close(self):
        cv2.destroyWindow(self.name)

    def __del__(self):
        self.close()
