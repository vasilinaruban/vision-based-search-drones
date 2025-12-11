import os
import numpy as np
import cv2
from multiprocessing import Process
import sys
import os
import cv2
import json
from pymavlink import mavutil
import time
import numpy as np
from multiprocessing import Process

current = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current)
sys.path.append(parent_directory)


class MavlinkHandler:
    def __init__(self, connection_string='udpin:localhost:14550'):
        self.connection_string = connection_string
        self.mav_connection = None
        self.connect()

    def connect(self):
        """Connect to MavProxy and wait for heartbeat"""
        try:
            self.mav_connection = mavutil.mavlink_connection(self.connection_string)
            print("Waiting for MAVLink heartbeat...")
            self.mav_connection.wait_heartbeat(timeout=5.0)
            print(f"Heartbeat from system {self.mav_connection.target_system}")
            return True
        except Exception as e:
            print(f"MAVLink connection failed: {e}")
            return False

    def get_gps_data(self):
        """Get current GPS coordinates"""
        if self.mav_connection is None:
            if not self.connect():
                return None

        try:
            msg = self.mav_connection.recv_match(type='GLOBAL_POSITION_INT', blocking=True, timeout=1)
            if msg:
                return {
                    'latitude': msg.lat / 1e7,
                    'longitude': msg.lon / 1e7,
                    'altitude': msg.alt / 1e3,
                    'timestamp': time.time()
                }
            return None
        except Exception as e:
            print(f"Error getting GPS data: {e}")
            return None

    def center_gimbal(self):
        """Center the gimbal using MAV_CMD_DO_MOUNT_CONTROL"""
        if self.mav_connection is None:
            if not self.connect():
                return False

        try:
            # Send mount control command to center gimbal
            self.mav_connection.mav.command_long_send(
                self.mav_connection.target_system,    # Target system ID
                self.mav_connection.target_component, # Target component ID
                mavutil.mavlink.MAV_CMD_DO_MOUNT_CONTROL, # Command ID
                0,  # Confirmation
                10,  # Pitch (0� = level)
                0,  # Roll (ignored in MAV_MOUNT_MODE_MAVLINK_TARGETING)
                10,  # Yaw (0� = forward)
                0,  # Reserved
                0,  # Reserved
                3,   # MAV_MOUNT_MODE_MAVLINK_TARGETING
                1
            )
            print("Gimbal centering command sent")
            return True
        except Exception as e:
            print(f"Error centering gimbal: {e}")
            return False


    def save_gps_data(filename, gps_data):
        json_filename = os.path.splitext(filename)[0] + ".json"
        with open(json_filename, 'w') as f:
            json.dump(gps_data, f, indent=4)

class Camera(Process):
    """This class represents a camera process that captures frames from a video source 
    and performs various operations on the frames (preprocess).

    Attributes
    ----------
    net_size : tuple
        The size of the neural network input.
    queue : Queue
        The queue to put the processed frames into.
    source : int, str
        The video source (also path to file.mp4) to capture frames from.
    frames : generator
        A generator object that yields frames from the video capture.

    Methods
    -------
    get_frame(None)
        Returns the next frame from the frames generator.
    resize_frame(frame, net_size)
        Resizes the given frame using OpenCV's resize function.
    crop_frame(frame, net_size)
        Crops the given frame based on net_size.
    run(None) 
        Iterates over the frames generator, processes each frame, and puts it into the queue.

    """

    def __init__(self, source: int, queue, onnx=True, gt_queue=None):
        """
        Parameters
        ----------
        source : int, str
            The video source.
        queue : Queue
            The queue in which processed frames are placed. Then these frames will be fed 
            to the input of the neural network.
        onnx : bool, optional
            Whether to use ONNX model for postprocessing. Defaults to True.
        """
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        INPUT_SIZE = 640 #(550 if onnx else 544)
        self.net_size = (INPUT_SIZE, INPUT_SIZE) 
        self.mav_handler = MavlinkHandler()
        self._queue = queue
        self._gt_queue = gt_queue
        self.source = source

    @property
    def frames(self):
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise SystemExit("Bad source")
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    raise SystemExit("Camera stopped!")
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                yield frame
        except Exception as e:
            print(f"Stop recording loop. Exception {e}")
        finally:
            cap.release()
    
    def get_frame(self):
        """It yields the frame, making it available for further processing outside the function.
        """
        return next(self.frames)
    
    def resize_frame(self, frame, net_size):
        frame_size = frame.shape[:2]
        interpolation = cv2.INTER_CUBIC if any(x < y for x, y in zip(frame_size, net_size)) else cv2.INTER_AREA
        return cv2.resize(frame, net_size, interpolation=interpolation)

    def crop_frame(self, frame, net_size):
        net_size = net_size[0]
        hc, wc = frame.shape[0] // 2, frame.shape[1] // 2
        h0, w0 = hc - (net_size // 2), wc - (net_size // 2)
        assert (h0 >= 0 and w0 >= 0), 'The image size is not suitable to crop. Try Camera.resize_frame()'
        return frame[h0:h0+net_size, w0:w0+net_size]

    def run(self):
        for raw_frame in self.frames:
            frame = self.resize_frame(raw_frame, self.net_size)
            frame = np.expand_dims(frame, axis=0)
            gps_data = self.mav_handler.get_gps_data()

            if self._queue.full():
                continue

            if gps_data:
                self._queue.put((frame, gps_data))
            else:
                self._queue.put((frame, None))



