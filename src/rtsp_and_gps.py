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

# from base.stream import SIYIRTSP
# from base.siyi_sdk import SIYISDK


class FrameSaver:
    def __init__(self, base_folder="capture"):
        self.base_folder = base_folder
        self.session_folder = self.create_session_folder()
        self.frame_counter = 0
        self.save_every_n_frame = 2  # Save every second frame
        self.saved_frames_count = 0

    def create_session_folder(self):
        """Create session folder with auto-incrementing number"""
        session_num = 0
        while True:
            folder_name = f"{self.base_folder}_{session_num:04d}"
            if not os.path.exists(folder_name):
                os.makedirs(folder_name, exist_ok=True)
                return folder_name
            session_num += 1

    def save_frame(self, frame):
        """Save frame with sequential number"""
        filename = f"{self.session_folder}/frame_{self.saved_frames_count:06d}.jpg"
        cv2.imwrite(filename, frame)
        self.saved_frames_count += 1
        return filename

    def should_save_frame(self):
        """Determine if current frame should be saved"""
        self.frame_counter += 1
        return self.frame_counter % self.save_every_n_frame == 0


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
                10,  # Pitch (0° = level)
                0,  # Roll (ignored in MAV_MOUNT_MODE_MAVLINK_TARGETING)
                10,  # Yaw (0° = forward)
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



class GPSCamera(Process):

    INPUT_IMG_SIZE = 640

    def __init__(self, source: str, queue):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)

        # Initialize components
        
        self.frame_saver = FrameSaver()
        self.mav_handler = MavlinkHandler()
        
        self.net_size = (self.INPUT_IMG_SIZE, self.INPUT_IMG_SIZE)
        self._queue = queue
        self.source = source
        self.max_retries=5
        self.retry_delay=2
        self.retry_count = 0
        self.gst_pipeline = None

    def crop_frame(self, frame, net_size):
        frames = []
        net_size = net_size[0]
        frames.append(frame[:net_size, :net_size])
        frames.append(frame[:net_size, -net_size:])
        frames.append(frame[-net_size:, :net_size])
        frames.append(frame[-net_size:, -net_size:])
        return frames

    def run(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        #self.rtsp = SIYIRTSP(rtsp_url=self.rtsp_url, debug=False, cam_name=self.cam_str)
        print("Starting capture...")
        if not cap.isOpened():
            raise SystemExit("Bad source")
        try:
            while cap.isOpened():
                # Read frame from RTSP stream
                ret, raw_frame = cap.read()
                if not ret:
                    raise SystemExit("Camera stopped!")
                    
                # Process frame saving
                if self.frame_saver.should_save_frame():
                    gps_data = self.mav_handler.get_gps_data()
                    if gps_data:
                        #filename = self.frame_saver.save_frame(raw_frame)
                        #save_gps_data(filename, gps_data)
                        #print(f"Saved: {filename} with GPS data")
                        raw_frame = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
                        frames = self.crop_frame(raw_frame, self.net_size)
                        if (not self._queue.empty()):
                            continue
                        frame_expanded = np.expand_dims(raw_frame, axis=0)
                        self._queue.put((frame_expanded, gps_data))
                    else:
                        print("No GPS data available for frame")
                
                # Check for exit key (ESC)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
                    
        except KeyboardInterrupt:
            print("Capture stopped by user")
        finally:
            self.rtsp.stop()
            cv2.destroyAllWindows()
            print("Capture completed")




def capture_rtsp_and_gps(rtsp_url, max_retries=5, retry_delay=2):
    frame_saver = FrameSaver()
    gps_logger = GPSLogger()

    retry_count = 0
    gst_pipeline = None

    # Главный цикл обработки
    main_loop = GLib.MainLoop()

    while True:
        try:
            # Инициализация/переинициализация GStreamer pipeline
            if gst_pipeline is None or not gst_pipeline.running:
                if gst_pipeline is not None:
                    gst_pipeline.stop()
                
                print(f"Попытка подключения к RTSP потоку (попытка {retry_count + 1}/{max_retries})...")
                gst_pipeline = GStreamerPipeline(rtsp_url)
                gst_pipeline.start()
                
                retry_count = 0
                print("Успешное подключение к RTSP потоку")

            # Получаем кадр
            frame = gst_pipeline.get_frame()
            
            if frame is not None and frame_saver.should_save_frame():
                gps_data = gps_logger.get_gps_data()
                
                if gps_data:
                    filename = frame_saver.save_frame(frame)
                    save_gps_data(filename, gps_data)
                    print(f"Сохранено: {filename} с координатами {gps_data}")

            # Обработка событий GLib
            context = main_loop.get_context()
            context.iteration(False)
            
            # Небольшая задержка для снижения нагрузки
            GLib.timeout_add(10, lambda: None)  # Альтернатива time.sleep()

        except (ConnectionError, RuntimeError) as e:
            print(f"Ошибка: {e}")
            retry_count += 1
            
            if retry_count >= max_retries:
                print("Достигнуто максимальное количество попыток. Завершение работы.")
                break
                
            print(f"Повторная попытка через {retry_delay} секунд...")
            time.sleep(retry_delay)
            
        except KeyboardInterrupt:
            print("Завершение работы по запросу пользователя...")
            break
            
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            time.sleep(1)  # Защита от бесконечного цикла при непредвиденных ошибках

    # Освобождение ресурсов
    if gst_pipeline is not None:
        gst_pipeline.stop()
    cv2.destroyAllWindows()

    if __name__ == "__main__":
    # Настройки (замените на свои)
        RTSP_URL = "/dev/video0"

        print("Начало захвата RTSP и GPS данных...")
        capture_rtsp_and_gps(RTSP_URL)
