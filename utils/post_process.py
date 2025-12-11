import time
from itertools import product
import math
from math import sqrt
import cv2
import numpy as np
import onnxruntime
from multiprocessing import Process, Queue
import os
import math
import json
import numpy as np
import shutil
import cv2
from sklearn.cluster import DBSCAN
from geopy.distance import distance


MASK_SHAPE = (138, 138, 3)

COLORS = np.array([[0, 0, 0], [244, 67, 54], [233, 30, 99], [156, 39, 176], [103, 58, 183], [100, 30, 60],
                   [63, 81, 181], [33, 150, 243], [3, 169, 244], [0, 188, 212], [20, 55, 200],
                   [0, 150, 136], [76, 175, 80], [139, 195, 74], [205, 220, 57], [70, 25, 100],
                   [255, 235, 59], [255, 193, 7], [255, 152, 0], [255, 87, 34], [90, 155, 50],
                   [121, 85, 72], [158, 158, 158], [96, 125, 139], [15, 67, 34], [98, 55, 20],
                   [21, 82, 172], [58, 128, 255], [196, 125, 39], [75, 27, 134], [90, 125, 120],
                   [121, 82, 7], [158, 58, 8], [96, 25, 9], [115, 7, 234], [8, 155, 220],
                   [221, 25, 72], [188, 58, 158], [56, 175, 19], [215, 67, 64], [198, 75, 20],
                   [62, 185, 22], [108, 70, 58], [160, 225, 39], [95, 60, 144], [78, 155, 120],
                   [101, 25, 142], [48, 198, 28], [96, 225, 200], [150, 167, 134], [18, 185, 90],
                   [21, 145, 172], [98, 68, 78], [196, 105, 19], [215, 67, 84], [130, 115, 170],
                   [255, 0, 255], [255, 255, 0], [196, 185, 10], [95, 167, 234], [18, 25, 190],
                   [0, 255, 255], [255, 0, 0], [0, 255, 0], [0, 0, 255], [155, 0, 0],
                   [0, 155, 0], [0, 0, 155], [46, 22, 130], [255, 0, 155], [155, 0, 255],
                   [255, 155, 0], [155, 255, 0], [0, 155, 255], [0, 255, 155], [18, 5, 40],
                   [120, 120, 255], [255, 58, 30], [60, 45, 60], [75, 27, 244], [128, 25, 70]], dtype='uint8')



CASTOM_CLASSES = ('person', 'backgrnd')


class ResultSaver:
    """
    Handles saving detected objects and their GPS coordinates
    """

    def __init__(
        self,
        output_path: str = "./results_json",
        eps: float = 0.00009,
        min_samples: int = 2,
        metric: str = "euclidean",
        drone_height_m: float = 40.0,
        fov_deg: float = 90.0,
        image_size=(640, 640)):
        self.output_path = output_path
        self.check_out_path(self.output_path)

        list_dir = os.listdir(output_path)
        self.n_image = len(list_dir) // 2

        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.drone_height_m = drone_height_m
        self.fov_deg = fov_deg
        self.image_size = image_size

        self.dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric=self.metric)

        self.boxes_list: list[list[float]] = []
        self.gps_data_list: list[list[float]] = []
        self.angles_list: list[float] = []
        self.imgs_with_boxes: list[str] = []
    
    def check_out_path(self, output_path):
        if not os.path.isdir(output_path):
            os.makedirs(output_path, exist_ok=True)
    
    def get_sd_path(self, output_path):
        sd_path = '/mnt/sdcard/'
        result_path = sd_path + 'results_json'
        return result_path

    def apply_add_bboxes(self, frame, outputs, gps_data):
        """
        Save the given frame on the screen with bounding boxes
        """
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
            
        if frame is None:
            return frame, None
        
        # Unpack outputs
        ids_p, class_p, boxes_p, masks_p = outputs
        img_name = f'img_{self.n_image}.jpg'
        
        # Extract GPS data
        lat = gps_data['latitude']
        long = gps_data['longitude']
        angle = gps_data.get('angle', 0)  # Default angle if not provided
        gps_coords = [lat, long]
        
        # Draw bounding boxes
        for i in range(len(ids_p)):
            box = boxes_p[i]
            score = class_p[i]
            cl = ids_p[i]
            
            top, left, right, bottom = map(int, box)
            cv2.rectangle(img=frame, pt1=(top, left), pt2=(right, bottom), color=(255, 0, 0),
                          thickness=2)
            cv2.putText(img=frame, text=f"{CASTOM_CLASSES[cl]} {score:.2f}", org=(top, left - 6),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0, 0, 255), thickness=2)
        
            self.boxes_list.append(box)
            self.gps_data_list.append(gps_coords)
            self.angles_list.append(angle)
            self.imgs_with_boxes.append(img_name)
        
        return frame, img_name
        
    def save_result(self, frame, outputs, gps_data):
        frame, img_name = self.apply_add_bboxes(frame, outputs, gps_data)
        if img_name is None:
            return
            
        # Save img 
        bufer_img_path = f'./bufer_imgs/{img_name}'
        os.makedirs('./bufer_imgs/', exist_ok=True)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(bufer_img_path, frame)
        
        self.n_image += 1
    
    # SAVE JSON METHODS
    def pix2m(
        self,
        drone_lat: float,
        drone_lon: float,
        bbox_center_xy: tuple[float, float],
        image_size: tuple[int, int],
        angle: float, ) -> tuple[float, float]:
        img_w, img_h = image_size
        cx, cy = bbox_center_xy

        fov_rad = math.radians(self.fov_deg)
        aspect_ratio = img_h / img_w

        ground_width_m = 2 * self.drone_height_m * math.tan(fov_rad / 2)
        ground_height_m = ground_width_m * aspect_ratio

        # Смещение от центра изображения в метрах
        dx = (cx - img_w / 2) / img_w * ground_width_m
        dy = (cy - img_h / 2) / img_h * ground_height_m
        
        # dx → восток, dy → юг
        east_shift = distance(meters=dx).destination((drone_lat, drone_lon), bearing=90+angle)
        final_point = distance(meters=dy).destination((east_shift.latitude, east_shift.longitude), bearing=180+angle)

        return final_point.latitude, final_point.longitude
    
    def xyxy2coords(self, boxes, gps_data, angles_data):
        """
        Переводит пиксельные координаты xyxy в GPS-координаты каждого бокса.
        Использует только self.image_size, self.drone_height_m, self.fov_deg и gps_data.
        """
        res = []

        # xyxy -> центр бокса
        xyboxes = np.empty((len(boxes), 2))
        for i, box in enumerate(boxes):
            xyboxes[i, 0] = (box[0] + box[2]) / 2
            xyboxes[i, 1] = (box[1] + box[3]) / 2

        for (x_px, y_px), gps, angle in zip(xyboxes, gps_data, angles_data):
            lat, lon = self.pix2m(
                drone_lat=gps["latitude"],
                drone_lon=gps["longitude"],
                bbox_center_xy=(x_px, y_px),
                image_size=self.image_size,
                angle=angle,
            )
            res.append((lat, lon))

        return np.array(res)


    def get_clusters(self):
        """
        Кластеризация
        """
        boxes_np = np.array(self.boxes_list)  # xyxy
        gps_data_np = np.array(self.gps_data_list)
        angles_np = np.array(self.angles_list)
        imgs_np = np.array(self.imgs_with_boxes)
        
        if len(boxes_np) == 0:
            return np.array([]), np.array([]), np.array([])
            
        box_centers = self.xyxy2coords(boxes_np, gps_data_np, angles_np)
        labels = self.dbscan.fit_predict(box_centers)
        return box_centers, np.array(labels), imgs_np

    def get_mean_coords(self, n_clusters, box_centers, labels):
        mean_coords = []
        for k in range(n_clusters):
            class_member_mask = labels == k
            mean_coords.append(box_centers[class_member_mask].mean(axis=0))
        mean_coords = np.array(mean_coords)
        return np.round(mean_coords, decimals=7)
         

    def get_best_imgs(self, n_clusters, np_img_list, labels):
        best_imgs_with_object = []
        for k in range(n_clusters):
            class_member_mask = labels == k
            claster_imgs = np_img_list[class_member_mask]
            if len(claster_imgs) >= 2:
                claster_imgs = [claster_imgs[0].item(),
                                claster_imgs[-1].item()]
            best_imgs_with_object.append(claster_imgs)
        return best_imgs_with_object
    
    def create_object_entry(self, name, center_lat, center_lon, image_files):
        return {"name": name, "dd.dddddd_lat": f"{center_lat:.6f}",
                "dd.dddddd_lon": f"{center_lon:.6f}",
                "images": image_files }

    def save_to_json(self, objects_data, json_number):
        """
        Сохраняет данные об объектах в JSON-файл с заданной структурой.
        """
        result = {"Objects": objects_data}
        
        output_path = os.path.join(self.output_path, f'object_data_{json_number%2}.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        #output_path = os.path.join(self.sdcard_output_path, f'object_data_{json_number%2}.json')
        #with open(output_path, 'w', encoding='utf-8') as f:
        #    json.dump(result, f, indent=2, ensure_ascii=False)

    def save_json(self):
        """ Сохранение результатов в формате json
        """
        box_centers, labels, img_filtered = self.get_clusters()
        if len(box_centers) == 0:
            return
            
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters > 0:
            mean_coords = self.get_mean_coords(n_clusters, box_centers, labels)
            best_imgs_with_object = self.get_best_imgs(n_clusters, img_filtered, labels)
            objects_data = []
            for k in range(n_clusters):
                name = f'object_{k}'
                center_lat = mean_coords[k][0]
                center_lon = mean_coords[k][1]
                image_files = list(best_imgs_with_object[k])
                objects_data.append(self.create_object_entry(name, center_lat, center_lon, image_files))
            self.save_to_json(objects_data, self.json_number)
            self.json_number += 1
    
    def update_imgs(self, json_path):
        if not os.path.exists(json_path):
            return
            
        with open(json_path, 'r') as file:  
            data = json.load(file)
        
        current_imgs = os.listdir('./results_json/')
        current_imgs = [img for img in current_imgs if img.endswith('jpg')]
        best_imgs = []
        for obj in data['Objects']:
            obj_images = obj['images']
            for img in obj_images:
                best_imgs.append(img)
                source_img = './bufer_imgs/' + img
                destination_img = './results_json/' + img
                shutil.copyfile(source_img, destination_img)
                #destination_img = '/mnt/sdcard/results_json/' + img
                #shutil.copyfile(source_img, destination_img)
        for cur in current_imgs:
            if cur in best_imgs:
                continue
            else:
                os.remove('./results_json/'+cur)
    
    def copy_best_img(self):
        json_path = './results_json/object_data_0.json'
        if os.path.exists(json_path):
            self.update_imgs(json_path)
            return
        json_path = './results_json/object_data_1.json'
        if os.path.exists(json_path):
            self.update_imgs(json_path)


class Detection(Process):
    """
    Attributes
    ----------
    input_size : int
        Represents the size of the input frame.
    input : Queue
        The queue of input frames for the detection process.
    cfg : dict
        The configuration settings for the rknn-detection process. It include parameters such as 
        confidence thresholds, maximum number of output predictions, etc. (see main.py)
    q_out : Queue
        An instance of the "Queue" class with a maximum size of 3. It is used to store the 
        processed frames and prepared results for display.
    
    Methods
    -------
    permute(net_outputs)
        Permutes the elements in the net_outputs list according to a specific order.
    detect(inputs)
        Detect is the final layer of SSD. Decode location preds, apply non-maximum suppression 
        to location predictions based on conf scores and threshold to a top_k number of output 
        predictions for both confidence score and locations, as the predicted masks.
    prep_display(results)
        This method prepares the results for display. It extracts data from the inference results
        in the form: class_ids, scores, bboxes, masks
    run(None)
        Method runs in an infinite loop. It puts the frame and prepared results into the "q_out" queue.
    
    """
    
    def __init__(self, input, cfg=None):
        super().__init__(group=None, target=None, name=None, args=(), kwargs={}, daemon=True)
        self.input_size = 0
        self.input = input
        self.cfg = cfg
        self.q_out = Queue(maxsize=3)
    
    def run(self):
        while True:
            frame, inputs, gps_data = self.input.get()
            results = self.detect(inputs)
            self.q_out.put((frame, self.prep_display(results)))
    
    def permute(self, net_outputs):
        '''implementation dependent'''
        pass

    def detect(self, inputs):
        '''implementation dependent'''
        pass

    def prep_display(self, results):
        '''implementation dependent'''
        pass




class RKNNDetection(Detection):
    """This class represents an implementation of the RKNNDetection algorithm, which is a subclass of the 
    Detection class. It includes methods for initializing the algorithm, permuting the network outputs, 
    performing object detection, and preparing the results for display.
    
    Attributes
    ----------
    input_size : int
        The size of the input frame.
    anchors : list
        A list of anchor boxes used for object detection.

    Methods
    -------
    __init__(input, cfg)
        Initializes the RKNNDetection algorithm by setting the input size and generating the anchor boxes.
    permute(net_outputs)
        Permutes the arrays in net_outputs to have a specific shape.
    detect(onnx_inputs)
        Performs object detection by applying non-maximum suppression.
    prep_display(results)
        Prepares the results for display.
    
    """

    def __init__(self, queue, cfg=None, result_saver_kwargs=None):
        super().__init__(input=queue, cfg=cfg)
        self.queue = queue

        if result_saver_kwargs is None:
            result_saver_kwargs = {}

        self.result_saver = ResultSaver(**result_saver_kwargs)
    def run(self):
        while True:
            frame, inputs, gps_data = self.input.get()
            results = self.detect(inputs)
            display_results = self.prep_display(results)
            
            # Save results with GPS data
            if gps_data is not None:
                self.result_saver.save_result(frame, display_results, gps_data)
            
            self.q_out.put((frame, display_results))
            
            # Periodically save JSON results (e.g., every 10 frames)
            if self.result_saver.n_image % 10 == 0:
                self.result_saver.save_json()
                self.result_saver.copy_best_img()
    
    def detect(self, inputs):
        '''
        Returns
        -------
        class_ids, class_thre, box_thre, coef_thre, proto_p
        '''
        h_orig, w_orig = (self.input_size, self.input_size)
        print(inputs)

        data = inputs[0][0]  # (5, N)
        boxes = data[:4, :].T
        confidences = data[4, :]

        conf_threshold = 0.45
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]

        # Масштабируем обратно
        scale_x = w_orig / 640
        scale_y = h_orig / 640
        boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
        boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y

        boxes = boxes.astype(np.int32)

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=confidences.tolist(),
            score_threshold=conf_threshold,
            nms_threshold=0.4
        )

        if len(indices) == 0:
            indices = []
        else:
            indices = indices.flatten()

        return (boxes, indices, confidences )
    
    def prep_display(self, results):
        boxes, indices, confidences = results
        ids_p = []
        class_p = []
        box_p = []
        
        for i in indices:
            x_center, y_center, w, h = boxes[i]
            x1 = int((x_center - w / 2))
            y1 = int((y_center - h / 2))
            x2 = int((x_center + w / 2))
            y2 = int((y_center + h / 2))
            conf = confidences[i]
            
            # Create dummy data for missing parameters
            ids_p.append(0)  # Class ID (0 for 'first')
            class_p.append(conf)  # Confidence score
            box_p.append([x1, y1, x2, y2])  # Bounding box
        
        # Create dummy masks (since we're doing detection, not segmentation)
        mask_p = np.zeros((len(ids_p), *MASK_SHAPE), dtype=np.float32)
        
        return (
            np.array(ids_p), 
            np.array(class_p), 
            np.array(box_p), 
            mask_p
        )


class PostProcess():
    """Class to handle post-processing of yolact inference results.

    Attributes
    ----------
    detection : Detection
        Detection class object.

    Methods
    -------
    run()
        Starts the detection process.
    get_outputs()
        Retrieves the prepared results from the detection process.
        
    """
    
    def __init__(self, queue, cfg=None, onnx=True, result_saver_kwargs=None):
        self.cfg = cfg
        self.onnx = onnx

        if result_saver_kwargs is None:
            result_saver_kwargs = {}

        self.detection = RKNNDetection(
            queue=queue,
            cfg=cfg,
            result_saver_kwargs=result_saver_kwargs,
        )
    
    def run(self):
        self.detection.start()
    
    def get_outputs(self):
        return self.detection.q_out.get()




class Visualizer():
    def show_results(self, frame, outputs):
        
        """
        Show the given frame on the screen with bounding boxes
        """
        # Remove batch dimension if present
        if frame.ndim == 4 and frame.shape[0] == 1:
            frame = frame[0]
            
        # Convert to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Unpack outputs
        ids_p, class_p, box_p, mask_p = outputs
        
        # Draw bounding boxes
        for i in range(len(ids_p)):
            class_id = ids_p[i]
            confidence = class_p[i]
            box = box_p[i]
            
            # Draw rectangle
            color = COLORS[class_id + 1].tolist()
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
            
            # Draw label
            label = f"{CASTOM_CLASSES[class_id]}: {confidence:.2f}"
            cv2.putText(frame, label, (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        frame = cv2.resize(frame, (640,640))
        
        cv2.imshow('Object Detection', frame)
        cv2.waitKey(1)



def rknn_draw(img_origin, ids_p, class_p, box_p, mask_p, cfg=None, fps=None):
    """
    Generates an image with bounding boxes and labels for detected objects.

    Returns
    -------
    frame : numpy.ndarray
        The image with bounding boxes, masks and labels.
    """
    real_time = False
    if ids_p is None:
        return img_origin

    num_detected = ids_p.shape[0]

    img_fused = img_origin
    masks_semantic = mask_p * (ids_p[:, None, None] + 1)  # expand ids_p' shape for broadcasting
    # The color of the overlap area is different because of the '%' operation.
    masks_semantic = masks_semantic.astype('int').sum(axis=0) % (len(COCO_CLASSES))
    color_masks = COLORS[masks_semantic].astype('uint8')
    img_fused = cv2.addWeighted(color_masks, 0.4, img_origin, 0.6, gamma=0)

    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    for i in reversed(range(num_detected)):
        color = COLORS[ids_p[i] + 1].tolist()
        img_fused = draw_box(img_fused, box_p[i, :], color, ids_p[i], class_p[i])

    if real_time:
        fps_str = f'fps: {fps:.2f}'
        text_w, text_h = cv2.getTextSize(fps_str, font, scale, thickness)[0]
        # Create a shadow to show the fps more clearly
        img_fused = img_fused.astype(np.float32)
        img_fused[0:text_h + 8, 0:text_w + 8] *= 0.6
        img_fused = img_fused.astype(np.uint8)
        cv2.putText(img_fused, fps_str, (0, text_h + 2), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_fused, color_masks


def draw_gt(gt_masks):
    masks_semantic = gt_masks.astype('int').sum(axis=0) % (len(COCO_CLASSES))
    colors = get_colors(len(COCO_CLASSES))
    colors = np.array(colors, dtype=np.uint8)
    color_masks = colors[masks_semantic].astype('uint8')
    return color_masks

def draw_box(frame, box, color, class_id, score):
    hide_score = False
    scale = 0.6
    thickness = 1
    font = cv2.FONT_HERSHEY_DUPLEX

    x1, y1, x2, y2 = box
    class_name = COCO_CLASSES[class_id]
    
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    text_str = f'{class_name}: {score:.2f}' if not hide_score else class_name
    text_w, text_h = cv2.getTextSize(text_str, font, scale, thickness)[0]
    cv2.rectangle(frame, (x1, y1), (x1 + text_w, y1 + text_h + 5), color, -1)
    cv2.putText(frame, text_str, (x1, y1 + 15), font, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return frame

def get_colors(num):
    colors = [[0, 0, 0]]
    np.random.seed(0)
    for _ in range(num):
        color = np.random.randint(0, 256, [3]).astype(np.uint8)
        colors.append(color.tolist())
    return colors


iou_thres = [x / 100 for x in range(5, 50, 5)]
def evaluate(outputs, ground_truth):
    gt, gt_masks, img_h, img_w = ground_truth
    ap_data = {'box': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thres],
               'mask': [[APDataObject() for _ in COCO_CLASSES] for _ in iou_thres]}
    
    ids_p, class_p, boxes_p, masks_p = outputs
    ap_obj = ap_data['box'][0][0]
    prep_metrics(ap_data, ids_p, class_p, boxes_p, masks_p, gt, gt_masks, img_h, img_w, iou_thres)
    accuracy, precision, recall = ap_obj.get_accuracy()
    gt_mask = draw_gt(gt_masks)
    return gt_mask, (accuracy, precision, recall)

def add_eval_data(frame, accuracy, precision, recall):
    text_acc = f"accuracy {accuracy}"
    text_pre = f"precision {precision}"
    text_rec = f"recall {recall}"
    cv2.putText(frame, text_acc, (15, 25), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 125, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_pre, (15, 50), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 125, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, text_rec, (15, 75), cv2.FONT_HERSHEY_DUPLEX, 0.6,
                (255, 125, 255), 1, cv2.LINE_AA)
    return frame





