import rospy
import cv2 as cv
import yaml
import torch
import os
import sys
import numpy as np

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud, ChannelFloat32

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# Please change YOLO root ditectory!
yolo_path = '/workspace/HDD1/ros_workspace/yolo_ws/src/yolov9_ros'
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

sys.path.insert(0, yolo_path)
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode
from utils.augmentations import (Albumentations, augment_hsv, classify_albumentations, classify_transforms, copy_paste,
                                 letterbox, mixup, random_perspective)

class node:
    def __init__(self):
        rospy.init_node('yolov9_ros', anonymous=True)
        self._mConfigPath = rospy.get_param('path_to_config','')
        self._mCVBridge = CvBridge()

        self._msSubImgTopicName = ""
        self._msPubImgTopicName = ""

        # ===============
        #   yolo parm
        # ===============
        self.weights="" # 'yolo.pt',  # model path or triton URL
        self.source="ROOT" # 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        self.data="ROOT" # 'data/coco.yaml',  # dataset.yaml path
        self.imgsz=(480, 640)  # inference size (height, width)
        self.conf_thres=0.25  # confidence threshold
        self.iou_thres=0.45  # NMS IOU threshold
        self.max_det=1000 # maximum detections per image
        self.device=''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.view_img=False  # show results
        self.save_txt=False  # save results to *.txt
        self.save_conf=False  # save confidences in --save-txt labels
        self.save_crop=False  # save cropped prediction boxes
        self.nosave=False  # do not save images/videos
        self.classes=None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms=False  # class-agnostic NMS
        self.augment=False  # augmented inference
        self.visualize=False  # visualize features
        self.update=False  # update all models
        self.project="ROOT" # 'runs/detect',  # save results to project/name
        self.name='exp'  # save results to project/name
        self.exist_ok=False  # existing project/name ok, do not increment
        self.line_thickness=3  # bounding box thickness (pixels)
        self.half=False  # use FP16 half-precision inference
        self.dnn=False  # use OpenCV DNN for ONNX inference
        self.vid_stride=1  # video frame-rate stride

        self.initalize()

        self._mImgSubscriber = rospy.Subscriber(self._msSubImgTopicName, Image, self.callbackImage, queue_size = 1000)
        self._mInfoPublisher = rospy.Publisher(self._msPubImgTopicName, PointCloud, queue_size = 1000)
        if self.mPubResultImg:
            self._mResultImgPublisher = rospy.Publisher(self.mPubResultImg, Image, queue_size = 1000)

        rospy.spin()
    def initalize(self):
        with open(self._mConfigPath, 'r') as stream:
            param = yaml.safe_load(stream)
            param_ros = param['ros']
            param_yolo = param['yolo']

            self._msSubImgTopicName = param_ros['sub_img_topic_name']
            self._msPubImgTopicName = param_ros['pub_detect_topic_name']

            config_yolo_path = param_yolo['yolo_path']
            if config_yolo_path != yolo_path:
                print("check your yolo path in config.yaml and detect_ros.py file!") # top of this file
                print("shut down")
                return
            self.weights = param_yolo['weight_path']
            self.nImgWidth = param_yolo['image_width']
            self.nImgHeight = param_yolo['image_height']
            self.imgsz = (self.nImgHeight,self.nImgWidth)

            self.visualize = param_yolo['visualize']
            self.mPubResultImg=param_yolo['pub_detect_image_topic_name']

            # Load model
            self.device = select_device(self.device)
            self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
            self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
            self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size

            bs = 1
            self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *(self.imgsz)))  # warmup
            self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile())

            print("\nimage_topic:",self._msSubImgTopicName)
            print("pub_topic:",self._msPubImgTopicName)
            if self.mPubResultImg:
                print("result_img_topic:",self.mPubResultImg)

            self.allocateGPUMemory()

            print("\nInitialize Finish!")

    def allocateGPUMemory(self):
        cvImg = np.zeros((self.nImgHeight, self.nImgWidth, 3), np.uint8)
        im = letterbox(cvImg, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        with self.dt[1]:
            pred = self.model(im, augment=self.augment, visualize=False)
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

    def callbackImage(self, imgMsg):
        cvImg = self._mCVBridge.imgmsg_to_cv2(imgMsg, desired_encoding='bgr8')
        time = imgMsg.header.stamp
        # =====================
        #       Inference
        # =====================
        im = letterbox(cvImg, self.imgsz, stride=self.stride, auto=self.pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.model.device)
            im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with self.dt[1]:
            pred = self.model(im, augment=self.augment, visualize=False)

        # NMS
        with self.dt[2]:
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            gn = torch.tensor(cvImg.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            im0 = cvImg.copy()

            if self.visualize or self.mPubResultImg:
                annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

            msg = PointCloud()
            msg.header.stamp = time
            size_of_result = ChannelFloat32()
            size_of_result.name = "size"
            size_of_result.values.append(len(det))
            msg.channels.append(size_of_result)

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls) # names[c] => object name
                    label = f'{self.names[c]} {conf:.2f}'

                    # cv.rectangle(im0, (int(xyxy[0]),int(xyxy[1]),int(xyxy[2]),int(xyxy[3])),(255,255,255),3)
                    one_object = ChannelFloat32()
                    one_object.name = f'{self.names[c]}'
                    one_object.values.append(conf)
                    one_object.values.append(int(xyxy[0])) # x1
                    one_object.values.append(int(xyxy[1])) # y1
                    one_object.values.append(int(xyxy[2])) # x2
                    one_object.values.append(int(xyxy[3])) # y2
                    msg.channels.append(one_object)

                    if self.visualize or self.mPubResultImg:
                        annotator.box_label(xyxy, label, color=colors(c, True))
            
            self._mInfoPublisher.publish(msg)

            if self.visualize:
                cv.imshow("test_infer",im0)
                cv.waitKey(1)
            if self.mPubResultImg:
                rosImg = self._mCVBridge.cv2_to_imgmsg(im0)
                self._mResultImgPublisher.publish(rosImg)
        
if __name__ == "__main__":
    rosNode = node()

    