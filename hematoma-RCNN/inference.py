import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import psutil
from model import create_model
import subprocess
from statistics import mean
from config import (
    NUM_CLASSES, DEVICE, NUM_WORKERS, CLASSES
)
from datasets import (
    create_test_dataset, 
    create_test_loader
)

from torchmetrics.detection.mean_ap import MeanAveragePrecision

def get_gpu_load():
    """
    Returns the current GPU utilization as a string percentage.
    """
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
        encoding='utf-8'
    )
    return result.strip()


def test(test_data_loader, model):
    print('Testing')
    model.eval()
    target = []
    preds = []

    time.sleep(3)
    cpu_usage_before = psutil.cpu_percent(interval=None)
    gpu_load_before = get_gpu_load() if torch.cuda.is_available() else "N/A"
    cpu_list = []
    gpu_list = []
    for i, data in enumerate(test_data_loader):
        images, targets = data
        #images = [cv2.resize(image, (640, 640)) for image in images]

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        
        for t in targets:
            if t['boxes'].size() == torch.Size([0]):
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                #boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.zeros([0], dtype=torch.int64)
                #labels = torch.as_tensor(labels, dtype=torch.int64)
                area = torch.zeros([0], dtype=torch.float32)
                iscrowd = torch.zeros([0], dtype=torch.int64)
                t["boxes"] = boxes.to(DEVICE)
                t["labels"] = labels.to(DEVICE)
                t["area"] = area.to(DEVICE)
                t["iscrowd"] = iscrowd.to(DEVICE)

        
        with torch.no_grad():
            outputs = model(images, targets)
            cpu_usage_after = psutil.cpu_percent(interval=None)
            gpu_load_after = get_gpu_load() if torch.cuda.is_available() else "N/A"
            
        cpu_list.append(cpu_usage_after)
        gpu_list.append(int(gpu_load_after))
        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict) 
        #####################################
    cpu_usage_avg = mean(cpu_list)
    cpu_usage_diff = cpu_usage_avg - cpu_usage_before
    print(f"CPU usage increase due to inference in test function: {cpu_usage_diff}%")

    gpu_useage_avg = mean(gpu_list)
    print(f"GPU Load before execution: {gpu_load_before}%")
    print(f"GPU Load after execution: {gpu_useage_avg}%")


    metric = MeanAveragePrecision()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':

    # this will help us create a different color for each class
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # load the best model and trained weights
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('C:/Users/akumar80/Documents/Avisha Kumar Lab Work/HematomaDetectionRCNN/outputs/best_model.pth', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()

    # directory where all the images are present
    DIR_TEST = #'C:/Users
    test_images = glob.glob(f"{DIR_TEST}/*.png")
    print(f"Test instances: {len(test_images)}")
    # define the detection threshold...
    # ... any detection having score below this will be discarded
    detection_threshold = 0.5 #0.8
    # to count the total number of images iterated through
    frame_count = 0
    # to keep adding the FPS for each image
    total_fps = 0 

    test_dataset = create_test_dataset()

    test_loader = create_test_loader(test_dataset, NUM_WORKERS)

    metric_summary = test(test_loader, model)

    print(f"mAP@0.50:0.95: {metric_summary['map']}")
    print(f"mAP@0.50: {metric_summary['map_50']}")   
    print(f"mAP@0.75: {metric_summary['map_75']}")   
    print(f"mAR@1: {metric_summary['mar_1']}")   
    print(f"mAR@10: {metric_summary['mar_10']}")  
    print(f"mAR@large: {metric_summary['mar_large']}")  
    print(f"mAR@medium: {metric_summary['mar_medium']}")  
    print(f"mAR@small: {metric_summary['mar_small']}")

    time.sleep(3)
    cpu_usage_before = psutil.cpu_percent(interval=None)
    gpu_load_before = get_gpu_load() if torch.cuda.is_available() else "N/A"
    cpu_list = []
    gpu_list = []

    for i in range(len(test_images)):
        # get the image file name for saving output later on
        image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
        image = cv2.imread(test_images[i])
        orig_image = image.copy()
        # BGR to RGB
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # make the pixel range between 0 and 1
        image /= 255.0
        # bring color channels to front
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # convert to tensor
        image = torch.tensor(image, dtype=torch.float).cuda()
        # add batch dimension
        image = torch.unsqueeze(image, 0)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(image.to(DEVICE))
            cpu_usage_after = psutil.cpu_percent(interval=None)
            gpu_load_after = get_gpu_load() if torch.cuda.is_available() else "N/A"
            
        cpu_list.append(cpu_usage_after)
        gpu_list.append(int(gpu_load_after))
        end_time = time.time()
        
        # get the current fps
        fps = 1 / (end_time - start_time)
        # add `fps` to `total_fps`
        total_fps += fps
        # increment frame count
        frame_count += 1
        # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
            
            # # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[CLASSES.index(class_name)]
                cv2.rectangle(orig_image,
                            (int(box[0]), int(box[1])),
                            (int(box[2]), int(box[3])),
                            color, 2)
                cv2.putText(orig_image, class_name, 
                            (int(box[0]), int(box[1]-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                            2, lineType=cv2.LINE_AA)
            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            cv2.imwrite(#f"C:/Users
        print(f"Image {i+1} done...")
        print('-'*50)
    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    cpu_usage_avg = mean(cpu_list)
    cpu_usage_diff = cpu_usage_avg - cpu_usage_before
    print(f"CPU usage increase due to inference in test function: {cpu_usage_diff}%")

    gpu_useage_avg = mean(gpu_list)
    print(f"GPU Load before execution: {gpu_load_before}%")
    print(f"GPU Load after execution: {gpu_useage_avg}%")