from ObjectDetectionVal import bboxEvaluation
import os
import cv2
import numpy as np
import pandas as pd


def non_max_suppression_fast(boxes, overlapThresh=0.2):
    '''
    # Malisiewicz et al.
    :param boxes: list of bboxes in [x1, y1, x2, y2] format
    :param overlapThresh: minimum amount of overlap of boxes to be merged
    :return: uniq bboxes

    '''

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick], pick

if __name__ == '__main__':
    image_dir = "/media/vahid/Elements/Data/yolo_face_plate_dataset/val/images"
    gt_dir = "/media/vahid/Elements/Data/yolo_face_plate_dataset/val/labels/"
    detection_dir = "/media/vahid/Elements/Data/yolo_face_plate_dataset/val/out"

    # read the detections
    label_files = [item for item in os.listdir(detection_dir) if item.endswith(".txt")]
    data = []
    for text_file in label_files:
        with open(os.path.join(detection_dir, text_file)) as current_label_file:
            lines = current_label_file.read().splitlines()
            records = [item.split(" ") for item in lines]
            if len(records) > 0:
                if len(records[0]) > 2:
                    boxes = np.array([[float(item[1]), float(item[2]),
                                       float(item[3]), float(item[4])]
                                      for item in records if len(item) > 2])

                    boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                    boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                    boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                    boxes[:, 3] = boxes[:, 1] + boxes[:, 3]
                    # boxes, idx = non_max_suppression_fast(boxes)
                    idx = range(len(boxes))
                    labels = [int(item[0]) for id, item in enumerate(records) if (len(item) > 2 and id in idx)]
                    conf = [float(item[5]) for id, item in enumerate(records) if (len(item) > 2 and id in idx)]


                    for label, box, confi in zip(labels, boxes, conf):
                        if confi > 0.3:
                            data.append([text_file, label, box, confi])

    predicted_df = pd.DataFrame(data, columns=["image_name", "label", "bbox", "conf"])

    evaluator = bboxEvaluation(gt_dir, image_dir)
    evaluator.calc_mAP(predicted_df, print_details=True)

