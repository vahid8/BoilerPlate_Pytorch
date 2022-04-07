from ObjectDetectionVal import bboxEvaluation
import os
import cv2
import numpy as np
import pandas as pd

if __name__ == '__main__':
    image_dir = "mAP_test_data/src"
    gt_dir = "mAP_test_data/gt"
    detection_dir = "mAP_test_data/detection"

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

                    labels = [int(item[0]) for item in records if len(item) > 2]
                    conf = [float(item[5]) for item in records if len(item) > 2]

                    for label, box, confi in zip(labels, boxes, conf):
                        data.append([text_file, label, box, confi])

    predicted_df = pd.DataFrame(data, columns=["image_name", "label", "bbox", "conf"])

    evaluator = bboxEvaluation(gt_dir, image_dir)
    evaluator.calc_mAP(predicted_df, print_details=True)

