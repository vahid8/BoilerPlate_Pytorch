import os
import cv2
import numpy as np
import pandas as pd

class bboxEvaluation:
    def __init__(self, label_dir, image_dir):
        self.label_dir = label_dir
        self.image_dir = image_dir
        self.gt_df = self.load_validation_labels()

    def calc_mAP(self, predicted_df, print_details=False):
        # Calculation is done separately for each label
        all_AP = []
        all_labels = []
        all_P = []
        all_R = []

        for uniq_label in predicted_df['label'].unique():
            all_labels.append(uniq_label)
            current_predicted_df = predicted_df[predicted_df["label"] == uniq_label]
            current_gt_df = self.gt_df[self.gt_df["label"] == uniq_label]
            total_gt_bbox_current_label = len(current_gt_df)
            # calculation of iou for each image separately
            all_images_iou = []
            all_images_conf = []

            for name, group in current_predicted_df.groupby("image_name"):
                current_gt = current_gt_df[current_gt_df['image_name'] == name]
                image_iou = self.calc_iou(group['bbox'].tolist(), current_gt['bbox'].tolist())
                image_conf = group['conf'].tolist()

                all_images_iou.extend(image_iou)
                all_images_conf.extend(image_conf)

            # Now create table for the current label
            # sort based on confidence
            all_images_iou = [x for _, x in sorted(zip(all_images_conf, all_images_iou))]
            # do calculations base on iou threshold 0.5
            array_b = np.where(np.array(all_images_iou) >= 0.5, 1, 0)
            true_positive_sum = 0
            false_positive_sum = 0
            all_images_precision = []
            all_images_recall = []
            for true_positive in array_b:
                true_positive_sum += true_positive
                false_positive_sum += (1 - true_positive)
                all_images_precision.append(true_positive_sum / (true_positive_sum + false_positive_sum))
                all_images_recall.append(true_positive_sum / total_gt_bbox_current_label)

            # Now calculate the AP (Area under Precision-Recall curve)
            prev_p = 1
            prev_r = 0
            cum_area = 0
            for p, r in zip(all_images_precision, all_images_recall):
                area = (p+prev_p)*(r-prev_r)/2
                prev_r = r
                prev_p = p
                cum_area += area

            all_P.append(round(prev_p, 2))
            all_R.append(round(prev_r, 2))
            all_AP.append(cum_area)

        if print_details:
            print("label".ljust(10,' ')+"|"+"meanP".ljust(10,' ')+"|"+"meanR".ljust(10,' ')+"|"+"AP@0.5".ljust(10, ' '))
            print("".ljust(10, '-') + "|" + "".ljust(10, '-'))
            for l, p, r, AP in zip(all_labels, all_P, all_R, all_AP):
                print(f"{str(l).ljust(10,' ')}|{str(p*100).ljust(10,' ')}|{str(r*100).ljust(10,' ')}|"
                      f"{str(round(AP*100,2)).ljust(10,' ')}")

        print("-"*50)
        print("meanP(%)".ljust(20, ' ') + "MeanR(%)".ljust(20, ' ') + "mAP@0.5(%)".ljust(20, ' '))
        print(f"{str(round((sum(all_P)/len(all_P))*100,2)).ljust(20,' ')}{str(round((sum(all_R)/len(all_R))*100,2)).ljust(20,' ')}"
              f"{str(round((sum(all_AP)/len(all_AP))*100,2)).ljust(20,' ')}")


    def load_validation_labels(self):
        data = []
        label_files = [item for item in os.listdir(self.label_dir) if item.endswith(".txt")]
        for text_file in label_files:
            with open(os.path.join(self.label_dir, text_file)) as current_label_file:
                lines = current_label_file.read().splitlines()
                records = [item.split(" ") for item in lines]
                if len(records) > 0:
                    if len(records[0]) > 2:
                        boxes = np.array([[float(item[1]), float(item[2]),
                                           float(item[3]), float(item[4])]
                                          for item in records])
                        boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
                        boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
                        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
                        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

                        labels = [int(item[0]) for item in records]

                        for label, box in zip(labels, boxes):
                            data.append([text_file, label, box])

        return pd.DataFrame(data, columns=["image_name", "label", "bbox"])


    @staticmethod
    def calc_iou(pred_bboxes, ground_truth_bboxes):

        '''
        calculate iou for one image for a set of ground truth and predicted bboxes

        :param pred_bboxes: boxes in [[x1,y1, x2, y2],[x1,y1, x2, y2], ...] format with the same label
        :param ground_truth_bboxes: boxes in [[x1,y1, x2, y2],[x1,y1, x2, y2], ...] format with the same label
        :param iou_threshold: threshold to return true (TP) or false (FP) for detected bboxes
        :param conf_level: list of confidence levels of predictions

        :return:
        '''
        # create a numpy array from upper left corners of gt
        gt_up_left_corners = np.array([[item[0], item[1]] for item in ground_truth_bboxes])

        iou_all = list()
        for item in pred_bboxes:
            if len(gt_up_left_corners)>0:
                # create a numpy array from upper left corners of gt
                current_up_left_corner = np.array([item[0], item[1]]).reshape(1, 2)
                # find the nearest bbox in ground truth
                idx = np.argmin(np.linalg.norm((current_up_left_corner - gt_up_left_corners), axis=1))
                ##-------------- calculate the iou
                # calculate the intersection (the area between min of upper left and max of lower right
                temp_gt = ground_truth_bboxes[idx]

                # check if there is any intersection ( check start and end point of bbox against upper left point of gt
                x_sign = (item[0] - temp_gt[0]) * (item[2] - temp_gt[0])
                y_sign = (item[1] - temp_gt[1])* (item[3] - temp_gt[1])

                if x_sign <= 0 and y_sign <= 0:
                    upper_left_x = min([item[0], temp_gt[0]])
                    upper_left_y = min([item[1], temp_gt[1]])

                    lower_right_x = max([item[2], temp_gt[2]])
                    lower_right_y = max([item[3], temp_gt[3]])

                    intersection_area = (upper_left_x - lower_right_x) * (upper_left_y - lower_right_y)

                    # calculate the union (all)
                    area_1 = (item[2] - item[0]) * (item[3] - item[1])
                    area_2 = (temp_gt[2] - temp_gt[0]) * (temp_gt[3] - temp_gt[1])
                    union = area_2 + area_1 - intersection_area

                    # calculate iou
                    iou = intersection_area / (union + 1e-8)

                else:  # No intersection
                    iou = 0
            else:
                iou = 0

            iou_all.append(iou)

        return iou_all


    @staticmethod
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
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]
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
        return boxes[pick]
