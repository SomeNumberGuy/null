import numpy as np
from scipy.spatial import distance

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) of two bounding boxes.
    """
    x1_tl, y1_tl, w1, h1 = box1
    x2_tl, y2_tl, w2, h2 = box2

    x1_br, y1_br = x1_tl + w1, y1_tl + h1
    x2_br, y2_br = x2_tl + w2, y2_tl + h2

    # Calculate coordinates of intersection area
    x_tl = max(x1_tl, x2_tl)
    y_tl = max(y1_tl, y2_tl)
    x_br = min(x1_br, x2_br)
    y_br = min(y1_br, y2_br)

    # Calculate intersection area
    intersection_area = max(0, x_br - x_tl + 1) * max(0, y_br - y_tl + 1)

    # Calculate area of both bounding boxes
    area_box1 = (w1 + 1) * (h1 + 1)
    area_box2 = (w2 + 1) * (h2 + 1)

    # Calculate union area
    union_area = area_box1 + area_box2 - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_metrics(ground_truth, predictions, confidence_threshold=0.5, iou_threshold=0.5):
    """
    Calculate precision, recall, and F1-score for object detection.
    """
    # Filter predictions by confidence threshold
    predictions = [pred for pred in predictions if pred[4] >= confidence_threshold]

    # Sort predictions by confidence score
    predictions.sort(key=lambda x: x[4], reverse=True)

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    # Calculate true positives, false positives, and false negatives
    for gt_box in ground_truth:
        gt_match = False
        for pred_box in predictions:
            if calculate_iou(gt_box[:4], pred_box[:4]) >= iou_threshold and gt_box[4] == pred_box[5]:
                true_positives += 1
                gt_match = True
                break
        if not gt_match:
            false_negatives += 1

    false_positives = len(predictions) - true_positives

    # Calculate precision, recall, and F1-score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def calculate_mAP(ground_truth, predictions, iou_threshold=0.5):
    """
    Calculate mean average precision (mAP) for object detection.
    """
    average_precisions = []
    for class_label in range(1, max(max(gt_box[4] for gt_box in ground_truth), max(pred_box[5] for pred_box in predictions)) + 1):
        class_gt = [gt_box for gt_box in ground_truth if gt_box[4] == class_label]
        class_pred = [pred_box for pred_box in predictions if pred_box[5] == class_label]

        n_positives = len(class_gt)
        true_positives = np.zeros(len(class_pred))

        for i, pred_box in enumerate(class_pred):
            pred_box_gt_iou = [calculate_iou(pred_box[:4], gt_box[:4]) for gt_box in class_gt]
            if len(pred_box_gt_iou) > 0:
                if max(pred_box_gt_iou) >= iou_threshold:
                    true_positives[i] = 1

        cumul_true_positives = np.cumsum(true_positives)

        precision = cumul_true_positives / (np.arange(len(class_pred)) + 1)
        recall = cumul_true_positives / n_positives

        recall = np.concatenate(([0], recall, [1]))
        precision = np.concatenate(([0], precision, [0]))

        # Compute the precision envelope
        for i in range(len(precision) - 2, -1, -1):
            precision[i] = max(precision[i], precision[i + 1])

        # Compute average precision as the area under the precision-recall curve
        indices = np.where(recall[1:] != recall[:-1])[0] + 1
        average_precision = np.sum((recall[indices] - recall[indices - 1]) * precision[indices])

        average_precisions.append(average_precision)

    mAP = np.mean(average_precisions)

    return mAP

# Example usage
ground_truth_data = [(10, 10, 20, 20, 1), (30, 30, 40, 40, 2)]  # Format: (x_tl, y_tl, w, h, label)
prediction_data = [(10, 10, 20, 20, 0.9, 1), (25, 25, 30, 30, 0.8, 2)]  # Format: (x_tl, y_tl, w, h, confidence, label)

precision, recall, f1 = calculate_metrics(ground_truth_data, prediction_data)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)

mAP = calculate_mAP(ground_truth_data, prediction_data)
print("mAP:", mAP)
