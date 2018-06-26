import torch
import torch.nn as nn
import torch.nn.functional as F


def cwh2point(boxes):
    """
    Convert boxes to (x_min, y_min, x_max, y_max) representation from (center_x, center_y, width, height).
    """

    return torch.cat((
                        boxes[:, :2] - boxes[:, 2:]/2,     # x_min, y_min
                        boxes[:, :2] + boxes[:, 2:]/2),    # x_max, y_max
                     1)


def point2cwh(boxes):
    """
    Reverse of cwh2point.
    """

    return torch.cat((
                        (boxes[:, 2:] + boxes[:, :2]) / 2,    # cx, cy
                        boxes[:, 2:] - boxes[:, :2]),         # w, h
                     1)


def intersect(box_a, box_b):
    """
    Compute intersection of two sets of boxes.
    """

    max_xy = torch.min(box_a[:, 2:].unsqueeze(1), box_b[:, 2:].unsqueeze(0))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1), box_b[:, :2].unsqueeze(0))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_area(b): 
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def jaccard(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes.
    """

    inter = intersect(box_a, box_b)
    union = box_area(box_a).unsqueeze(1) + box_area(box_b).unsqueeze(0) - inter
    return inter / union


def match_prior_with_truth(priors, bbox_truth):
    """
    Match each prior box with the ground truth box.
    """

    # size (num_true_boxes, num_prior_boxes)
    overlaps = jaccard(
        bbox_truth,
        cwh2point(priors)
    )

    prior_overlap, prior_idx = overlaps.max(1)

    #if prior_idx.numel() != prior_idx.unique().numel():
    #    print("WARNING: two ground truth boxes matched to one prior box.")

    #print("True boxes with small overlap:", torch.sum(prior_overlap < 0.4).item(), "/", bbox_truth.size()[0])

    overlaps[torch.arange(bbox_truth.size()[0]).long(), prior_idx] = 10.

    truth_overlap, truth_idx = overlaps.max(0)
    
    labeled_pos = truth_overlap > 0.4
    labeled_pos_idx = labeled_pos.nonzero()[:, 0]

    nonlabeled_pos_idx = (labeled_pos == 0).nonzero()[:, 0]

    return labeled_pos_idx, truth_idx[labeled_pos_idx], nonlabeled_pos_idx


def localization_loss(bbox_predictions, priors, targets):
    # formula (2) for SSD paper
    ground_truth_centers = (targets[:, :2] - priors[:, :2]) / priors[:, 2:]
    ground_truth_wh = torch.log(targets[:, 2:] / priors[:, 2:])
    ground_truth_vector = torch.cat([ground_truth_centers, ground_truth_wh], 1)
    
    return F.smooth_l1_loss(bbox_predictions, ground_truth_vector, size_average=True)

def multibox_loss(predictions, priors, targets):

    loc_loss = 0
    clf_loss = 0
    
    for bbox_prediction, class_prediction, target in zip(*predictions, targets):

        if target.numel() != 0:
        
            prior_idx_that_matched, truth_idx_for_prior, prior_idx_bg = match_prior_with_truth(priors, target[:,:-1])

            # bbox loss

            priors_that_matched = priors[prior_idx_that_matched]
            target_cwh = point2cwh(target[:,:-1])
            target_for_prior = target_cwh[truth_idx_for_prior]

            loc_loss += localization_loss(bbox_prediction[prior_idx_that_matched], 
                                          priors_that_matched, 
                                          target_for_prior)

            # classification loss

            predictions_for_matched_priors = class_prediction[prior_idx_that_matched]
            labels_for_matched_priors = target[truth_idx_for_prior,-1].long()
            matched_clf_loss = F.cross_entropy(predictions_for_matched_priors, 
                                               labels_for_matched_priors, 
                                               size_average=False) / priors_that_matched.size()[0]
            
            prediction_for_bg_priors = class_prediction[prior_idx_bg]
            labels_for_bg_priors = torch.zeros_like(prior_idx_bg)
            hard_negatives_num = prior_idx_that_matched.size()[0] * 2

        else:
            
            matched_clf_loss = 0
            prediction_for_bg_priors = class_prediction
            labels_for_bg_priors = torch.zeros(priors.shape[0]).long().cuda()
            hard_negatives_num = 5
        
        bg_clf_loss = F.cross_entropy(prediction_for_bg_priors,
                                      labels_for_bg_priors,
                                      size_average=False,
                                      reduce=False)
        hard_negative_bg_idx = bg_clf_loss.topk(hard_negatives_num)[1]
        bg_clf_loss = torch.sum(bg_clf_loss[hard_negative_bg_idx])

        clf_loss += matched_clf_loss + (bg_clf_loss / hard_negatives_num)

    return loc_loss, clf_loss / len(targets)

