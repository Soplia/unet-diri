from __future__ import print_function, division
import torch
import torch.nn.functional as F

def dice_loss(prediction, target):
    """Calculating the dice loss
    Args:
        prediction = predicted image
        target = Targeted image
    Output:
        dice_loss"""

    smooth = 1.0

    i_flat = prediction.view(-1)
    t_flat = target.view(-1)

    intersection = (i_flat * t_flat).sum()

    return 1 - ((2. * intersection + smooth) / (i_flat.sum() + t_flat.sum() + smooth))

def calc_loss(prediction, target, bce_weight=0.5):
    """Calculating the loss and metrics
    Args:
        prediction = predicted image
        target = Targeted image
        metrics = Metrics printed
        bce_weight = 0.5 (default)
    Output:
        loss : dice loss of the epoch """
    
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    prediction = torch.sigmoid(prediction)
    dice = dice_loss(prediction, target)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    return loss

def relu_evidence(logits):
    return F.relu(logits)
def L2Loss(inputs):
    return torch.sum(inputs ** 2) / 2

def KL(alpha, numOfClass):
    beta = torch.ones((1, numOfClass), dtype = torch.float32, requires_grad= False)
    S_alpha = torch.sum(alpha, dim= 1, keepdims= True)
    S_beta = torch.sum(beta, dim = 1, keepdims= True)
    lnB = torch.lgamma(input= S_alpha) - torch.sum(torch.lgamma(alpha), dim= 1, keepdims= True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim= 1, keepdims= True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(input= S_alpha)
    dg1 = torch.digamma(input= alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim = 1, keepdims= True) + lnB + lnB_uni
    return kl

def DirichletLoss(p, alpha, numOfClass, global_step= 1, annealing_step= 1, lam= 1):
    S = torch.sum(alpha, dim = 1, keepdims = True)
    logLikeHood = torch.sum ((p - (alpha / S)) ** 2, dim = 1, keepdims= True) + \
                              torch.sum (alpha * (S - alpha) / (S * S * (S + 1)), dim = 1, keepdims= True)
    
    KL_reg = min(1.0, float(global_step) / annealing_step) * \
                     KL((alpha - 1) * (1 - p) + 1, numOfClass)
    return logLikeHood + lam * KL_reg

def PretreatBefCalLoss(resultAftModel, label, numOfClass= 2):
    oriShape = resultAftModel.shape
    alpha = torch.zeros(oriShape[0] * oriShape[2] * oriShape[3], oriShape[1])
    reaLabel = torch.zeros(oriShape[0] * oriShape[2] * oriShape[3], oriShape[1])
    # 这边有了之后需要重新改正上面的代码
    for batch in range(oriShape[0]):
        for channel in range(numOfClass): 
            for i in range(oriShape[2]):
                for j in range(oriShape[3]):
                    alpha[batch * oriShape[2] * oriShape[3] + i * oriShape[2] + j][channel] = \
                    relu_evidence(resultAftModel[batch][channel][i][j]) + 1
                    reaLabel[batch * oriShape[2] * oriShape[3] +i * oriShape[2] + j][channel] = \
                        label[batch][0][i][j]
    return alpha, reaLabel

def threshold_predictions_v(predictions, thr=150):
    thresholded_preds = predictions[:]
   # hist = cv2.calcHist([predictions], [0], None, [2], [0, 2])
   # plt.plot(hist)
   # plt.xlim([0, 2])
   # plt.show()
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 255
    return thresholded_preds

def threshold_predictions_p(predictions, thr=0.01):
    thresholded_preds = predictions[:]
    #hist = cv2.calcHist([predictions], [0], None, [256], [0, 256])
    low_values_indices = thresholded_preds < thr
    thresholded_preds[low_values_indices] = 0
    low_values_indices = thresholded_preds >= thr
    thresholded_preds[low_values_indices] = 1
    return thresholded_preds

def MaxmunVote(resultAftModel):
    oriShape = resultAftModel.shape
    pred = torch.zeros(oriShape[0], 1, oriShape[2], oriShape[2])
    for batch in range(oriShape[0]):
        for channel in range(oriShape[1]):
            resultAftModel[batch][channel] = threshold_predictions_p(resultAftModel[batch][channel])
            pred[batch][0] += resultAftModel[batch][channel]
        pred[batch][0] = pred[batch][0] >= 2
    return pred

def RefineWithUncertainty(pred, reaLabel, uncertainty, th= .5):
    '''
    If the uncertainty is greater than the threshold, 
    the real label is used to replace the predicted label
    '''
    oriShape = pred.shape
    pred  = pred.cpu().detach().numpy()
    for batch in range(oriShape[0]):
        index = uncertainty >= th
        index = index.numpy()
        pred[index] = reaLabel.cpu().detach().numpy()[index]
    return pred

def CalculateUncertainty(alpha, shape):
    K = alpha.shape[1]
    S = torch.sum(alpha, dim= 1)
    uncertainty = K / S
    return uncertainty.reshape(shape)