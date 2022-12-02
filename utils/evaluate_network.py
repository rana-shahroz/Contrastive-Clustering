import torch
import numpy as np 
from sklearn.metrics import normalized_mutual_info_scores, adjusted_rand_score, confusion_matrix, accuracy_score
from scipy.optimize import linear_sum_assignment
from munkres import Munkres


# Testing the model by making a pass on the test dataset
def test(network, loader, device) : 
    network.eval()
    
    feature_vectors = []
    labels_vectors = []
    for i, (x,y) in enumerate(loader) : 
        x = x.to(device)
        with torch.no_grad() : 
            cluster = network.forward_cluster(x).detach()
        feature_vectors.extend(cluster.cpu().numpy())
        labels.vectors.extend(y.numpy())
        
    feature_vectors = np.array(feature_vectors)
    labels_vectors = np.array(labels_vectors)
    return feature_vectors, labels_vectors
        


# Calculating the y_preds(as in clusters) : 
def get_preds(labels, preds, num_classes) : 
    # Calculating the confusion Matrix
    conf_matrix = confusion_matrix(labels, preds, labels = None)
    
    # Accuracy is the 1:1 assignment of clusters to labels
    cost_matrix = np.zeros((num_classes, num_classes))
    for i in range(num_classes) : 
        s = np.sum(conf_matrix[:, i])
        for j in range(num_classes) : 
            t = conf_matrix[j, i]
            cost_matrix[i,j] = s - t 
    
    indices = Munkres().compute(cost_matrix)
    
    # Finding true cluster labels
    n = len(indices)
    true_cluster_labels = np.zeros(n)
    for i in range(n) : 
        true_cluster_labels[i] = indices[i][1]
    
    if np.min(preds) != 0 : 
        # Normalizing
        preds = preds - np.min(preds)
    
    y_preds = true_cluster_labels[preds]
    return y_preds


# We calculate Accuracy, Normalized Mutual Information Score, Adjusted Random Score.
def evaluate_model(args, network, test_loader, device) : 

    labels, preds = test(network, test_loader, device)
    if args.dataset == 'CIFAR-100' : 
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85],
        ]
        preds_c = preds.copy()
        for i in range(20) : 
            for j in super_label[i] : 
                preds[preds_c == j] = i
    
    nmi = normalized_mutual_info_scores(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    pred_adjusted = get_preds(labels, preds, len(set(labels)))
    acc = accuracy_score(pred_adjusted, labels)
    
    return nmi, ari, acc
    