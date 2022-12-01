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
def evaluate_model(network, test_loader, device) : 
    labels, preds = test(network, test_loader, device)
    nmi = normalized_mutual_info_scores(labels, preds)
    ari = adjusted_rand_score(labels, preds)
    pred_adjusted = get_preds(labels, preds, len(set(labels)))
    acc = accuracy_score(pred_adjusted, labels)
    
    return nmi, ari, acc
    