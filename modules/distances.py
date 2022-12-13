import torch 
import math


def cosine_similarity(dim=2) : 
    return torch.nn.CosineSimilarity(dim=dim)

def pairwise_distance(p = 2.0):
    return torch.nn.PairwiseDistance(p=p)

def get_similarity_metric(name) :
    return cosine_similarity()