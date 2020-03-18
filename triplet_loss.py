from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from torch.nn.modules.distance import PairwiseDistance
from torch.nn.functional import pdist
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import torch

#### IMPORTANT: Note this code has been adapted from Github repository of the paper 'Time-Contrastive Networks: Self-Supervised Learning from 
####            Video', published in IEEE, ICRA 2018. Link to the repo: https://github.com/tensorflow/models/tree/master/research/tcn

def triplet_cond(z, margin, seq_len, dim_z):
    num_data = seq_len
    embedding = z.permute(1, 0)
    times = np.tile(np.arange(seq_len, dtype=np.int32), 1)
    times = np.reshape(times, [times.shape[0], 1])
    sequence_ids = np.concatenate([np.ones(seq_len)*i for i in range(seq_len)])
    sequence_ids = np.reshape(sequence_ids, [sequence_ids.shape[0], 1])

    pos_radius = 2 #int(len(z)*0.2)
    neg_radius = 5 #int(len(z)*0.5)

    # Get a positive mask, i.e. indices for each time index
    # that are inside the positive range.
    in_pos_range = np.less_equal(np.abs(times - times.transpose()), pos_radius)

    # Get a negative mask, i.e. indices for each time index
    # that are inside the negative range (> t + (neg_mult * pos_radius)
    # and < t - (neg_mult * pos_radius).
    in_neg_range = np.greater(np.abs(times - times.transpose()), neg_radius)

    sequence_adjacency = sequence_ids == sequence_ids.T
    sequence_adjacency_not = np.logical_not(sequence_adjacency)

    #pdist_matrix = euclidean_distances(embedding, squared=True)
     
    pdist_matrix = torch.norm(embedding[:, None] - embedding, dim = 2, p = 2)

    loss_np = 0.0
    num_positives = 0.0
    for i in range(num_data):
      for j in range(num_data):
        if in_pos_range[i, j] and i != j and sequence_adjacency[i, j]:
          num_positives += 1.0

          pos_distance = pdist_matrix[i][j]
          neg_distances = []

          for k in range(num_data):
            if in_neg_range[i, k] or sequence_adjacency_not[i, k]:
              neg_distances.append(pdist_matrix[i][k])
          
          neg_distances.sort()  # sort by distance
          
          chosen_neg_distance = neg_distances[0]

          for l in range(len(neg_distances)):
            chosen_neg_distance = neg_distances[l]
            if chosen_neg_distance > pos_distance:
              break
 
          zero_ = torch.autograd.Variable(torch.FloatTensor(np.array(0.0)), requires_grad = True) 
          loss_np += torch.max(zero_.cuda(), margin - chosen_neg_distance + pos_distance)

    loss_np /= num_positives

    return loss_np
