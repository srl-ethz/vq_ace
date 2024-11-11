#The code is modified from https://colab.research.google.com/github/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb#scrollTo=u7JXMffGwQIM


import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings
    


class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, decay, epsilon=1e-5, counterMax = 256):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self.register_buffer('_codebook_counters', torch.full((num_embeddings,), counterMax))
        self._counterMax = counterMax
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        
        self._decay = decay
        self._epsilon = epsilon

    def encodings_from_indices(self, indices):
        encodings = torch.zeros(indices.shape[0], self._num_embeddings, device=indices.device)
        encodings.scatter_(1, indices, 1)
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight)
        return encodings, quantized

    def forward(self, inputs):
        inputs = inputs.contiguous() # B,X,X,C, C=self._embedding_dim
        
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings, quantized = self.encodings_from_indices(encoding_indices)
        quantized = quantized.view(input_shape)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

            self._codebook_counters -= 1
            used_embeddings = torch.sum(encodings, dim=0) > 0
            self._codebook_counters[used_embeddings] = self._counterMax

            # Replace codes with largest distance when counters reach 0
            for i in range(self._num_embeddings):
                if self._codebook_counters[i] <= 0:
                    # Find the vector with the largest distance in the batch
                    distances_min = torch.min(distances, dim=1).values # to min distance toward any code
                    max_dist_index = torch.argmax(distances_min)
                    self._embedding.weight.data[i] = flat_input[max_dist_index]
                    self._ema_w.data[i] = flat_input[max_dist_index]
                    # self._embedding.weight.data[i].normal_()
                    # self._ema_w.data[i].normal_()
                    self._codebook_counters[i] = self._counterMax
        
        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        
        # Straight Through Estimator
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return {"commitment_loss": e_latent_loss}, quantized.contiguous(), perplexity, encodings
    
