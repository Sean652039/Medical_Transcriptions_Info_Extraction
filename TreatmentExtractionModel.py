from torch import nn
import torch


class TreatmentLDA(nn.Module):
    def __init__(self, input_size, num_topics):
        super(TreatmentLDA, self).__init__()
        # Document-topic distribution
        self.theta = nn.Parameter(torch.randn(input_size, num_topics))
        # Topic-word distribution
        self.phi = nn.Parameter(torch.randn(num_topics, input_size))

    def forward(self, x):
        # Calculate document-topic distribution
        theta = torch.softmax(self.theta, dim=1)
        # Calculate topic-word distribution
        phi = torch.softmax(self.phi, dim=0)
        # Calculate document-word distribution
        x_hat = torch.matmul(theta, phi)
        return x_hat
