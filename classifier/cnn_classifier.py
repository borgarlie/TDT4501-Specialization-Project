import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_Text(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes, num_kernels, kernel_sizes, dropout_p):
        super(CNN_Text, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_kernels = num_kernels
        self.kernel_sizes = kernel_sizes
        self.dropout_p = dropout_p

        # V = args.embed_num  # vocab_size
        # D = args.embed_dim  # hidden_size
        #
        # C = args.class_num  # num_classes
        # Ci = 1  # ? 1?
        # Co = args.kernel_num  # number of each kernel, num_kernels
        # Ks = args.kernel_sizes  # kernel_sizes

        self.embed = nn.Embedding(self.vocab_size, self.hidden_size)

        self.convs1 = nn.ModuleList(
            [nn.Conv2d(1, self.num_kernels, (kernel, self.hidden_size)) for kernel in self.kernel_sizes])
        '''
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
        '''
        # can remove comment

        self.dropout = nn.Dropout(self.dropout_p)
        self.fc1 = nn.Linear(len(self.kernel_sizes) * self.num_kernels, self.num_classes)

    def forward(self, input):
        x = self.embed(input)

        x = x.unsqueeze(1)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]

        x = torch.cat(x, 1)  # Does this do anything now ?

        '''
        x1 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3), 1) # (N,len(Ks)*Co)
        '''
        # can remove comment

        x = self.dropout(x)
        logit = self.fc1(x)
        return logit
