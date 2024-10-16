import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from torch import nn, Tensor
from variation_module import MCDualMixin
import time

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

'''
    Concrete dropout was not used during master thesis
'''


import torch
import torch.nn as nn

class ConcreteDropout(nn.Module):
    def __init__(self, dropout=True, concrete=True, p_fix=0.01, weight_regularizer=1e-5,
                 dropout_regularizer=1e-5, conv="lin", Bayes=True):
        super().__init__()
        self.dropout = dropout
        self.concrete = concrete
        self.p_fix = p_fix
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        self.conv = conv
        self.Bayes = Bayes

        self.p_logit = nn.Parameter(torch.FloatTensor([0]).to(device))

    def forward(self, x, layer, stop_dropout=False):
        if self.concrete:
            p = torch.sigmoid(self.p_logit)
        else:
            p = torch.tensor(self.p_fix, device=device)

        if (self.dropout and not stop_dropout) or self.Bayes:
            out = layer(self._concrete_dropout(x, p, self.concrete))
        else:
            out = layer(x).to(device)

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        regularization, weights_regularizer, dropout_regularizer = 0, 0, 0
        if self.Bayes:
            weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)
            if self.concrete:
                dropout_regularizer = p * torch.log(p)
                dropout_regularizer += (1. - p) * torch.log(1. - p)
                input_dimensionality = x[0].numel() if self.conv == "lin" else list(x.size())[1]
                dropout_regularizer *= self.dropout_regularizer * input_dimensionality
            regularization = weights_regularizer + dropout_regularizer

        return out, regularization

    def _concrete_dropout(self, x, p, concrete):
        if not concrete:
            if self.conv == "lin":
                drop_prob = torch.bernoulli(torch.ones(x.shape, device=device) * p).to(device)
            elif self.conv == "1D":
                drop_prob = torch.bernoulli(torch.ones(list(x.size())[0], list(x.size())[1], 1, device=device) * p).to(device)
                drop_prob = drop_prob.repeat(1, 1, list(x.size())[2])
            else:
                drop_prob = torch.bernoulli(torch.ones(list(x.size())[0], list(x.size())[1], 1, 1, device=device) * p).to(device)
                drop_prob = drop_prob.repeat(1, 1, list(x.size())[2], list(x.size())[3]).to(device)
        else:
            eps = 1e-7
            temp = 0.1

            if self.conv == "lin":
                unif_noise = torch.rand_like(x, device=device).to(device)
            elif self.conv == "1D":
                unif_noise = torch.rand(list(x.size())[0], list(x.size())[1], 1, device=device).to(device)
                unif_noise = unif_noise.repeat(1, 1, list(x.size())[2])
            else:
                unif_noise = torch.rand(list(x.size())[0], list(x.size())[1], 1, 1, device=device).to(device)
                unif_noise = unif_noise.repeat(1, 1, list(x.size())[2], list(x.size())[3])

            drop_prob = (torch.log(p + eps)
                         - torch.log(1 - p + eps)
                         + torch.log(unif_noise + eps)
                         - torch.log(1 - unif_noise + eps))

            drop_prob = torch.sigmoid(drop_prob / temp)

        random_tensor = 1 - drop_prob
        retain_prob = 1 - p
        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x


'''
    CNN
'''

class StochasticCNN_1D(nn.Module, MCDualMixin):
    def __init__(self, num_features_df, value_embedding, emb_dims=None, hidden_size=1024,
                 weight_regularizer=.1e-5, dropout_regularizer=1e-5,
                 conv_dropout_traditional=False, dropout_dense_only=False,
                 input_size=1, c=1, hs=True, kernel_size=3, nr_kernels = 20,
                 nr_conv_layers=2, seq_len=6,
                 dropout=True, concrete=False, p_fix=0.2, Bayes=True):

        '''
        ARGUMENTS:
        emb_dims: list of tuples (a, b) for each categorical variable,
                  with a: number of levels, and b: embedding dimension
        hidden_size: number of nodes in dense layers
        weight_regularizer: parameter for weight regularization in reformulated ELBO
        dropout_regularizer: parameter for dropout regularization in reformulated ELBO
        conv_dropout_traditional: if "True" then traditional dropout between convolutional layers
        dropout_dense_only: if "True" then only dropout in dense layers, not in convolutional layers
        input_size: number of range features
        c: number of outputs (one for remaining time prediction)
        hs: "True" if heteroscedastic, "False" if homoscedastic
        kernel_size: size of kernels in convolutional layers
        nr_kernels: number of kernels in convolutional layers
        nr_conv_layers: number of convolutional layers
        seq_len: sequence length
        dropout: in case of deterministic model, apply dropout if "True", otherwise no dropout
        concrete: dropout parameter is fixed when "False". If "True", then concrete dropout
        p_fix: dropout parameter in case "concrete"="False"
        Bayes: BNN if "True", deterministic model if "False" (only sampled once for inference)
        '''
        self.value_embedding = value_embedding
        self.heteroscedastic = hs
        self.nr_conv_layers = nr_conv_layers

        super().__init__()

        self.embedding = nn.Embedding(num_features_df + 1, value_embedding, padding_idx=0).to(device)


        '''self.no_of_embs = 0
        if emb_dims:
            self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                             for x, y in emb_dims])
            self.no_of_embs = sum([y for x, y in emb_dims])
        '''
        self.conv = "1D"
        if conv_dropout_traditional:
            self.conv = "lin"

        dropout_conv = dropout
        if dropout_dense_only:
            dropout_conv = False 

        len_conv1 = self.value_embedding + 1
        self.conv1 = nn.Conv1d(len_conv1, nr_kernels, kernel_size=kernel_size).to(device)
        self.maxpool1 = nn.MaxPool1d(3,1)
        if self.nr_conv_layers == 3:
            self.conv2 = nn.Conv1d(nr_kernels, nr_kernels * 2, kernel_size=kernel_size).to(device)
            self.maxpool2 = nn.MaxPool1d(3,1).to(device)
            self.conv3a = nn.Conv1d(nr_kernels*2, nr_kernels, kernel_size=kernel_size).to(device)
        elif self.nr_conv_layers == 2:
            self.conv3b = nn.Conv1d(nr_kernels, nr_kernels, kernel_size=kernel_size).to(device)
        else:
            pass
        self.maxpool3 = nn.MaxPool1d(3, 1)
        self.len_lin3 = nr_kernels * (seq_len - self.nr_conv_layers * (kernel_size - 1) - self.nr_conv_layers * (3 - 1))
        print(self.len_lin3)
        self.linear4 = nn.Linear(self.len_lin3, hidden_size)
        self.linear5 = nn.Linear(hidden_size, int(hidden_size / 10))

        self.linear6_mu = nn.Linear(int(hidden_size / 10), c)
        if self.heteroscedastic:
            self.linear6_logvar = nn.Linear(int(hidden_size / 10), 1)

        # concrete dropout for convolutional layers
        self.conc_drop1 = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        self.conc_drop2 = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        self.conc_drop3a = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        self.conc_drop3b = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        # concrete dropout for dense layers
        self.conc_drop4 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)
        self.conc_drop5 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)
        self.conc_drop6_mu = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                            weight_regularizer=weight_regularizer,
                                            dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)
        if self.heteroscedastic:
            self.conc_drop6_logvar = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                                weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)

        self.relu = nn.ReLU()

        
    def forward(self, x_cat, x_range, stop_dropout=False):
        '''
        ARGUMENTS:
        x_cat: categorical variables. Torch tensor (batch size x sequence length x number of variables)
        x_range: range variables. Torch tensor (batch size x sequence length x number of variables)
        stop_dropout: if "True" prevents dropout during inference for deterministic models

        OUTPUTS:
        mean: outputs (point estimates). Torch tensor (batch size x number of outputs)
        log_var: log of uncertainty estimates. Torch tensor (batch size x number of outputs)
        regularization.sum(): sum of KL regularizers over all model layers
        '''
        x_cat = x_cat.to(device)
        x_range = x_range.to(device)
        regularization = torch.empty(7, device=x_range.device)    #x.device = cuda:0 here

        embeddings = self.embedding(x_cat).to(device)
        # Shape: (batch size, Seq len, embedding dim)
        x_range = x_range.unsqueeze(-1).to(device)
        x = torch.cat((embeddings, x_range), dim=2).to(device)


        '''
        if self.no_of_embs != 0:
            x = [emb_layer(x_cat[:, :, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, -1)
            x = torch.cat([x, x_range], -1)
        else:
            x = x_range '''

        x = torch.transpose(x, 1, 2)  # reshape from (N, seq_len, nr_features) to (N, nr_features, seq_len)

        x, regularization[0] = self.conc_drop1(x, nn.Sequential(self.conv1, self.relu, self.maxpool1), stop_dropout)
        if self.nr_conv_layers == 2:
            x, regularization[2] = self.conc_drop3b(x, nn.Sequential(self.conv3b, self.relu, self.maxpool3), stop_dropout)
        elif self.nr_conv_layers == 3:
            x, regularization[1] = self.conc_drop2(x, nn.Sequential(self.conv2, self.relu, self.maxpool2), stop_dropout)
            x, regularization[2] = self.conc_drop3a(x, nn.Sequential(self.conv3a, self.relu, self.maxpool3), stop_dropout)
        else:
            pass
        x3 = x.view(-1, self.len_lin3)
        x4, regularization[3] = self.conc_drop4(x3, nn.Sequential(self.linear4, self.relu).to(device), stop_dropout)
        x5, regularization[4] = self.conc_drop5(x4, nn.Sequential(self.linear5, self.relu).to(device), stop_dropout)
        mean, regularization[5] = self.conc_drop6_mu(x5, self.linear6_mu.to(device), stop_dropout)
        if self.heteroscedastic:
            log_var, regularization[6] = self.conc_drop6_logvar(x5, self.linear6_logvar.to(device), stop_dropout)
        else:
            regularization[6] = 0
            log_var = torch.empty(mean.size())

        #log_var = torch.clamp(log_var, min=-10, max=10)
        #mean = self.relu(mean)

        #log_var = self.relu(log_var)

        log_var = torch.clamp(log_var, min = -10, max = 10)

        return mean, log_var, regularization.sum()
    

'''
    CNN no Uncertainty
'''

class StochasticCNN_1DNoUnc(nn.Module, MCDualMixin):
    def __init__(self, num_features_df, value_embedding, emb_dims=None, hidden_size=1024,
                 weight_regularizer=1e-5, dropout_regularizer=1e-5,
                 conv_dropout_traditional=False, dropout_dense_only=False,
                 input_size=1, c=1, hs=True, kernel_size=3, nr_kernels = 20,
                 nr_conv_layers=2, seq_len=6,
                 dropout=True, concrete=False, p_fix=0.2, Bayes=True):

        '''
        ARGUMENTS:
        emb_dims: list of tuples (a, b) for each categorical variable,
                  with a: number of levels, and b: embedding dimension
        hidden_size: number of nodes in dense layers
        weight_regularizer: parameter for weight regularization in reformulated ELBO
        dropout_regularizer: parameter for dropout regularization in reformulated ELBO
        conv_dropout_traditional: if "True" then traditional dropout between convolutional layers
        dropout_dense_only: if "True" then only dropout in dense layers, not in convolutional layers
        input_size: number of range features
        c: number of outputs (one for remaining time prediction)
        hs: "True" if heteroscedastic, "False" if homoscedastic
        kernel_size: size of kernels in convolutional layers
        nr_kernels: number of kernels in convolutional layers
        nr_conv_layers: number of convolutional layers
        seq_len: sequence length
        dropout: in case of deterministic model, apply dropout if "True", otherwise no dropout
        concrete: dropout parameter is fixed when "False". If "True", then concrete dropout
        p_fix: dropout parameter in case "concrete"="False"
        Bayes: BNN if "True", deterministic model if "False" (only sampled once for inference)
        '''
        self.value_embedding = value_embedding
        self.heteroscedastic = hs
        self.nr_conv_layers = nr_conv_layers

        super().__init__()

        self.embedding = nn.Embedding(num_features_df + 1, value_embedding, padding_idx=0).to(device)


        '''self.no_of_embs = 0
        if emb_dims:
            self.emb_layers = nn.ModuleList([nn.Embedding(x, y)
                                             for x, y in emb_dims])
            self.no_of_embs = sum([y for x, y in emb_dims])
        '''
        self.conv = "1D"
        if conv_dropout_traditional:
            self.conv = "lin"

        dropout_conv = dropout
        if dropout_dense_only:
            dropout_conv = False 

        len_conv1 = self.value_embedding + 1
        self.conv1 = nn.Conv1d(len_conv1, nr_kernels, kernel_size=kernel_size).to(device)
        self.maxpool1 = nn.MaxPool1d(3,1)
        if self.nr_conv_layers == 3:
            self.conv2 = nn.Conv1d(nr_kernels, nr_kernels * 2, kernel_size=kernel_size).to(device)
            self.maxpool2 = nn.MaxPool1d(3,1).to(device)
            self.conv3a = nn.Conv1d(nr_kernels*2, nr_kernels, kernel_size=kernel_size).to(device)
        elif self.nr_conv_layers == 2:
            self.conv3b = nn.Conv1d(nr_kernels, nr_kernels, kernel_size=kernel_size).to(device)
        else:
            pass
        self.maxpool3 = nn.MaxPool1d(3, 1)
        self.len_lin3 = nr_kernels * (seq_len - self.nr_conv_layers * (kernel_size - 1) - self.nr_conv_layers * (3 - 1))
        print(self.len_lin3)
        self.linear4 = nn.Linear(self.len_lin3, hidden_size)
        self.linear5 = nn.Linear(hidden_size, int(hidden_size / 10))

        self.linear6_mu = nn.Linear(int(hidden_size / 10), c)
        if self.heteroscedastic:
            self.linear6_logvar = nn.Linear(int(hidden_size / 10), 1)

        # concrete dropout for convolutional layers
        self.conc_drop1 = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        self.conc_drop2 = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        self.conc_drop3a = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        self.conc_drop3b = ConcreteDropout(dropout=dropout_conv, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv=self.conv, Bayes=Bayes).to(device)
        # concrete dropout for dense layers
        self.conc_drop4 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)
        self.conc_drop5 = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                          weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)
        self.conc_drop6_mu = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                            weight_regularizer=weight_regularizer,
                                            dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)
        if self.heteroscedastic:
            self.conc_drop6_logvar = ConcreteDropout(dropout=dropout, concrete=concrete, p_fix=p_fix,
                                                weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer, conv="lin", Bayes=Bayes).to(device)

        self.relu = nn.ReLU()

        
    def forward(self, x_cat, x_range, stop_dropout=False):
        '''
        ARGUMENTS:
        x_cat: categorical variables. Torch tensor (batch size x sequence length x number of variables)
        x_range: range variables. Torch tensor (batch size x sequence length x number of variables)
        stop_dropout: if "True" prevents dropout during inference for deterministic models

        OUTPUTS:
        mean: outputs (point estimates). Torch tensor (batch size x number of outputs)
        log_var: log of uncertainty estimates. Torch tensor (batch size x number of outputs)
        regularization.sum(): sum of KL regularizers over all model layers
        '''
        x_cat = x_cat.to(device)
        x_range = x_range.to(device)
        regularization = torch.empty(7, device=x_range.device)    #x.device = cuda:0 here

        embeddings = self.embedding(x_cat).to(device)
        # Shape: (batch size, Seq len, embedding dim)
        x_range = x_range.unsqueeze(-1).to(device)
        x = torch.cat((embeddings, x_range), dim=2).to(device)


        '''
        if self.no_of_embs != 0:
            x = [emb_layer(x_cat[:, :, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
            x = torch.cat(x, -1)
            x = torch.cat([x, x_range], -1)
        else:
            x = x_range '''

        x = torch.transpose(x, 1, 2)  # reshape from (N, seq_len, nr_features) to (N, nr_features, seq_len)

        x, regularization[0] = self.conc_drop1(x, nn.Sequential(self.conv1, self.relu, self.maxpool1), stop_dropout)
        if self.nr_conv_layers == 2:
            x, regularization[2] = self.conc_drop3b(x, nn.Sequential(self.conv3b, self.relu, self.maxpool3), stop_dropout)
        elif self.nr_conv_layers == 3:
            x, regularization[1] = self.conc_drop2(x, nn.Sequential(self.conv2, self.relu, self.maxpool2), stop_dropout)
            x, regularization[2] = self.conc_drop3a(x, nn.Sequential(self.conv3a, self.relu, self.maxpool3), stop_dropout)
        else:
            pass
        x3 = x.view(-1, self.len_lin3)
        x4, regularization[3] = self.conc_drop4(x3, nn.Sequential(self.linear4, self.relu).to(device), stop_dropout)
        x5, regularization[4] = self.conc_drop5(x4, nn.Sequential(self.linear5, self.relu).to(device), stop_dropout)
        mean, regularization[5] = self.conc_drop6_mu(x5, self.linear6_mu.to(device), stop_dropout)
        if self.heteroscedastic:
            log_var, regularization[6] = self.conc_drop6_logvar(x5, self.linear6_logvar.to(device), stop_dropout)
        else:
            regularization[6] = 0
            log_var = torch.empty(mean.size())
        
        #mean = self.relu(mean)

        return mean, regularization.sum()




'''
    LSTM Cell
'''

class StochasticLSTMCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: Optional[float]=None, weight_regularizer=1e-5, dropout_regularizer=1e-5):
        """
        Args:
        - dropout: should be between 0 and 1
        """
        super(StochasticLSTMCell, self).__init__()
        self.wr = weight_regularizer
        self.dr = dropout_regularizer
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        if dropout is None:
            self.p_logit = nn.Parameter(torch.empty(1).normal_())
        elif not 0 < dropout < 1:
            raise Exception("Dropout rate should be between in (0, 1)")
        else:
            self.p_logit = dropout

        self.Wi = nn.Linear(self.input_size, self.hidden_size)
        self.Wf = nn.Linear(self.input_size, self.hidden_size)
        self.Wo = nn.Linear(self.input_size, self.hidden_size)
        self.Wg = nn.Linear(self.input_size, self.hidden_size)
        
        self.Ui = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uf = nn.Linear(self.hidden_size, self.hidden_size)
        self.Uo = nn.Linear(self.hidden_size, self.hidden_size)
        self.Ug = nn.Linear(self.hidden_size, self.hidden_size)
        
        self.init_weights()

    def init_weights(self):
        k = torch.tensor(self.hidden_size, dtype=torch.float32).reciprocal().sqrt()
        
        self.Wi.weight.data.uniform_(-k,k)
        self.Wi.bias.data.uniform_(-k,k)
        
        self.Wf.weight.data.uniform_(-k,k)
        self.Wf.bias.data.uniform_(-k,k)
        
        self.Wo.weight.data.uniform_(-k,k)
        self.Wo.bias.data.uniform_(-k,k)
        
        self.Wg.weight.data.uniform_(-k,k)
        self.Wg.bias.data.uniform_(-k,k)
        
        self.Ui.weight.data.uniform_(-k,k)
        self.Ui.bias.data.uniform_(-k,k)
        
        self.Uf.weight.data.uniform_(-k,k)
        self.Uf.bias.data.uniform_(-k,k)
        
        self.Uo.weight.data.uniform_(-k,k)
        self.Uo.bias.data.uniform_(-k,k)
        
        self.Ug.weight.data.uniform_(-k,k)
        self.Ug.bias.data.uniform_(-k,k)
        
    # Note: value p_logit at infinity can cause numerical instability
    def _sample_mask(self, B):
        """Dropout masks for 4 gates, scale input by 1 / (1 - p)"""
        if isinstance(self.p_logit, float):
            p = self.p_logit
        else:
            p = torch.sigmoid(self.p_logit)
        GATES = 4
        eps = torch.tensor(1e-7)
        t = 1e-1
        
        ux = torch.rand(GATES, B, self.input_size)
        uh = torch.rand(GATES, B, self.hidden_size)

        if self.input_size == 1:
            zx = (1-torch.sigmoid((torch.log(eps) - torch.log(1+eps)
                                   + torch.log(ux+eps) - torch.log(1-ux+eps))
                                 / t))
        else:
            zx = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)
                                   + torch.log(ux+eps) - torch.log(1-ux+eps))
                                 / t)) / (1-p)
        zh = (1-torch.sigmoid((torch.log(p+eps) - torch.log(1-p+eps)
                               + torch.log(uh+eps) - torch.log(1-uh+eps))
                             / t)) / (1-p)
        return zx, zh

    def regularizer(self):        
        if isinstance(self.p_logit, float):
            p = torch.tensor(self.p_logit)
        else:
            p = torch.sigmoid(self.p_logit)
        
        # Weight
        weight_sum = torch.tensor([
            torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("weight")
        ]).sum() / (1.-p)
        
        # Bias
        bias_sum = torch.tensor([
            torch.sum(params**2) for name, params in self.named_parameters() if name.endswith("bias")
        ]).sum()
        
        #if isinstance(self.p_logit, float):
        #    dropout_reg = torch.zeros(1)
        #else:
             # Dropout
            #dropout_reg = self.input_size * (p * torch.log(p) + (1-p)*torch.log(1-p))
        return self.wr * weight_sum, self.wr * bias_sum #, self.dr * dropout_reg
        
    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        input shape (sequence, batch, input dimension)
        output shape (sequence, batch, output dimension)
        return output, (hidden_state, cell_state)
        """

        T, B = input.shape[0:2]

        if hx is None:
            h_t = torch.zeros(B, self.hidden_size, dtype=input.dtype)
            c_t = torch.zeros(B, self.hidden_size, dtype=input.dtype)
        else:
            h_t, c_t = hx

        hn = torch.empty(T, B, self.hidden_size, dtype=input.dtype)

        # Masks
        zx, zh = self._sample_mask(B)
        zh = zh.to(device)
        zx = zx.to(device)
        h_t = h_t.to(device)
        c_t = c_t.to(device)
        input = input.to(device)

        
        for t in range(T):
            x_i, x_f, x_o, x_g = (input[t] * zx_ for zx_ in zx)
            h_i, h_f, h_o, h_g = (h_t * zh_ for zh_ in zh)

            i = torch.sigmoid(self.Ui(h_i) + self.Wi(x_i))
            f = torch.sigmoid(self.Uf(h_f) + self.Wf(x_f))
            o = torch.sigmoid(self.Uo(h_o) + self.Wo(x_o))
            g = torch.tanh(self.Ug(h_g) + self.Wg(x_g))

            c_t = f * c_t + i * g
            h_t = o * torch.tanh(c_t)
            hn[t] = h_t
        
        return hn, (h_t, c_t)


'''
    LSTM which is using the cells
'''

class StochasticLSTM(nn.Module):
    """LSTM stacked layers with dropout and MCMC"""

    def __init__(self, input_size: int, hidden_size: int, dropout:Optional[float]=None, num_layers: int=1):
        super(StochasticLSTM, self).__init__()
        self.num_layers = num_layers
        print("num layers")
        print(num_layers)
        self.first_layer = StochasticLSTMCell(input_size, hidden_size, dropout)
        self.hidden_layers = nn.ModuleList([StochasticLSTMCell(hidden_size, hidden_size, dropout) for i in range(num_layers-1)])
    
    def regularizer(self):
        total_weight_reg, total_bias_reg = self.first_layer.regularizer()
        for l in self.hidden_layers:
            weight, bias = l.regularizer()
            total_weight_reg += weight
            total_bias_reg += bias
        return total_weight_reg, total_bias_reg

    def forward(self, input: Tensor, hx: Optional[Tuple[Tensor, Tensor]]=None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        B = input.shape[1]
        h_n = torch.empty(self.num_layers, B, self.first_layer.hidden_size)
        c_n = torch.empty(self.num_layers, B, self.first_layer.hidden_size)
        
        outputs, (h, c) = self.first_layer(input, hx)
        h_n[0] = h
        c_n[0] = c

        for i, layer in enumerate(self.hidden_layers):
            outputs, (h, c) = layer(outputs, (h, c))
            h_n[i+1] = h
            c_n[i+1] = c

        return outputs, (h_n, c_n)

'''
    Class to use the whole LSTM
'''
class Net(nn.Module, MCDualMixin):
    def __init__(self, num_features_df, hidden_size, num_layers, value_embedding):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(num_features_df + 1, value_embedding, padding_idx=0)
        input_size = value_embedding + 1
        # def __init__(self, input_size: int, hidden_size: int, dropout:Optional[float]=None, num_layers: int=1):
        self.rnn = StochasticLSTM(input_size, hidden_size, dropout=0.2, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_mean = nn.Linear(int(hidden_size / 2), 1)
        self.fc_log_var = nn.Linear(int(hidden_size / 2), 1)
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()
    
    def regularizer(self):        
        # Weight and bias regularizer
        weight_sum, bias_sum = self.rnn.regularizer()
        sum = weight_sum + bias_sum
        return sum
    
    def get_output_shape(self, x):
        B = x.shape[1]
        return (B,1), (B,1)

    def forward(self, input_value, input_time):
        embeddings = self.embedding(input_value)
        input_time = input_time.unsqueeze(-1) 
        # Input size: Batch, window, input_dim (embedding + 1)
        combined = torch.cat((embeddings, input_time), dim=2)
        combined = combined.transpose(0, 1).to(device)


        result, _ = self.rnn(combined)
        result = result.to(device)
        result = result[-1, :, :] 
        result = self.fc1(result)
        mean = self.fc_mean(result)
        log_var = self.fc_log_var(result)
        regularizer = self.regularizer()

        #mean = self.relu(mean)
        #log_var = self.relu(log_var)

        log_var = torch.clamp(log_var, min = -10, max = 10)



        return mean, log_var, regularizer
    

'''
    LSTM without Uncertainty
'''

class NetNoUnc(nn.Module, MCDualMixin):
    def __init__(self, num_features_df, hidden_size, num_layers, value_embedding):
        super(NetNoUnc, self).__init__()
        self.embedding = nn.Embedding(num_features_df + 1, value_embedding, padding_idx=0)
        input_size = value_embedding + 1
        # def __init__(self, input_size: int, hidden_size: int, dropout:Optional[float]=None, num_layers: int=1):
        self.rnn = StochasticLSTM(input_size, hidden_size, dropout=0.2, num_layers=num_layers)
        self.fc1 = nn.Linear(hidden_size, int(hidden_size / 2))
        self.fc_mean = nn.Linear(int(hidden_size / 2), 1)

        self.relu = nn.ReLU()
    
    def regularizer(self):        
        # Weight and bias regularizer
        weight_sum, bias_sum = self.rnn.regularizer()
        sum = weight_sum + bias_sum
        return sum
    
    def get_output_shape(self, x):
        B = x.shape[1]
        return (B,1), (B,1)

    def forward(self, input_value, input_time):
        embeddings = self.embedding(input_value)
        input_time = input_time.unsqueeze(-1) 
        # Input size: Batch, window, input_dim (embedding + 1)
        combined = torch.cat((embeddings, input_time), dim=2)
        combined = combined.transpose(0, 1).to(device)


        result, _ = self.rnn(combined)
        result = result.to(device)
        result = result[-1, :, :] 
        result = self.fc1(result)
        mean = self.fc_mean(result)

        #mean = self.relu(mean)
        return mean, torch.tensor(0)
    

    

'''
    Simple Ensemble
'''

class SimpleEnsemble(nn.Module):
    def __init__(self, numberModels, dropout_prob = 0.00):
        super(SimpleEnsemble, self).__init__()
        self.fc1 = nn.Linear(2 * numberModels, 8)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc6 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, mean, var):
        # mean = self.relu(mean)
        #var = self.relu(var)
        x = torch.cat((mean, var), dim=1).to(device)
        ## 

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc6(x)
        return x
    
    

'''
    Complex Ensemble
'''

class Ensemble(nn.Module):
    def __init__(self, numberModels, dropout_prob = 0.00):
        super(Ensemble, self).__init__()
        self.fc1 = nn.Linear(2 * numberModels, 20)
        self.fc_middle = nn.Linear(20, 20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc5 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, mean, var):
        # mean = self.relu(mean)
        #var = self.relu(var)
        x = torch.cat((mean, var), dim=1).to(device)
        x = self.fc1(x)
        x = self.fc_middle(x)
        x = self.fc2(x)
        x = (self.fc5(x))
        return x

'''
    Simple Regression
'''

class SimpleRegressionUncertainty(nn.Module):
    def __init__(self, numberModels):
        super(SimpleRegressionUncertainty, self).__init__()
        self.fc1 = nn.Linear(2 * numberModels, 1)
        self.relu = nn.ReLU()

    def forward(self, mean, var):
        # mean = self.relu(mean)
        #var = self.relu(var)
        x = torch.cat((mean, var), dim=1).to(device)

        x = (((self.fc1(x))))
        return x
    
    
'''
    Simple Regression without uncertainty
'''  
class SimpleEnsembleNoUncertainty(nn.Module):
    def __init__(self, numberModels, dropout_prob = 0.00):
        super(SimpleEnsembleNoUncertainty, self).__init__()
        self.fc1 = nn.Linear(numberModels, 8)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc5 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, mean):
        # mean = self.relu(mean)
        x = mean.to(device)
        x = self.fc1(x)
        x = self.fc2(x)
        x = ((self.fc5(x)))
        return x
    

'''
    Ensemble without uncertainty
'''  
class EnsembleNoUncertainty(nn.Module):
    def __init__(self, numberModels, dropout_prob = 0.00):
        super(EnsembleNoUncertainty, self).__init__()
        self.fc1 = nn.Linear(numberModels, 20)
        self.fc_middle = nn.Linear(20, 20)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(20, 10)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc5 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, mean):
        # mean = self.relu(mean)
        x = mean.to(device)
        #print("input")
        #print(x)
        x = self.fc1(x)
        x = self.fc_middle(x)
        x = self.fc2(x)
        x = ((self.fc5(x)))
        return x

'''
    Simple Ensemble without uncertainty
''' 

class SimpleRegressionNoUncertainty(nn.Module):
    def __init__(self, numberModels):
        super(SimpleRegressionNoUncertainty, self).__init__()
        self.fc1 = nn.Linear(numberModels, 1)
        self.relu = nn.ReLU()

    def forward(self, mean):
        # mean = self.relu(mean)
        x = mean.to(device)
        x = ((self.fc1(x)))
        return x
    




class BranchForEveryInput(nn.Module):
    def __init__(self, hidden_size=32):
        super(BranchForEveryInput, self).__init__()

        # Feature branch
        self.model_1 = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

        # Uncertainty branch
        self.model_2 = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

        self.model_3 = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.ReLU()
        )

        # Combined branch
        self.combined_branch = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.relu = torch.nn.ReLU()
    

    ## Extract this right out!
    def forward(self, x, u):
        # x = self.relu(x)
        # u = self.relu(u)

        first_uncertainty = u[:, 0].to(device)

        first_prediction = x[:, 0].to(device)

        second_uncertainty = u[:, 1].to(device)

        second_prediction = x[:, 1].to(device)

        first = torch.stack((first_prediction, first_uncertainty), dim=1).to(device)

        second = torch.stack((second_prediction, second_uncertainty), dim=1).to(device)


        output_first = self.model_1(first).to(device)
        output_second = self.model_1(second).to(device)

        combined = torch.cat((output_first, output_second), dim=1).to(device)

        output = (self.combined_branch(combined))
        
        return output
    
