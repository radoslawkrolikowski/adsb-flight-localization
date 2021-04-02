import torch


class GhostBatchNorm(torch.nn.Module):
    """Implementation of the Ghost Batch Normalization.
    
    Parameters
    ----------
    num_features: int
        Size of the second dimension of the input tensor
    n_chunks: int, optional (default=128)
        Number of small, 'ghost' batches
    chunk_size: int, optional (default=0)
        Size of the 'ghost' batch. Use interchangeably with n_chunks
    track_running_stats: boolean, optional (default=False)
        Whether to track the running mean and variance
    momentum: float, optional (default=0.02)
        The value used for the running_mean and running_var computation
        
    Returns
    -------
    torch.Tensor
        Normalized batch
        
    """
    
    def __init__(self, num_features, n_chunks=128, chunk_size=0, track_running_stats=False, momentum=0.02):
        
        super(GhostBatchNorm, self).__init__()
        
        self.num_features = num_features
        self.n_chunks = n_chunks
        self.chunk_size = chunk_size 
        self.trs = track_running_stats
        self.momentum = momentum
        
        self.batch_norm = torch.nn.BatchNorm1d(self.num_features, 
                                         track_running_stats=self.trs,
                                         momentum=self.momentum)

        
    def forward(self, batch):

        if self.chunk_size:
            self.n_chunks = batch.size(0) // self.chunk_size
            
        batch_chunks = torch.chunk(batch, self.n_chunks, dim=0)
                
        norm_batch_chunks = [self.batch_norm(chunk) for chunk in batch_chunks]
        
        return torch.cat(norm_batch_chunks, dim=0)



class Sparsemax(torch.nn.Module):
    """Implementation of the Sparsemax class.
    
    Parameters
    ---------
    dim: int, optional (default=-1)
        The dimension we want to cast the operation over
            
    """
    
    def __init__(self, dim=-1):
        
        super(Sparsemax, self).__init__()
        
        self.dim = dim
        
    def forward(self, input_tensor):
        return SparsemaxFunction.apply(input_tensor, self.dim)
        
        
        
class SparsemaxFunction(torch.autograd.Function):
    """Implementation of the Sparsemax function.
    
    """
    
    @staticmethod
    def forward(ctx, input_tensor, dim=-1):
        """Forward function.
        
        Parameters
        ----------
        ctx : context object 
            The context object that can be used to stash information for backward computation
        input_tensor : torch.Tensor
            Input tensor
        dim: int, optional (default=-1)
            The dimension we want to cast the operation over

        """
        
        ctx.dim = dim
        
        # Stabilize the function by subtracting the maximum value
        input_tensor -= input_tensor.max(dim, keepdim=True)[0]
        
        # Sort input tensor in descending order
        input_sorted = torch.sort(input_tensor, dim=dim, descending=True)[0]
        
        # Determine sparsity of projection
        range_values = torch.arange(input_tensor.size(dim)).view(1, -1)
        bound = 1 + range_values * input_sorted

        input_cumsum = input_sorted.cumsum(dim) - input_sorted

        k = torch.where(bound > input_cumsum)[0].max()

        # Calculate taus
        taus = (input_cumsum[k] - 1) / k
        output = torch.max(input_tensor - taus, torch.zeros_like(input_tensor))
        
        # Save the output for backpropagation
        ctx.save_for_backward(output)

        return output 
        
    @staticmethod
    def backward(ctx, grad_output):
        """Backward function.
        
        """

        output, *_ = ctx.saved_tensors
        dim = ctx.dim

        # Compute gradient
        nonzeros = torch.ne(output, 0)
        suma = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        grad_input = nonzeros * (grad_output - suma.unsqueeze(1).expand_as(grad_output))

        return grad_input, None


class AttentiveTransformer(torch.nn.Module):
    """Implementation of the AttentiveTransformer (https://arxiv.org/pdf/1908.07442.pdf).
    
    Parameters
    ----------
    input_dim: int
        Input size
    output_dim: int
        Output size
    n_chunks: int, optional (default=128)
        Number of small, 'ghost' batches
    chunk_size: int, optional (default=0)
        Size of the 'ghost' batch. Use interchangeably with n_chunks
    track_running_stats: boolean, optional (default=False)
        Whether to track the running mean and variance
    momentum: float, optional (default=0.02)
        The value used for the running_mean and running_var computation
    ghost_batch_norm: boolean, optional (default=False)
        Whether to use the Ghost Batch Normalization
        
    Returns
    -------
    torch.Tensor
         Learnable mask for soft selection of the features
    
    """
    
    def __init__(self, input_dim, output_dim, n_chunks=128, chunk_size=0, track_running_stats=False, momentum=0.02, ghost_batch_norm=False):
        
        super(AttentiveTransformer, self).__init__()
        
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
        if ghost_batch_norm:
            self.bn = GhostBatchNorm(output_dim, 
                                      n_chunks=n_chunks,
                                      chunk_size=chunk_size,
                                      track_running_stats=track_running_stats,
                                      momentum=momentum)
        else:
            self.bn = torch.nn.BatchNorm1d(output_dim,
                                            track_running_stats=track_running_stats,
                                            momentum=momentum)           
            
        self.sparsemax = Sparsemax(dim=-1)
        
    def forward(self, features, priors): 
        
        x = self.linear(features)
        x = self.bn(x)        
        x = torch.mul(x, priors)
        mask = self.sparsemax(x)

        return mask
    
    
class GLU(torch.nn.Module):
    """Implementation of the Gated Linear Unit (GLU).
    
    Parameters
    ----------
    input_dim: int
        Input size
    output_dim: int
        Output size
    fc: torch.nn.Module.Linear, optional (default=None)
        Shared fully connected layer
    n_chunks: int, optional (default=128)
        Number of small, 'ghost' batches
    chunk_size: int, optional (default=0)
        Size of the 'ghost' batch. Use interchangeably with n_chunks
    track_running_stats: boolean, optional (default=False)
        Whether to track the running mean and variance
    momentum: float, optional (default=0.02)
        The value used for the running_mean and running_var computation
    ghost_batch_norm: boolean, optional (default=False)
        Whether to use the Ghost Batch Normalization
        
    Returns
    -------
    torch.Tensor
         GLU output tensor
    
    """
    
    def __init__(self, input_dim, output_dim, fc=None, n_chunks=128, chunk_size=0, track_running_stats=False, momentum=0.02,
                 ghost_batch_norm=False):
        
        super(GLU, self).__init__()
    
        self.output_dim = output_dim
        
        if fc:
            self.linear = fc
        else:
            self.linear = torch.nn.Linear(input_dim, 2 * output_dim)
    
        if ghost_batch_norm:
            self.bn = GhostBatchNorm(2 * output_dim,
                                      n_chunks=n_chunks,
                                      chunk_size=chunk_size,
                                      track_running_stats=track_running_stats,
                                      momentum=momentum)
        else:
            self.bn = torch.nn.BatchNorm1d(2 * output_dim,
                                            track_running_stats=track_running_stats,
                                            momentum=momentum)

    def forward(self, x):
        
        x = self.linear(x)
        x = self.bn(x)
        out = torch.mul(x[:, :self.output_dim], torch.sigmoid(x[:, self.output_dim:]))
        
        return out

 
class FeatureTransformer(torch.nn.Module):
    """Implementation of the FeatureTransformer.
    
    Parameters
    ---------
    input_dim: int
        Input size
    output_dim: int
        Output size
    shared_GLU_fc: torch.nn.ModuleList, optional (default=None)
        Shared GLU fully connected layers
    n_independent: int, optional (default=2)
        Number of independent GLU blocks
    n_chunks: int, optional (default=128)
        Number of small, 'ghost' batches
    chunk_size: int, optional (default=0)
        Size of the 'ghost' batch. Use interchangeably with n_chunks
    track_running_stats: boolean, optional (default=False)
        Whether to track the running mean and variance
    momentum: float, optional (default=0.02)
        The value used for the running_mean and running_var computation
    ghost_batch_norm: boolean, optional (default=False)
        Whether to use the Ghost Batch Normalization
        
    Returns
    -------
    torch.Tensor
         Output tensor of the FeatureTransformer
        
    """
    
    def __init__(self, input_dim, output_dim, shared_GLU_fc, n_independent=2, n_chunks=128, chunk_size=0, track_running_stats=False,
                 momentum=0.02, ghost_batch_norm=False):
        
        super(FeatureTransformer, self).__init__()

        self.shared_GLU_fc = shared_GLU_fc
        self.n_independent = n_independent
        self.glu_layers = torch.nn.ModuleList()
        
        is_first_GLU = False
        
        if shared_GLU_fc:
            self.glu_layers.append(GLU(input_dim,
                                   output_dim,
                                   fc=shared_GLU_fc[0],
                                   n_chunks=n_chunks,
                                   chunk_size=chunk_size,
                                   track_running_stats=track_running_stats,
                                   momentum=momentum,
                                   ghost_batch_norm=ghost_batch_norm))
            
            is_first_GLU = True
            
            for fc in shared_GLU_fc[1:]:
                self.glu_layers.append(GLU(output_dim,
                                       output_dim,
                                       fc=fc,
                                       n_chunks=n_chunks,
                                       chunk_size=chunk_size,
                                       track_running_stats=track_running_stats,
                                       momentum=momentum,
                                       ghost_batch_norm=ghost_batch_norm))
            
        if n_independent > 0:
            
            if not is_first_GLU:
                self.glu_layers.append(GLU(input_dim,
                                       output_dim,
                                       n_chunks=n_chunks,
                                       chunk_size=chunk_size,
                                       track_running_stats=track_running_stats,
                                       momentum=momentum,
                                       ghost_batch_norm=ghost_batch_norm))
                
                is_first_GLU = True
                

            for i in range(is_first_GLU, n_independent):
                self.glu_layers.append(GLU(output_dim,
                                       output_dim,
                                       n_chunks=n_chunks,
                                       chunk_size=chunk_size,
                                       track_running_stats=track_running_stats,
                                       momentum=momentum,
                                       ghost_batch_norm=ghost_batch_norm))
        

    def forward(self, x):
        
        scale = torch.sqrt(torch.FloatTensor([0.5]).to(x.device))
                    
        # The first GLU has no scale multiplication
        x = self.glu_layers[0](x)
        
        for glu in self.glu_layers[1:]:

            x = torch.add(x, glu(x))
            x = x * scale
                
        return x       
        
    
class TabNet(torch.nn.Module):
    """Implementation of the TabNet network (https://arxiv.org/pdf/1908.07442.pdf).
    
    Parameters
    ----------
    input_dim: int
        Input size
    output_dim: int
        Output size
    n_d: int
        Size of the decision layer
    n_a: int
        Size of the attention layer
    gamma: float, optional (default=1.2)
        The scaling factor for attention
    epsilon: float, optional (default=1e-10)
        A very small value added to the mask while increasing its sparsity
    n_shared: int, optional (default=2)
        Number of shared GLU fully connected layers
    n_independent: int, optional (default=2)
        Number of independent GLU blocks
    n_steps: int
        Number of decision steps
    n_chunks: int, optional (default=128)
        Number of small, 'ghost' batches
    chunk_size: int, optional (default=0)
        Size of the 'ghost' batch. Use interchangeably with n_chunks
    track_running_stats: boolean, optional (default=False)
        Whether to track the running mean and variance
    momentum: float, optional (default=0.02)
        The value used for the running_mean and running_var computation
    ghost_batch_norm: boolean, optional (default=False)
        Whether to use the Ghost Batch Normalization
        
    Returns
    -------
    out: torch.Tensor
        Tensor of predictions
    sparse_loss: torch.Tensor      
        Tensor of sparse losses
    masks: list
        List of masks
    mask_explain: torch.Tensor
         List of explanatory masks
    
    """
    
    def __init__(self, input_dim, output_dim, n_d, n_a, gamma=1.2, epsilon=1e-10, n_shared=2, n_independent=2, n_steps=2, n_chunks=128,
                 chunk_size=0, track_running_stats=False, momentum=0.02, ghost_batch_norm=False):
        
        super(TabNet, self).__init__()
                
        self.n_shared = n_shared
        self.n_independent = n_independent
        self.n_steps = n_steps
        self.n_d = n_d
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.bn = torch.nn.BatchNorm1d(input_dim)

        self.linear = torch.nn.Linear(n_d, output_dim)
        
        
        if self.n_shared > 0:
            shared_GLU_fc = torch.nn.ModuleList()

            # First GLU block reduces the dimension of the input features to a dimension equal n_d + n_a
            shared_GLU_fc.append(torch.nn.Linear(input_dim, 2 * (n_d + n_a)))            
            
            for i in range(1, self.n_shared):
                    shared_GLU_fc.append(torch.nn.Linear(n_d + n_a, 2 * (n_d + n_a)))
                    
        else:
            shared_GLU_fc = None
            
        self.first_transformer = FeatureTransformer(input_dim,
                                                    n_d + n_a,
                                                    shared_GLU_fc,
                                                    n_independent=n_independent,
                                                    n_chunks=n_chunks,
                                                    chunk_size=chunk_size,
                                                    track_running_stats=track_running_stats,
                                                    momentum=momentum,
                                                    ghost_batch_norm=ghost_batch_norm)
        
        self.feature_transformer = torch.nn.ModuleList()
        self.attentive_transformer = torch.nn.ModuleList()
        
        for step in range(self.n_steps):
            
            self.feature_transformer.append(FeatureTransformer(input_dim,
                                                               n_d + n_a,
                                                               shared_GLU_fc,
                                                               n_independent=n_independent,
                                                               n_chunks=n_chunks,
                                                               chunk_size=chunk_size,
                                                               track_running_stats=track_running_stats,
                                                               momentum=momentum,
                                                               ghost_batch_norm=ghost_batch_norm))
            
            self.attentive_transformer.append(AttentiveTransformer(n_a,
                                                                   input_dim,
                                                                   n_chunks=n_chunks,
                                                                   chunk_size=chunk_size,
                                                                   track_running_stats=track_running_stats,
                                                                   momentum=momentum,
                                                                   ghost_batch_norm=ghost_batch_norm))
            
    def forward(self, x):
        """Forward propagate through the neural network model.

        """

        x = self.bn(x)  
        priors = torch.ones(x.shape).to(x.device)
        att = self.first_transformer(x)[:, self.n_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        mask_explain = torch.zeros(x.shape).to(x.device)
        out = torch.zeros(x.size(0), self.n_d).to(x.device)
        
        masks = []

        for step in range(self.n_steps):
            
            mask = self.attentive_transformer[step](att, priors)
            x_t = self.feature_transformer[step](torch.mul(x, mask))
            sparse_loss += torch.mean(torch.sum(torch.mul(mask, torch.log(mask + self.epsilon)), dim=1))
            
            # Update the priors
            priors = torch.mul(self.gamma - mask, priors)
            
            # Update the attention
            att = x_t[:, self.n_d:]
            
            d = torch.nn.ReLU()(x_t[:, :self.n_d])
            out += d
            
            # Save the mask
            masks.append(mask)
            
            # Calculate feature importance            
            step_importance = torch.sum(d, dim=1)
            mask_explain += torch.mul(mask, step_importance.unsqueeze(dim=1))
            
        out = self.linear(out.double())
        
        return out, sparse_loss, masks, mask_explain


    def add_loss_fn(self, loss_fn):
        """Add loss function to the model.
        
        """
        self.loss_fn = loss_fn
        

    def add_optimizer(self, optimizer):
        """Add optimizer to the model.
        
        """
        self.optimizer = optimizer
        
        
    def add_device(self, device=torch.device('cpu')):
        """Specify the device.
        
        """
        self.device = device
    