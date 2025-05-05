import torch
import torch.nn.functional as F

def SATA(x: torch.Tensor, M_att: torch.Tensor, alpha:float=1.0):
   """
   x: token embedding tensor  , [ batch_size, tokens (N), channels]
   M_att : attention map , [batch_size, N, N]
   alpha: bound controling factor
   """
   batch_size, num_token, channels = x.shape
   if num_token<2 or alpha<=0:
      return x, None
   ##### Spatial Autocorrelation    
   # remove class token
   cls_token=x[...,0,:].view(batch_size,1,-1)
   x = x[...,1:,:]
   M_att = M_att[...,1:,:]
   M_att = M_att[...,:,1:]
   (batch_size, N , dim) = tuple(x.size())
   ###### Local Moran's index
   a = F.adaptive_avg_pool1d(x, 1) # compute global context attribute (Eq.6)
   a = torch.reshape(a, (batch_size, N, -1))
   z = (a-a.mean(1,keepdim=True))/a.std(1, keepdim=True) # Eq. 4
        
   z_t = z.transpose(-1, -2)
   zxz_t = z@z_t# B x N x N

   I_l = torch.reshape(torch.diagonal(zxz_t@M_att, dim1=1, dim2=2), (batch_size, N, 1))  # B x N x 1
   s = (I_l-I_l.mean(1,keepdim=True))/I_l.std(1,keepdim=True) # spatial autocorrelation score, Eq. 5 # B x N x 1
    
   ### Tokens Splitting and Grouping
   output_tokens, residual_tokens = split_group_by_scores(x,s_score=s,alpha=alpha)

   ## add class token
   output_tokens = torch.cat([cls_token, output_tokens],dim=1)

   return  output_tokens, residual_tokens  


def split_group_by_scores(x, s_score, alpha=1.):
    
   ####### Tokens Splitting 
   # computing lower and upper bounds
    batch_size, num_token, channels = x.shape

    med_score,_ = torch.median(s_score, dim=1,keepdim=True)
    mean_score = torch.mean(s_score,dim=1,keepdim=True)
    
    lower_bound = (mean_score - torch.abs(med_score))*alpha
    upper_bound = (mean_score + torch.abs(med_score))*alpha

    set_B_mask = (s_score <= upper_bound) & (s_score >= lower_bound) # Eq.8
   
   #### Unification with regards to the batch_size
   # Step 1: Calculate the unified size for set B with regards to the batch size
    num_B_elements = torch.sum(set_B_mask).item()  
    unified_size = int(num_B_elements / batch_size)  
    unified_num_B = unified_size * batch_size  
    num_B_to_swap = num_B_elements - unified_num_B  # Determine how many elements need to be swapped out
    
    if num_B_to_swap > 0:   
       # Step 2: Sort the scores of elements in set_B_mask along with indices
       sorted_scores, sorted_indices = torch.sort(s_score[set_B_mask].view(-1))  # Sort the scores and get sorted indices
    
       # Step 3: Extract num_elements_to_swap highest-scored elements from set_B_mask and set to False
       selected_indices = sorted_indices[:num_B_to_swap] 
       true_indices = torch.where(set_B_mask)  # Get the indices of true elements in set_B_mask
       set_B_mask[true_indices[0][selected_indices], true_indices[1][selected_indices], 
                  true_indices[2][selected_indices]] = False  # Set the selected elements to False
    ############################################
    set_A_mask = ~set_B_mask # Eq.7

    set_B = x.masked_select(set_B_mask.expand_as(x)).view(batch_size,-1,channels)        
    set_A = x.masked_select(set_A_mask.expand_as(x)).view(batch_size,-1,channels)
    
    A_num= set_A.size(1)
    
   ####### Tokens Grouping
    if A_num>2:
        merged_tokens, residual_tokens = token_merging(set_A,channels)
        output_tokens = torch.cat([set_B, merged_tokens], dim=1)
    else:
        residual_tokens = None
        output_tokens = set_B
        
    return output_tokens, residual_tokens

def token_merging(x,token_size):

    batch_size = x.size(0)
    src, dst = x[..., ::2, :], x[..., 1::2, :]
    scores = src @ dst.transpose(-1, -2)

    node_max, node_idx = scores.max(dim=-1)
    src_idx = node_max.argsort(dim=-1, descending=True)[..., None]    
    dst_idx = node_idx[..., None].gather(dim=-2, index=src_idx)


    res_tokens = src.gather(dim=-2, index=src_idx.expand(batch_size, -1, token_size))
    mrg_tokens = dst.scatter_reduce(-2, dst_idx.expand(batch_size, -1, token_size), src, reduce="mean")
    return mrg_tokens,res_tokens