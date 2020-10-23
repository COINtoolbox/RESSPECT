# Copyright 2020 resspect software
# Author: Noble Kennamer
#
# created on 26 June 2020
#
# Licensed GNU General Public License v3.0;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.gnu.org/licenses/gpl-3.0.en.html
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

__all__ = ['compute_conditional_entropies_B', 'joint_probs_M_K_impl',
           'joint_probs_M_K', 'entropy_from_M_K', 'entropy_from_probs_b_M_C',
           'exact_batch', 'split_arrays', 'entropy_joint_probs_B_M_C',
           'importance_weighted_entropy_p_b_M_C', 'sample_M_K',
           'from_M_K', 'take_expand', 'fast_multi_choices', 
           'batch_sample']

import numpy as np


## Conditional Entropy
def compute_conditional_entropies_B(probs_B_K_C):
    """Do something....
    
    Parameters
    ----------
    probs_B_K_C: XXXX 
        XXXXXXXXX
        
    Returns
    -------
        XXXXXXXXX
    
    """
    B, K, C = probs_B_K_C.shape
    
    return np.sum(-1*probs_B_K_C * np.log(probs_B_K_C), axis=(1, 2)) / K


## Exact Methods
def joint_probs_M_K_impl(probs_N_K_C, prev_joint_probs_M_K):
    """ Do something XXXX.
    
    Parameters
    ----------
    probs_N_K_C: XXXXX
        XXXXXXX
    prev_joint_probs_M_K: XXXXX
        XXXXXXX
    
    Returns
    -------
        XXXXXXXXXX
    """
    
    assert prev_joint_probs_M_K.shape[1] == probs_N_K_C.shape[1]

    N, K, C = probs_N_K_C.shape
    prev_joint_probs_K_M_1 = prev_joint_probs_M_K.transpose()[:, :, None]

    # Exponential memory consumption. Increases by a 
    # factor of C each iteration.
    for i in range(N):
        i_K_1_C = probs_N_K_C[i][:, None, :]
        joint_probs_K_M_C = prev_joint_probs_K_M_1 * i_K_1_C
        prev_joint_probs_K_M_1 = joint_probs_K_M_C.reshape((K, -1, 1))

    prev_joint_probs_M_K = prev_joint_probs_K_M_1.squeeze(2).transpose()
    
    return prev_joint_probs_M_K


def joint_probs_M_K(probs_N_K_C, prev_joint_probs_M_K=None):
    """ Do something XXXX
    
    Parameters
    ----------
    probs_N_K_C: XXXXX
        XXXXXXXXXX
    prev_joint_probs_M_K: XXXX (optional)
        XXXXXXX. Default is None.
        
    Returns
    -------
        XXXXXXXXXX
    
    """
    
    if prev_joint_probs_M_K is not None:
        assert prev_joint_probs_M_K.shape[1] == probs_N_K_C.shape[1]

    N, K, C = probs_N_K_C.shape
    
    if prev_joint_probs_M_K is None:
        prev_joint_probs_M_K = np.ones((1, K), dtype=np.float64)
        
    return joint_probs_M_K_impl(probs_N_K_C, prev_joint_probs_M_K)


def entropy_from_M_K(joint_probs_M_K):
    """Do something ... 
    
    Parameters
    ----------
    joint_probs_M_K: XXXX
        XXXXXXX
        
    Returns
    -------
        XXXXXXX
    
    """
    
    probs_M = np.mean(joint_probs_M_K, axis=1, keepdims=False)
    nats_M = -np.log(probs_M) * probs_M
    entropy = np.sum(nats_M)
    
    return entropy

def entropy_from_probs_b_M_C(probs_b_M_C):
    """Do something ....
    
    Parameters
    ----------
    probs_b_M_C: XXXX
        XXXXXX
        
    Returns
    -------
        XXXXX
    """
    
    return np.sum(-probs_b_M_C * np.log(probs_b_M_C), axis=(1, 2))


def exact_batch(probs_B_K_C, prev_joint_probs_M_K=None):
    """Do something ...
    
    Parameters
    ----------
    probs_B_K_C: XXXX
        XXXXXXX
    prev_joint_probs_M_K: XXXX (optional)
        XXXXX. Default is None.
        
    Returns
    -------
        XXXXXXXX
    """
    
    if prev_joint_probs_M_K is not None:
        assert prev_joint_probs_M_K.shape[1] == probs_B_K_C.shape[1]

    B, K, C = probs_B_K_C.shape

    if prev_joint_probs_M_K is None:
        prev_joint_probs_M_K = np.ones((1, K), dtype=np.float64)

    joint_probs_B_M_C = entropy_joint_probs_B_M_C(probs_B_K_C, prev_joint_probs_M_K)

    # Now we can compute the entropy.
    entropy_B = np.zeros((B,), dtype=np.float64)

    chunk_size = 256
    
    for i in range(0, B, chunk_size):
        end_i = i+chunk_size
        entropy_B[i:end_i] = entropy_from_probs_b_M_C(joint_probs_B_M_C[i:end_i])

    return entropy_B

# Not computing an Entropy
def entropy_joint_probs_B_M_C(probs_B_K_C, prev_joint_probs_M_K):
    """Do something ....
    
    Parameters
    ----------
    probs_B_K_C: XXXX
        XXXXXXX
    prev_joint_probs_M_K: XXXX
        XXXXXX
        
    Returns
    -------
        XXXXXX 
    """
    B, K, C = probs_B_K_C.shape
    M = prev_joint_probs_M_K.shape[0]
    joint_probs_B_M_C = np.empty((B, M, C), dtype=np.float64)

    for i in range(B):
        np.matmul(prev_joint_probs_M_K, probs_B_K_C[i], out=joint_probs_B_M_C[i])

    joint_probs_B_M_C /= K
    
    return joint_probs_B_M_C

def split_arrays(arr1, arr2, chunk_size):
    """Do something ...
    
    arr1: XXXX
        XXX
    arr2: XXX
        XXXXX
    chunk_size: XXX
        XXXXX
    
    Returns
    -------
        XXXXXX
    """
    
    assert arr1.shape[0] == arr2.shape[0]
    
    arrays = []
    
    for i in range(0, arr1.shape[0], chunk_size):
        end_i = i+chunk_size
        arrays.append((arr1[i:end_i], arr2[i:end_i]))
        
    return arrays

## Sampling approaches
def sample_M_K(probs_N_K_C, S=1000):
    """Do something ...
    
    Parameters
    ----------
    probs_N_K_C: XXX
        XXXX
    S: int (optional)
        XXXXX. Default is 1000. 
        
    Returns
    -------
        XXXXXX
    """
    K = probs_N_K_C.shape[1]

    choices_N_K_S = fast_multi_choices(probs_N_K_C, S)

    expanded_choices_N_K_K_S = choices_N_K_S[:, None, :, :]
    expanded_probs_N_K_K_C = probs_N_K_C[:, :, None, :]

    probs_N_K_K_S = take_expand(expanded_probs_N_K_K_C, 
                                index=expanded_choices_N_K_K_S, dim=-1)
                                
    # exp sum log seems necessary to avoid 0s?
    probs_K_K_S = np.exp(np.sum(np.log(probs_N_K_K_S), axis=0, 
                                keepdims=False))
    samples_K_M = probs_K_K_S.reshape((K, -1))

    samples_M_K = samples_K_M.transpose()
    
    return samples_M_K

def from_M_K(samples_M_K):
    """Do something ...
    
    Parameters
    ----------
    samples_M_K: XXX
        XXXXX
        
    Returns
    -------
        XXXXXXXX
    """
    probs_M = np.mean(samples_M_K, axis=1, keepdims=False)
    nats_M = -np.log(probs_M)
    entropy = np.mean(nats_M)
    
    return entropy


def take_expand(data, index, dim):
    """Do something ...
    
    Parameters
    ----------
    data: XXX
        XXXXX
    index: XXXX
        XXXXX
    dim: XXXX
        XXXXX
        
    Returns
    -------
        XXXXXXX    
    """
    
    max_shape = [max(dr, ir) for dr, ir in zip(data.shape, index.shape)]
    new_data_shape = list(max_shape)
    new_data_shape[dim] = data.shape[dim]

    new_index_shape = list(max_shape)
    new_index_shape[dim] = index.shape[dim]


    data = data.repeat(new_data_shape[2], axis=2)
    index = index.repeat(new_index_shape[1], axis=1)

    return np.take_along_axis(data, index, axis=dim)

def fast_multi_choices(probs_b_C, M):
    """Do something ...
    
    Parameters
    ----------
    probs_b_C: XXXX
        XXXXXX
    M: XXXX
        XXXXXX
        
    Returns
    -------
        XXXXXX    
    """
    probs_B_C = probs_b_C.reshape((-1, probs_b_C.shape[-1]))
    s = probs_B_C.cumsum(axis=1).repeat(M, axis=0)
    # Probabilities might not sum to 1. perfectly due to numerical errors.
    s[:, -1] = 1.
    r = np.random.rand(probs_B_C.shape[0] * M)[:, np.newaxis]
    choices = (s <= r).sum(axis=1)
    choices_b_M = choices.reshape(probs_b_C.shape[:-1] + (M,))
    return choices_b_M

def batch_sample(probs_B_K_C, samples_M_K):
    """Do something ....
    
    Parameters
    ----------
    probs_B_K_C: XXX
        XXXXXXX
    samples_M_K: XXXX
        XXXXX
        
    Returns
    -------
        XXXXXXX    
    """
    probs_B_K_C = probs_B_K_C
    samples_M_K = samples_M_K

    M, K = samples_M_K.shape
    B, K_, C = probs_B_K_C.shape
    assert K == K_

    p_B_M_C = np.empty((B, M, C), dtype=np.float64)

    for i in range(B):
        np.matmul(samples_M_K, probs_B_K_C[i], out=p_B_M_C[i])

    p_B_M_C /= K

    q_1_M_1 = samples_M_K.mean(axis=1, keepdims=True)[None]

    # Now we can compute the entropy.
    # We store it directly on the CPU to save GPU memory.
    entropy_B = np.zeros((B,), dtype=np.float64)

#     chunk_size = 256
#     for entropy_b, p_b_M_C in split_tensors(entropy_B, p_B_M_C, chunk_size):
#         entropy_b.copy_(importance_weighted_entropy_p_b_M_C(p_b_M_C, q_1_M_1, M), non_blocking=True)

    chunk_size = 256
    for i in range(0, B, chunk_size):
        end_i = i+chunk_size
        entropy_B[i:end_i] = importance_weighted_entropy_p_b_M_C(p_B_M_C[i:end_i], q_1_M_1, M)

    return entropy_B

def importance_weighted_entropy_p_b_M_C(p_b_M_C, q_1_M_1, M: int):
    """Do something ....
    
    Parameters
    ----------
    p_b_M_C: XXXX
        XXXXXX
    q_1_M_1, M: int
        XXXXX
    
    Returns
    -------
        XXXXXX
    """
    return np.sum(-np.log(p_b_M_C) * p_b_M_C / q_1_M_1, axis=(1, 2)) / M
