# Python API 参考

## version

> k2 provides a Python API to collect information about the environment in which k2 was built: `python3 -m k2.version`

> Please attach the above output while creating an issue on GitHub.

> If you are using PyTorch with k2, please also attach the environment information about PyTorch which can be obtained by running: `python3 -m torch.utils.collect_env`

## k2

add_epsilon_self_loops
arc_sort
cat
closure
compose
compose_arc_maps
connect
convert_dense_to_fsa_vec
create_fsa_vec
create_sparse
ctc_graph
ctc_loss
ctc_topo
determinize
do_rnnt_pruning
expand_ragged_attributes
get_aux_labels
get_best_matching_stats
get_lattice
get_rnnt_logprobs
get_rnnt_logprobs_joint
get_rnnt_logprobs_pruned
get_rnnt_logprobs_smoothed
get_rnnt_prune_ranges
get_rnnt_prune_ranges_deprecated
index_add
index_fsa
index_select
intersect
intersect_dense
intersect_dense_pruned
intersect_device
invert
is_rand_equivalent
joint_mutual_information_recursion
levenshtein_alignment
levenshtein_graph
linear_fsa
linear_fsa_with_self_loops
linear_fst
linear_fst_with_self_loops
mutual_information_recursion
mwer_loss
one_best_decoding
properties_to_str
prune_on_arc_post
pruned_ranges_to_lattice
random_fsa
random_fsa_vec
random_paths
remove_epsilon
remove_epsilon_and_add_self_loops
remove_epsilon_self_loops
replace_fsa
reverse
rnnt_loss
rnnt_loss_pruned
rnnt_loss_simple
rnnt_loss_smoothed
shortest_path
simple_ragged_index_select
swoosh_l
swoosh_l_forward
swoosh_l_forward_and_deriv
swoosh_r
swoosh_r_forward
swoosh_r_forward_and_deriv
to_dot
to_str
to_str_simple
to_tensor
top_sort
trivial_graph
union
CtcLoss
DecodeStateInfo
DenseFsaVec
DeterminizeWeightPushingType
Fsa
MWERLoss
Nbest
OnlineDenseIntersecter
RaggedShape
RaggedTensor
RnntDecodingConfig
RnntDecodingStream
RnntDecodingStreams
SymbolTable

## k2.ragged

`k2.ragged.cat(srcs: List[_k2.ragged.RaggedTensor], axis: int) -> _k2.ragged.RaggedTensor`

Concatenate a list of ragged tensors along a given axis.

Parameters
- `srcs`: A list (or a tuple) of ragged tensors to concatenate. 
  They MUST all have the same dtype and on the same device.
- `axis`: Only 0 and 1 are supported right now. 
  If it is 1, then `srcs[i].dim0` must all have the same value.

Returns
- Return a concatenated tensor.

---


create_ragged_shape2
create_ragged_tensor
index
index_and_sum
random_ragged_shape
regular_ragged_shape
RaggedShape
RaggedTensor