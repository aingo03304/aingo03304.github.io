---
layout: post
title: "Deep Dive into XLA (Draft)"
date: 2020-03-01 01:43:00 +0900
image_url: ""
mathjax: true
comments: true
---

# CPU Backend Level Optimization
## Convolution canonicalization  
An HLO pass that canonicalizes the dimension numbers of all top-level convolutions in the given module. 

In order to hit the fast path of using Eigen's convolution implementation, a convolution's dimension numbers need to satisfy certain constraints (so called canonical convolutions). 

This pass expands non-canonical convolutions into reshapes and canonical convolutions, so that these non-canonical convolutions can run faster.
## Parallel task assigner  
ParallelTaskAssigner computes target parallel task counts for all HLOs in the module, then assigns parallel task counts to HLOs in the entry computation, or to HLOs in embedded computations invoked by (potentially nested) kWhile or kCall instructions. 

Each HLO which is assigned parallel task counts is outlined into its own embedded computation, which is compiled as a parallel compute function, and which is invoked from a kCall instruction that is lowered in codegen to a runtime parallel fork/join call.

# GPU Backend Level Optimization
## Alias passthrough params  
This pass aliases input and output buffers that are associated with a parameter that is passed through to the module root unmodified.

This pass assumes that parameters and the root use unnested shapes, which is the case for XLA:GPU.

This pass must run prior to copy insertion.

## cuBLAS GEMM pad for tensor cores  
Adds padding to dot operations to make them run faster on GPUs with tensor cores (https://devblogs.nvidia.com/programming-tensor-cores-cuda-9/).

f16 dots are padded to have input/output shapes with dimensions that are multiples of 8, so that we can use tensor cores.

Don't run this pass on GPUs without tensor cores -- it will make them slower!
This pass depends on xla::DotDecomposer pass, so it should go strictly later.

## cuDNN batchnorm rewriter  
Rewrites BatchNorm HLOs into calls into cudnn where possible.

A call into cudnn for performing a batchnorm op is represented as a `CustomCall` HLO with `custom_call_target` equal to one of

- `kCudnnBatchNormForwardInferenceCallTarget`
- `kCudnnBatchNormForwardTrainingCallTarget`, or
- `kCudnnBatchNormBackwardCallTarget`.

A `CustomCall` created by this pass has the same operands corresponding batchnorm HLO, except the epsilon() and feature_index() properties of the batchnorm HLO are converted into proper operands, added to the end of the `CustomCall`
's operands list.

The inputs/outputs of the cudnn calls for `BatchNormTraining` and `BatchNormGrad` do not correspond exactly to the HLOs.  In particular, the training cudnn call returns `1/sqrt(variance + epsilon)`, while the HLO returns plain variance.  Similarly, the grad cudnn call expects `1/sqrt(variance + epsilon)` as input, whereas the HLO expects plain variance.

This pass adds HLOs in front of / behind the `CustomCall`s to fix up the inputs/outputs as appropriate, and we rely on the `AlgebraicSimplifier` to remove these where possible.

Currently batchnorm ops over F32s are converted into cudnn calls, so long as epsilon is not too small.  This pass leaves other batchnorm ops unmodified.

The GPU backend does not implement a lowering for the batchnorm HLOs -- it expects them to be lowered to cudnn calls via this pass or to HLO soup via `BatchNormRewriter`.

## cuDNN fused conv rewriter  
Rewrite the custom call targeting cudnnConvolutionForward to cudnnConvolutionBiasActivationForward by fusing applicable point-wise operations following forward convolution.  This transform must run after cudnn_conv_rewriter. It is straightforward for floating point convolutions: 

transforming
```
max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias))
```
to
```
cudnnConvolutionBiasActivationForward(x, w, bias, alpha1, alpha2, side)
```

Integer convolution requires additional patterns to match CuDNN semantics:
#1 from
```
cast<int8>(clamp<-128, 127>(conv(int8_x, int8_w)))
```
to
```
cudnnConvolutionForward<int8>(int8_x, int8_w)
```
or #2 from
```
cast<float>(conv(int8_x, int8_w))
```
to
```
cudnnConvolutionForward<float>(int8_x, int8_w)
```
or #3 from
```
cast<int8>(clamp<-128, 127>(max(0, alpha1 *
           cast<float>(conv(int8_x, int8_w)) +
           alpha2 * cast<float>(int8_side) +
           broadcast(bias)))
```
to
```
cudnnConvolutionBiasActivationForward<int8>(int8_x, int8_w, bias, alpha1, alpha2, int8_side)
```
or #4 from
```
max(0, alpha1 * cast<float>(conv(int8_x, int8_w)) + alpha2 * float_side + broadcast(bias))
```
to
```
cudnnConvolutionBiasActivationForward<float>(int8_x, int8_w, bias, alpha1,
alpha2, float_side)
```
Rewrite the custom call targeting `cudnnConvolutionForward` to `cudnnConvolutionBiasActivationForward` by fusing applicable point-wise operations following forward convolution. This transform must run after `cudnn_conv_rewriter`.
It is straightforward for floating point convolutions:
transforming
```
max(0, alpha1 * conv(x, w) + alpha2 * side_input + broadcast(bias))
```
to
```
cudnnConvolutionBiasActivationForward(x, w, bias, alpha1, alpha2, side)
```

Integer convolution requires additional patterns to match CuDNN semantics:
#1 from
```
cast<int8>(clamp<-128, 127>(conv(int8_x, int8_w)))
```
to
```
cudnnConvolutionForward<int8>(int8_x, int8_w)
```
or #2 from
```
cast<float>(conv(int8_x, int8_w))
```
to
```
cudnnConvolutionForward<float>(int8_x, int8_w)
```
or #3 from
```
cast<int8>(clamp<-128, 127>(max(0, alpha1 *
           cast<float>(conv(int8_x, int8_w)) +
           alpha2 * cast<float>(int8_side) +
           broadcast(bias)))
```
to
```
cudnnConvolutionBiasActivationForward<int8>(int8_x, int8_w, bias, alpha1, alpha2, int8_side)
```
or #4 from
```
max(0, alpha1 * cast<float>(conv(int8_x, int8_w)) + alpha2 * float_side + broadcast(bias))
```
to
```
cudnnConvolutionBiasActivationForward<float>(int8_x, int8_w, bias, alpha1, alpha2, float_side)
```

## cuDNN pad for convolutions  
Two zero-paddings for CuDNN thunking are done in this transform: padding for tensor cores and padding for integer convolutions.  This transform also add slice instruction to remove unnecessary output features.

## cuSolver rewriter  
Rewrites Cholesky calls into `CustomCall` HLOs that call into cuSolver.

## Fusion merger  
An HLO pass that attempts to merge fusion instructions to reduce kernel launch overhead and improve data locality.

Fusion instructions are merged into their users if two conditions are met:

1) The flops_to_bytes ratio of the fusion instruction is below the threshold value of 1.0.
2) The result of merging the fusion instruction into its users would not increase bytes transferred.

## GEMM algorithm picker  

## GEMM rewriter  
cuBLAS GEMM in the most general form can run the following operation:

(kAdd
(kMultiply (kDot A B) alpha)
(kMultiply C beta))

where A, B, C are matrixes and `alpha` and `beta` are host constants. The additional requirement is that C has no other users (otherwise, it does not make sense to fuse it inside the custom call).

Both multiplication and addition can be avoided (equivalent to setting `alpha` to one and `beta` to zero).

This pass pattern-matches the most general form of this instruction (we assume transposes are already folded), and rewrites it into a custom call where (A, B, C) are three operands respectively, and `alpha` and `beta` are stored in the backend config.

## GPU conv algorithm picker  
Modifies `CustomCall`s to cudnn convolutions, choosing the best algorithm for each and adding explicit scratch space to the `CustomCall`s.

## GPU conv padding legalization  
An HLO pass that canonicalizes convolution instructions for GPU codegen. It inserts Pad instructions before Convolution instructions with uncanonicalized padding, so that they can be lowered to Cudnn/Miopen convolution.

## GPU conv rewriter  
Rewrites plain convolutions, backwards-filter convolutions, and backwards-input convolutions into `CustomCall` HLOs that call into Cudnn/Miopen. 

For integer convolution, it requires the following pattern:
```
conv<InputT=int32, ResultT=int32>(
convert<int32>(int8_x), convert<int32>(int8_y))
```
We transform it to:
```
custom_call<int32>(int8_x, int8_y, target=cudnnForwardConvolution)
```
Note that this pattern is necessary but not sufficient to map convolutions to CuDNN. More patterns will be matched in `cudnn_fused_conv_rewriter`.

## GPU copy insertion  
Besides the modifications made by the generic `xla::CopyInsertion`, this GPU-specific copy insertion also materializes operands of library calls by inserting kCopy instructions.

## GPU sanitize constant names  
Sanitizes HLO instruction names for the GPU backend. Currently, it only replaces . and - in the HLO constant instruction names with _ to please the LLVM PTX backend.

## Horizontal fusion  
This optimization pass horizontally fuses computations for reducing kernel launch overhead while increasing kernel launch dims on GPU. The initial motivation of this horizontal fusion is due to the observation that the training optimizer phase (e.g., AdamOptimizer and L2Loss, etc.) typically has many small kernels as a result of applying the same formula on many training parameters (or variables in Tensorflow). Fusing these small kernels, hence, provides performance gain.

Theoretically speaking, we may implement a cycle detection algorithm to make sure no cycles are created after fusion. However, cycle detection check is somewhat cumbersome; also, we observe that naive horizontal fusion of arbitrary kernels may not be profitable due to control divergence and possible increase of memory bandwidth pressure due to uncoalesced memory accesses (note that horizontal fusion does not change the amount of memory read+written at all). In practice, a simple yet effective heuristic is used to avoid these issues while addressing the known beneficial cases. That is, we simply search for fusion candidates by looking for instructions whose outputs are all consumed by the same instruction. This catches the cases in the training optimizer phase, as the candidate instructions are typically consumed only by the ROOT tuple of the entry computation.

The following illustrates the mechanism of the horizontal fusion. Before fusion, there are two trivial kernels in the illustrating example. One has only a Mul op, while the other consists of only an Add op. Since they are only consumed by the same (ROOT) tuple instruction, horizontal fusion is triggered.
```
i0 i1   i2 i3
| |     | |
v v     v v
Mul     Add
|       |
v       v
(ROOT) tuple

We horizontally fuse them into the below pattern.

i0 i1   i2 i3       +++ (Slice) Input Fusion
 | |     | |          +
 v v     v v          +
 Mul     Add          +
  |       |           +
  v       v           +
Reshape0  Reshape1    +
  |       |           +
  v       v           +
 Concatenate          +
  |       |           +
  v       v           +
  Slice0  Slice1     +++
  |       |
  v       v
Reshape2  Reshape3
  |       |
  v       v
 (ROOT) tuple
```

Note that this fusion style provides an important advantage that kernels of different shapes can be horizontally fused. The first pair of reshapes (i.e., Reshape0 and Reshape1) reshape the dims to 1 dimension, so that the outputs of the fused kernels can (always) be concatenated. The second pair of reshapes (Reshape2 and Reshape3) restore the original shapes to the output tensors.

No extra copies are introduced by the horizontal fusion. Besides Reshape2 and Reshape3, the other instructions are fused into an input fusion; the output dims of the concatenate will be used as the kernel launch dims. Instruction bitcasts can be used for Reshape2 and Reshape3 as long as the outputs of Mul and Add are row-major.

## Multi output fusion  
Multi-output fusion of sibling and producer-consumer instructions for the GPU backend.

## Reduction degenerate dim remover  
Enforces the invariant that reduction input and output have no degenerate (size 1) dimension. Since these dimensions are physically meaningless, they are removed using bitcasts.

For example,
```
f[1] out = reduce(f[100, 1, 1] input, dimensions={0, 1})
```
becomes:

```
f[100] tmp1 = f[100] bitcast(f[100, 1, 1], input)
f[] tmp2 = reduce(f[100] tmp1, dimensions={0})
f[1] out = f[] bitcast(tmp2)
```

## Reduction dimension grouper  
Groups adjacent (logically and physically) reduced dimensions in reduction input.

Precondition: ReductionLayoutNormalizer has been run (physical proximity and logical proximity become the same).

For example,
```
f[] out = reduce(f[10,20,30] input, dimensions={0,1,2})
```
becomes:
```
f[600] tmp = f[600] bitcast(f[10,20,30] input)
f[] out = reduce(f[600] tmp, dimensions={0})
```

## Reduction layout normalizer  
Enforces default (minor-to-major) layout on all reduction inputs.
Note that since reduction output can request a custom layout,
this pass only guarantees standard layout for the input.

For example,
```
f[20,30]{0,1} out = reduce(f[10,20,30]{2,0,1} input, dimensions={0})
```
becomes:
```
f[20,10,30] tmp = f[20,10,30] bitcast(f[10,20,30]{2,0,1} input)
f[20,30]{0,1} out = reduce(f[20,10,30]{2,1,0} tmp, dimensions={1})
```

## Reduction splitter  
Splits a reduce op into two consecutive reduce ops if
* the reduce dimensions are not contiguous and
* at least one reduce dimension is large (i.e. corresponds to a large input
shape dimension).

Reductions with non-contiguous dimensions are emitted as simple element-wise loops. This is inefficient when reducing large input shape dimensions. Splitting such reductions allows using more efficient reduction emitters.

This pass splits reduce ops into two consecutive reduce ops. Run it to a fixpoint to split reduce ops along multiple large dimensions.

Precondition: ReductionDimensionGrouper has been run and adjacent reduce dimentsions have been grouped. Reduction layouts have been normalized.

## Tree reduction rewriter  
Rewrites reductions in a way they can be implemented without atomics.

Rule application: rewrite a single HLO reduce operation into two.

- Case 1: Row reduction, batched dimension is present, larger than Z-tiling size.

Rewriting:
```
f32[B] out = reduce(f32[A, B, C] input, dimensions={0, 2})
```
Into:
```
f32[A, B] tmp = reduce(f32[A, B, C] input, dimensions={2})
f32[B] out = reduce(f32[A, B] tmp, dimensions={0})
```
- Case 2: Row reduction

Let M be the thread tiling multiplied by the warp size.
We go from (assuming C > M):
```
f32[B] out = reduce(f32[A, B, C] input, dimensions={0, 2})
```
to:
```
f32[A, B, P] padded = pad(input) // Let P = ceil(C/M) * M.
f32[A, B, Q, M] reshaped = bitcast(padded) // Let Q = ceil(C/M)
f32[B, Q] inner_reduce = reduce(reshaped, dimensions={0, 3})
f32[B] outer_reduce = reduce(inner_reduce, dimensions={1})
```
- Case 3: Column reduction

Let T be the tiling size for the column reduction.

We go from (assuming B > T):
```
f32[A, C] out = reduce(f32[A, B, C] input, dimensions={1})
```
to:
```
f32[A, P, C] padded = pad(input) // Let P = ceil(B/T) * T.
f32[A, Q, T, C] reshaped = bitcast(padded) // Let Q = ceil(B/T)
f32[A, Q, C] inner_reduce = reduce(reshaped, dimensions={2})
f32[A, C] outer_reduce = reduce(inner_reduce, dimensions={1})
```

## Variadic op splitter  
Splits variadic ops with many operands into pieces such that we don't exceed the parameter space on the GPU. Currently only concatenate ops are split up.


# Frontend Level Optimization

1. Algebraic simplifier
2. All gather decomposer
3. All reduce combiner
4. All reduce simplifier
5. All reduce cross combiner
6. Batch dot simplification
7. Batchnorm expander
8. bfloat16 conversion folding
9. bfloat16 normalization
10. bfloat16 propagation
11. Call inliner
12. Conditional canonicalizer
13. Conditional code motion
14. Conditional simplifier
15. Conditional to select
16. Convolution group converter
17. Copy insertion
18. Defuser
19. Despecializer
20. Dot decomposer
21. Dynamic index splitter
22. Dynamic padder
23. Flatten call graph
24. HLO constant folding
25. HLO cse
26. HLO dce
27. HLO domain isolator
28. HLO domain remover
29. HLO domain verifier
30. HLO element type converter
31. HLO get dimension size rewriter
32. HLO memory scheduler
33. HLO module dce
34. HLO rematerialization
35. HLO subcomputation unification
36. HLO verifier
37. Indexed array analysis
38. Instruction fusion
39. Layout assignment
40. Map inliner
41. Memory space propagation
42. Multi output fusion
43. Op expand pass
44. Optimize input output buffer alias
45. Reshape mover
46. Root instruction sinker
47. Scatter expander
48. Sharding propagation
49. Slice sinker
50. Sort simplifier
51. TopK rewriter
52. Transpose folding
53. Tree reduction rewriter
54. Tuple simplifier
55. While loop constant sinking
56. While loop invariant code motion
57. While loop simplifier
58. While loop trip count annotator
59. Zero sized HLO elimination