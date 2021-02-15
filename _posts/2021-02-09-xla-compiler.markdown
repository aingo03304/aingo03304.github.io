---
layout: post
title: "Deep Dive into XLA (Draft)"
date: 2020-03-01 01:43:00 +0900
image_url: "/assets/xla/xlalogo.png"
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
Adds padding to dot operations to make them run faster on GPUs with tensor cores [https:devblogs.nvidia.com/programming-tensor-cores-cuda-9/](https:devblogs.nvidia.com/programming-tensor-cores-cuda-9/).

f16 dots are padded to have input/output shapes with dimensions that are multiples of 8, so that we can use tensor cores.

Don't run this pass on GPUs without tensor cores -- it will make them slower!
This pass depends on xla::DotDecomposer pass, so it should go strictly later.

## cuDNN batchnorm rewriter  
Rewrites BatchNorm HLOs into calls into cudnn where possible.

A call into cudnn for performing a batchnorm op is represented as a `CustomCall` HLO with `custom_call_target` equal to one of

- `kCudnnBatchNormForwardInferenceCallTarget`
- `kCudnnBatchNormForwardTrainingCallTarget`, or
- `kCudnnBatchNormBackwardCallTarget`.

A `CustomCall` created by this pass has the same operands corresponding batchnorm HLO, except the epsilon() and feature_index() properties of the batchnorm HLO are converted into proper operands, added to the end of the `CustomCall`'s operands list.

The inputs/outputs of the cudnn calls for `BatchNormTraining` and `BatchNormGrad` do not correspond exactly to the HLOs.  In particular, the training cudnn call returns `1/sqrt(variance + epsilon)`, while the HLO returns plain variance.  Similarly, the grad cudnn call expects `1/sqrt(variance + epsilon)` as input, whereas the HLO expects plain variance.

This pass adds HLOs in front of / behind the `CustomCall`s to fix up the inputs/outputs as appropriate, and we rely on the `AlgebraicSimplifier` to remove these where possible.

Currently batchnorm ops over F32s are converted into cudnn calls, so long as epsilon is not too small.  This pass leaves other batchnorm ops unmodified.

The GPU backend does not implement a lowering for the batchnorm HLOs -- it expects them to be lowered to cudnn calls via this pass or to HLO soup via `BatchNormRewriter`.

## cuDNN fused conv rewriter  
Rewrite the custom call targeting cudnnConvolutionForward to cudnnConvolutionBiasActivationForward by fusing applicable point-wise operations following forward convolution.  This transform must run after `cudnn_conv_rewriter`. It is straightforward for floating point convolutions: 

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
f32[A, B, P] padded = pad(input) Let P = ceil(C/M) * M.
f32[A, B, Q, M] reshaped = bitcast(padded) Let Q = ceil(C/M)
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
f32[A, P, C] padded = pad(input) Let P = ceil(B/T) * T.
f32[A, Q, T, C] reshaped = bitcast(padded) Let Q = ceil(B/T)
f32[A, Q, C] inner_reduce = reduce(reshaped, dimensions={2})
f32[A, C] outer_reduce = reduce(inner_reduce, dimensions={1})
```

## Variadic op splitter  
Splits variadic ops with many operands into pieces such that we don't exceed the parameter space on the GPU. Currently only concatenate ops are split up.


# Frontend Level Optimization

## Algebraic simplifier
A pass which performs algebraic simplifications.

## All gather decomposer
AllGatherDecomposer is a pass which converts unsupported all-gathers into dynamic-update-slices and all-reduces.

## All reduce combiner
Combines small non-dependent AllReduce ops into larger combined AllReduce ops. A typical AllReduce implementation has a minimum latency-induced time for a AllReduce op so a single combined op can be more efficient than many small ones.

## All reduce simplifier
A pass that detects all-reduces whose inputs are already the same across replicas using the replication analysis, then replaces those all-reduces with local computations. E.g., a sum all-reduce on replicated input will be replaced by a multiply with the replica count.

## All reduce cross combiner
When the HLO graph contains a cross-module AllReduce (N separate AllReduce
ops that share the same channel_id for MPMD partitioning, or 1 AllReduce op
for SPMD partitioning), followed by some simple linear operations, followed
by a cross-replica AllReduce (also known as cross-replica sum, or CRS), we
can combine the CMAR and the CRAR, to use an efficient AllReduce
implementation that fully utilizes the interconnect bandwidth.

Such sequences appear in spatially partitioned models (either MPMD or SPMD).
This pass must run right after spatial partitioning, when the code is still
in a single HLO module.

The steps are:
1) Find CMARs followed by simple ops followed by CRARs.
2) Group CMARs by channel_id. They must all be rewritten. For SPMD
   partitioning, there will only be a single CMAR for each channel_id.
3) Prove that the CMAR patterns in each core produce the same result.
4) Eliminate the CMAR, and if it feeds an addition/subtraction, divide the
   other operand by the number of spatial partitions.
5) Turn the CRAR into an all-core AllReduce.

The pass also handles the case where multiple CMARs lead to the same CRAR,
and eliminates all CMARs. This graph:

       Y
       |
 X   CMAR_2   Z
 |      \    /
CMAR_1     +
   \     /
      +
      |
    CRAR

gets rewritten to:

          Z   num_partitions
           \  /
      Y    div
       \   /
   X     +
    \   /
      +
      |
 all-core AR


## Batch dot simplification
Simplifies batch dot operations.

Normally these would live in the algebraic simplifier, but we want to run
this to fixpoint (this pass reaches fixed point in one execution) before we
run the DotDecomposer.

## Batchnorm expander
A pass which rewrites batch norm operations into more operations. Breaking a
big operation into smaller operations helps leverage our generic fusion
logic.

## bfloat16 conversion folding
A pass which folds F32 <-> BF16 conversions to their operands or users, when
it is supported by the backend.

This pass follows the passed-in backend-specific BF16 support rules, but can
introduce mixed precision in individual HLOs which breaks the assumption of
some other HLO passes. So it should be used at the end of the HLO
optimization pipeline followed by a DCE pass. If other passes are needed
after this pass, run BFloat16MixedPrecisionRemoval first to undo some of the
changed made by this pass.

## bfloat16 normalization
A pass which adds F32 <-> BF16 conversions for HLO instructions that do not
support BF16 input/output or mixed precision, according to the passed-in
backend-specific BF16 support rules.

## bfloat16 mixed precision removal
A pass that unconditionally removes the mixed F32/BF16 uses in HLO
instructions (excluding convert) by adding F32 <-> BF16 conversions. Unlike
BFloat16Normalization, this pass does not use a backend-specific
BFloat16Support, and does not change HLOs that have BF16 data if they do not
use mixed precision; it removes mixed precision even if the backend supports
it. This pass is used to make the HLO module valid for other HLO passes which
do not support mixed precision.

## bfloat16 propagation
HLO pass which reduces the precision of some HLO instructions to BF16
according to the backend-specific BFloat16Support rule provided by the
caller.

This pass can be used to reduce instruction precision without affecting the
numerical accuracy of the module, i.e., the final output of the module would
be bitwise identical to that without this pass; this is possible if the
backend already reduces precision to BF16 on some HLO instructions.

This pass will not modify the signature of a computation, unless it is a
fusion computation or its only caller is a while.

!!! WARNING !!! This pass can introduce mixed precision in individual HLOs,
which has two issues:

1) It does not guarantee to respect the passed-in BFloat16Support
specification in terms of mixed precision, so the backend may not support an
HLO that has mixed precision produced by this pass. To address this issue,
run BFloat16Normalization with the same BFloat16Support after this pass.

2) In general, mixed precision may break the assumptions of some other HLO
passes even if the specific backend supports the individual HLOs. Such
assumptions include that there are no HLOs using mixed precision, or that the
precision of an HLO's output is determined by its inputs. It should be used
at the end of the HLO optimization pipeline but before
BFloat16ConversionFolding. If other passes are needed after this pass, run
BFloat16MixedPrecisionRemoval first to undo some of the changes made by this
pass.

## Call inliner
For every kCall operation in the main computation, we inline the body of the
called function, and proceed recursively.

## Conditional canonicalizer
Canonicalize output of conditionals, make non-tuple outputs into tuple with
single element output. After this pass, all conditional instructions have
tuple outputs.

## Conditional code motion
HLO pass that moves identical ops in/out of conditional.
- The definition of identical are the shape of the operands are identical
and their properties are identical.
- Only the identical ops that won't share operands with other ops will
be moved out of conditional.

## Conditional simplifier
HLO pass that removes kConditional with a constant predicate, replacing them
with their true or false computation as appropriate.

## Conditional to select
A pass which transforms conditionals to selects in places where conditionals
are legal, but not currently supported by the backends (e.g. inside kMap)

## Convolution group converter
A pass which rewrites convolutions with feature_group_count > 1 into
convolutions with feature_group_count = 1.

## Copy insertion
Copy insertion is a legalization HLO pass which inserts copies (kCopy
instructions) to eliminate several kinds of problems in the HLO module.

  (1) Entry parameter or a constant live out of the entry computation.  Entry
      computation arguments and constants have different lifetimes than the
      computation result and cannot share the same allocation. Parameters and
      constants live out of non-entry computations do not need copies.

  (2) Different values which are simultaneously live and which must be held
      in the same buffer. This can occur in while bodies. Specifically, the
      while loop state (the arguments to the while instruction) is updated
      in-place and the update may clobber the value from the previous
      iteration before the previous value is dead. Computations called from
      kCall instructions do not need such copies because kCall has no update
      in-place semantics.

  (3) The buffer set of the root instruction of the entry computation must be
      unambiguous and distinct. That is, InstructionAliasSet::IsAmbiguous and
      InstructionAliasSet::IsDistinct return true.

## Defuser
A pass which replaces all fusion instructions with the equivalent un-fused
instructions.

## Despecializer
Pass which strips control dependencies from all instructions in the module.
Creates an HloPassPipeline containing multiple HloPasses that can
despecialize an optimized HloModule. This is useful to run an HloModule
optimized for one specific platform on a different platform (undoing platform
specific passes) with matching numerics for comparison.

Current despecialization passes are HloDescheduler, ControlDepRemover,
Defuser and BFloat16MixedPrecisionRemoval.

## Dot decomposer
DotDecomposer is a pass which converts dots into a canonical form where
non-contracting and contracting dimensions are reshaped together and batch
dimensions are the most major dimensions.

## Dynamic index splitter
Convert R1 index operands to DynamicSlice and DynamicUpdateSlice ops into
separate scalars.

## Dynamic padder
With bounded shapes, only part of the shape contains effective data and the
rest contains padded data, whose value can be anything depending on the
source of the data. When a bounded shape is directly consumed by an
instruction that collapses dimensions (reduce for example), the padding data
would affect result of the instruction.

DynamicPadder uses DynamicDimensionInference to detect bounded shapes in a
hlo module, it then inserts certain instructions to reset the padding into an
identity value so that in doesn't affect the result of subsequent
instruction. For example, it'd reset the padding to 0 before a bounded shape
is consumed by a reduce-sum.

Dynamic_padder removes dynamic shapes from the entry computation, and inserts
custom calls (with dynamic shapes), which are lowered by specialized
emitters: PadToStatic and SliceToDynamic.

Each instruction can have one of the three modes in supporting dynamic
lowering.

## Flatten call graph
Flattening associates each call site with a unique computation (for
sequential calling contexts) This simplifies buffer assignment and
points-to analysis (see b/36865746 for details).

## HLO constant folding
A pass which performs constant folding in order to avoid unnecessary
computation on constants.

## HLO cse
A pass which performs common-subexpression elimination. Identical constants
and identical instructions with the same operands are commoned. The pass
iterates over the instructions in topological order which enables the pass to
find arbitrarily large common expressions.

## HLO dce
HLO pass which removes dead instructions from each computation in the module
and removes dead computations from the module.

An instruction is dead if it is not reachable from the root. A computation is
dead if it is not the entry computation of the module and it is not reachable
from the entry computation.

This pass does not remove dead parameter instructions, as parameter
instructions cannot be deleted.

## HLO domain isolator
Domain isolation is the task of placing kDomain instructions between HLO
instructions having different sharding. A kDomain instruction is essentially
used to break an HLO graph edge connecting two instructions with different
sharding. If a set of connected instructions have all the same sharding, no
kDomain instruction will be placed.

## HLO domain remover
Removes all the kDomain instructions of a given kind from the input module,
and calls the normalizer to propagate the properties on the possibly new born
instructions.

## HLO domain verifier
Verifies that the domain instructions are consistent, and the each domain is
surrounded by the same metadata.

Verify that the whole kDomain frontier bounding the instruction reach set,
has matching metadata.
A kDomain instruction has two sides of metadata, a user facing and an
operand facing.
A reachable instruction set can make contact with a kDomain instruction on
a user facing side (the kDomain is operand of the instruction), or on a
operand facing side (the kDomain is user of the instruction).
And depending on the contact side, the proper metadata object
(user_side_metadata() vs. operand_side_metadata()) needs to be used for
consistency checks.
Returns the DomainMetadata pointer which surrounds the domain, and
represents the common metadata within such domain. If the returned
DomainMetadata pointer is nullptr, the input domain had no kDomain
boundary.

## HLO element type converter
A pass that eliminates certain element types as the input or output of ops by
inserting Convert ops. This allows a backend to support an element type while
only actually implementing the Convert op for that element type. This is
generally not the fastest approach, but it works.

## HLO get dimension size rewriter
Pass to replace a kGetDimensionSize instruction with a hlo instruction
representing the dynamic size if the dimension is dynamic, otherwise a
constant instruction representing the static size.

## HLO memory scheduler
A pass which schedules the HLO instructions in a module. The HloModule's
schedule field is set to the resulting HloSchedule using
HloModule::set_schedule.

## HLO trivial scheduler
A pass which produces a naive, but correct schedule. The schedule is produced
using a DFS traversal of the graph with no attempt to minimize memory use.

## HLO descheduler
A trivial pass which clears the schedule currently set on the
HloModule. After this pass runs HloModule::has_schedule will return false.

## HLO module dce
HLO pass which removes dead code from computations in the module using
HloModule-scoped analysis (HloLivenessAnalysis).

Sweeps through live instructions which cross computation boundaries (kWhile),
and removes code at dead shape indices.


## HLO rematerialization
HLO pass which rematerializes instructions to reduce peak memory use, where
memory use is defined as the total size of all live HLO instruction
values. Parameters and constants are included in memory use estimates.

CSE will undo the effects of this optimization and should not be run after
this pass. In general, this pass should be run very late, immediately before
code generation.

## HLO subcomputation unification
Unify subcomputations of a `HloModule`: if any computations are equal, choose
one arbitrarily to use and delete the others.

## HLO verifier
HLO pass that verifies invariants of HLO instructions for each computation in
the module.

## Indexed array analysis
A pass that prints all non-trivial results returned by IndexedArrayAnalysis.
This pass is a no-op if !VLOG_IS_ON(2) so it should be fine to
unconditionally add to the regular HLO pass pipeline.

## Instruction fusion
HLO pass which performs instruction fusion. Instructions are fused
"vertically", meaning producing instructions are fused into their consumers
with the intent that the loops which compute their values will be fused in
code generation. Derived classes define ShouldFuse method to select which
instructions to fuse.

## Layout assignment
HLO pass which assigns layouts to all instructions in the HLO module while
satisfying all necessary invariants and minimizing cost.

## Map inliner
A pass which performs map inlining. This replaces kMap instructions with
their equivalent sequence of array operations. For example:
  map({X, Y}, add) -> add(X, Y)).

## Memory space propagation
This is a legalization pass that propagates the memory space in the layout to
the fusion computations.

## Multi output fusion
This class implements the fusing of sibling fusion instructions that sharing
common operands.
It constructs the following associated data structures.
 (1) candidates_: stores the instruction and the set of instructions it can
     fuse to.
 (2) candidates_index_: maps instruction to id.
 (3) reachability_: reachability map in this computation.
 (4) all_fusion_candidates_: the vector of candidate instructions.
 (5) worklist_: a priority queue that contains pairs of instructions to be
     fused and their fusion profit scores.

 Function Perform() applies the optimization. It picks up the most profitable
 pair in the worklist_, checks if it's legal to fuse and fuses the pair.
 After fusion, it updates the associated structures such as reachability_,
 candidates_ and worklist_.
 Note that the reachability map is updated based on the original computation.
 This works because the reachability is monotonically increasing with
 instruction fusion.This class implements the fusing of sibling fusion instructions that sharing
common operands.
It constructs the following associated data structures.
 (1) candidates_: stores the instruction and the set of instructions it can
     fuse to.
 (2) candidates_index_: maps instruction to id.
 (3) reachability_: reachability map in this computation.
 (4) all_fusion_candidates_: the vector of candidate instructions.
 (5) worklist_: a priority queue that contains pairs of instructions to be
     fused and their fusion profit scores.

 Function Perform() applies the optimization. It picks up the most profitable
 pair in the worklist_, checks if it's legal to fuse and fuses the pair.
 After fusion, it updates the associated structures such as reachability_,
 candidates_ and worklist_.
 Note that the reachability map is updated based on the original computation.
 This works because the reachability is monotonically increasing with
 instruction fusion.

## Op expand pass
This pass is an abstract superclass for passes that replace operations that
match a pattern. It is intended to be subclassed, not used directly.

This pass is useful for legalizing HLO instructions that a particular backend
does not support into other HLO instructions.

## Optimize input output buffer alias
This pass opportunistically finds input and output buffers that can be
aliased, and writes the alias config into the HloModule.

The input and the output buffers can be in any shape, and each output buffer
can alias with an input buffer with the same shape. Each input buffer may
only alias with a single output buffer. For example, for the following
parameter and the output buffers,

 Parameters : { P1(f32[3]), P2(s32[3]), P3(f32[3,12]), P4(f32[16,12]), ... }
 Outputs    : { O1(s32[3]), O2(f32[3]), O3(f32[16,12]), ... }

one potential aliasing would be (O1, P2), (O2, P1), (O3, P4), ..

## Reshape mover
A pass which moves Reshapes and Transposes to let later passes combine them.
This now only moves them outputward across elementwise ops all whose operands
are equivalent Reshapes or Transposes, but in future could potentially move
them inputward also.

## Root instruction sinker
Given a scheduled HLO module, this pass sinks the ROOT of the instruction to
the bottom of the non-fusion computations. To avoid dependency violations of
moving the ROOT instruction, it creates a new ROOT instruction that looks
like the following:
  - For tuple ROOT type:
       new_root = tuple(gte(old_root), gte(old_root), ...)
  - For non-tuple ROOT type:
       new_root = bitcast(old_root)

## Scatter expander


## Sharding propagation
Propagates sharding information around the graph. HLOs that have shardings
are kept as-is, those that do not have shardings are given shardings based on
a simple local greedy heuristic.

## Slice sinker
An HLO pass that sinks slice operations used by a group of elementwise
operations and merges the group of elementwise operations.

## Sort simplifier
HLO pass which removes unused operands from sort, where an unused operand is
defined as an operand at some index 'x' at which the output is not used.

## TopK rewriter
This pass pattern-matches soups of HLOs executing a TopK operation and
replaces them with a TopK CustomCall when the given values are supported by
the CustomCall and it is more efficient to use that implementation.

## Transpose folding
HLO pass that folds transpose operators into Dot operators, where the Dot
operator is implemented by a GEMM kernel that can transpose its inputs.

## Tree reduction rewriter
Increase precision for the reduction operation by applying the reduce-window
first.

E.g. suppose we want to reduce f32[1024] to a scalar. This pass first applies
a reduce-window (with kSame padding) of size `reduce_window_size`, and then
reduces the resulting array f32[32]. The rewrite is not applied if any of the
reduced dimensions is smaller than the `reduce_window_size`.

Applying this pass until a fixed point performs a variant of pairwise
summation (https:en.wikipedia.org/wiki/Pairwise_summation), which is
guaranteed to have an asymptotically smaller error bound provided that
intermediate roundoff errors are random and have random sign.

If this pass lowers the performance too much, the window size can always be
increased to a larger value.

## Tuple simplifier
A pass which simplifies patterns of Tuple and GetTupleElement instructions in
the module.

## While loop constant sinking
Sinks while loop invariant values that happen to be constants into the while
loop body and conditional. This is probably not a win in isolation but may
unlock further optimizations like constant folding.

  state = (..., const, ...)
  while (pred(state)) {
    (..., v, ...) = state
    use(v)
    state = (..., v, ...)
  }

=>

  state = (..., const, ...)
  while (pred(state)) {
    (..., v, ...) = state
    use(const)
    state = (..., v, ...)
  }

Note that it leaves the `v` in place to keep that component of the state
tuple trivially loop invariant.  WhileLoopSimplifier will later get rid of
`v`.

TODO(b/79121449):  We should also sink broadcasts of constants.

## While loop invariant code motion
HLO pass that rewrites while loops to hoist loop invariant instructions in
the while body into the computation that contains the while instruction.

## While loop simplifier
HLO pass that makes the following transformations on while loops:

 - A while loop with static trip count of 0 is deleted.

 - A while loop with static trip count of 1 is replaced by its body (sans
   loop).

 - Elements of a while loop's tuple that the loop doesn't use are removed
   from the tuple.

 - If the while loop's parameter is a nested tuple, it's flattened to a
   single-level tuple.  This is good because it usually reduces the number of
   kTuple instructions, but also because it unlocks additional optimizations
   (e.g. removing unused loop parameters).

Flattening nested while loop tuples adds a whole mess of likely unnecessary
kGetTupleElement and kTuple operations to the graph.  We expect that tuple
simplifier will be run afterwards.


## While loop trip count annotator
Pass that annotates `while` loops with known trip counts.

The annotation is stored as a backend-config on the while loop node.

This pass should run after all passes that might semantically modify a while
loop, e.g. by unrolling it.  Otherwise, a loop could end up with a
backend-config that doesn't match its true trip-count.

This pass does some pattern-matching on loop bodies and conditions, so it
should run after most HLO simplifications and before fusion and layout
assignment, which make pattern matching much more difficult by e.g.
introducing `copy` nodes.

## Zero sized HLO elimination
HLO pass that replaces zero sized Hlos with a zero sized constant literal.