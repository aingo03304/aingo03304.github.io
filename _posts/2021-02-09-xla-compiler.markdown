---
layout: post
title: "Deep Dive into XLA (Draft)"
date: 2020-03-01 01:43:00 +0900
image_url: ""
mathjax: true
comments: true
---

# GPU backend Level Optimization
1. Parallel task assignment
2. Alias passthrough params
3. cuBLAS GEMM pad for tensor cores
4. cuDNN batchnorm rewriter
5. cuDNN fused conv rewriter
6. cuDNN pad for convolutions
7. cuSolver rewriter
8. Fusion merger
9. GEMM algorithm picker
10. GEMM rewriter
11. GPU conv algorithm picker
12. GPU conv padding legalization
13. GPU conv rewriter
14. GPU copy insertion
15. GPU sanitize constant names
16. Horizontal fusion
17. Multi output fusion
18. Reduction degenerate dim remover
19. Reduction dimension grouper
20. Reduction layout normalizer
21. Reduction splitter
22. Tree reduction rewriter
23. Variadic op splitter


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