# Clifford / geometric algebra deep learning: papers from 2026

A recall-oriented sweep of everything published or first posted between **2026-01-01 and 2026-10-25**
that uses Clifford algebra (also called geometric algebra, GA) inside a neural network or a
machine-learning method, or that studies such networks. Entries are grouped by topic; a paper that fits
several topics is listed once and cross-referenced.

Conventions used below:

* **CGENN** = Clifford Group Equivariant Neural Network (Ruhe et al., 2023), the layer family whose linear
  maps and geometric products commute with the Clifford group action.
* **GATr** = Geometric Algebra Transformer (Brehmer et al., 2023), a transformer whose tokens are
  multivectors of the projective geometric algebra (PGA) `Cl(3,0,1)`. **L-GATr** is its Lorentz-equivariant
  variant on the spacetime algebra `Cl(1,3)` (Spinner et al., 2024). **P-GATr** is the PGA variant used in robotics.
* **CGA** = conformal geometric algebra `Cl(4,1)`; **PGA** = projective geometric algebra `Cl(3,0,1)`.
* **Multivector** = a general element of a Clifford algebra (a sum of scalar, vector, bivector, ... parts);
  a **rotor** is an even-grade multivector of unit norm that encodes a rotation; a **versor** is a product of
  vectors, encoding a general orthogonal transformation.
* Venue tags: `[arXiv]` preprint only, `[conf]` conference paper, `[journal]`, `[OpenReview sub]` a 2026
  submission visible on OpenReview (ICLR 2027 / TMLR), `[preprint server]` SSRN, Research Square or Preprints.org.
* Items first posted before 2026 but **published in a 2026 venue** are marked `(preprint 2025)`.

---

## 1. Surveys, reviews and position papers

- **Geometric Algebra for Neural Networks: A Survey** — T. Zhong, K. Fong, K. Wang, C. Allen-Blanchette (Princeton / Stanford). `[arXiv]` 2026-10-21. <https://arxiv.org/abs/2610.30121>. The first full survey of GA-native networks; its companion list is <https://github.com/CAB-Lab-Princeton/AwesomeGANN>. The best single starting point.
- **Review of Clifford algebra-based AI methods in signal analysis** — R. Byrtus, F. Prinzi, P. Vašík, S. Vitabile. `[journal]` Phil. Trans. R. Soc. A 384(2326), 2026-08-13. <https://doi.org/10.1098/rsta.2025.0099>. Review of Clifford-algebra neural methods for signals with a medical-imaging focus; part of the theme issue *Modern applications of geometric algebra*.
- **Merits of geometric algebra applied to cryptography and machine learning** — D. W. Silva. `[journal]` Phil. Trans. R. Soc. A, 2026-08-13. <https://doi.org/10.1098/rsta.2025.0107>. Clifford fully homomorphic encryption plus "geometric neural networks" that classify encrypted `Cl(3,0)` point clouds (multivector weights instead of matrices).
- **One algebra for all: geometric algebra methods for neurosymbolic XR scene authoring, animation and neural rendering** — M. Kamarianakis, A. Protopsaltis, G. Papagiannakis. `[journal]` Phil. Trans. R. Soc. A, 2026-08-13. <https://doi.org/10.1098/rsta.2025.0109>. Position paper; GA for neural rendering and generative-AI scene editing.
- **Generative geospatial modelling with geometric algebra** — Phil. Trans. R. Soc. A, 2026-08-13. <https://doi.org/10.1098/rsta.2025.0103>. GA as a bridge to large-language-model spatial reasoning.
- **Overview of W. K. Clifford's geometric algebras** — E. Hitzer et al. `[journal]` Phil. Trans. R. Soc. A, 2026-08-13. <https://doi.org/10.1098/rsta.2025.0094>. Background overview that covers the algebras used in deep learning (PGA, CGA, spacetime algebra).
- **Toward a Functional Geometric Algebra for Natural Language Semantics** — J. Pustejovsky. `[arXiv]` 2026-04-28. <https://arxiv.org/abs/2604.25902>. Argues that Clifford algebras should replace linear algebra as the substrate of semantic representations in neural language models.
- **Geometric Structures for Learning and Optimization in Robotics** — S. Calinon. `[journal]` Annual Review of Control, Robotics, and Autonomous Systems, 2026-05-05. <https://doi.org/10.1146/annurev-control-022624-013742>. Review; geometric algebra is one of its three frameworks for robot skill learning.
- **Unveiling Hidden Convexity in Deep Learning: A sparse signal processing perspective** — E. Zeger, M. Pilanci et al. `[journal]` IEEE Signal Processing Magazine, May 2026. <https://doi.org/10.1109/msp.2026.3677783>. Shows how geometric-algebra (wedge-product) structure emerges in the convex reformulation of deep networks.
- **Advanced Graph Neural Networks for Smart Mining: A Systematic Literature Review of Equivariant, Topological, Symplectic, and Generative Models** — L. Rojas, L. Jorquera, J. Garcia. `[journal]` Mathematics 14(5):763, 2026-02-25. <https://doi.org/10.3390/math14050763>. PRISMA review whose scope explicitly includes Clifford (geometric algebra) GNNs.
- **Symmetry group equivariant convolutions for representation learning: a survey** — R. Basheer, D. Mishra. `[journal]` Int. J. Data Science and Analytics 22(1), vol. dated 2026 (online 2025-12-29). <https://doi.org/10.1007/s41060-025-00994-7>. General equivariance survey listed by the GA survey above.
- **A Comprehensive Review of CliffordNet** — Mamun. Zenodo, Sept 2026. <https://zenodo.org/records/22992570>. Secondary review of one paper; low weight.

## 2. New GA / Clifford architectures (general purpose)

- **CliffordNet: All You Need is Geometric Algebra** — Z. Ji. `[arXiv]` 2026-01-11 (v2 later). <https://arxiv.org/abs/2601.06793>. Vision backbone where the geometric product `uv = u·v + u∧v` is the only mixing operator; linear complexity; code <https://github.com/ParaMind2025/CAN>.
- **Clifford Kolmogorov-Arnold Networks (ClKAN)** — M. Wolff, F. Alesiani, C. Duhme, X. Jiang. `[arXiv]` 2026-02-05, `[conf]` IJCNN 2026. <https://arxiv.org/abs/2602.05977>, <https://doi.org/10.1109/ijcnn69854.2026.11728617>. KAN layers with Clifford-valued spline functions; randomized quasi-Monte-Carlo grids to tame the `2^n` blow-up.
- **Versor: A Geometric Sequence Architecture** — T. M. Huy, E. Hirst. `[arXiv]` 2026-02-10. <https://arxiv.org/abs/2602.10195>. CGA (`Cl(4,1)`) states evolved by rotors; Recursive Rotor Accumulator (linear-time recurrence) and Geometric Product Attention; custom bit-masked Clifford kernels. Code <https://github.com/VersorAI/Versor>.
- **Clifford Sheaf Neural Networks (CSNN)** — K. Kamiya, J. Nicholls. `[arXiv]` 2026-10-01. <https://arxiv.org/abs/2610.01322>. Cellular-sheaf network whose stalks are Clifford algebras; restriction maps are K-term sandwich products, giving an equivariant, grade-mixing sheaf Laplacian.
- **Conditional Clifford-Steerable CNNs for PDE Modeling** — B. L. Szarvas, M. Zhdanov. `[conf]` ICML 2026, PMLR 306 (preprint 2025, arXiv 2510.14007, v2 2026). <https://arxiv.org/abs/2510.14007>, <https://openreview.net/forum?id=LJ8rxwl9N6>. Clifford-steerable convolutions whose kernels are conditioned on the input multivector field; `O(p,q)` / `E(p,q)` equivariant.
- **Hypercomplexification of U-Net by Clifford Algebra Embeddings** — R. Byrtus, F. Prinzi, P. Vašík, S. Vitabile. `[conf]` LNCS (ENGAGE 2025 proceedings), published 2026-06-04. <https://doi.org/10.1007/978-3-032-22539-9_5>. U-Net with Clifford-algebra embedding blocks; breast-cancer segmentation on DCE-MRI.
- **Projective Geometric Algebra Graph Neural Networks: Edge-Modulated Message Passing via the Geometric Product (PGA-GNN)** — anonymous. `[OpenReview sub]` TMLR, 2026-09-19. <https://openreview.net/forum?id=zA86PE77B9>. Strictly `E(3)`-equivariant message passing where edge messages are geometric products in PGA; a local alternative to GATr's global attention.
- **Wider Groups, Fewer Weights: Efficient Pseudo-Orthogonal Equivariance via Geometric Algebras** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-18. <https://openreview.net/forum?id=chx1AMpGVP>. Builds equivariant layers for groups larger than the Clifford group that act identically to `O(p,q)`, trading expressiveness for far fewer parameters (follow-up to GLGENN).
- **TS-CAN: Time Series Forecasting via the Clifford Geometric Product** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-19. <https://openreview.net/forum?id=ruus1Sc1Uq>. Replaces dot-product interactions in forecasters with the geometric product (in-phase + quadrature components).
- **ALGEBRAIC INTERACTION: A Recipe for Neural Architecture Generation** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-17. <https://openreview.net/forum?id=8RRovMh8Vr>. Swaps the "feature algebra" of linear attention, Mamba and equivariant nets; Clifford algebras are one of the instantiations.
- **Rotors: Lightweight Geometric Recurrence for Edge Intelligence** — anonymous. `[OpenReview sub]` ICLR 2027 (desk rejected), 2026-09-18. <https://openreview.net/forum?id=wMspnEY5us>. Sequence model whose state is rotated on cooperating `SO(D)` manifolds; `O(1)` inference memory.
- **Rotor Recurrences: What Compact Recurrent States Can Predict** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-18. <https://openreview.net/forum?id=pqFoZLIy1e>. Theory: sharp prediction limits for rotor-transition recurrences in 3 and 4 dimensions.
- **Path-Locked Neural Composition: Exact Evaluation-Path Invariance with Geometric Algebra** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-16. <https://openreview.net/forum?id=bkIXwcoMQ2>. Learned motion calibration composed with exact PGA motor products; visual odometry on ETH3D.
- **Standard Attention as a Small-Angle Limit of Riemannian Geometric Algebra Transformers** — Z. Joseph. `[preprint server]` Research Square, 2026-02-11. <https://doi.org/10.21203/rs.3.rs-8811263/v1>. Attention derived as the small-angle limit of rotor flow on `Spin(3)` inside `Cl(3,0)`.
- **Weyl Algebra Equivariant Neural Networks** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-19. <https://openreview.net/forum?id=IFp4LqmXqP>. Not Clifford-based, but positioned explicitly against Clifford-algebra equivariance (symplectic / differential-operator symmetries that Clifford algebras cannot express).

## 3. Theory and analysis of GA layers

- **When Do Geometric Algebra Layers Beat Scalarization? A Controlled Study on SO(3)-Equivariant Vector Laws** — F. Polly. `[arXiv]` 2026-07-07. <https://arxiv.org/abs/2607.06634>. Controlled comparison of `Cl(3,0)` networks against invariant-scalar MLP baselines; GA wins only when the target composes group elements in depth.
- **Grade Discovery as an Identifiability Diagnostic for Clifford-Valued Features** — anonymous. `[OpenReview sub]` TMLR (rejected), 2026-05-26. <https://openreview.net/forum?id=8Q9Xmttlum>. Tests whether the grade (scalar / vector / bivector / pseudoscalar) of a feature channel can be recovered from data.
- **On Lie Groups Preserving Subspaces of Degenerate Clifford Algebras** — E. Filimoshina, D. Shirokov. `[arXiv]` 2026-01-12, `[journal]` Advances in Applied Clifford Algebras 2026-01-10. <https://arxiv.org/abs/2601.07191>, <https://doi.org/10.1007/s00006-025-01431-5>. Lie groups in degenerate (PGA-type) algebras, motivated by building equivariant networks.
- **Unveiling Hidden Convexity in Deep Learning** — see §1.

## 4. Scientific machine learning: molecules, materials, PDEs, fluids, geoscience

- **CliffordIP: Clifford algebra equivariant interatomic potentials for heterogeneous catalysis** — C. Polat, E. Serpedin, M. Kurban, H. Kurban. `[journal]` npj Computational Materials, 2026-08-04. <https://doi.org/10.1038/s41524-026-02259-8>. Message-passing potential on `Cl(3,0)` 8-dimensional multivectors, `O(3)`-equivariant via `Pin(3)`; OC20 / OC22 benchmarks.
- **Geometric Algebra Meets Cartesian Tensors: Higher-Order Equivariance for Interatomic Potentials (CliffordSTF)** — C. Polat, E. Serpedin, M. Kurban, H. Kurban. `[arXiv]` 2026-06-28. <https://arxiv.org/abs/2606.29584>. Couples Clifford multivectors to symmetric-traceless tensor tracks to fix the poor force directions of `L ≤ 1` Clifford potentials.
- **Design and Evaluation of a Geometric Algebra-Based Graph Neural Network for Molecular Property Prediction** — K. H. Petersen, M. N. Schmidt. `[conf]` NLDL 2026, PMLR 307. <https://proceedings.mlr.press/v307/petersen26a.html>.
- **When Does Equivariance Help? Canonical Alignment in Neural Fluid Surrogates** — P. Rygiel, J. Suk, K. K. Yeung, C. Brune, J. M. Wolterink. `[arXiv]` 2026-05-12; `[OpenReview sub]` ICLR 2027. <https://arxiv.org/abs/2605.18816>. Compares GATr-type equivariant surrogates with canonically aligned non-equivariant ones on vascular CFD meshes.
- **Geometric deep learning for local growth prediction on abdominal aortic aneurysm surfaces** — D. Alblas, P. Rygiel, J. Suk, ..., J. M. Wolterink. `[journal]` Artificial Intelligence in Medicine 172:103323, Feb 2026. <https://doi.org/10.1016/j.artmed.2025.103323>. PGA / GATr-based model on aneurysm surface meshes.
- **Learning Low-Order Approximations of the Quark Propagator in Lattice QCD** — anonymous. `[OpenReview sub]` ICLR 2027, 2026-09-16. <https://openreview.net/forum?id=GuDDklAx8s>. Gauge-equivariant networks expressed in the Clifford (gamma-matrix) basis for scientific discovery.
- **A Geometric Algebra-Inspired Spatio-temporal Learning Model for Sea Level Variability Reconstruction Under Incomplete Observations** — D. Li, Y. Huang, C. Huang, W. Luo, Z. Yu, L. Yuan. `[journal]` Advances in Applied Clifford Algebras, 2026-09-11. <https://doi.org/10.1007/s00006-026-01457-3>.
- **Geometric Algebra-Based Time Series Reconstruction of Sea Level Variability (Northeast Pacific, 1993–2023)** — Y. Huang, D. Li, W. Luo, Z. Yu, L. Yuan. `[conf]` LNCS (ENGAGE 2025 proceedings), 2026. <https://doi.org/10.1007/978-3-032-22539-9_9>.
- **Conditional Clifford-Steerable CNNs for PDE Modeling** — see §2.
- **Causal Discovery using Geometric Algebra** — A. Sudjianto, S. Setiawan. `[preprint server]` SSRN 2026. <https://doi.org/10.2139/ssrn.7291638>. Pairwise causal-direction scoring recast with outer products and multivariate angles.

## 5. High-energy physics (GA transformers, L-GATr line)

Papers whose own method is a geometric-algebra model:

- **Lorentz-Equivariant Geometric Algebra Transformers for Top-Quark Event Reconstruction** — E. Abasov, L. Dudko, A. Markina, M. Perfilov, P. Volkov, G. Vorotnikov, A. Zaborenko. `[arXiv]` 2026-10-21. <https://arxiv.org/abs/2610.29678>. L-GATr on `tt̄/tWb` separation, FCNC detection and full event reconstruction.
- **Geometric algebra as the input language of collider foundation models** — E. Abasov et al. `[arXiv]` 2026-05-15. <https://arxiv.org/abs/2605.15910>. Represents a whole collider event as one `Cl(1,3)⊗V_flav` multivector; theorem on which Lorentz invariants higher grades add; L-GATr-type transformer.
- **What Do Lorentz-Equivariant Jet Taggers Learn?** — `[arXiv]` 2026-06-19; `[conf]` ICML 2026 AI4Physics workshop. <https://arxiv.org/abs/2606.21790>. Probes, equivariance tests and grade ablations on L-GATr, L-GATr-slim and LLoCa-T.
- **Virtues and Vices of Equivariant Transformers** — L. Favaro, T. Plehn, H. Qu, J. Spinner. `[arXiv]` 2026-08-03. <https://arxiv.org/abs/2608.02735>. Lorentz-equivariant (spacetime-algebra) transformers for large-scale jet and flavour tagging, inference-optimised.
- **Explicit or Implicit? Encoding Physics at the Precision Frontier** — V. Bresó-Pla, K. Greif, V. Mikuni, B. Nachman, T. Plehn, T. Wamorkar, D. Whiteson. `[arXiv]` 2026-03-09; `[journal]` Phys. Rev. D 114, 036034 (2026). <https://arxiv.org/abs/2603.08802>. Explicit symmetry (L-GATr) versus large-scale pre-training.
- **QGrAT: Quantum Grade-Aware Transformer for Jet Tagging** — `[arXiv]` 2026-10-12. <https://arxiv.org/abs/2610.17351>. Quantum transformer whose features are organised by the grades of the spacetime Clifford algebra.
- **How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?** — J. Agarwal, S. Khare, D. Kumar. `[arXiv]` 2026-10-21. <https://arxiv.org/abs/2610.29439>. Distils into L-GATr-slim students (20k parameters).
- **End-to-end event reconstruction for precision physics at future colliders** — D. Garcia, L. Herrmann, G. Krzmanc, M. Selvaggi. `[arXiv]` 2026-03-04. <https://arxiv.org/abs/2603.04084>. GATr networks combined with object-condensation clustering.
- Late-2025 precursor worth knowing: **Economical Jet Taggers – Equivariant, Slim, and Quantized** — A. Petitjean, T. Plehn, J. Spinner, U. Köthe. arXiv 2512.17011 (Dec 2025); introduces L-GATr-slim used by several 2026 papers.

2026 HEP papers that cite or benchmark against L-GATr but whose own method is not GA-based (from the INSPIRE-HEP citation graph; listed for completeness): 2601.00950, 2601.11716, 2601.11719, 2601.13308, 2601.16417, 2602.03728, 2602.06496, 2602.06895, 2602.15781, 2602.24282, 2603.02310, 2603.22407, 2604.02415, 2605.12229, 2605.16036, 2605.21789, 2605.22891, 2605.26821, 2606.04165, 2606.14854, 2606.23791, 2606.26228, 2606.27420, 2607.01354, 2607.16144, 2607.23377, 2607.23591, 2608.05812, 2608.06450, 2608.13633, 2608.16061, 2608.21509, 2608.28449, 2609.09785, 2609.26899, 2609.32874, 2610.06784, 2610.09320, 2610.12408, 2610.15919, 2610.17177, 2610.27106, 2610.30245.

## 6. Vision, remote sensing and medical imaging

- **CliffordNet** — see §2.
- **UHD Low-Light Image Enhancement via Real-Time Enhancement Methods with Clifford Information Fusion** — X. Wang, C. Wu, D. Zhao, G. Gao et al. `[arXiv]` 2026-04-10. <https://arxiv.org/abs/2604.09321>. Feature fusion in a 2D Clifford multivector space (scalar, vector, bivector) with Clifford similarity.
- **Geometric algebra transformer with spectral-spatial-volumetric feature extraction for hyperspectral image classification (H-GATr)** — J. Fabi, C. Schiavella, I. Amerini. `[journal]` Pattern Recognition Letters, April 2026. <https://doi.org/10.1016/j.patrec.2026.02.001>. Weight-free multivector embedding of spectral / spatial / volumetric cues fed to a GATr.
- **Geometric algebra enhanced twin-branch contrastive network design for few-shot hyperspectral image classification** — R. Zhang, W. Cao, Q. Liu. `[journal]` Pattern Recognition, Nov 2026. <https://doi.org/10.1016/j.patcog.2026.113549>.
- **Geometric Algebra-based Dual-branch Graph Convolutional Network for Deformable Medical Image Registration with Bi-Directional Flow** — W. Cao, M. Anwar, Z. Yan. `[journal]` IEEE J. Biomedical and Health Informatics, 2026. <https://doi.org/10.1109/jbhi.2026.3718748>.
- **GIENet: Graph-Infused Encoder with Geometric Algebra Single-Pass Fusion for Unsupervised Deformable Registration** — Y. Zhou, W. Cao, W. Feng, Y. Zhang. `[preprint server]` SSRN 2026. <https://doi.org/10.2139/ssrn.6205817>.
- **Conformal Geometric Algebra Graph Learning Theory for 3D Human Skeleton Action** — W. Cao, B. Wang, J. Hu. `[conf]` ECTI DAMT & NCON 2026, 2026-02-04. <https://doi.org/10.1109/ectidamtncon67592.2026.11459996>.
- **Conformal geometric algebra graph learning theory for non-Euclidean data** — W. Cao, B. Wang. `[conf]` ICCAIS 2025 (SPIE proceedings published 2026-04-24). <https://doi.org/10.1117/12.3112939>.
- **Hypercomplexification of U-Net by Clifford Algebra Embeddings** — see §2.
- **Preprocessing of Retinal Fundus Image Using Clifford Algebra** — S. Roy, A. Mitra, S. K. Setua. `[preprint server]` Preprints.org, 2026-05-11. <https://doi.org/10.20944/preprints202605.0666.v1>. Clifford colour space and rotor-based illumination correction as a preprocessing stage for diagnosis pipelines (not itself a neural network).

## 7. Graphs, multimodal and representation learning

- **LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning** — X. Li, Z. Wu, Z. Chen, H. Sun, D. Su, G. Zeng, H. Qin, R.-H. Li, G. Wang. `[arXiv]` 2026-01-29 (2601.21453) and re-posted 2026-08-25 (2608.24795). <https://arxiv.org/abs/2601.21453>, <https://arxiv.org/abs/2608.24795>. Modality-aware geometric manifold grounded in Clifford algebra; decoupled propagate-then-aggregate GNN.
- **ICE: Task-Aligned Clifford Latent Fields for Multimodal Graph Foundation Models** — `[arXiv]` 2026-09-24; `[OpenReview sub]` ICLR 2027. <https://arxiv.org/abs/2609.29398>. Higher-grade channels recover pair relations across foundation graphs.
- **Clifford Sheaf Neural Networks**, **PGA-GNN** — see §2.
- **Recursive Binding on a Budget: Subspace Carving in Order-p Tensor Memories** — T. Pence, D. Yamada, V. Singh. `[arXiv]` 2026-06-10; `[OpenReview sub]` ICML 2026. <https://arxiv.org/abs/2606.11391>. Shows tensor-product representations are a special case of binding in a Clifford algebra and gives a Clifford formulation of the method.
- **Semantic Plane Alignment: Measuring Summarization Completeness via Tensor-based Geometric Algebra** — A. Sudjianto. `[preprint server]` SSRN 2026. <https://doi.org/10.2139/ssrn.6168708>. GA (planes spanned by embeddings) as an evaluation metric for summaries.
- **Toward a Functional Geometric Algebra for Natural Language Semantics** — see §1.

## 8. Robotics, autonomous driving and control learning

- **Efficient Equivariant Transformer for Self-Driving Agent Modeling (DriveGATr)** — S. Xu, D. Chen, K. Wong, C. Zhang, K. Fallah, R. Urtasun (Waabi). `[arXiv]` 2026-04-01; `[conf]` CVPR 2026. <https://arxiv.org/abs/2604.01466>, <https://cvpr.thecvf.com/virtual/2026/poster/36432>. Scene elements as multivectors of the 2D PGA `R*_{2,0,1}`; `SE(2)`-equivariant attention without pairwise relative encodings.
- **Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning (hPGA-DP)** — X. Sun, Y. Wang, S. Yang, Y. Chen, D. Rakita. `[conf]` ICRA 2026 (preprint 2025, arXiv 2507.05695). <https://doi.org/10.1109/icra57385.2026.11696239>, <https://arxiv.org/abs/2507.05695>. P-GATr encoder / decoder around a standard diffusion denoiser.
- **Path-Locked Neural Composition** — see §2.
- **From Language to Task Maps: Compiling Semantic Relations While Preserving Task-Relevant Freedom** — J. Park, J. Lee, J. Lee, S. Hwang. `[arXiv]` 2026-09-28. <https://arxiv.org/abs/2609.34412>. Compiles language-specified relations into CGA task maps composed with learned policies (GA-adjacent).
- **The Cartan-Topos Protocol** — M. Hernández, E. Sánchez-Soto. `[arXiv]` 2026-05-30. <https://arxiv.org/abs/2606.00714>. Clifford rotors / motors for `SE(3)` pose synchronisation in multi-agent coordination (not deep learning).
- **Geometric Structures for Learning and Optimization in Robotics** — see §1.
- GA robotics papers from 2026 that are *not* learning-based, for context: CGA inverse kinematics of 3R robots (<https://arxiv.org/abs/2609.00311>, <https://arxiv.org/abs/2609.00316>); GA-based optimal control with analytical derivatives (IEEE/ASME TMECH, <https://doi.org/10.1109/tmech.2026.3727499>); PGA base-parameter identification (IJRR, <https://doi.org/10.1177/02783649261431352>); Clifford-algebra snake-robot kinematics (J. Franklin Inst., <https://doi.org/10.1016/j.jfranklin.2025.108297>); PGA flexible multibody dynamics (<https://doi.org/10.1007/s11044-026-10154-9>).
- Related but **not** GA: *Screw Attention: Rigid-Body Algebra Inside a Transformer* (<https://arxiv.org/abs/2610.00904>) uses screw theory / spatial transforms, not Clifford algebra.

## 9. Wireless channel modelling and neural fields

- **A Geometric Algebra-informed NeRF Framework for Generalizable Wireless Channel Prediction (GAI-NeRF)** — J. Shen, L. L. Enamorado, S. Mao, X. Wang. `[arXiv]` 2026-04-13; `[conf]` IEEE INFOCOM 2026. <https://arxiv.org/abs/2604.11983>, <https://doi.org/10.1109/infocom59046.2026.11571298>. GA attention for ray–object interactions inside a neural radiance field.
- **A Geometric Algebra-Informed 3DGS Framework for Wireless Channel Prediction (GAI-GS)** — same group. `[arXiv]` 2026-05-18; `[OpenReview sub]` CVPR 2026. <https://arxiv.org/abs/2605.19065>. Same idea on 3D Gaussian splatting.

## 10. Sequence models, language and LLM tooling

- **Versor**, **TS-CAN**, **Rotors**, **Rotor Recurrences**, **Riemannian GA transformers** — see §2.
- **GA-VisAgent: A Multi-Agent application for code generation and visualization in interactive learning** — W. Jian, Z. Jianbo, X. Yuhao, L. Zhenxia. `[arXiv]` 2026-05-02. <https://arxiv.org/abs/2605.01299>. LLM agents (built on a GA-specialised language model, GAGPT) that generate and visualise geometric-algebra code.
- **Adaptive Domain Models: Bayesian Evolution, Warm Rotation, and Principled Training for Geometric and Neuromorphic AI** — H. Haynes. `[arXiv]` 2026-03-18 (v7). <https://arxiv.org/abs/2603.18104>. Hybrid design with grade-typed Clifford representations for geometry and spiking layers for temporal patterns.
- **Generative geospatial modelling with geometric algebra**, **Functional GA for NL semantics**, **Semantic Plane Alignment** — see §1 and §7.

## 11. Clifford-valued neural-network dynamics (stability and synchronisation)

This is the control-theory stream: recurrent Hopfield / BAM / Cohen–Grossberg / inertial networks whose states and
weights are Clifford numbers, analysed with Lyapunov methods. Not deep learning, but squarely "Clifford algebra
neural networks". 2026 items found:

- G. Rajchakit — *A restart-graded L1 scheme for fractional-order Clifford-valued delay neural networks*, Nonlinear Science, Dec 2026. <https://doi.org/10.1016/j.nls.2026.100203>
- G. Rajchakit — *Observer-Based Output-Feedback Synchronization of Fractional-Order Fuzzy Clifford-Valued Neural Networks*, Complex Systems Stability & Control, 2026-10-23. <https://doi.org/10.53941/cssc.2026.100028>
- G. Rajchakit — *Stability and Control of Fractional-Order Fuzzy Stochastic Clifford-Valued Neural Networks with Hybrid Markovian Switching, Cyber-Attacks, and Event-Triggered Multi-Agent Synchronization*, Nonlinear Analysis and Computer Simulations, 2026-10-08. <https://doi.org/10.53941/nacs.2026.100016>
- G. Rajchakit, C. P. Lim — *Exponential lag synchronization of inertial Clifford-valued neural networks ... applications to multi-spectral image watermarking*, Neurocomputing, Oct 2026. <https://doi.org/10.1016/j.neucom.2026.134343>
- G. Rajchakit, C. P. Lim — *Global Mittag–Leffler synchronization of fractional-order Clifford-valued neural networks ... adaptive quantized pinning control*, Int. J. Dynamics and Control, 2026-08-07. <https://doi.org/10.1007/s40435-026-02228-5>
- G. Rajchakit, C. P. Lim, M. M. Tajudeen, T. Huang — *Quasi-Projective Synchronization of Heterogeneous Fractional-Order Clifford-Valued Neural Networks via Intermittent Pinning Control*, Int. J. Robust and Nonlinear Control, 2026-09-07. <https://doi.org/10.1002/rnc.70702>
- G. Rajchakit, C. P. Lim, M. M. Tajudeen, T. Huang — *H∞ synchronisation of fractional-order Clifford-valued reaction–diffusion neural networks with leakage delays*, Int. J. Systems Science, 2026-10-21. <https://doi.org/10.1080/00207721.2026.2736095>
- R. Sriraman, N. Manoj — *Finite-time stabilization of fractional-order fuzzy Clifford-valued neural networks using novel inequalities*, Physica Scripta, 2026-07-02. <https://doi.org/10.1088/1402-4896/ae8133>
- N. Manoj, R. Sriraman — *Relaxed synchronization analysis of fractional-order Clifford-valued delayed neural networks via novel inner product-based inequality*, Physica Scripta, 2026-10-20. <https://doi.org/10.1088/1402-4896/aeb198>
- N. Manoj, R. Sriraman — *Projective Synchronization of Fractional-Order BAM Clifford-Valued Delayed Memristive Neural Networks*, Math. Methods Appl. Sci., 2026-02-19. <https://doi.org/10.1002/mma.70592>
- S. Ramalingam, M. Nagappan — *Finite-time stability of fractional-order delayed Clifford-valued neural networks via a novel non-separation approach*, Comput. Appl. Math., 2026-03-10. <https://doi.org/10.1007/s40314-026-03690-9>
- M. Nagappan et al. — *Synchronization dynamics of fractional-order Clifford-valued BAM delayed neural networks and their applications*, J. Math. Computer Sci., 2026-09-23. <https://doi.org/10.22436/jmcs.043.03.02>
- S. Kumari, M. Malik — *Slide Mode Control Based Finite and Fixed-Time Projective Synchronization for Delayed Clifford-Valued Neural Networks*, Math. Methods Appl. Sci., 2026-07-30. <https://doi.org/10.1002/mma.70915>
- D. Agrawal, S. Abbas, A. Debbouche — *Aperiodic intermittent control strategy for synchronization of Clifford-valued neural networks on time scales using matrix measure*, Appl. Math. Comput., Sept 2026. <https://doi.org/10.1016/j.amc.2026.130034>
- D. Agrawal, S. Abbas — *A unified matrix measure strategy for QGPL synchronization of Clifford-valued competitive model with parameter mismatch: Application to LSTM model*, Neurocomputing, June 2026. <https://doi.org/10.1016/j.neucom.2026.133377>
- Y. Deng, H. Li, M. Tan — *Synchronization of neutral-type Clifford-valued inertial neural networks with mixed delays by non-reduced-order and non-separation approaches*, Neurocomputing, Dec 2026. <https://doi.org/10.1016/j.neucom.2026.134991>
- Q. Yang, Y. Qiao, L. Duan, J. Miao — *Multistability of equilibria for Clifford-valued Cohen–Grossberg neural networks with discontinuous activation functions and time delays*, Math. Comput. Simulation, July 2026. <https://doi.org/10.1016/j.matcom.2025.12.015>
- C. Aouiti, F. Dridi — *Advances in the Mathematical Theory of WPAA Dynamics for Impulsive High Order Neural Systems in Clifford Algebras*, J. Nonlinear Dynamics and Applications, 2026-01-21. <https://doi.org/10.62762/jnda.2025.838385>
- C. Li, Y. Zhang, K. I. Kou, Y. Liu — *Fixed-time synchronization of Clifford biquaternion neural networks with two-sided coefficients and application to multispectral image protection*, Neural Networks, vol. dated Jan 2027 (online 2026). <https://doi.org/10.1016/j.neunet.2026.109528>
- Y. Li, R. Xu — *Global Exponential Weyl Almost Periodic Synchronization for Hypercomplex-Valued High-Order Hopfield Neural Networks with Delays*, Qualitative Theory of Dynamical Systems, Feb 2026. <https://doi.org/10.1007/s12346-026-01459-3>
- M. Kobayashi — *Noise Robust Projection Rule for Split-Quaternion-Valued Hopfield Networks*, IEICE Trans. Fundamentals, 2026. <https://doi.org/10.1587/transfun.2026map0002>. Split quaternions are a 4-dimensional Clifford algebra; the paper frames itself as Clifford-algebra neural networks.
- G. Rajchakit — *Global Fixed-Time Synchronization of Fractional-Order Dual-Split-Quaternion Valued Memristive Neural Networks*, Complex Systems Stability & Control, 2026-07-07. <https://doi.org/10.53941/cssc.2026.100013> (hypercomplex, Clifford-adjacent).

Quaternion-valued neural-network stability papers (around 20 found in 2026) are omitted because they do not
invoke Clifford algebra explicitly.

## 12. Hypercomplex-valued networks (Clifford-adjacent)

- **Ghost Features and Spooky Transfer Learning for Hypercomplex-Valued Neural Networks** — G. Vieira, M. E. Valle et al. `[arXiv]` 2026-08-07. <https://arxiv.org/abs/2608.07735>. Hypercomplex layers whose real part reproduces a trained real layer; imaginary parts add "ghost features". Code <https://github.com/mevalle/v-nets/>.
- **Fully tensorial approach to hypercomplex-valued neural networks** — A. Niemczynowicz, R. A. Kycia. `[journal]` Information Sciences, Nov 2026 (arXiv 2407.00449). <https://doi.org/10.1016/j.ins.2026.123796>. Unified tensor implementation covering Clifford algebras.
- **Hypercomplex Phase Retrieval** — `[arXiv]` 2026-02-27. <https://arxiv.org/abs/2602.23946>. Clifford-algebra signal processing (not a neural network).

## 13. Other machine-learning uses of geometric algebra

- **Adaptive Sphere Classification via Conformal Geometric Algebra** — I. Athmani. `[preprint server]` Research Square, 2026-03-31. <https://doi.org/10.21203/rs.3.rs-9244644/v1>. A classifier whose learning rule is the CGA point-in-sphere inner product.
- **Hybrid Geometric Algebra Correntropy Criterion** — H. Zhao, X. Hou, X. Long. `[book chapter]` *Information Theoretic Learning-based Filter*, Springer, 2026-10-06. <https://doi.org/10.1007/978-3-032-29623-8_7>. GA-valued adaptive filtering.
- **Machine Learning the Phase Structure and Algebraic Invariants of Clifford- and Grassmann-Valued Cellular Automata** — F. K. Mirza. `[preprint server]` Research Square, 2026-07-31. <https://doi.org/10.21203/rs.3.rs-10394430/v1>.
- **Quantum classification and search algorithms using spinorial representations** — `[arXiv]` 2026-03-17. <https://arxiv.org/abs/2603.16564>. Quantum algorithms built on Clifford algebras and spinors (GA-adjacent).
- **Geometric Algebra Quantum Gate Decomposition** — Amraoui, Toffano. `[journal]` Axioms 15(7):523, 2026-07-13. <https://doi.org/10.3390/axioms15070523>. Rotor-based gate synthesis (GA-adjacent, not learning).
- **QGrAT** — see §5.

## 14. Software and resources appearing in 2026

- AwesomeGANN curated list (Princeton CAB Lab): <https://github.com/CAB-Lab-Princeton/AwesomeGANN>
- Versor kernels: <https://github.com/VersorAI/Versor>; CliffordNet: <https://github.com/ParaMind2025/CAN>
- Phil. Trans. R. Soc. A theme issue *Modern applications of geometric algebra*, vol. 384 issue 2326 (Aug 2026), preface <https://doi.org/10.1098/rsta.2026.0174>, includes a GA implementation / memory-requirements overview <https://doi.org/10.1098/rsta.2025.0104>.

---

## Name collisions deliberately excluded

* "Clifford" quantum circuits, Clifford gates, Clifford group in the stabilizer sense, Clifford+T synthesis.
* The Clifford torus (e.g. *Learning Holographic Reduced Representations with Clifford Variational Autoencoders*, arXiv 2609.28409, projects onto a Clifford torus, not a Clifford algebra).
* "Multivector" retrieval / reranking (ColBERT-style token embeddings) and "multivector" model-predictive control of electric drives.
* Rotor machines, rotorcraft, kicked rotors.

## How this list was built and known gaps

Sources queried (all restricted to 2026): the arXiv API with ~85 keyword variants (1,646 candidates, filtered
by GA and learning keywords and read by hand), OpenReview search (ICLR 2026/2027, ICML 2026, CVPR 2026, TMLR,
workshops), Crossref (~3,000 records, filtered), INSPIRE-HEP citation graph of L-GATr / GATr / CGENN, alphaXiv
and Consensus semantic search, the reference list of the October 2026 GA survey and its AwesomeGANN companion
list, and targeted web searches. Semantic Scholar, OpenAlex and dblp were rate-limited from this network, so
the citation graph outside high-energy physics (papers that use GATr or CGENN without naming them in title or
abstract) is the main known gap. Anonymous OpenReview submissions are listed because they are public, but they
are not peer reviewed.
