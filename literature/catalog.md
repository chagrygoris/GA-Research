# Clifford / geometric algebra deep learning: paper catalogue

Generated from `literature/papers.yaml` on 2026-10-31 by `python -m ga_lit render`. Do not edit by hand; edit the YAML or use the CLI.

**290 accepted papers** (125 core, 108 adjacent, 57 context), 0 waiting in the inbox. Browse interactively: `literature/site/index.html`.

Relevance: **core** = geometric algebra inside the model or method; **adjacent** = GA-related machine learning or a GA method next to a learned component; **context** = cites or benchmarks against GA models without using one.

## 2026

### Surveys, reviews and position papers

Overviews of geometric-algebra (GA) neural networks and essays arguing for GA as a representation.

- **Geometric Algebra for Neural Networks: A Survey** — Zhong, Fong, Wang, Allen-Blanchette. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.30121) · [code](https://github.com/CAB-Lab-Princeton/AwesomeGANN)
  First full survey of GA-native neural networks (Princeton / Stanford); its companion AwesomeGANN list is the best single starting point. — also: Software and resources
- **A Comprehensive Review of CliffordNet: Geometric Algebra as the Foundation for Efficient Vision Architectures** *(context)* — Mamun. *Zenodo, 2026-09-01.* [url](https://zenodo.org/records/22992570)
  Secondary review of a single paper (CliffordNet); low weight. — also: Vision, remote sensing and medical imaging
- **One algebra for all: geometric algebra methods for neurosymbolic XR scene authoring, animation and neural rendering** *(adjacent)* — Kamarianakis, Protopsaltis, Papagiannakis. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [arxiv](https://arxiv.org/abs/2511.15398) · [doi](https://doi.org/10.1098/rsta.2025.0109)
  Position paper on GA for neural rendering and generative-AI scene editing in extended reality.
- **Merits of geometric algebra applied to cryptography and machine learning** — Silva. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0107)
  Clifford fully homomorphic encryption plus "geometric neural networks" that classify encrypted Cl(3,0) point clouds with multivector weights instead of matrices. — also: New GA / Clifford architectures
- **Review of Clifford algebra-based AI methods in signal analysis** — Byrtus, Prinzi, Vašík, Vitabile. *Philosophical Transactions of the Royal Society A 384(2326), 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0099)
  Review of Clifford-algebra neural methods for signals with a medical-imaging focus; part of the theme issue "Modern applications of geometric algebra". — also: Vision, remote sensing and medical imaging
- **Overview of W. K. Clifford's geometric algebras** *(context)* — Hitzer. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0094)
  Background overview covering the algebras used in deep learning (projective, conformal, spacetime algebra).
- **Geometric Structures for Learning and Optimization in Robotics** *(adjacent)* — Calinon. *Annual Review of Control, Robotics, and Autonomous Systems, 2026-05-05.* [doi](https://doi.org/10.1146/annurev-control-022624-013742)
  Review of geometric approaches to robot skill learning; geometric algebra is one of its three frameworks. — also: Robotics, driving and control learning
- **Toward a Functional Geometric Algebra for Natural Language Semantics** — Pustejovsky. *arXiv, 2026-04-28.* [arxiv](https://arxiv.org/abs/2604.25902)
  Argues that Clifford algebras should replace linear algebra as the substrate of semantic representations in neural language models. — also: Sequence models, language and LLM tooling
- **Advanced Graph Neural Networks for Smart Mining: A Systematic Literature Review of Equivariant, Topological, Symplectic, and Generative Models** *(context)* — Rojas, Jorquera, Garcia. *Mathematics 14(5):763, 2026-02-25.* [doi](https://doi.org/10.3390/math14050763)
  PRISMA review whose scope explicitly includes Clifford (geometric algebra) graph neural networks. — also: Graphs, multimodal and representation learning
- **Symmetry group equivariant convolutions for representation learning: a survey** *(context)* — Basheer, Mishra. *International Journal of Data Science and Analytics 22(1) (online 2025-12-29), 2026-01-01.* [doi](https://doi.org/10.1007/s41060-025-00994-7)
  General equivariance survey listed by the October 2026 GA survey.

### New GA / Clifford architectures

General-purpose layers and models whose weights or activations are multivectors.

- **Clifford Sheaf Neural Networks** — Kamiya, Nicholls. *arXiv, 2026-10-01.* [arxiv](https://arxiv.org/abs/2610.01322)
  Cellular-sheaf network whose stalks are Clifford algebras; K-term sandwich restriction maps give an equivariant, grade-mixing sheaf Laplacian. — also: Graphs, multimodal and representation learning
- **Projective Geometric Algebra Graph Neural Networks: Edge-Modulated Message Passing via the Geometric Product** — anonymous. *TMLR submission, 2026-09-19.* [openreview](https://openreview.net/forum?id=zA86PE77B9)
  Strictly E(3)-equivariant message passing where edge messages are geometric products in projective GA; a local alternative to GATr's global attention. — also: Graphs, multimodal and representation learning
- **TS-CAN: Time Series Forecasting via the Clifford Geometric Product** — anonymous. *ICLR 2027 submission, 2026-09-19.* [openreview](https://openreview.net/forum?id=ruus1Sc1Uq)
  Replaces dot-product interactions in time-series forecasters with the geometric product (in-phase plus quadrature components). — also: Sequence models, language and LLM tooling
- **Weyl Algebra Equivariant Neural Networks** *(context)* — anonymous. *ICLR 2027 submission, 2026-09-19.* [openreview](https://openreview.net/forum?id=IFp4LqmXqP)
  Not Clifford-based, but positioned explicitly against Clifford-algebra equivariance (symplectic / differential-operator symmetries Clifford algebras cannot express). — also: Theory and analysis of GA layers
- **Rotors: Lightweight Geometric Recurrence for Edge Intelligence** — anonymous. *ICLR 2027 submission (desk rejected), 2026-09-18.* [openreview](https://openreview.net/forum?id=wMspnEY5us)
  Sequence model whose state is rotated on cooperating SO(D) manifolds; O(1) inference memory. — also: Sequence models, language and LLM tooling
- **Wider Groups, Fewer Weights: Efficient Pseudo-Orthogonal Equivariance via Geometric Algebras** — anonymous. *ICLR 2027 submission, 2026-09-18.* [openreview](https://openreview.net/forum?id=chx1AMpGVP)
  Equivariant layers for groups larger than the Clifford group that act identically to O(p,q), trading expressiveness for far fewer parameters. — also: Theory and analysis of GA layers
- **Algebraic Interaction: A Recipe for Neural Architecture Generation** — anonymous. *ICLR 2027 submission, 2026-09-17.* [openreview](https://openreview.net/forum?id=8RRovMh8Vr)
  Swaps the "feature algebra" of linear attention, Mamba and equivariant networks; Clifford algebras are one instantiation and improve linear attention accuracy.
- **Path-Locked Neural Composition: Exact Evaluation-Path Invariance with Geometric Algebra** — anonymous. *ICLR 2027 submission, 2026-09-16.* [openreview](https://openreview.net/forum?id=bkIXwcoMQ2)
  Learned motion calibration composed with exact projective-GA motor products; visual odometry on ETH3D. — also: Robotics, driving and control learning
- **Hypercomplexification of U-Net by Clifford Algebra Embeddings** — Byrtus, Prinzi, Vašík, Vitabile. *LNCS (ENGAGE 2025 proceedings), 2026-06-04.* [doi](https://doi.org/10.1007/978-3-032-22539-9_5)
  U-Net with Clifford-algebra embedding blocks; breast-cancer segmentation on dynamic contrast-enhanced MRI. — also: Vision, remote sensing and medical imaging
- **Conditional Clifford-Steerable CNNs for PDE Modeling** — Szarvas, Zhdanov. *ICML 2026 (PMLR 306; arXiv preprint Oct 2025), 2026-04-30.* [arxiv](https://arxiv.org/abs/2510.14007) · [openreview](https://openreview.net/forum?id=LJ8rxwl9N6)
  Clifford-steerable convolutions whose kernels are conditioned on the input multivector field; O(p,q) / E(p,q) equivariant PDE surrogates. Preprint 2025, published 2026. — also: Scientific machine learning
- **Versor: A Geometric Sequence Architecture** — Huy, Hirst. *arXiv, 2026-02-10.* [arxiv](https://arxiv.org/abs/2602.10195) · [code](https://github.com/VersorAI/Versor)
  Conformal GA Cl(4,1) states evolved by rotors; Recursive Rotor Accumulator (linear-time recurrence) and Geometric Product Attention; custom bit-masked Clifford kernels. — also: Sequence models, language and LLM tooling
- **Clifford Kolmogorov-Arnold Networks** — Wolff, Alesiani, Duhme, Jiang. *arXiv; IJCNN 2026, 2026-02-05.* [arxiv](https://arxiv.org/abs/2602.05977) · [doi](https://doi.org/10.1109/ijcnn69854.2026.11728617)
  Kolmogorov-Arnold network layers with Clifford-valued spline functions; randomized quasi-Monte-Carlo grids tame the 2^n blow-up of higher-dimensional algebras.
- **CliffordNet: All You Need is Geometric Algebra** — Ji. *arXiv, 2026-01-11.* [arxiv](https://arxiv.org/abs/2601.06793) · [code](https://github.com/ParaMind2025/CAN)
  Vision backbone where the geometric product uv = u·v + u∧v is the only mixing operator; linear complexity; 77.8% CIFAR-100 with 1.4M parameters. — also: Vision, remote sensing and medical imaging

### Theory and analysis of GA layers

Expressivity, identifiability, group-theoretic and optimisation studies of GA networks.

- **Rotor Recurrences: What Compact Recurrent States Can Predict** — anonymous. *ICLR 2027 submission, 2026-09-18.* [openreview](https://openreview.net/forum?id=pqFoZLIy1e)
  Sharp prediction limits for rotor-transition recurrences in three and four dimensions. — also: Sequence models, language and LLM tooling
- **When Do Geometric Algebra Layers Beat Scalarization? A Controlled Study on SO(3)-Equivariant Vector Laws** — Polly. *arXiv, 2026-07-07.* [arxiv](https://arxiv.org/abs/2607.06634)
  Controlled comparison of Cl(3,0) networks against invariant-scalar MLP baselines; GA wins by an order of magnitude only when the target composes group elements in depth.
- **Grade Discovery as an Identifiability Diagnostic for Clifford-Valued Features** — anonymous. *TMLR submission (rejected), 2026-05-26.* [openreview](https://openreview.net/forum?id=8Q9Xmttlum)
  Tests whether the grade (scalar / vector / bivector / pseudoscalar) of a feature channel can be recovered from paired observations.
- **Unveiling Hidden Convexity in Deep Learning: A sparse signal processing perspective** *(adjacent)* — Zeger, Pilanci. *IEEE Signal Processing Magazine, 2026-05-01.* [doi](https://doi.org/10.1109/msp.2026.3677783)
  Shows how geometric-algebra (wedge-product) structure emerges in convex reformulations of deep networks. — also: Surveys, reviews and position papers
- **Standard Attention as a Small-Angle Limit of Riemannian Geometric Algebra Transformers** — Joseph. *Research Square, 2026-02-11.* [doi](https://doi.org/10.21203/rs.3.rs-8811263/v1)
  Dot-product attention derived as the small-angle limit of rotor flow on Spin(3) inside Cl(3,0). — also: New GA / Clifford architectures
- **On Lie Groups Preserving Subspaces of Degenerate Clifford Algebras** *(adjacent)* — Filimoshina, Shirokov. *Advances in Applied Clifford Algebras; arXiv, 2026-01-10.* [arxiv](https://arxiv.org/abs/2601.07191) · [doi](https://doi.org/10.1007/s00006-025-01431-5)
  Lie groups in degenerate (projective-GA-type) algebras, motivated by building equivariant networks.

### Scientific machine learning

Molecules, materials, PDE surrogates, fluids, geoscience and lattice field theory.

- **Learning Low-Order Approximations of the Quark Propagator in Lattice QCD** *(adjacent)* — anonymous. *ICLR 2027 submission, 2026-09-16.* [openreview](https://openreview.net/forum?id=GuDDklAx8s)
  Gauge-equivariant networks expressed in the Clifford (gamma-matrix) basis for scientific discovery of propagator expansions. — also: High-energy physics
- **A Geometric Algebra-Inspired Spatio-temporal Learning Model for Sea Level Variability Reconstruction Under Incomplete Observations** — Li, Huang, Huang, Luo et al.. *Advances in Applied Clifford Algebras, 2026-09-11.* [doi](https://doi.org/10.1007/s00006-026-01457-3)
  GA-inspired spatio-temporal learning model for reconstructing sea-level fields from incomplete observations. — also: Sequence models, language and LLM tooling
- **CliffordIP: Clifford algebra equivariant interatomic potentials for heterogeneous catalysis** — Polat, Serpedin, Kurban, Kurban. *npj Computational Materials, 2026-08-04.* [doi](https://doi.org/10.1038/s41524-026-02259-8)
  Message-passing potential on Cl(3,0) 8-dimensional multivectors, O(3)-equivariant via Pin(3); 20-24% lower energy error on catalysis subsets of OC20 / OC22.
- **Geometric Algebra Meets Cartesian Tensors: Higher-Order Equivariance for Interatomic Potentials** — Polat, Serpedin, Kurban, Kurban. *arXiv, 2026-06-28.* [arxiv](https://arxiv.org/abs/2606.29584)
  CliffordSTF couples Clifford multivectors to symmetric-traceless tensor tracks to fix the poor force directions of L <= 1 Clifford potentials.
- **Geometric Algebra-Based Time Series Reconstruction of Sea Level Variability - A Case Study Along the Northeast Pacific Coast (1993-2023)** — Huang, Li, Luo, Yu et al.. *LNCS (ENGAGE 2025 proceedings), 2026-06-04.* [doi](https://doi.org/10.1007/978-3-032-22539-9_9)
  Companion case study of the GA sea-level reconstruction model. — also: Sequence models, language and LLM tooling
- **When Does Equivariance Help? Canonical Alignment in Neural Fluid Surrogates** — Rygiel, Suk, Yeung, Brune et al.. *arXiv; ICLR 2027 submission, 2026-05-12.* [arxiv](https://arxiv.org/abs/2605.18816) · [openreview](https://openreview.net/forum?id=VLMVwDLU6q)
  Compares GATr-type equivariant surrogates with canonically aligned non-equivariant ones on vascular CFD meshes.
- **Geometric deep learning for local growth prediction on abdominal aortic aneurysm surfaces** — Alblas, Rygiel, Suk, Kappe et al.. *Artificial Intelligence in Medicine 172:103323, 2026-02-01.* [doi](https://doi.org/10.1016/j.artmed.2025.103323)
  Projective-GA (GATr-based) model predicting local growth on aneurysm surface meshes. — also: Vision, remote sensing and medical imaging
- **Design and Evaluation of a Geometric Algebra-Based Graph Neural Network for Molecular Property Prediction** — Petersen, Schmidt. *NLDL 2026 (PMLR 307), 2026-01-19.* [openreview](https://openreview.net/forum?id=aHHQByxsjB) · [url](https://proceedings.mlr.press/v307/petersen26a.html)
  GA-based graph neural network evaluated on molecular property prediction at the Northern Lights Deep Learning conference. — also: Graphs, multimodal and representation learning

### High-energy physics

The Lorentz-equivariant geometric algebra transformer (L-GATr) line and other spacetime-algebra models.

- **Sequential SBI for Resonance Searches at the LHC** *(context)* — Bahl. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.30245) · [inspire](https://inspirehep.net/literature/3217957)
  Cites L-GATr.
- **Lorentz-Equivariant Geometric Algebra Transformers for Top-Quark Event Reconstruction** — Abasov, Dudko, Markina, Perfilov et al.. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.29678) · [inspire](https://inspirehep.net/literature/3217369)
  L-GATr on ttbar/tWb separation, FCNC detection and full event reconstruction; companion code repository.
- **How Far Can Foundation Models Be Distilled for Efficient Jet Tagging?** — Agarwal, Khare, Kumar. *arXiv, 2026-10-21.* [arxiv](https://arxiv.org/abs/2610.29439) · [inspire](https://inspirehep.net/literature/3217213)
  Distils foundation models into L-GATr-slim students; a 20k-parameter student beats every published tagger trained from scratch.
- **This is SParTA: A modular, efficient and scalable transformer architecture for jet-tagging** *(context)* — Baruah. *arXiv, 2026-10-19.* [arxiv](https://arxiv.org/abs/2610.27106) · [inspire](https://inspirehep.net/literature/3216611)
  Benchmarks against L-GATr.
- **QGrAT: Quantum Grade-Aware Transformer for Jet Tagging** — anonymous. *arXiv, 2026-10-12.* [arxiv](https://arxiv.org/abs/2610.17351) · [inspire](https://inspirehep.net/literature/3214450)
  Quantum transformer whose features are organised by the grades of the spacetime Clifford algebra. — also: Other machine-learning uses of GA
- **MadNIS, Ready for Production at LO** *(context)* — Heimel. *arXiv, 2026-10-12.* [arxiv](https://arxiv.org/abs/2610.17177) · [inspire](https://inspirehep.net/literature/3214424)
  Cites L-GATr.
- **RINO: Renormalization Group Distillation with No Labels** *(context)* — Hao. *arXiv, 2026-10-11.* [arxiv](https://arxiv.org/abs/2610.15919)
  Cites L-GATr.
- **A compact, interrogable Tree Tensor Network Jet Tagger on Lund declustering trees** *(context)* — Napolitano. *arXiv, 2026-10-08.* [arxiv](https://arxiv.org/abs/2610.12408)
  Cites L-GATr.
- **A Unified Geometric Framework for Understanding Collider Event Manifolds** *(context)* — Cai. *arXiv, 2026-10-07.* [arxiv](https://arxiv.org/abs/2610.09320)
  Cites L-GATr.
- **How to scale your HEP ML models: A recipe for robust architecture comparisons at scale** *(context)* — Vigl. *arXiv, 2026-10-05.* [arxiv](https://arxiv.org/abs/2610.06784)
  Cites L-GATr.
- **Machine learning for the LHC physics program: a 2025-2026 stocktake** *(context)* — Thaler. *arXiv, 2026-09-26.* [arxiv](https://arxiv.org/abs/2609.32874)
  Review citing L-GATr.
- **Towards Foundation Models on Hardware Accelerators for Particle Physics** *(context)* — Benyas. *arXiv, 2026-09-22.* [arxiv](https://arxiv.org/abs/2609.26899)
  Cites the economical (L-GATr-slim) taggers.
- **More QCD Masterclass Lectures on Jet Physics and Machine Learning** *(context)* — Larkoski. *arXiv, 2026-09-09.* [arxiv](https://arxiv.org/abs/2609.09785)
  Lecture notes citing L-GATr and the GA input-language paper.
- **A High- and Variable-Dimensional Measurement of the Z+jets Differential Cross Section with the ATLAS Experiment** *(context)* — Greif. *arXiv, 2026-08-28.* [arxiv](https://arxiv.org/abs/2608.28449)
  Cites L-GATr.
- **Know What You Don't Flow** *(context)* — Butter. *arXiv, 2026-08-21.* [arxiv](https://arxiv.org/abs/2608.21509)
  Cites L-GATr.
- **Assessing Parameter Redundancy in Transformers for Jet Tagging** *(context)* — Cheng. *arXiv, 2026-08-17.* [arxiv](https://arxiv.org/abs/2608.16061)
  Benchmarks against L-GATr.
- **Unknown Unknowns: Model Misspecification in Machine Learning for Physics** *(context)* — Cruz-Martinez. *arXiv, 2026-08-13.* [arxiv](https://arxiv.org/abs/2608.13633)
  Cites L-GATr.
- **Generative Amplification with Surrogate Monte Carlo** *(context)* — Bahl. *arXiv, 2026-08-06.* [arxiv](https://arxiv.org/abs/2608.06450)
  Cites L-GATr.
- **Machine Learning is Good for Physics - and Vice Versa** *(context)* — Krämer. *arXiv, 2026-08-06.* [arxiv](https://arxiv.org/abs/2608.05812)
  Essay citing L-GATr.
- **Virtues and Vices of Equivariant Transformers** — Favaro, Plehn, Qu, Spinner. *arXiv, 2026-08-03.* [arxiv](https://arxiv.org/abs/2608.02735)
  Lorentz-equivariant (spacetime-algebra) transformers for large-scale jet and flavour tagging, inference-optimised.
- **Neural Control Variates at LO and NLO** *(context)* — Heimel. *arXiv, 2026-07-26.* [arxiv](https://arxiv.org/abs/2607.23591)
  Cites L-GATr.
- **Predict before you train: Scaling Laws for particle physics foundation models** *(context)* — Uslu. *arXiv, 2026-07-25.* [arxiv](https://arxiv.org/abs/2607.23377)
  Cites L-GATr.
- **Learning Standard Model structure from LHC data with Riemannian flow matching** *(context)* — Kato. *arXiv, 2026-07-17.* [arxiv](https://arxiv.org/abs/2607.16144)
  Cites L-GATr.
- **Local Conformal Predictions for Calibrated Surrogates** *(context)* — Dubey. *arXiv, 2026-07-01.* [arxiv](https://arxiv.org/abs/2607.01354)
  Cites L-GATr.
- **Machine learning fully hadronic events with spectral functions** *(context)* — Altakach. *arXiv, 2026-06-25.* [arxiv](https://arxiv.org/abs/2606.27420)
  Cites L-GATr.
- **Interpreting "Interpretability" and Explaining "Explainability" in Machine Learning in Physics** *(context)* — Gambhir. *arXiv, 2026-06-24.* [arxiv](https://arxiv.org/abs/2606.26228)
  Cites L-GATr.
- **One Generator, Any Process: LLM-Conditioning for the LHC** *(context)* — Bahl. *arXiv, 2026-06-22.* [arxiv](https://arxiv.org/abs/2606.23791)
  Cites L-GATr.
- **What Do Lorentz-Equivariant Jet Taggers Learn?** — anonymous. *arXiv; ICML 2026 AI4Physics workshop, 2026-06-19.* [arxiv](https://arxiv.org/abs/2606.21790) · [openreview](https://openreview.net/forum?id=xr5xIic0Ei)
  Probes, equivariance tests and grade ablations on L-GATr, L-GATr-slim and LLoCa-T jet taggers. — also: Theory and analysis of GA layers
- **Event Generation with Parallel Langevin Sampling and Learned Stein Diagnostics** *(context)* — Verheyen. *arXiv, 2026-06-12.* [arxiv](https://arxiv.org/abs/2606.14854)
  Cites L-GATr.
- **CaloTrilogy: Toward a Breakthrough in One-Step, End-to-End, Physics-Guided Shower Generation** *(context)* — Jiang. *arXiv, 2026-06-02.* [arxiv](https://arxiv.org/abs/2606.04165)
  Cites L-GATr.
- **Particle-Lund Multimodality in Jet Taggers** *(context)* — Gouskos. *arXiv, 2026-05-26.* [arxiv](https://arxiv.org/abs/2605.26821)
  Benchmarks against L-GATr.
- **Pointwise Metrics Mislead: An Evaluation Protocol for Multimodal Inverse Problems** *(context)* — Baattrup. *arXiv, 2026-05-21.* [arxiv](https://arxiv.org/abs/2605.22891)
  Cites L-GATr.
- **Patch Hierarchical Attention Transformer for Efficient Particle Jet Tagging** *(context)* — Wang. *arXiv, 2026-05-20.* [arxiv](https://arxiv.org/abs/2605.21789)
  Benchmarks against L-GATr.
- **The Monte Carlo Ecosystem in High-Energy Physics: A Primer** *(context)* — Beekveld. *arXiv, 2026-05-15.* [arxiv](https://arxiv.org/abs/2605.16036)
  Cites L-GATr.
- **Geometric algebra as the input language of collider foundation models** — Abasov, Dudko, Markina, Perfilov et al.. *arXiv, 2026-05-15.* [arxiv](https://arxiv.org/abs/2605.15910) · [inspire](https://inspirehep.net/literature/3156709)
  A whole collider event as one Cl(1,3)⊗V_flav multivector; theorem on which Lorentz invariants higher grades add; L-GATr-type transformer.
- **Open LHC Monte Carlo Event Generation** *(context)* — Bothmann. *arXiv, 2026-05-12.* [arxiv](https://arxiv.org/abs/2605.12229)
  Cites L-GATr.
- **Generative models on phase space** *(context)* — Bogorad. *arXiv, 2026-04-02.* [arxiv](https://arxiv.org/abs/2604.02415)
  Cites L-GATr.
- **MadNIS at NLO** *(context)* — Crescenzo. *arXiv, 2026-03-23.* [arxiv](https://arxiv.org/abs/2603.22407)
  Cites L-GATr.
- **Explicit or Implicit? Encoding Physics at the Precision Frontier** — Bresó-Pla, Greif, Mikuni, Nachman et al.. *arXiv; Physical Review D 114, 036034 (2026), 2026-03-09.* [arxiv](https://arxiv.org/abs/2603.08802) · [openreview](https://openreview.net/forum?id=uhT0wi62Et)
  Explicit symmetry (L-GATr) versus large-scale pre-training at the precision frontier of collider physics.
- **End-to-end event reconstruction for precision physics at future colliders** — Garcia, Herrmann, Krzmanc, Selvaggi. *arXiv, 2026-03-04.* [arxiv](https://arxiv.org/abs/2603.04084)
  Geometric algebra transformer networks combined with object-condensation clustering for particle-flow reconstruction.
- **The Latent Information Geometry of Jet Classification** *(context)* — Kuntz. *arXiv, 2026-03-02.* [arxiv](https://arxiv.org/abs/2603.02310)
  Cites L-GATr.
- **Unfolding without Iterations, Adversaries, or Surrogates** *(context)* — Ore. *arXiv, 2026-02-27.* [arxiv](https://arxiv.org/abs/2602.24282)
  Cites L-GATr.
- **Neural Scaling Laws for Boosted Jet Tagging** *(context)* — Vigl. *arXiv, 2026-02-17.* [arxiv](https://arxiv.org/abs/2602.15781)
  Cites L-GATr.
- **MadSpace - Event Generation for the Era of GPUs and ML** *(context)* — Heimel. *arXiv, 2026-02-06.* [arxiv](https://arxiv.org/abs/2602.06895)
  Cites L-GATr.
- **CoLLM: AI engineering toolbox for end-to-end deep learning in collider analyses** *(context)* — Esmail. *arXiv, 2026-02-06.* [arxiv](https://arxiv.org/abs/2602.06496)
  Cites L-GATr.
- **Modern Machine Learning and Particle Physics Phenomenology at the LHC** *(context)* — Ubiali. *arXiv, 2026-02-03.* [arxiv](https://arxiv.org/abs/2602.03728)
  Review citing L-GATr.
- **PanopTag: Simultaneously Tagging All Jets in a Particle Collision Event** *(context)* — Qureshi. *arXiv, 2026-01-23.* [arxiv](https://arxiv.org/abs/2601.16417)
  Cites L-GATr.
- **Scaling laws for amplitude surrogates** *(context)* — Bahl. *arXiv, 2026-01-19.* [arxiv](https://arxiv.org/abs/2601.13308)
  Cites L-GATr (amplitude surrogates).
- **jBOT: Semantic Jet Representation Clustering Emerges from Self-Distillation** *(context)* — Tsoi. *arXiv, 2026-01-16.* [arxiv](https://arxiv.org/abs/2601.11719)
  Cites L-GATr.
- **AllShowers: One model for all calorimeter showers** *(context)* — Buss. *arXiv, 2026-01-16.* [arxiv](https://arxiv.org/abs/2601.11716)
  Cites L-GATr.
- **How to Trust Learned Loop Amplitudes** *(context)* — Bahl. *arXiv, 2026-01-02.* [arxiv](https://arxiv.org/abs/2601.00950)
  Cites L-GATr (amplitude surrogates).

### Vision, remote sensing and medical imaging

Image backbones, hyperspectral classification, registration, segmentation and enhancement.

- **Geometric algebra enhanced twin-branch contrastive network design for few-shot hyperspectral image classification** — Zhang, Cao, Liu. *Pattern Recognition, 2026-11-01.* [doi](https://doi.org/10.1016/j.patcog.2026.113549)
  GA-enhanced twin-branch contrastive network for few-shot hyperspectral classification.
- **Subspace Clifford Networks** — Ji. *arXiv, 2026-10-27.* [arxiv](https://arxiv.org/abs/2610.37226)
  Subspace Clifford Networks: replace CliffordNet's global cyclic rolling with grouped subspace interactions. — also: New GA / Clifford architectures
- **Preprocessing of Retinal Fundus Image Using Clifford Algebra** *(adjacent)* — Roy, Mitra, Setua. *Preprints.org, 2026-05-11.* [doi](https://doi.org/10.20944/preprints202605.0666.v1)
  Clifford colour space and rotor-based illumination correction as a preprocessing stage for diagnosis pipelines (not itself a neural network).
- **UHD Low-Light Image Enhancement via Real-Time Enhancement Methods with Clifford Information Fusion** — Wang, Wu, Zhao, Gao. *arXiv, 2026-04-10.* [arxiv](https://arxiv.org/abs/2604.09321)
  Feature fusion in a 2D Clifford multivector space (scalar, vector, bivector) with Clifford similarity; millisecond 4K/8K inference.
- **Geometric algebra transformer with spectral-spatial-volumetric feature extraction for hyperspectral image classification** — Fabi, Schiavella, Amerini. *Pattern Recognition Letters, 2026-04-01.* [doi](https://doi.org/10.1016/j.patrec.2026.02.001) · [openreview](https://openreview.net/forum?id=jfb7cr7u30)
  H-GATr embeds spectral, spatial and volumetric cues as multivectors without extra parameters and feeds them to a geometric algebra transformer.
- **Conformal Geometric Algebra Graph Learning Theory for 3D Human Skeleton Action** — Cao, Wang, Hu. *ECTI DAMT & NCON 2026, 2026-02-04.* [doi](https://doi.org/10.1109/ectidamtncon67592.2026.11459996)
  Conformal-GA graph learning applied to 3D skeleton action recognition. — also: Graphs, multimodal and representation learning
- **GIENet: Graph-Infused Encoder with Geometric Algebra Single-Pass Fusion for Unsupervised Deformable Registration** — Zhou, Cao, Feng, Zhang. *SSRN, 2026-01-01.* [doi](https://doi.org/10.2139/ssrn.6205817)
  Graph-infused encoder with a GA single-pass fusion decoder for unsupervised 3D deformable registration. — also: Graphs, multimodal and representation learning
- **Geometric Algebra-based Dual-branch Graph Convolutional Network for Deformable Medical Image Registration with Bi-Directional Flow** — Cao, Anwar, Yan. *IEEE Journal of Biomedical and Health Informatics, 2026-01-01.* [doi](https://doi.org/10.1109/jbhi.2026.3718748)
  GA-based dual-branch graph convolutional network for deformable registration with bidirectional flow. — also: Graphs, multimodal and representation learning

### Graphs, multimodal and representation learning

Graph neural networks, sheaves, binding / vector-symbolic representations with Clifford structure.

- **ICE: Task-Aligned Clifford Latent Fields for Multimodal Graph Foundation Models** — anonymous. *arXiv; ICLR 2027 submission, 2026-09-24.* [arxiv](https://arxiv.org/abs/2609.29398) · [openreview](https://openreview.net/forum?id=wIfzslZuuF)
  Clifford latent fields whose higher-grade channels recover pair relations across multimodal foundation graphs.
- **Recursive Binding on a Budget: Subspace Carving in Order-p Tensor Memories** *(adjacent)* — Pence, Yamada, Singh. *arXiv; ICML 2026 submission, 2026-06-10.* [arxiv](https://arxiv.org/abs/2606.11391) · [openreview](https://openreview.net/forum?id=Oc5d0gsRTs)
  Shows tensor-product representations are a special case of binding in a Clifford algebra and gives a Clifford formulation of the method. — also: Theory and analysis of GA layers
- **Conformal geometric algebra graph learning theory for non-Euclidean data** — Cao, Wang. *ICCAIS 2025 (SPIE proceedings), 2026-04-24.* [doi](https://doi.org/10.1117/12.3112939)
  Theory of graph learning on non-Euclidean data in conformal GA. — also: Theory and analysis of GA layers
- **LION: A Clifford Neural Paradigm for Multimodal-Attributed Graph Learning** — Li, Wu, Chen, Sun et al.. *arXiv (re-posted as 2608.24795 in Aug 2026), 2026-01-29.* [arxiv](https://arxiv.org/abs/2601.21453) · [openreview](https://openreview.net/forum?id=TgmpQaICO2)
  Modality-aware geometric manifold grounded in Clifford algebra; decoupled propagate-then-aggregate graph network for multimodal-attributed graphs.

### Robotics, driving and control learning

Policies, scene models and pose pipelines built on projective or conformal GA.

- **Synthesis of discrete mechanisms: a hybrid approach based on geometric algebra and genetic algorithms** *(adjacent)* — Kadan, Petrov. *Yugra State University Bulletin, 2026-10-13.* [doi](https://doi.org/10.18822/byusu20260372-76)
  Synthesises kinematic mechanisms from a discrete set of components by searching projective-geometric-algebra representations with a genetic algorithm. — also: Other machine-learning uses of GA
- **Screw Attention: Rigid-Body Algebra Inside a Transformer** *(context)* — anonymous. *arXiv, 2026-10-01.* [arxiv](https://arxiv.org/abs/2610.00904)
  Uses screw theory and spatial transforms, not Clifford algebra; listed because it is the closest non-GA competitor for rigid-body-aware attention.
- **From Language to Task Maps: Compiling Semantic Relations While Preserving Task-Relevant Freedom** *(adjacent)* — Park, Lee, Lee, Hwang. *arXiv, 2026-09-28.* [arxiv](https://arxiv.org/abs/2609.34412)
  Compiles language-specified relations into conformal-GA task maps composed with learned policies through RMPflow.
- **Hybrid Diffusion Policies with Projective Geometric Algebra for Efficient Robot Manipulation Learning** — Sun, Wang, Yang, Chen et al.. *ICRA 2026 (arXiv preprint Jul 2025), 2026-06-01.* [arxiv](https://arxiv.org/abs/2507.05695) · [doi](https://doi.org/10.1109/icra57385.2026.11696239) · [openreview](https://openreview.net/forum?id=ZOxNLgPuhA)
  hPGA-DP uses a projective GA transformer (P-GATr) as state encoder and action decoder around a standard diffusion denoiser; preprint 2025, published 2026.
- **The Cartan-Topos Protocol: A Unified Geometric and Categorical Framework for Resilient Multi-Agent Coordination** *(adjacent)* — Hernández, Sánchez-Soto. *arXiv, 2026-05-30.* [arxiv](https://arxiv.org/abs/2606.00714)
  Clifford rotors and motors for SE(3) pose synchronisation in multi-agent coordination; not deep learning.
- **Efficient Equivariant Transformer for Self-Driving Agent Modeling** — Xu, Chen, Wong, Zhang et al.. *arXiv; CVPR 2026, 2026-04-01.* [arxiv](https://arxiv.org/abs/2604.01466) · [openreview](https://openreview.net/forum?id=vmIehea8D9) · [url](https://cvpr.thecvf.com/virtual/2026/poster/36432)
  DriveGATr encodes scene elements as multivectors of the 2D projective GA R*(2,0,1); SE(2)-equivariant attention without pairwise relative encodings; Waymo Open Motion. — also: New GA / Clifford architectures

### Wireless channels and neural fields

GA attention inside neural radiance fields and Gaussian splatting for radio propagation.

- **A Geometric Algebra-Informed 3DGS Framework for Wireless Channel Prediction** — Shen, Enamorado, Mao, Wang. *arXiv; CVPR 2026 submission, 2026-05-18.* [arxiv](https://arxiv.org/abs/2605.19065) · [openreview](https://openreview.net/forum?id=fcXTtuAMvs)
  GAI-GS couples 3D Gaussian splatting with a GA-based attention mechanism for wireless modelling.
- **A Geometric Algebra-informed NeRF Framework for Generalizable Wireless Channel Prediction** — Shen, Enamorado, Mao, Wang. *arXiv; IEEE INFOCOM 2026, 2026-04-13.* [arxiv](https://arxiv.org/abs/2604.11983) · [doi](https://doi.org/10.1109/infocom59046.2026.11571298)
  GAI-NeRF puts geometric-algebra attention over ray-object interactions inside a neural radiance field for channel prediction.

### Sequence models, language and LLM tooling

Rotor-based recurrences, time-series forecasters, semantics and GA-aware language-model tools.

- **Generative geospatial modelling with geometric algebra** *(adjacent)* — anonymous. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2025.0103)
  GA as a bridge to large-language-model spatial reasoning for geospatial generation. — also: Surveys, reviews and position papers
- **GA-VisAgent: A Multi-Agent application for code generation and visualization in interactive learning** *(adjacent)* — Jian, Jianbo, Yuhao, Zhenxia. *arXiv, 2026-05-02.* [arxiv](https://arxiv.org/abs/2605.01299)
  LLM agents built on a GA-specialised language model (GAGPT) that generate and visualise geometric-algebra code. — also: Software and resources
- **Adaptive Domain Models: Bayesian Evolution, Warm Rotation, and Principled Training for Geometric and Neuromorphic AI** *(adjacent)* — Haynes. *arXiv, 2026-03-18.* [arxiv](https://arxiv.org/abs/2603.18104)
  Hybrid design with grade-typed Clifford representations for geometry and spiking layers for temporal patterns, serving as domain checkers for language-model pipelines. — also: New GA / Clifford architectures
- **Semantic Plane Alignment: Measuring Summarization Completeness via Tensor-based Geometric Algebra** *(adjacent)* — Sudjianto. *SSRN, 2026-01-01.* [doi](https://doi.org/10.2139/ssrn.6168708)
  Planes spanned by embeddings (GA outer products) as an evaluation metric for summary completeness. — also: Other machine-learning uses of GA

### Clifford-valued network dynamics

Stability and synchronisation theory of recurrent networks with Clifford-number states (control theory).

- **A restart-graded L1 scheme for fractional-order Clifford-valued delay neural networks: Weak singularities and sharp convergence** *(adjacent)* — Rajchakit. *Nonlinear Science, 2026-12-01.* [doi](https://doi.org/10.1016/j.nls.2026.100203)
  Numerical scheme with convergence proof for fractional-order Clifford-valued delay networks.
- **Synchronization of neutral-type Clifford-valued inertial neural networks with mixed delays by non-reduced-order and non-separation approaches** *(adjacent)* — Deng, Li, Tan. *Neurocomputing, 2026-12-01.* [doi](https://doi.org/10.1016/j.neucom.2026.134991)
  Inertial (second-order) Clifford-valued networks analysed without order reduction.
- **Observer-Based Output-Feedback Synchronization of Fractional-Order Fuzzy Clifford-Valued Neural Networks: A Cascade Certificate and the Quantified Price of Output Feedback** *(adjacent)* — Rajchakit. *Complex Systems Stability & Control, 2026-10-23.* [doi](https://doi.org/10.53941/cssc.2026.100028)
  Leader-following synchronisation when only part of the multivector state is measured.
- **H-infinity synchronisation of fractional-order Clifford-valued reaction-diffusion neural networks with leakage delays via sampled-data control** *(adjacent)* — Rajchakit, Lim, Tajudeen, Huang. *International Journal of Systems Science, 2026-10-21.* [doi](https://doi.org/10.1080/00207721.2026.2736095)
  Reaction-diffusion Clifford-valued networks under sampled-data control.
- **Relaxed synchronization analysis of fractional-order Clifford-valued delayed neural networks via novel inner product-based inequality** *(adjacent)* — Manoj, Sriraman. *Physica Scripta, 2026-10-20.* [doi](https://doi.org/10.1088/1402-4896/aeb198)
  Inner-product inequalities for Clifford numbers that avoid real-valued decomposition.
- **Stability and Control of Fractional-Order Fuzzy Stochastic Clifford-Valued Neural Networks with Hybrid Markovian Switching, Cyber-Attacks, and Event-Triggered Multi-Agent Synchronization** *(adjacent)* — Rajchakit. *Nonlinear Analysis and Computer Simulations, 2026-10-08.* [doi](https://doi.org/10.53941/nacs.2026.100016)
  Mean-square Mittag-Leffler synchronisation under deception attacks and event-triggered communication.
- **Fixed-time synchronization of Clifford biquaternion neural networks with two-sided coefficients and application to multispectral image protection** *(adjacent)* — Li, Zhang, Kou, Liu. *Neural Networks (volume dated Jan 2027), 2026-10-01.* [doi](https://doi.org/10.1016/j.neunet.2026.109528)
  Eight-dimensional Clifford biquaternion networks with two-sided coefficients; fixed-time synchronisation and multispectral image encryption. — also: Hypercomplex-valued networks
- **Exponential lag synchronization of inertial Clifford-valued neural networks with proportional delays via sampled-data pinning control and applications to multi-spectral image watermarking** *(adjacent)* — Rajchakit, Lim. *Neurocomputing, 2026-10-01.* [doi](https://doi.org/10.1016/j.neucom.2026.134343)
  Lag synchronisation with a watermarking application.
- **Synchronization dynamics of fractional-order Clifford-valued BAM delayed neural networks and their applications** *(adjacent)* — Nagappan, Ramalingam, Rajchakit, Rajchakit et al.. *Journal of Mathematics and Computer Science, 2026-09-23.* [doi](https://doi.org/10.22436/jmcs.043.03.02)
  Synchronisation of fractional-order Clifford-valued BAM networks.
- **Quasi-Projective Synchronization of Heterogeneous Fractional-Order Clifford-Valued Neural Networks via Intermittent Pinning Control** *(adjacent)* — Rajchakit, Lim, Tajudeen, Huang. *International Journal of Robust and Nonlinear Control, 2026-09-07.* [doi](https://doi.org/10.1002/rnc.70702)
  Heterogeneous drive and response networks with Clifford-valued states.
- **Aperiodic intermittent control strategy for synchronization of Clifford-valued neural networks on time scales using matrix measure** *(adjacent)* — Agrawal, Abbas, Debbouche. *Applied Mathematics and Computation, 2026-09-01.* [doi](https://doi.org/10.1016/j.amc.2026.130034)
  Matrix-measure synchronisation criteria on time scales, unifying continuous and discrete time.
- **Global Mittag-Leffler synchronization of fractional-order Clifford-valued neural networks with mixed time-varying delays via adaptive quantized pinning control** *(adjacent)* — Rajchakit, Lim. *International Journal of Dynamics and Control, 2026-08-07.* [doi](https://doi.org/10.1007/s40435-026-02228-5)
  Synchronisation criteria under finite-resolution communication without decomposing the Clifford-valued system.
- **Slide Mode Control Based Finite and Fixed-Time Projective Synchronization for Delayed Clifford-Valued Neural Networks** *(adjacent)* — Kumari, Malik. *Mathematical Methods in the Applied Sciences, 2026-07-30.* [doi](https://doi.org/10.1002/mma.70915)
  Sliding-mode controllers for finite- and fixed-time projective synchronisation.
- **Finite-time stabilization of fractional-order fuzzy Clifford-valued neural networks using novel inequalities** *(adjacent)* — Sriraman, Manoj. *Physica Scripta, 2026-07-02.* [doi](https://doi.org/10.1088/1402-4896/ae8133)
  Non-decomposition finite-time stabilisation analysis.
- **Multistability of equilibria for Clifford-valued Cohen-Grossberg neural networks with discontinuous activation functions and time delays** *(adjacent)* — Yang, Qiao, Duan, Miao. *Mathematics and Computers in Simulation, 2026-07-01.* [doi](https://doi.org/10.1016/j.matcom.2025.12.015)
  Counts and locates multiple equilibria of Clifford-valued Cohen-Grossberg networks.
- **A unified matrix measure strategy for QGPL synchronization of Clifford-valued competitive model with parameter mismatch: Application to LSTM model** *(adjacent)* — Agrawal, Abbas. *Neurocomputing, 2026-06-01.* [doi](https://doi.org/10.1016/j.neucom.2026.133377)
  Quasi-generalised projective lag synchronisation of a Clifford-valued competitive model with an LSTM application.
- **Finite-time stability of fractional-order delayed Clifford-valued neural networks via a novel non-separation approach** *(adjacent)* — Ramalingam, Nagappan. *Computational and Applied Mathematics, 2026-03-10.* [doi](https://doi.org/10.1007/s40314-026-03690-9)
  Finite-time stability without separating the Clifford components.
- **Projective Synchronization of Fractional-Order BAM Clifford-Valued Delayed Memristive Neural Networks: An Inner Product-Based Inequality** *(adjacent)* — Manoj, Sriraman. *Mathematical Methods in the Applied Sciences, 2026-02-19.* [doi](https://doi.org/10.1002/mma.70592)
  Bidirectional associative memory networks with memristive Clifford-valued weights.
- **Global Exponential Weyl Almost Periodic Synchronization for Hypercomplex-Valued High-Order Hopfield Neural Networks with Delays** *(adjacent)* — Li, Xu. *Qualitative Theory of Dynamical Systems, 2026-02-01.* [doi](https://doi.org/10.1007/s12346-026-01459-3)
  Hypercomplex-valued high-order Hopfield networks (Clifford algebras as the unifying case). — also: Hypercomplex-valued networks
- **Advances in the Mathematical Theory of WPAA Dynamics for Impulsive High Order Neural Systems in Clifford Algebras** *(adjacent)* — Aouiti, Dridi. *Journal of Nonlinear Dynamics and Applications, 2026-01-21.* [doi](https://doi.org/10.62762/jnda.2025.838385)
  Weighted pseudo-almost automorphic solutions of impulsive high-order Hopfield networks in Clifford algebras.
- **Noise Robust Projection Rule for Split-Quaternion-Valued Hopfield Networks** *(adjacent)* — Kobayashi. *IEICE Transactions on Fundamentals, 2026-01-01.* [doi](https://doi.org/10.1587/transfun.2026map0002)
  Split quaternions are a four-dimensional Clifford algebra; the paper frames itself as Clifford-algebra neural networks. — also: Hypercomplex-valued networks

### Hypercomplex-valued networks

Quaternion, biquaternion and general hypercomplex layers framed as Clifford algebras.

- **Fully tensorial approach to hypercomplex-valued neural networks** *(adjacent)* — Niemczynowicz, Kycia. *Information Sciences (arXiv 2407.00449), 2026-11-01.* [arxiv](https://arxiv.org/abs/2407.00449) · [doi](https://doi.org/10.1016/j.ins.2026.123796)
  Unified tensor implementation of hypercomplex (including Clifford) layers. — also: Software and resources
- **Performance Increases in Physics-Informed Machine Learning using Hypercomplex-based Automatic Differentiation (HYPAD)** *(adjacent)* — Roberts, Aristizabal, Millwater. *ACM Transactions on Mathematical Software, 2026-10-16.* [doi](https://doi.org/10.1145/3858612)
  HYPAD: hypercomplex-number automatic differentiation that speeds up derivative computation when training physics-informed neural networks. — also: Scientific machine learning
- **Ghost Features and Spooky Transfer Learning for Hypercomplex-Valued Neural Networks** *(adjacent)* — Vieira, Valle. *arXiv, 2026-08-07.* [arxiv](https://arxiv.org/abs/2608.07735) · [code](https://github.com/mevalle/v-nets/)
  Hypercomplex layers whose real part reproduces a trained real layer; the imaginary parts add "ghost features" that can be transferred into trained networks. — also: New GA / Clifford architectures
- **Global Fixed-Time Synchronization of Fractional-Order Dual-Split-Quaternion Valued Memristive Neural Networks: Full Proofs and 6D Relativistic Image Encryption** *(adjacent)* — Rajchakit. *Complex Systems Stability & Control, 2026-07-07.* [doi](https://doi.org/10.53941/cssc.2026.100013)
  Dual-split-quaternion algebra (Clifford-adjacent hypercomplex) for synchronisation and image encryption. — also: Clifford-valued network dynamics
- **Hypercomplex Phase Retrieval** *(context)* — anonymous. *arXiv, 2026-02-27.* [arxiv](https://arxiv.org/abs/2602.23946)
  Clifford-algebra hypercomplex signal processing for phase retrieval; not a neural network.

### Other machine-learning uses of GA

Classifiers, filters, causal discovery and quantum machine learning built on GA.

- **Hybrid Geometric Algebra Correntropy Criterion** *(adjacent)* — Zhao, Hou, Long. *Information Theoretic Learning-based Filter (Springer), 2026-10-06.* [doi](https://doi.org/10.1007/978-3-032-29623-8_7)
  GA-valued adaptive filtering with a correntropy criterion.
- **Machine Learning the Phase Structure and Algebraic Invariants of Clifford- and Grassmann-Valued Cellular Automata** *(adjacent)* — Mirza. *Research Square, 2026-07-31.* [doi](https://doi.org/10.21203/rs.3.rs-10394430/v1)
  Machine learning applied to discrete dynamical systems driven by Clifford and Grassmann algebras.
- **Geometric Algebra Quantum Gate Decomposition** *(context)* — Amraoui, Toffano. *Axioms 15(7):523, 2026-07-13.* [doi](https://doi.org/10.3390/axioms15070523)
  Rotor-based quantum gate synthesis; GA-adjacent, not learning.
- **Adaptive Sphere Classification via Conformal Geometric Algebra** — Athmani. *Research Square, 2026-03-31.* [doi](https://doi.org/10.21203/rs.3.rs-9244644/v1)
  Supervised classifier whose learning rule is the conformal-GA point-in-sphere inner product.
- **Quantum classification and search algorithms using spinorial representations** *(adjacent)* — anonymous. *arXiv, 2026-03-17.* [arxiv](https://arxiv.org/abs/2603.16564)
  Quantum classification and search algorithms built on Clifford algebras and spinors.
- **Causal Discovery using Geometric Algebra** *(adjacent)* — Sudjianto, Setiawan. *SSRN, 2026-01-01.* [doi](https://doi.org/10.2139/ssrn.7291638)
  Pairwise causal-direction scoring recast with outer products and multivariate angles. — also: Scientific machine learning

### Software and resources

Libraries, curated lists and theme issues.

- **Preface to 'Modern applications of geometric algebra' (Phil. Trans. R. Soc. A theme issue 384/2326)** *(context)* — Hitzer, Hildenbrand, Papagiannakis. *Philosophical Transactions of the Royal Society A, 2026-08-13.* [doi](https://doi.org/10.1098/rsta.2026.0174) · [url](https://doi.org/10.1098/rsta.2025.0104)
  Theme issue bundling the 2026 GA papers above; includes a GA implementation and memory-requirements overview (rsta.2025.0104). — also: Surveys, reviews and position papers

## 2025

### Surveys, reviews and position papers

Overviews of geometric-algebra (GA) neural networks and essays arguing for GA as a representation.

- **Geometric Algebra Transformers: Revolutionizing Geometric Data** — Cohen. *ICIAM2023 Springer Series, 2025-08-19.* [doi](https://doi.org/10.1007/978-981-95-0351-3_2)
  Book chapter introducing geometric algebra transformers. — also: New GA / Clifford architectures

### New GA / Clifford architectures

General-purpose layers and models whose weights or activations are multivectors.

- **Soft Geometric Inductive Bias for Object Centric Dynamics** — Linander, Heins, Tschantz, Perin et al.. *arXiv, 2025-12-17.* [arxiv](https://arxiv.org/abs/2512.15493)
  Object-centric world models with a soft geometric-algebra inductive bias (Clifford transformer). — also: Robotics, driving and control learning
- **Single-Core Superscalar Optimization of Clifford Neural Layers** — Huang, Ciranni, Spadaccini, Zurita. *arXiv, 2025-09-29.* [arxiv](https://arxiv.org/abs/2510.03290)
  Superscalar single-core optimisation of Clifford convolutional layers. — also: Software and resources
- **Equivariant Neural Networks with Geometric Algebras: A New Approach** — Filimoshina, Shirokov. *2025 International Joint Conference on Neural Networks (IJCNN), 2025-06-30.* [doi](https://doi.org/10.1109/ijcnn64981.2025.11227310)
  New approach to equivariant neural networks with geometric algebras (IJCNN 2025).
- **Fast Clifford Neural Layers** — Xia, Neuwinger, Xiao. *arXiv, 2025-06-22.* [arxiv](https://arxiv.org/abs/2507.01040)
  Optimised single-core CPU inference for Clifford convolution and multivector activation layers. — also: Software and resources
- **GLGENN: A Novel Parameter-Light Equivariant Neural Networks Architecture Based on Clifford Geometric Algebras** — Filimoshina, Shirokov. *arXiv, 2025-06-11.* [arxiv](https://arxiv.org/abs/2506.09625) · [openreview](https://openreview.net/forum?id=H0ySAzwu8k)
  GLGENN: parameter-light networks equivariant to all pseudo-orthogonal transformations, via generalised Lipschitz groups.
- **ViNE-GATr: scaling geometric algebra transformers with virtual nodes embeddings** — Suk, Hehn, Behboodi, Cesa. *ICLR 2025 Workshop MLMP Poster, 2025-03-06.* [openreview](https://openreview.net/forum?id=Eb7fQBdA3D)
  ViNE-GATr: virtual-node embeddings that scale GATr to large geometric data.

### Theory and analysis of GA layers

Expressivity, identifiability, group-theoretic and optimisation studies of GA networks.

- **Composing Linear Layers from Irreducibles** — Pence, Yamada, Singh. *arXiv, 2025-07-15.* [arxiv](https://arxiv.org/abs/2507.11688) · [openreview](https://openreview.net/forum?id=z5FGi0vyCr)
  Synthesises linear layers from Clifford-algebra bivector (rotor) primitives.
- **The neural networks with tensor weights and emergent fermionic Wick rules in the large-width limit** *(adjacent)* — Huang, Zhou. *arXiv, 2025-07-07.* [arxiv](https://arxiv.org/abs/2507.05303)
  Tensor-weight complex networks whose infinite-width limit yields Clifford-algebra (fermionic Wick) statistics.

### Scientific machine learning

Molecules, materials, PDE surrogates, fluids, geoscience and lattice field theory.

- **Conditional Clifford-Steerable CNNs with Complete Kernel Basis for PDE Modeling** — Szarvas, Zhdanov. *Submitted to ICLR 2026, 2025-09-19.* [openreview](https://openreview.net/forum?id=GiNy68FgKR)
  Completes the Clifford-steerable CNN kernel basis with conditional kernels for PDE modelling. — also: Theory and analysis of GA layers
- **Geometric Algebra-Enhanced Bayesian Flow Network for RNA Inverse Design** — Wang, Gao, Zhao. *NeurIPS 2025 poster, 2025-09-18.* [doi](https://doi.org/10.52202/085713-0341) · [openreview](https://openreview.net/forum?id=x0KcjteNds)
  Geometric-algebra-enhanced Bayesian flow network for RNA inverse design.
- **Wall Shear Stress Estimation in Abdominal Aortic Aneurysms: Towards Generalisable Neural Surrogate Models** — Rygiel, Suk, Brune, Yeung et al.. *arXiv, 2025-07-30.* [arxiv](https://arxiv.org/abs/2507.22817)
  E(3)-equivariant model with projective-GA descriptors for wall-shear-stress estimation in aortic aneurysms.
- **Clifford Group Equivariant Diffusion Models for 3D Molecular Generation** — Liu, Vadgama, Ruhe, Bekkers et al.. *arXiv, 2025-04-22.* [arxiv](https://arxiv.org/abs/2504.15773) · [openreview](https://openreview.net/forum?id=XxIDwO5H5q)
  Clifford diffusion models: E(n)-equivariant diffusion with Clifford multivectors for 3D molecule generation.
- **Fengbo: a Clifford Neural Operator pipeline for 3D PDEs in Computational Fluid Dynamics** — Pepe, Montanari, Lasenby. *ICLR 2025 Poster, 2025-01-22.* [openreview](https://openreview.net/forum?id=VsxbWTDHjh)
  Fengbo: a Clifford neural-operator pipeline for 3D computational fluid dynamics.

### High-energy physics

The Lorentz-equivariant geometric algebra transformer (L-GATr) line and other spacetime-algebra models.

- **Economical Jet Taggers - Equivariant, Slim, and Quantized** — Petitjean, Plehn, Spinner, Köthe. *arXiv (Dec 2025), 2025-12-18.* [arxiv](https://arxiv.org/abs/2512.17011) · [inspire](https://inspirehep.net/literature/3094401)
  Late-2025 precursor that introduces L-GATr-slim, the slim Lorentz-equivariant tagger used by several 2026 papers.
- **Probing a Quarkophobic ${\mathbf{W}}^\prime$ at the High-Luminosity LHC via Vector Boson Fusion and Lorentz-Equivariant Point Cloud Learning** — U.S., A., J.D.. *arXiv, 2025-02-23.* [arxiv](https://arxiv.org/abs/2502.16630) · [inspire](https://inspirehep.net/literature/2893828)
  First BSM search (quarkophobic W') using L-GATr point-cloud learning.
- **Symmetry-aware networks for the LHC** — Simon. *INSPIRE record, 2025-01-01.* [doi](https://doi.org/10.11588/heidok.00036846) · [inspire](https://inspirehep.net/literature/2961245)
  PhD thesis on symmetry-aware networks (the L-GATr line) for the LHC.

### Vision, remote sensing and medical imaging

Image backbones, hyperspectral classification, registration, segmentation and enhancement.

- **Dual-branch geometric algebra graph network with pyramid fusion for deformable medical image registration** — Anwar, Cao. *Journal of King Saud University Computer and Information Sciences, 2025-12-09.* [doi](https://doi.org/10.1007/s44443-025-00412-9)
  Dual-branch geometric-algebra graph network with pyramid fusion for deformable medical image registration.
- **Neural-GASh: A CGA-based neural radiance prediction pipeline for real-time shading** — Geronikolakis, Kamarianakis, Protopsaltis, Papagiannakis. *arXiv, 2025-07-18.* [arxiv](https://arxiv.org/abs/2507.13917)
  Neural-GASh: real-time neural radiance shading with CGA-encoded vertex inputs. — also: Wireless channels and neural fields
- **Geometric Inductive Priors in Diffusion-Based Optical Flow Estimation** — Pepe, Mendonca, Lasenby. *BEW 2025 Oral, 2025-07-09.* [openreview](https://openreview.net/forum?id=0CnP0YAFZ4)
  GA-DDVM: a diffusion optical-flow model that works on geometric-algebra representations.
- **A Geometric Algebra-Based Model for Enhanced Hyperspectral Anomaly Detection** — Liu, Luo, Wang, Huang et al.. *IEEE Trans. Geosci. Remote. Sens. 2025, 2025-01-01.* [openreview](https://openreview.net/forum?id=053lHSab8h)
  Geometric-algebra model for hyperspectral anomaly detection.

### Graphs, multimodal and representation learning

Graph neural networks, sheaves, binding / vector-symbolic representations with Clifford structure.

- **Geometric Algebra Multi-Order Graph Neural Network for Traffic Prediction** — Zang, Cui, Wang, Lei et al.. *IEEE Trans. Big Data 2025, 2025-12-31.* [doi](https://doi.org/10.1109/tbdata.2024.3442533) · [openreview](https://openreview.net/forum?id=PFUyWKGYYt)
  Geometric-algebra multi-order graph neural network for traffic prediction.
- **Graph Geometric Algebra networks for graph representation learning** — Zhong, Cao. *Scientific Reports, 2025-01-02.* [doi](https://doi.org/10.1038/s41598-024-84483-0)
  Graph geometric algebra networks for graph representation learning.
- **Expressiveness Analysis and Enhancing Framework for Geometric Knowledge Graph Embedding Models** *(adjacent)* — Song, Yin, Liu, Liao et al.. *IEEE Trans. Knowl. Data Eng. 2025, 2025-01-01.* [openreview](https://openreview.net/forum?id=fZhr1S6Jli)
  Expressiveness analysis of geometric knowledge-graph embeddings, comparing models built on different geometric algebras. — also: Theory and analysis of GA layers

### Robotics, driving and control learning

Policies, scene models and pose pipelines built on projective or conformal GA.

- **GAGrasp: Geometric Algebra Diffusion for Dexterous Grasping** — Zhong, Allen-Blanchette. *arXiv, 2025-03-06.* [arxiv](https://arxiv.org/abs/2503.04123) · [doi](https://doi.org/10.1109/icra55743.2025.11127957) · [openreview](https://openreview.net/forum?id=t67yG4Kkxg)
  GAGrasp: SE(3)-equivariant geometric-algebra diffusion for dexterous grasp generation.

### Sequence models, language and LLM tooling

Rotor-based recurrences, time-series forecasters, semantics and GA-aware language-model tools.

- **Clifford Algebraic Rotor Embeddings : Maybe embeddings should start to CARE** — Sriram, Paliwal, Ecker, Geijn. *arXiv, 2025-11-11.* [arxiv](https://arxiv.org/abs/2511.11665) · [openreview](https://openreview.net/forum?id=V7hM3JpJKE)
  CARE: Clifford-algebra rotor positional embeddings that generalise RoPE. — also: New GA / Clifford architectures
- **Geometric Dynamics of Consumer Credit Cycles: A Multivector-based Linear-Attention Framework for Explanatory Economic Analysis** — Sudjianto, Setiawan. *arXiv, 2025-09-08.* [arxiv](https://arxiv.org/abs/2510.15892) · [doi](https://doi.org/10.2139/ssrn.5443034)
  Multivector linear-attention model of consumer-credit dynamics, with bivectors capturing rotational feedback.
- **ReCo: Reminder Composition Mitigates Hallucinations in Vision-Language Models** *(adjacent)* — Chytas, Choi, Kim, Singh. *arXiv, 2025-06-27.* [arxiv](https://arxiv.org/abs/2506.22636) · [openreview](https://openreview.net/forum?id=NpQ8QZjL3o)
  ReCo: a small trainable module built on geometric-algebra composition that reduces VLM hallucination. — also: Vision, remote sensing and medical imaging

### Clifford-valued network dynamics

Stability and synchronisation theory of recurrent networks with Clifford-number states (control theory).

- **Fast fixed-/preassigned-time synchronization of Clifford-valued neural networks for medical image encryption** *(adjacent)* — Zhang, Kou, Zhang, Liu. *Neurocomputing, 2025-10-01.* [doi](https://doi.org/10.1016/j.neucom.2025.130984)
  Fixed- and preassigned-time synchronisation of Clifford-valued networks for medical image encryption.
- **Further results on global stability of Clifford-valued neural networks subject to time-varying delays** *(adjacent)* — Manoj, Sriraman, Gurusamy, Shang. *Neurocomputing, 2025-10-01.* [doi](https://doi.org/10.1016/j.neucom.2025.130886)
  Global stability of Clifford-valued networks with time-varying delays.
- **Event-triggered control for exponential synchronization of reaction–diffusion fractional-order Clifford-valued delayed neural networks and its application to image encryption** *(adjacent)* — Sriraman, Manoj, Agarwal, Vigo-Aguiar et al.. *Neurocomputing, 2025-10-01.* [doi](https://doi.org/10.1016/j.neucom.2025.130604)
  Event-triggered synchronisation of reaction-diffusion fractional-order Clifford-valued networks, applied to image encryption.
- **Decomposition method-based global Mittag-Leffler synchronization for fractional-order Clifford-valued neural networks with transmission delays and impulses** *(adjacent)* — Sriraman, Manoj, Pratap. *Neural Networks, 2025-09-01.* [doi](https://doi.org/10.1016/j.neunet.2025.107564)
  Mittag-Leffler synchronisation of fractional-order Clifford-valued networks with transmission delays and impulses.
- **Synchronization of WPAA-Solution of Clifford-Valued Neural Network with Multiple Time-Varying Delays and Impulse Effects on Hybrid Domains** *(adjacent)* — Agrawal, Abbas. *Acta Applicandae Mathematicae, 2025-04-25.* [doi](https://doi.org/10.1007/s10440-025-00729-7)
  Synchronisation of weighted pseudo almost automorphic solutions of Clifford-valued networks on hybrid domains.
- **Mean-square stability of stochastic Clifford-valued Cohen-Grossberg neural networks with variable coefficients and time-varying delays** *(adjacent)* — Luo, Jiang, Tan. *Neurocomputing, 2025-04-01.* [doi](https://doi.org/10.1016/j.neucom.2025.129500)
  Mean-square stability of stochastic Clifford-valued Cohen-Grossberg networks with variable coefficients.
- **Global Mittag-Leffler stability and synchronization of fractional-order Clifford-valued delayed neural networks with reaction-diffusion terms and its application to image encryption** *(adjacent)* — Manoj, Sriraman. *Information Sciences, 2025-04-01.* [doi](https://doi.org/10.1016/j.ins.2024.121773)
  Mittag-Leffler stability of fractional-order reaction-diffusion Clifford-valued networks, applied to image encryption.
- **Fixed-time synchronization of Clifford-valued neural networks with D operator via aperiodic semi-intermittent control** *(adjacent)* — Wang, Li, Zheng. *Advances in Continuous and Discrete Models, 2025-02-13.* [doi](https://doi.org/10.1186/s13662-025-03911-1)
  Fixed-time synchronisation of Clifford-valued networks with a D operator via semi-intermittent control.
- **Besicovitch almost periodic solutions to Clifford-valued high-order Hopfield fuzzy neural networks with a $ D $ operator** *(adjacent)* — Li, , Ning, Li et al.. *AIMS Mathematics, 2025-01-01.* [doi](https://doi.org/10.3934/math.2025549)
  Besicovitch almost periodic solutions of Clifford-valued high-order Hopfield fuzzy networks with a D operator.

### Hypercomplex-valued networks

Quaternion, biquaternion and general hypercomplex layers framed as Clifford algebras.

- **Hypercomplex neural networks: Exploring quaternion, octonion, and beyond in deep learning** *(adjacent)* — Devadas, Hiremani, Preethi, T et al.. *MethodsX, 2025-12-01.* [doi](https://doi.org/10.1016/j.mex.2025.103644)
  Overview of quaternion, octonion and higher hypercomplex networks. — also: Surveys, reviews and position papers
- **Complex, Hypercomplex and Fuzzy-Valued Neural Networks** *(adjacent)* — Niemczynowicz, Perfilieva, García-Raffi, Kycia. *, 2025-11-06.* [doi](https://doi.org/10.1201/9781003515302)
  Book on complex-, hypercomplex- and fuzzy-valued neural networks. — also: Surveys, reviews and position papers
- **Asymptotically Stable Quaternion-valued Hopfield-structured Neural Network with Periodic Projection-based Supervised Learning Rules** *(adjacent)* — Wang, Ma, Pang. *arXiv, 2025-10-18.* [arxiv](https://arxiv.org/abs/2510.16607)
  Quaternion-valued Hopfield-structured network with stable supervised learning rules. — also: Clifford-valued network dynamics
- **A Study on Convolutional Vector-Valued Neural Networks for Color Image Classification** *(adjacent)* — Rodrigues, Valle. *Anais do XXII Encontro Nacional de Inteligência Artificial e Computacional (ENIAC 2025), 2025-09-29.* [doi](https://doi.org/10.5753/eniac.2025.11952)
  Convolutional vector-valued (including hypercomplex) networks for colour image classification. — also: Vision, remote sensing and medical imaging
- **Generalized Hypercomplex Neural Networks: Advancing Multi-Dimensional Feature Representation with Adaptive Quaternion-Based Architectures** *(adjacent)* — Hiremani, Devadas, Rani, K.R et al.. *, 2025-01-01.* [doi](https://doi.org/10.2139/ssrn.5084804)
  Generalised hypercomplex networks with adaptive quaternion-based layers.

### Other machine-learning uses of GA

Classifiers, filters, causal discovery and quantum machine learning built on GA.

- **A New Geometric Algebra-Based Classification of Hand Bradykinesia in Parkinson’s Disease Measured Using a Sensory Glove** *(adjacent)* — Saggio, Roselli, Pietrosanti, Romano et al.. *Algorithms, 2025-08-19.* [doi](https://doi.org/10.3390/a18080527)
  Geometric-algebra-based classification of Parkinson's bradykinesia from sensory-glove data.

### Software and resources

Libraries, curated lists and theme issues.

- **Geometric Algebra Software for Kinematics** *(context)* — Storti, Hadfield, Rockhold. *Volume 5: 21st IEEE/ASME International Conference on Mechatronic and Embedded Systems and Applications (MESA); 49th Mechanisms and Robotics Conference (MR), 2025-08-17.* [doi](https://doi.org/10.1115/detc2025-168739)
  Geometric-algebra software for kinematics. — also: Robotics, driving and control learning
- **GAGPT and Its Application to the Interactive Learning of Geometric Algebra** *(adjacent)* — Wang, Du, Zhao, Luo et al.. *Advances in Applied Clifford Algebras, 2025-05-30.* [doi](https://doi.org/10.1007/s00006-025-01385-8)
  GAGPT: a GPT-based assistant for interactive learning of geometric algebra. — also: Sequence models, language and LLM tooling

## 2024

### Surveys, reviews and position papers

Overviews of geometric-algebra (GA) neural networks and essays arguing for GA as a representation.

- **Deep Learning Using Geometric Algebra** — Bayro-Corrochano. *Geometric Algebra Applications Vol. III, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-66342-0_14)
  Book chapter on deep learning with geometric algebra. — also: New GA / Clifford architectures

### New GA / Clifford architectures

General-purpose layers and models whose weights or activations are multivectors.

- **Metric Learning for Clifford Group Equivariant Neural Networks** — Ali, Kulytė, Borde, Liò. *arXiv, 2024-07-13.* [arxiv](https://arxiv.org/abs/2407.09926) · [openreview](https://openreview.net/forum?id=4tw1U41t6Q)
  Learns the metric of the Clifford algebra inside CGENNs. — also: Theory and analysis of GA layers
- **Clifford-Steerable Convolutional Neural Networks** — Zhdanov, Ruhe, Weiler, Lucic et al.. *arXiv, 2024-02-22.* [arxiv](https://arxiv.org/abs/2402.14730) · [openreview](https://openreview.net/forum?id=eMvxweO8DK)
  Clifford-Steerable CNNs: E(p,q)-equivariant CNNs on multivector fields, including Poincaré equivariance. — also: Scientific machine learning

### Theory and analysis of GA layers

Expressivity, identifiability, group-theoretic and optimisation studies of GA networks.

- **Black Boxes and Looking Glasses: Multilevel Symmetries, Reflection Planes, and Convex Optimization in Deep Networks** — Zeger, Pilanci. *arXiv, 2024-10-05.* [arxiv](https://arxiv.org/abs/2410.04279) · [openreview](https://openreview.net/forum?id=vpo2K9Xivv)
  Convex Lasso reformulation of absolute-value networks whose features are expressed with geometric algebra.
- **Gradient of Clifford Neural Networks** — Maruyama, Alesiani. *D3S3 2024 Poster, 2024-09-30.* [openreview](https://openreview.net/forum?id=SjoBKs8gR6)
  Derives gradients and backpropagation for Clifford neural networks.
- **Randomized Geometric Algebra Methods for Convex Neural Networks** — Wang, Kim, Chu, Subramaniam et al.. *arXiv, 2024-06-04.* [arxiv](https://arxiv.org/abs/2406.02806)
  Randomised GA algorithms for training convex networks to global optimality, including LLM fine-tuning.
- **Universal Approximation Theorem for Vector- and Hypercomplex-Valued Neural Networks** *(adjacent)* — Valle, Vital, Vieira. *arXiv, 2024-01-04.* [arxiv](https://arxiv.org/abs/2401.02277) · [doi](https://doi.org/10.1016/j.neunet.2024.106632)
  Universal approximation theorem for vector- and hypercomplex-valued (including Clifford) networks. — also: Hypercomplex-valued networks

### Scientific machine learning

Molecules, materials, PDE surrogates, fluids, geoscience and lattice field theory.

- **Self-Supervised Learning for Ordered Three-Dimensional Structures** — Spellings, Martirossyan, Dshemuchadse. *arXiv, 2024-11-22.* [arxiv](https://arxiv.org/abs/2411.14680) · [openreview](https://openreview.net/forum?id=MkR9hN9twX)
  Self-supervised pretraining of rotation-equivariant geometric-algebra networks on ordered 3D structures.
- **Generating Highly Designable Proteins with Geometric Algebra Flow Matching** — Wagner, Seute, Viliuga, Wolf et al.. *arXiv, 2024-11-07.* [arxiv](https://arxiv.org/abs/2411.05238) · [doi](https://doi.org/10.52202/079017-2479) · [openreview](https://openreview.net/forum?id=k8O4pDqfXY)
  Clifford Frame Attention: geometric-algebra flow matching for designable protein backbones.
- **STAResNet: a Network in Spacetime Algebra to solve Maxwell's PDEs** — Pepe, Buchholz, Lasenby. *arXiv, 2024-08-24.* [arxiv](https://arxiv.org/abs/2408.13619)
  STAResNet: a ResNet in spacetime algebra for Maxwell's PDEs.
- **SINR: Equivariant Neural Vector Fields** *(adjacent)* — Ruhe, Forré. *ICML 2024 Workshop GRaM, 2024-06-17.* [openreview](https://openreview.net/forum?id=LgWZvbcjM1)
  SINR: steerable implicit neural representations for vector, multivector and tensor fields. — also: Vision, remote sensing and medical imaging
- **A Geometric Algebra-based machine learning method for Typhoon intensity forecasting** — haiyan, Huang, Li, Luo et al.. *, 2024-05-07.* [doi](https://doi.org/10.21203/rs.3.rs-4353217/v1)
  Geometric-algebra-based machine learning for typhoon intensity forecasting.
- **CLIFFORD NEURAL OPERATORS ON ATMOSPHERIC DATA INFLUENCED PARTIAL DIFFERENTIAL EQUATIONS** — Roy, Leong, Shinde, Phillips et al.. *AI4DiffEqtnsInSci @ ICLR 2024 Poster, 2024-03-03.* [openreview](https://openreview.net/forum?id=wFv3EeJtan)
  Clifford neural operators for atmospheric PDEs.
- **GA-ReLU: an activation function for Geometric Algebra Networks applied to 2D Navier-Stokes PDEs** — Pepe, Buchholz, Lasenby. *AI4DiffEqtnsInSci @ ICLR 2024 Poster, 2024-03-03.* [openreview](https://openreview.net/forum?id=T54Hn1e98P)
  GA-ReLU: an activation function for GA networks, tested on 2D Navier-Stokes. — also: New GA / Clifford architectures
- **Clifford Fourier Neural Operators for Learning the Multi-Components Elastic Wavefields** — Zhang, Innanen. *First EAGE Workshop on The Role of AI in FWI, 2024-01-01.* [doi](https://doi.org/10.3997/2214-4609.202481003)
  Clifford Fourier neural operators for multi-component elastic wavefields.
- **Geometric Algebra Models of Proteins for Three-Dimensional Structure Prediction: A Detailed Analysis** — Pepe, Lasenby, Chacon. *Springer Proceedings in Mathematics &amp; Statistics, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-55985-3_6)
  Detailed analysis of geometric-algebra protein features for 3D structure prediction.
- **Geometric Algebra Models of Proteins for Three-Dimensional Structure Prediction** — Pepe, Lasenby, Chacón. *Lecture Notes in Computer Science, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-34031-4_6)
  Geometric-algebra models of protein geometry as inputs to deep structure prediction.

### High-energy physics

The Lorentz-equivariant geometric algebra transformer (L-GATr) line and other spacetime-algebra models.

- **Interpolating amplitudes** *(context)* — Víctor, Gudrun, Vitaly, Anton. *arXiv, 2024-12-12.* [arxiv](https://arxiv.org/abs/2412.09534) · [doi](https://doi.org/10.21468/SciPostPhys.19.5.123) · [inspire](https://inspirehep.net/literature/2858900)
  Compares interpolation methods for loop amplitudes, including an MLP and L-GATr.
- **A Lorentz-Equivariant Transformer for All of the LHC** — Brehmer, Bresó, Haan, Plehn et al.. *arXiv, 2024-11-01.* [arxiv](https://arxiv.org/abs/2411.00446) · [doi](https://doi.org/10.21468/SciPostPhys.19.4.108) · [inspire](https://inspirehep.net/literature/2844989)
  L-GATr across LHC tasks: amplitude regression, record jet tagging and the first Lorentz-equivariant flow-matching event generator. — also: New GA / Clifford architectures
- **Geometric Graph Neural Network based track finding** — Garcia, Selvaggi, Francois. *Submitted to ICLR 2025, 2024-09-23.* [doi](https://doi.org/10.1051/epjconf/202533701125) · [openreview](https://openreview.net/forum?id=PgVo0t9rC2)
  GATr-based geometric GNN for charged-particle track finding. — also: Graphs, multimodal and representation learning
- **Lorentz-Equivariant Geometric Algebra Transformers for High-Energy Physics** — Spinner, Bresó, Haan, Plehn et al.. *arXiv, 2024-05-23.* [arxiv](https://arxiv.org/abs/2405.14806) · [doi](https://doi.org/10.52202/079017-0699) · [openreview](https://openreview.net/forum?id=X34GKv8sYT) · [inspire](https://inspirehep.net/literature/2789600)
  L-GATr: Lorentz-equivariant GATr in spacetime algebra for amplitude regression, jet tagging and generative modelling. — also: New GA / Clifford architectures

### Vision, remote sensing and medical imaging

Image backbones, hyperspectral classification, registration, segmentation and enhancement.

- **A Geometric algebra-enhanced network for skin lesion detection with diagnostic prior** — Wang, Ju, Zhu, Zhu et al.. *The Journal of Supercomputing, 2024-12-23.* [doi](https://doi.org/10.1007/s11227-024-06833-z)
  Geometric-algebra-enhanced network for skin-lesion detection with diagnostic priors.
- **Fuzzy Attention-based Geometric Algebra Convolution in Recursive Fusion Network for Medical Image Registration** — Zhou, Cao, Zhang. *2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM), 2024-12-03.* [doi](https://doi.org/10.1109/bibm62325.2024.10821956)
  Fuzzy-attention geometric-algebra convolution in a recursive fusion network for medical image registration.
- **Geometric Algebra Planes: Convex Implicit Neural Volumes** *(adjacent)* — Sivgin, Fridovich-Keil, Wetzstein, Pilanci. *arXiv, 2024-11-20.* [arxiv](https://arxiv.org/abs/2411.13525) · [openreview](https://openreview.net/forum?id=mVOz28mPHr)
  GA-Planes: convex-trainable implicit neural volumes from geometric-algebra-inspired feature planes. — also: Theory and analysis of GA layers
- **LaB-GATr: geometric algebra transformers for large biomedical surface and volume meshes** — Suk, Imre, Wolterink. *arXiv, 2024-03-12.* [arxiv](https://arxiv.org/abs/2403.07536) · [openreview](https://openreview.net/forum?id=19YEmVJCEt)
  LaB-GATr: GATr with learned pooling for large biomedical surface and volume meshes. — also: New GA / Clifford architectures
- **CGAPoseNet+GCAN: A Geometric Clifford Algebra Network for Geometry-aware Camera Pose Regression** — Pepe, Lasenby, Buchholz. *2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2024-01-03.* [doi](https://doi.org/10.1109/wacv57701.2024.00646)
  CGAPoseNet+GCAN: a Geometric Clifford Algebra Network head for geometry-aware camera pose regression. — also: Robotics, driving and control learning
- **CliffPhys: Camera-Based Respiratory Measurement Using Clifford Neural Networks** — Ghezzi, Boccignone, Grossi, Lanzarotti et al.. *ECCV (85) 2024, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-73013-9_13) · [openreview](https://openreview.net/forum?id=UvjVBmU5q6)
  CliffPhys: Clifford-algebra networks on optical-flow and depth fields for camera-based respiration measurement.
- **GASSF-Net: Geometric Algebra Based Spectral-Spatial Hierarchical Fusion Network for Hyperspectral and LiDAR Image Classification** — Wang, Ye, Huang, Ju et al.. *Remote. Sens. 2024, 2024-01-01.* [doi](https://doi.org/10.3390/rs16203825) · [openreview](https://openreview.net/forum?id=Up4l39tZqZ)
  GASSF-Net: GA-based spectral-spatial fusion for hyperspectral and LiDAR classification.
- **Point Cloud Upsampling with Geometric Algebra Driven Inverse Heat Dissipation** — Xu, Dai, Zheng, Li et al.. *ACM Multimedia 2024, 2024-01-01.* [openreview](https://openreview.net/forum?id=JjlO36YACl)
  Point-cloud upsampling with GA-driven inverse heat dissipation.
- **Clifford Convolutional Neural Networks: Concepts, Implementation, and an Application for Lymphoblast Image Classification** — Vieira, Valle, Lopes. *Springer Proceedings in Mathematics &amp; Statistics, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-55985-3_10)
  Concepts and implementation of Clifford convolutional networks, with lymphoblast image classification. — also: New GA / Clifford architectures
- **Clifford Convolutional Neural Networks for Lymphoblast Image Classification** — Vieira, Valle, Lopes. *Lecture Notes in Computer Science, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-34031-4_7)
  Clifford convolutional networks for lymphoblast image classification.

### Graphs, multimodal and representation learning

Graph neural networks, sheaves, binding / vector-symbolic representations with Clifford structure.

- **TeMME: Temporal Knowledge Graph Completion Using Multi-grade Multivector Embeddings** — Lu, Yu, Fan, Zhan et al.. *PRICAI (4) 2024, 2024-12-31.* [openreview](https://openreview.net/forum?id=5XWlgyy40k)
  TeMME: temporal knowledge-graph completion with multi-grade multivector embeddings.
- **A Clifford Algebraic Approach to E(n)-Equivariant High-order Graph Neural Networks** — Tran, Vo, Huu, Nguyen. *arXiv, 2024-10-07.* [arxiv](https://arxiv.org/abs/2410.04692)
  Clifford-algebraic E(n)-equivariant high-order graph neural networks.
- **Geometric Algebra based encoding for graph prompting** — Chytas, Chakraborty, Singh. *AccMLBio Poster, 2024-06-17.* [openreview](https://openreview.net/forum?id=LR2VeJEDFw)
  Geometric-algebra graph encoder for prompting LLMs with graphs. — also: Sequence models, language and LLM tooling
- **Multivector Neurons: Better and Faster O(n)-Equivariant Clifford Graph Neural Networks** — Liu, Ruhe, Forré. *arXiv, 2024-06-06.* [arxiv](https://arxiv.org/abs/2406.04052) · [openreview](https://openreview.net/forum?id=Jd62y3QfAQ)
  Multivector neurons: fast O(n)-equivariant Clifford message-passing GNNs. — also: New GA / Clifford architectures
- **Hierarchy-based Clifford Group Equivariant Message Passing Neural  Networks** — Maruyama, Alesiani. *AI4DiffEqtnsInSci @ ICLR 2024 Poster, 2024-03-03.* [openreview](https://openreview.net/forum?id=x16hXhSbS1)
  Clifford-group-equivariant U-Net message passing with hierarchical pooling. — also: Scientific machine learning
- **Clifford Group Equivariant Simplicial Message Passing Networks** — Liu, Ruhe, Eijkelboom, Forré. *arXiv, 2024-02-15.* [arxiv](https://arxiv.org/abs/2402.10011) · [openreview](https://openreview.net/forum?id=Zz594UBNOH)
  Clifford-group-equivariant message passing on simplicial complexes. — also: New GA / Clifford architectures
- **Embedding Knowledge Graphs in Degenerate Clifford Algebras** — Teyou, Demir, Ngomo. *arXiv, 2024-02-06.* [arxiv](https://arxiv.org/abs/2402.04870) · [openreview](https://openreview.net/forum?id=8dkw4oipJH)
  Knowledge-graph embeddings in degenerate Clifford algebras with nilpotent basis vectors.
- **A Study of LFT Embeddings in the Second Order Clifford Algebra Cl(R²,º)** — Amouzouvi, Vahdati, Song, Bainson et al.. *IEEE Access, 2024-01-01.* [doi](https://doi.org/10.1109/access.2024.3514303) · [openreview](https://openreview.net/forum?id=v9xeBAymtI)
  Knowledge-graph embeddings using linear fractional transformations in the second-order Clifford algebra.

### Wireless channels and neural fields

GA attention inside neural radiance fields and Gaussian splatting for radio propagation.

- **Differentiable and Learnable Wireless Simulation with Geometric Transformers** — Hehn, Peschl, Orekondy, Behboodi et al.. *arXiv, 2024-06-21.* [arxiv](https://arxiv.org/abs/2406.14995) · [openreview](https://openreview.net/forum?id=9TClCDZXeh)
  Wireless GATr: differentiable and learnable radio-propagation simulation with geometric algebra transformers.

### Sequence models, language and LLM tooling

Rotor-based recurrences, time-series forecasters, semantics and GA-aware language-model tools.

- **A Novel Spinor-Based Embedding Model for Transformers** — White. *arXiv, 2024-09-26.* [arxiv](https://arxiv.org/abs/2410.00038)
  Spinor-based word embeddings for transformers.
- **Geometric Algebra Meets Large Language Models: Instruction-Based Transformations of Separate Meshes in 3D, Interactive and Controllable Scenes** *(adjacent)* — Kolyvakis, Kamarianakis, Papagiannakis. *arXiv, 2024-08-05.* [arxiv](https://arxiv.org/abs/2408.02275) · [doi](https://doi.org/10.1109/isemv67326.2025.00018) · [openreview](https://openreview.net/forum?id=Wnsv1aVC6y)
  LLM instructions mapped to conformal-GA transformations for 3D scene editing. — also: Vision, remote sensing and medical imaging

### Clifford-valued network dynamics

Stability and synchronisation theory of recurrent networks with Clifford-number states (control theory).

- **Stability and synchronization of octonion-valued neural networks with leakage and mixed delays on time scales** *(adjacent)* — Popa. *Comput. Appl. Math. 2024, 2024-12-31.* [doi](https://doi.org/10.1007/s40314-024-02820-5) · [openreview](https://openreview.net/forum?id=Tkny3U2iEZ)
  Stability and synchronisation of octonion-valued networks with leakage and mixed delays on time scales. — also: Hypercomplex-valued networks
- **Stability analysis of Clifford-valued memristor-based neural networks with impulsive disturbances and its application to image encryption** *(adjacent)* — Zhao, Qiao. *Applied Mathematics and Computation, 2024-08-01.* [doi](https://doi.org/10.1016/j.amc.2024.128710)
  Stability of Clifford-valued memristor networks with impulses, applied to image encryption.
- **Lagrange stability criteria for hypercomplex neural networks with time varying delays** *(adjacent)* — Baluni, Yadav, Das. *Communications in Nonlinear Science and Numerical Simulation, 2024-04-01.* [doi](https://doi.org/10.1016/j.cnsns.2023.107765)
  Lagrange stability criteria for hypercomplex networks with time-varying delays. — also: Hypercomplex-valued networks
- **Synchronization of Hypercomplex Neural Networks with Mixed Time-Varying Delays** *(adjacent)* — Baluni, Yadav, Das, Cao. *Cognitive Computation, 2024-03-11.* [doi](https://doi.org/10.1007/s12559-024-10253-9)
  Synchronisation of hypercomplex-valued networks with mixed time-varying delays. — also: Hypercomplex-valued networks
- **Exponential synchronisation for delayed Clifford-valued coupled switched neural networks via static event-triggering rule** *(adjacent)* — Xing, Luan, Deng. *International Journal of Systems Science, 2024-01-18.* [doi](https://doi.org/10.1080/00207721.2023.2301498)
  Event-triggered exponential synchronisation of delayed Clifford-valued coupled switched networks.
- **Finite-time stability of weighted pseudo almost periodic solutions for Clifford-valued shunting inhibitory cellular neural networks with delays** *(adjacent)* — Gao, Dai. *International Journal of Control, 2024-01-03.* [doi](https://doi.org/10.1080/00207179.2023.2285901)
  Finite-time stability of weighted pseudo almost periodic solutions for Clifford-valued shunting-inhibitory networks.
- **Synchronization of Clifford-valued neural networks with leakage, time-varying, and infinite distributed delays on time scales** *(adjacent)* — Popa. *AIMS Mathematics, 2024-01-01.* [doi](https://doi.org/10.3934/math.2024915)
  Synchronisation of Clifford-valued networks with leakage, time-varying and infinite distributed delays on time scales.
- **Almost periodic solutions for Clifford-valued stochastic shunting inhibitory cellular neural networks with mixed delays** *(adjacent)* — Shao, Li. *AIMS Mathematics, 2024-01-01.* [doi](https://doi.org/10.3934/math.2024655)
  Stepanov almost periodic solutions of Clifford-valued stochastic shunting-inhibitory cellular networks.
- **Anti-Periodic Solutions on Clifford-Valued High-Order Cohen--Grossberg Neural Networks with Multi-Proportional Delays** *(adjacent)* — Luo, Qiu. *, 2024-01-01.* [doi](https://doi.org/10.2139/ssrn.4710728)
  Anti-periodic solutions of Clifford-valued high-order Cohen-Grossberg networks with proportional delays.
- **PSEUDO ALMOST PERIODIC SOLUTION OF FRACTIONAL-ORDER CLIFFORD-VALUED HIGH-ORDER HOPFIELD NEURAL NETWORKS** *(adjacent)* — Huo, Li. *Journal of Applied Analysis &amp; Computation, 2024-01-01.* [doi](https://doi.org/10.11948/20220447)
  Pseudo almost periodic solutions of fractional-order Clifford-valued high-order Hopfield networks.
- **Synchronization analysis of novel delayed dynamical Clifford-valued neural networks on timescales** *(adjacent)* — Arbi, Tahri. *Journal of Algorithms &amp; Computational Technology, 2024-01-01.* [doi](https://doi.org/10.1177/17483026241241492)
  Synchronisation of delayed Clifford-valued shunting-inhibitory cellular networks on time scales.

### Hypercomplex-valued networks

Quaternion, biquaternion and general hypercomplex layers framed as Clifford algebras.

- **Auxiliary Learning for Hypercomplex Neural Networks** *(adjacent)* — Buscarino, Famoso, Fortuna, Puglisi. *2024 IEEE 19th Conference on Industrial Electronics and Applications (ICIEA), 2024-08-05.* [doi](https://doi.org/10.1109/iciea61579.2024.10664949)
  Auxiliary-learning training scheme for hypercomplex networks.
- **KHNNs: hypercomplex neural networks computations via Keras using TensorFlow and PyTorch** *(adjacent)* — Niemczynowicz, Kycia. *arXiv, 2024-06-29.* [arxiv](https://arxiv.org/abs/2407.00452) · [doi](https://doi.org/10.1016/j.softx.2025.102163)
  KHNN: Keras library for hypercomplex-valued dense and convolutional layers (TensorFlow and PyTorch). — also: Software and resources
- **Demystifying the Hypercomplex: Inductive biases in hypercomplex deep learning** *(adjacent)* — Comminiello, Grassucci, Mandic, Uncini. *IEEE Signal Processing Magazine, 2024-05-01.* [doi](https://doi.org/10.1109/msp.2024.3401622)
  Tutorial on the inductive biases of hypercomplex deep learning. — also: Surveys, reviews and position papers
- **Towards Explaining Hypercomplex Neural Networks** *(adjacent)* — Lopez, Grassucci, Capriotti, Comminiello. *arXiv, 2024-03-26.* [arxiv](https://arxiv.org/abs/2403.17929) · [doi](https://doi.org/10.1109/ijcnn60899.2024.10650793)
  Explainability methods for hypercomplex networks.
- **Hypercomplex Techniques in Signal and Image Processing Using Network Graph Theory: Identifying core research directions** *(adjacent)* — Alcayde, Ventura, Montoya. *IEEE Signal Processing Magazine, 2024-03-14.* [doi](https://doi.org/10.1109/msp.2024.3365463)
  Network-graph analysis of research directions in hypercomplex signal and image processing. — also: Surveys, reviews and position papers
- **PHNNs: Lightweight Neural Networks via Parameterized Hypercomplex Convolutions** *(adjacent)* — Grassucci, Zhang, Comminiello. *IEEE Trans. Neural Networks Learn. Syst. 2024, 2024-01-01.* [doi](https://doi.org/10.1109/tnnls.2022.3226772) · [openreview](https://openreview.net/forum?id=ABt2BYyJHc)
  PHNNs: parameterised hypercomplex convolutions that learn the algebra rules, framed through Clifford algebras. — also: Vision, remote sensing and medical imaging

### Other machine-learning uses of GA

Classifiers, filters, causal discovery and quantum machine learning built on GA.

- **Optimization of Weight Function for (3+1)d Phonon Propagation in Weyl Fermion Sea Expressed by Clifford Algebra Using Elman Rnn and Echo State Network** *(adjacent)* — Furui, Santos. *Applied Mathematics and Sciences An International Journal (MathSJ), 2024-07-13.* [doi](https://doi.org/10.5121/mathsj.2024.11301)
  Elman and echo-state networks optimise a weight function for phonon propagation with Clifford-algebra (biquaternion) Weyl fermions.
- **Quantum Boltzmann machines: Clifford algebras and asymptotic analysis** *(adjacent)* — A.S., N.A.. *INSPIRE record, 2024-06-13.* [doi](https://doi.org/10.1088/1402-4896/ad52d3) · [inspire](https://inspirehep.net/literature/2798321)
  Symmetry and asymptotic analysis of quantum Boltzmann machines whose Hamiltonians are built from Clifford generators.
- **Geometric Algebra Quantum Convolutional Neural Network: A model using geometric (Clifford) algebras and quantum computing** *(adjacent)* — Altamirano-Escobedo, Bayro-Corrochano. *IEEE Signal Processing Magazine, 2024-03-14.* [doi](https://doi.org/10.1109/msp.2024.3369015)
  Geometric-algebra quantum convolutional network combining Clifford algebras and quantum computing.
- **Applications of Quantum Computing and Geometric Algebra Convolutional Neural Networks** *(adjacent)* — Bayro-Corrochano. *Geometric Algebra Applications Vol. III, 2024-01-01.* [doi](https://doi.org/10.1007/978-3-031-66342-0_21)
  Book chapter on quantum computing and geometric-algebra convolutional networks.

### Software and resources

Libraries, curated lists and theme issues.

- **GA-Unity: A Production-Ready Unity Package for Seamless Integration of Geometric Algebra in Networked Collaborative Applications** *(context)* — Kamarianakis, Lydatakis, Papagiannakis. *arXiv, 2024-06-17.* [arxiv](https://arxiv.org/abs/2406.11560)
  GA-Unity: Unity package for geometric algebra in networked collaborative applications.

## 2023

### Surveys, reviews and position papers

Overviews of geometric-algebra (GA) neural networks and essays arguing for GA as a representation.

- **Survey of New Applications of Geometric Algebra** *(adjacent)* — Hitzer, Kamarianakis, Papagiannakis, Vašík. *, 2023-02-20.* [doi](https://doi.org/10.22541/au.167687105.52780013/v1) · [inspire](https://inspirehep.net/literature/2822586)
  Survey of 101 GA application papers from 2022, including machine-learning uses.

### New GA / Clifford architectures

General-purpose layers and models whose weights or activations are multivectors.

- **Euclidean, Projective, Conformal: Choosing a Geometric Algebra for Equivariant Transformers** — Haan, Cohen, Brehmer. *arXiv, 2023-11-08.* [arxiv](https://arxiv.org/abs/2311.04744) · [openreview](https://openreview.net/forum?id=MeuKVydtOf)
  Generalises GATr to any geometric algebra and compares Euclidean, projective and conformal versions. — also: Theory and analysis of GA layers
- **Geometric Algebra Transformer** — Brehmer, Haan, Behrends, Cohen. *arXiv, 2023-05-28.* [arxiv](https://arxiv.org/abs/2305.18415) · [doi](https://doi.org/10.52202/075280-1541) · [openreview](https://openreview.net/forum?id=M7r2CO4tJC)
  GATr: an E(3)-equivariant transformer on projective-GA multivectors for geometric data across domains.
- **Clifford Group Equivariant Neural Networks** — Ruhe, Brandstetter, Forré. *arXiv, 2023-05-18.* [arxiv](https://arxiv.org/abs/2305.11141) · [doi](https://doi.org/10.52202/075280-2748) · [openreview](https://openreview.net/forum?id=n84bzMrGUD) · [inspire](https://inspirehep.net/literature/2661033)
  CGENN: O(n)- and E(n)-equivariant layers that act on multivectors through the Clifford group. — also: Theory and analysis of GA layers
- **Geometric Clifford Algebra Networks** — Ruhe, Gupta, Keninck, Welling et al.. *arXiv, 2023-02-13.* [arxiv](https://arxiv.org/abs/2302.06594) · [openreview](https://openreview.net/forum?id=DNAJdkHPQ5)
  Geometric Clifford Algebra Networks (GCANs): Pin(p,q,r) group actions in plane-based GA as layers for dynamical systems. — also: Scientific machine learning

### Theory and analysis of GA layers

Expressivity, identifiability, group-theoretic and optimisation studies of GA networks.

- **From Complexity to Clarity: Analytical Expressions of Deep Neural Network Weights via Clifford's Geometric Algebra and Convexity** — Pilanci. *arXiv, 2023-09-28.* [arxiv](https://arxiv.org/abs/2309.16512) · [openreview](https://openreview.net/forum?id=6XPwSwEWsV)
  Optimal weights of deep ReLU networks are wedge products of training samples; training reduces to convex GA problems.

### Scientific machine learning

Molecules, materials, PDE surrogates, fluids, geoscience and lattice field theory.

- **Clifford Group Equivariant Neural Network Layers for Protein Structure Prediction** — Pepe, Buchholz, Lasenby. *NLDL 2024, 2023-11-03.* [openreview](https://openreview.net/forum?id=JNfpsiGS5E)
  CGENN layers inside a protein-structure-prediction pipeline.
- **Modeling orientational features via Geometric Algebra for 3D protein coordinates prediction** — Pepe, Lasenby. *, 2023-01-18.* [doi](https://doi.org/10.22541/au.167403971.10736333/v1)
  Geometric-algebra orientational features for deep-learning protein-structure prediction.
- **Using a Graph Transformer Network to Predict 3D Coordinates of Proteins via Geometric Algebra Modelling** — Pepe, Lasenby, Chacón. *Lecture Notes in Computer Science, 2023-01-01.* [doi](https://doi.org/10.1007/978-3-031-30923-6_7)
  Graph transformer that predicts 3D protein coordinates from geometric-algebra models of the backbone.

### Vision, remote sensing and medical imaging

Image backbones, hyperspectral classification, registration, segmentation and enhancement.

- **Geometric algebra-based multiscale encoder-decoder networks for 3D motion prediction** — Zhong, Cao. *Applied Intelligence, 2023-08-31.* [doi](https://doi.org/10.1007/s10489-023-04908-7)
  Geometric-algebra multiscale encoder-decoder for 3D human motion prediction.
- **Geometric algebra-based multiview interaction networks for 3D human motion prediction** — Zhong, Cao. *Pattern Recognition, 2023-06-01.* [doi](https://doi.org/10.1016/j.patcog.2023.109427)
  Geometric-algebra multiview interaction networks for 3D human motion prediction.
- **CGA-PoseNet: Camera Pose Regression via a 1D-Up Approach to Conformal Geometric Algebra** — Pepe, Lasenby. *arXiv, 2023-02-10.* [arxiv](https://arxiv.org/abs/2302.05211)
  CGA-PoseNet: camera pose regression that predicts a conformal-GA motor via the 1D-Up embedding. — also: Robotics, driving and control learning

### Graphs, multimodal and representation learning

Graph neural networks, sheaves, binding / vector-symbolic representations with Clifford structure.

- **UniSG^GA: A 3D scenegraph powered by Geometric Algebra unifying geometry, behavior and GNNs towards generative AI** — Kamarianakis, Protopsaltis, Angelis, Zikas et al.. *arXiv, 2023-06-18.* [arxiv](https://arxiv.org/abs/2306.10621)
  UniSG^GA: a 3D scene graph with GA-encoded geometry and behaviour, designed for GNN-based generative models.
- **Clifford Embeddings - A Generalized Approach for Embedding in Normed Algebras** — Demir, Ngomo. *ECML/PKDD (3) 2023, 2023-01-01.* [openreview](https://openreview.net/forum?id=EddLV8fHQ4)
  Knowledge-graph embeddings in Clifford algebras Cl(p,q) that generalise complex and quaternion models and pick the algebra per graph.
- **Block Decomposition with Multi-granularity Embedding for Temporal Knowledge Graph Completion** *(context)* — Yue, Ren, Zeng, Zhang et al.. *DASFAA (2) 2023, 2023-01-01.* [openreview](https://openreview.net/forum?id=9ilzCQOOPw)
  Block-decomposition temporal knowledge-graph embedding; matched "multivector" but its use of geometric algebra is unverified (abstract unavailable).

### Clifford-valued network dynamics

Stability and synchronisation theory of recurrent networks with Clifford-number states (control theory).

- **Global Exponential Synchronization of Clifford-Valued Memristive Fuzzy Neural Networks with Delayed Impulses** *(adjacent)* — Zhao, Qiao. *Cognitive Computation, 2023-12-07.* [doi](https://doi.org/10.1007/s12559-023-10221-9)
  Exponential synchronisation of Clifford-valued memristive fuzzy networks with delayed impulses.
- **Global exponential stability for delayed Clifford-valued coupled neural networks with impulsive effects** *(adjacent)* — Xing, Luan, Deng. *Journal of the Franklin Institute, 2023-12-01.* [doi](https://doi.org/10.1016/j.jfranklin.2023.09.024)
  Global exponential stability of delayed Clifford-valued coupled networks with impulses.
- **Multistability of almost periodic solution for Clifford-valued Cohen–Grossberg neural networks with mixed time delays** *(adjacent)* — Zhang, Qiao, Duan, Miao. *Chaos, Solitons &amp; Fractals, 2023-11-01.* [doi](https://doi.org/10.1016/j.chaos.2023.114100)
  Multistability of almost periodic solutions for Clifford-valued Cohen-Grossberg networks with mixed delays.
- **Weighted Pseudo Almost Periodic Synchronization for Clifford-Valued Neural Networks with Leakage Delay and Proportional Delay** *(adjacent)* — Gao, Huang, Dai. *Acta Applicandae Mathematicae, 2023-07-05.* [doi](https://doi.org/10.1007/s10440-023-00587-1)
  Weighted pseudo almost periodic synchronisation of Clifford-valued networks with leakage and proportional delays.
- **Besicovitch-like almost periodic solutions of Clifford-valued fuzzy cellular neural networks with time-varying delays** *(adjacent)* — Li, Zhang, Li. *Franklin Open, 2023-03-01.* [doi](https://doi.org/10.1016/j.fraope.2023.100011)
  Besicovitch-like almost periodic solutions of Clifford-valued fuzzy cellular networks with time-varying delays.
- **A spectral radius-based global exponential stability for Clifford-valued recurrent neural networks involving time-varying delays and distributed delays** *(adjacent)* — Assali. *Computational and Applied Mathematics, 2023-01-18.* [doi](https://doi.org/10.1007/s40314-023-02188-y)
  Spectral-radius criterion for global exponential stability of Clifford-valued recurrent networks with time-varying and distributed delays.
- **Clifford-Valued Distributed Optimization Based on Recurrent Neural Networks** *(adjacent)* — Xia, Liu, Kou, Wang. *IEEE Trans. Neural Networks Learn. Syst. 2023, 2023-01-01.* [doi](https://doi.org/10.1109/tnnls.2021.3139865) · [openreview](https://openreview.net/forum?id=JGWto1xzTQ)
  Networks of Clifford-valued recurrent neural networks that solve constrained distributed convex optimisation through a generalised Clifford gradient.
- **System decomposition method-based global stability criteria for T-S fuzzy Clifford-valued delayed neural networks with impulses and leakage term** *(adjacent)* — Alanazi, Sriraman, Gurusamy, Athithan et al.. *AIMS Mathematics, 2023-01-01.* [doi](https://doi.org/10.3934/math.2023774)
  Global asymptotic stability of Takagi-Sugeno fuzzy Clifford-valued delayed networks with impulses and leakage, via system decomposition.
- **Besicovitch almost periodic stochastic processes and almost periodic solutions of Clifford-valued stochastic neural networks** *(adjacent)* — Li, , Wang. *Discrete and Continuous Dynamical Systems - B, 2023-01-01.* [doi](https://doi.org/10.3934/dcdsb.2022162)
  Existence of Besicovitch almost periodic solutions for Clifford-valued stochastic neural networks.
- **System Decomposition Method-Based Exponential Stability of Clifford-Valued BAM Delayed Neural Networks** *(adjacent)* — Sriraman, Balaji, Veerasivaji. *IEEE Access, 2023-01-01.* [doi](https://doi.org/10.1109/access.2023.3295420)
  Exponential stability of Clifford-valued BAM delayed networks by system decomposition.

### Hypercomplex-valued networks

Quaternion, biquaternion and general hypercomplex layers framed as Clifford algebras.

- **PHYDI: Initializing Parameterized Hypercomplex Neural Networks as Identity Functions** *(adjacent)* — Mancanelli, Grassucci, Uncini, Comminiello. *arXiv, 2023-10-11.* [arxiv](https://arxiv.org/abs/2310.07612) · [doi](https://doi.org/10.1109/mlsp55844.2023.10285926)
  PHYDI: identity initialisation that stabilises large parameterised hypercomplex networks.
- **Generalization of Neural Networks on Second-Order Hypercomplex Numbers** *(adjacent)* — Pavlov, Kozlov, Bakulin, Zuev et al.. *Mathematics, 2023-09-19.* [doi](https://doi.org/10.3390/math11183973)
  Neural networks over general second-order hypercomplex numbers (complex, dual, hyperbolic).
- **Understanding Vector-Valued Neural Networks and Their Relationship with Real and Hypercomplex-Valued Neural Networks** *(adjacent)* — Valle. *arXiv, 2023-09-14.* [arxiv](https://arxiv.org/abs/2309.07716) · [doi](https://doi.org/10.1109/msp.2024.3401621)
  Relates vector-valued networks to real- and hypercomplex-valued networks. — also: Theory and analysis of GA layers
- **Deep Separable Hypercomplex Networks** *(adjacent)* — Shahadat, Maida. *The International FLAIRS Conference Proceedings, 2023-05-08.* [doi](https://doi.org/10.32473/flairs.36.133540)
  Depthwise-separable hypercomplex convolutions that cut the parameter cost of hypercomplex CNNs. — also: Vision, remote sensing and medical imaging
- **Enhancing ResNet Image Classification Performance by using Parameterized Hypercomplex Multiplication** *(adjacent)* — Shahadat, Maida. *arXiv, 2023-01-11.* [arxiv](https://arxiv.org/abs/2301.04623)
  Parameterised hypercomplex multiplication in ResNet front ends for image classification. — also: Vision, remote sensing and medical imaging
- **Self-Organizing Hypercomplex-Valued Adaptive Network** *(adjacent)* — Hazubski, Hoppe. *, 2023-01-01.* [doi](https://doi.org/10.2139/ssrn.4495222)
  Self-organising adaptive network with hypercomplex-valued weights.

### Other machine-learning uses of GA

Classifiers, filters, causal discovery and quantum machine learning built on GA.

- **Machine Learning Clifford invariants of ADE Coxeter elements** *(adjacent)* — Chen, Dechant, He, Heyes et al.. *arXiv, 2023-09-29.* [arxiv](https://arxiv.org/abs/2310.00041) · [doi](https://doi.org/10.1007/s00006-024-01325-y) · [inspire](https://inspirehep.net/literature/2705182)
  Machine learning applied to Clifford invariants of ADE Coxeter elements.
- **Logarithmic-Function And Geometric Algebra Based Variable Step-size Least Mean Square Algorithm.** *(adjacent)* — shahzad, WANG. *, 2023-09-18.* [doi](https://doi.org/10.21203/rs.3.rs-3334278/v1)
  GA-based variable step-size least-mean-square adaptive filter with a logarithmic cost.
- **Clifford Algebras, Quantum Neural Networks and Generalized Quantum Fourier Transform** *(adjacent)* — Trindade, Lula-Rocha, Floquet. *Advances in Applied Clifford Algebras, 2023-06-13.* [doi](https://doi.org/10.1007/s00006-023-01279-7)
  Clifford algebras as a framework for quantum neural networks and a generalised quantum Fourier transform.

### Software and resources

Libraries, curated lists and theme issues.

- **gafro: Geometric Algebra for Robotics** *(context)* — Löw, Abbet, Calinon. *arXiv, 2023-10-29.* [arxiv](https://arxiv.org/abs/2310.19090)
  gafro: C++ geometric-algebra library for robot kinematics and optimisation. — also: Robotics, driving and control learning
- **Project Elements: A computational entity-component-system in a scene-graph pythonic framework, for a neural, geometric computer graphics curriculum** *(adjacent)* — Papagiannakis, Kamarianakis, Protopsaltis, Angelis et al.. *arXiv, 2023-02-15.* [arxiv](https://arxiv.org/abs/2302.07691)
  Educational Python graphics framework (entity-component-system and scene graph) that combines GA with neural components.
