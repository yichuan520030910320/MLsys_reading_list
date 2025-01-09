# MLsys_reading_list
### RAG
#### COT+RAG
1.RAT:(COT+RAG) PKU--> https://arxiv.org/html/2403.05313v1

2.PlanxRAG: MSR---> https://arxiv.org/pdf/2410.20753

3.https://arxiv.org/pdf/2410.01782 --> OPEN-RAG: Enhanced Retrieval-Augmented Reasoning with Open-Source Large Language Models
#### Scaling
1.Inference Scaling for Long-Context Retrieval Augmented Generation : Deepmind-->https://arxiv.org/abs/2410.04343 high score in iclr

2.Scaling Retrieval-Based Language Models with a Trillion-Token Datastore: UW rulin shao-->https://arxiv.org/abs/2407.12854
#### RAG system paper
1.Cornell:Towards Understanding Systems Trade-offs in Retrieval-Augmented Generation Model Inference-->https://arxiv.org/abs/2412.11854

2.juncheng&Ravi: RAGserve-->https://arxiv.org/html/2412.10543v1

3.EdgeRAG: https://arxiv.org/pdf/2412.21023
#### New Algorithm
1.Self RAG: https://arxiv.org/abs/2310.11511 400+ citation

#### Grounding & Trustworthy
1.Trust align rag: 888 in iclr-->https://arxiv.org/pdf/2409.11242

2.ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability https://arxiv.org/abs/2410.11414 7.33 in ICLR24

3.ENHANCING LARGE LANGUAGE MODELS’ SITUATED FAITHFULNESS TO EXTERNAL CONTEXTS:https://arxiv.org/abs/2410.14675 (when LLM meet wrong context)

#### GraphRAG and Knowledge Graph Related
1.STructRAG: https://arxiv.org/abs/2410.08815

2.GraphCOT:https://arxiv.org/abs/2404.07103

3.Benchmark:MMQA: Evaluating LLMs with Multi-Table Multi-Hop Complex Questions  https://openreview.net/forum?id=GGlpykXDCa 888 iclr24

4.SIRERAG: INDEXING SIMILAR AND RELATED INFORMATION FOR MULTIHOP REASONING:https://arxiv.org/abs/2412.06206 
#### When Long context meets RAG
1.RankRAG by Nvidia: high score in nips24: https://arxiv.org/abs/2407.02485

2.Accelerating Inference of Retrieval-Augmented Generation via Sparse Context Selection:-->https://arxiv.org/abs/2405.16178

related:https://openreview.net/forum?id=TDy5Ih78b4-->Provence: efficient and robust context pruning for retrieval-augmented generation (Prune for RAG)

3.Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG--> DeepMind JiaWei: https://arxiv.org/abs/2410.05983

4.https://www.databricks.com/blog/author/quinn-leng Databricks Tech Report
#### Retriever?
1.Sufficient Context: A New Lens on Retrieval Augmented Generation Systems--> https://arxiv.org/abs/2411.06037

2.LLM-Augmented Retrieval: Enhancing Retrieval Models Through Language Models and Doc-Level Embedding https://openreview.net/forum?id=7yncrX80CN
#### Chat and RAG
1.ChatQA:https://arxiv.org/pdf/2401.10225 NIPS high score NVIDIA Followed bt ChatQA2

#### Multimodal
1.VisRAG:https://arxiv.org/abs/2410.10594

### 

### Fault tolerance

1. Mitigating Stragglers in the Decentralized Training on Heterogeneous Clusters
2. [CPR: Understanding and Improving Failure Tolerant Training for Deep Learning Recommendation with Partial Recovery](https://cs.stanford.edu/people/trippel/pubs/cpr-mlsys-21.pdf)
3. [Efficient Replica Maintenance for Distributed Storage Systems](https://www.usenix.org/legacy/event/nsdi06/tech/full_papers/chun/chun.pdf)
4. GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints SOSP 23 RICE
5. https://www.abstractsonline.com/pp8/#!/10856/presentation/9287
6. [Elastic Averaging for Efficient Pipelined DNN Training](https://conf.researchr.org/track/PPoPP-2023/PPoPP-2023-papers#)
7. [Understanding the Effects of Permanent Faults in GPU's Parallelism Management and Control Units | Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis](https://dl.acm.org/doi/10.1145/3581784.3607086)
8. [Straggler-Resistant Distributed Matrix Computation via Coding Theory: Removing a Bottleneck in Large-Scale Data Processing](https://www.ece.iastate.edu/adityar/files/2020/05/RamDT_SPMag20.pdf)
9. [[1901.05162\] Coded Matrix Multiplication on a Group-Based Model](https://arxiv.org/abs/1901.05162)
10. [Parity models: erasure-coded resilience for prediction serving systems](https://dl.acm.org/doi/10.1145/3341301.3359654)
11. [[2002.02440\] A locality-based approach for coded computation](https://arxiv.org/abs/2002.02440)
12. [Straggler Mitigation in Distributed Optimization Through Data Encoding](https://proceedings.neurips.cc/paper_files/paper/2017/file/663772ea088360f95bac3dc7ffb841be-Paper.pdf)
13. [Learning Effective Straggler Mitigation from Experience and Modeling](https://par.nsf.gov/servlets/purl/10112719)
14. [Straggler-Resistant Distributed Matrix Computation via Coding Theory: Removing a Bottleneck in Large-Scale Data Processing](https://www.ece.iastate.edu/adityar/files/2020/05/RamDT_SPMag20.pdf)
15. [Bamboo: Making Preemptible Instances Resilient for Affordable Training of Large DNNs](https://www.usenix.org/conference/nsdi23/presentation/thorpe) NSDI23 UCLA
16. [Varuna: Scalable, Low-cost Training of Massive Deep Learning Models](https://arxiv.org/pdf/2111.04007.pdf) Eurosys21 MSR
17. [Swift: Expedited Failure Recovery for Large-scale DNN Training](https://i.cs.hku.hk/~cwu/papers/yczhong-ppopp23-poster.pdf) PPoPP23 HKU
18. [Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=Dzh5C9EAAAAJ&sortby=pubdate&citation_for_view=Dzh5C9EAAAAJ:SAZ1SQo2q1kC) SOSP23 Umich 


### LLM Program

#### system
1. CUHK https://arxiv.org/pdf/2407.00326 Optimize LLM application


### New Recommendation system

1. Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU Meta new recsys linear and attention mixed)


### MCTS powered LLM

#### Algorithmn

1. MCTSr: https://arxiv.org/html/2406.07394v1  Code: https://github.com/trotsky1997/MathBlackBox

2. Rest MCTS ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search THU NIPS24 code: https://github.com/THUDM/ReST-MCTS

3. SC-MCTS https://arxiv.org/abs/2410.01707 new lol THU


### LLM for Video Understanding

1. repo : https://github.com/yunlong10/Awesome-LLMs-for-Video-Understanding

### Diffusion Model System
#### Model
1.DDPM https://arxiv.org/abs/2006.11239 cannot skip timesteps

2.DDIM https://arxiv.org/abs/2010.02502 can skip timesteps

3.Latent Diffusion Models https://github.com/CompVis/latent-diffusion in latent space rather than pixel space

4.[NSDI24] Approximate Caching for Efficiently Serving Diffusion Models https://arxiv.org/abs/2312.04429

5.LoRA for diffusion model parameter efficient finetune

6.ControlNet https://arxiv.org/abs/2302.05543

7.Video Diffusion Model https://arxiv.org/abs/2204.03458 3D unet More: https://github.com/showlab/Awesome-Video-Diffusion
