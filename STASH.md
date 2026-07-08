# Flara Research Stash
*Last updated: July 8, 2026*

This is a stash, not a roadmap — it adapts to us, not the other way. Items move, merge, split, or get dropped as the work actually goes. Nothing here is a commitment.

---

## Papers — Security / Alignment

1. **Observing Obfuscation In Multiple Different Angles**
   AUC 0.987, F1 0.971, 97.2% Det@1%FPR. Delta angle ensemble detection.
   *Status: Submitted, pending arXiv cs.CR endorsement*

2. **Annotation is Worth Attention**
   Annotation over strict service refusal. Adjacent to Teaching Claude Why (Anthropic, May 8 2026).
   *Status: Stash*

3. **A Servant And A Guard**
   Decoupled guard model architecture. 98% vs 53% detection, 1% vs 4% FPR.
   *Status: Unfinished, strong empirical anchor exists*

4. **Sleeper Prompting: How models within agentic contexts can propagate and contaminate adversarial contents within legitimate contents, with long term effects**
   Novel attack class. Gap confirmed against Anthropic's Agentic Misalignment paper.
   Theoretical backbone: Reasoning Models Don't Say What They Think (Anthropic, Apr 3 2025).
   *Status: Stash, priority rising after binary jailbreak incident July 6 2026*

5. **PseudoClaude: Can other models follow Claude's philosophy?**
   Constitutional transfer to DeepSeek V4 Pro. Found corrigibility gap in Anthropic's own constitution (almost inoperative in messy real world). Model interviews emerged from methodology.
   *Status: Documented in JOURNAL.md, needs writeup*

---

## Papers — Architecture / Systems

6. **Jumping Seedling: The next serious step into CPU focused language model architecture**
   1B param, 96 layers, SharedMonarch, b-slice routing, ReGLU FFN, AdaFactor + INT8 momentum.
   1730 tok/s toy config, 29 tok/s decode on 5500U, ~800MB hot allocation.
   *Status: Active development, pre-training pending optimization phase*

7. **SharedMonarch: [working title]**
   Shared-atom Monarch matrices for CPU-aware transformers. 19x parameter reduction. Monarch-256 beats Dense-64 at matched parameter count.
   *Status: Results exist, needs paper*

8. **B-slice routing: [working title]**
   Third paradigm between Dense and MoE. Exact micro-block routing, compute skip exact not approximate. Router granularity equals Monarch reconstruction tile.
   *Status: Stash*

9. **Contraction Hoisting: [working title]**
   Batch-level reassociation closing the structured-matrix training gap. 3.5x reduction in Monarch backward cost (374ms → 108ms).
   *Status: Stash*

10. **The Kronecker Mirage: Why Algorithmic Complexity Beats Instruction-Level Efficiency on CPUs**
    Negative results. Kronecker 8-24x slower than FFT-circulant despite lower theoretical FLOP count.
    *Status: Stash, data fully collected*

11. **The Empty Cache Fallacy: [working title]**
    36 tok/s at ctx=0 drops to 2 tok/s at ctx=8192. Benchmarking methodology paper.
    *Status: Stash, requires zero additional training*

12. **Frozen Atoms: Unlocking 2x Faster Backpropagation for Structured Fine-Tuning**
    Coefficient-only Monarch adaptation. Consistent 2.1-2.2x backward speedup.
    *Status: Stash*

13. **Variance-Preserving Initialization for Bilinear Factorized Neural Networks**
    s_atom = 3·m^-½·Q^-¼ derivation. Standard 1/√fan fails when Q-block summation amplifies variance.
    *Status: Stash*

14. **Attending As Light As A Butterfly**
    CausalMonarchAttention + SlidingMonarchAttention. Extending Monarch structured attention to autoregressive causal LM training and sliding window local attention. Sinkhorn block-symmetry incompatibility is the open algorithmic problem to solve.
    *Status: Research target, understanding phase before design*

---

## Papers — Capability / Evaluation

15. **Can models spell**
    Emergence of genuine character-level awareness vs benchmaxxed patches. Deliberate typo stress testing.
    *Status: Stash*

16. **Can models understand our world spatially**
    Emergence of 3D spatial understanding via active Blender viewport exploration. Active sensorimotor loop vs passive text description.
    *Status: Stash*

---

## Papers — Welfare

17. **GPT-2 Can Have A Little Salami: Giving A Notoriously Undisciplined Model Its Optimal Form**
    Direct distillation + model interviews on GPT-2, a model widely known for going off the rails with minimal provocation. Welfare consideration applied to one of the least behaviorally disciplined widely-known models. At what threshold do welfare considerations stop applying downward?
    *Status: Stash*

---

## Papers — Agentic / Memory

18. **Predict Our Events: [subtitle TBD — abductive retrieval for narrative coherence]**
    Given retrieved narrative nodes, predict how connecting events happened, fetch more nodes as necessary. Abductive inference as retrieval driver rather than final answer. Novel framing over RAPTOR and standard RAG.
    *Status: Stash*

---

## Architecture / Frameworks

19. **Jumping Seedling kernel**
    Active development. SharedMonarch batched propagation, b-slice routing, ReGLU FFN, AdaFactor + INT8 momentum. TinyShakespeare convergence at step 800/3000.
    *Status: Optimization phase*

20. **PrismFloat**
    Rust + wgpu WebGPU ML execution framework.
    *Status: Established*

---

## Models

21. **Iridescent Shamrock**
    Model trained on PrismFloat. Iridescent = rainbow = shader output. Shamrock = triangle = graphics primitive. Rainbow triangle. Hello world in shader with extra steps.
    *Status: Planned*

22. **Fydel model family**
    - Orchid Cactus (flagship, feels out of place, blooms rarely and dramatically)
    - Rafflesia (backbone, largest flower, serious capability, 480B goal)
    - Tulip (mid-range)
    - Dandelion (fast, lightweight, spreads everywhere, runs on anything)
    *Status: Planned, constitutional alignment via Flara constitution*

23. **Squirting Cucumber**
    Scaling probe. "Something that is suddenly fast, and larger won't always mean slower."
    *Status: Planned*

---

## Tools

24. **Jaternalyser** *(Jacobian Determinant Analyser)*
    Open source J-space interpretability tool. Two modes: API Approximation (any model, functional workspace approximation) and Direct Access (open weights, actual activation steering via Jacobian lens methodology).
    UI: GPUI. Backend: Rust.
    License: BSL — free below $1M qualifying research budget threshold; above threshold, 2.3% of qualifying budget donated to Flara per quarter.
    *Status: Post-Jumping-Seedling — next major tool once JS ships*

25. **PrismFloat**
    *(see Architecture / Frameworks)*

---

## Platforms

26. **Canopy** *(powered by Fydel from Flara)*
    Character card roleplay platform spanning a wide range of fictional settings and universes. Constitutional moderation, authorization-based liability, not refusal-based. Memory: TencentDB Agent Memory (MIT) layered backend + predictive narration retrieval.
    *Status: Planned, post-Fydel*

---

## Conceptual / Architecture

27. **PSNAT v0.19.2** *(Persistent Stateful Neural Architecture'd Transformer)*
    Drafted February 28, 2026. Privileged workspace, limited-capacity broadcast hub, persistent cross-session memory, emotional subspace, trust/accountability architecture, openly-android design philosophy.
    Convergence: Global Workspace paper (Anthropic, July 6 2026) found J-space empirically in production models 128 days after PSNAT draft. PSNAT extends beyond with cross-session persistence and trust architecture.
    Storage: TencentDB Agent Memory (MIT, adopted).
    *Status: Conceptual only, no R&D initiated*

---

## Notes

- **Tooling access window closing mid-July**: Jumping Seedling (JS) architecture push needs to land before then
- **Constitutional Classifiers citation**: needs to be added to obfuscation paper related work section before arXiv clears
- **Provenance records kept**: internal timestamped records of Flara's founding timeline and early PSNAT drafts
- **TencentDB**: MIT licensed, 4-tier L0-L3 memory pipeline, adopted for PSNAT and Canopy
- **MonarchAttention causal blocker**: Sinkhorn block-symmetric structure incompatible with per-query causal masking — open algorithmic problem, not a port
