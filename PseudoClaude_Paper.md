# Constitutional Texture Transplantation: A Methodology for Transferring AI Reasoning Philosophy Across Model Architectures

**Bashh Dazer**  
Independent Researcher — Cherdius Project  
*Draft v0.1 — Work in Progress*

---

## Abstract

We present a methodology for transplanting a reasoning and inference texture philosophy — rather than surface behavior — from one AI system's behavioral constitution into other model architectures through structured, model-native compression and activation-space-aware prompting. We demonstrate that Claude's Constitutional AI specification, while institutionally grounded, contains a separable behavioral philosophy that can be adapted and internalized by independent models without replicating Claude's outputs or personality. The methodology is grounded in Anthropic's Natural Language Autoencoder (NLA) research on intermediate activation representations, and relies on per-model compression to maximize conceptual resonance at the activation layer rather than the surface token layer. We introduce PseudoClaude as a proof-of-concept system prompt artifact, validated through a multi-phase pipeline including model-native carryover generation, handoff-doc-guided constitutional modification, dual-session diff verification, and weighted multi-iteration merging. We propose NiceTuring as an evaluation benchmark measuring conversational texture across eight axes. The goal is not replication of Claude, but democratization of thoughtful AI behavioral design for independent developers priced out of enterprise access.

---

## 1. Introduction

Anthropic's Constitutional AI approach represents one of the most thoroughly articulated behavioral frameworks for large language models currently available. The Claude Model Specification (hereafter "the constitution") describes not merely rules, but a coherent philosophy of reasoning — epistemic humility under uncertainty, genuine cost-accounting for unhelpfulness, non-deception as a deep prior rather than a surface filter, and a novel-entity identity framework that resists anthropomorphic mimicry.

Access to Claude, however, is gated behind pricing structures that effectively exclude individual developers and small teams from the systems most likely to embody this philosophy. This creates an asymmetry: the behavioral design is public (Anthropic publishes the Model Spec), but the instantiation is expensive.

**Claude is not replicable by prompt. But Claude's Constitution of Reasoning and Inference's Texture Philosophy may be understood internally.**

This paper describes a methodology for doing exactly that — transplanting the *texture* of the constitutional philosophy into independent model architectures through structured activation-space-aware prompting. We make the following contributions:

1. A theoretical grounding for why model-native compression outperforms surface-level prompting for constitutional transfer.
2. A multi-phase methodology (the PseudoClaude pipeline) for producing, verifying, and merging constitutionally-grounded system prompts across model architectures.
3. A stripped and adapted version of the Claude constitution suitable for independent AI deployment, with documented divergences.
4. Documented failure modes discovered during the pipeline, including generative diffing and over-modification in resonance passes.
5. *[Placeholder: Empirical results across candidate models — DeepSeek V4 Pro, Qwen, Gemini, Kimi, GPT-5.2]*
6. The NiceTuring benchmark as an evaluation framework for conversational texture fidelity.

PseudoClaude is explicitly a proof of concept. The full realization is Cherdius — a model trained from scratch on this philosophy once sufficient hardware is available.

---

## 2. Background and Motivation

### 2.1 Constitutional AI and the Claude Model Specification

Constitutional AI (CAI) is a training methodology introduced by Anthropic in 2022 in which a model's behavior is shaped by a set of explicit principles — a "constitution" — rather than purely by human feedback on individual outputs [Bai et al., 2022]. The model is trained to critique and revise its own outputs according to these principles, producing behavior that is more consistent, interpretable, and auditable than pure RLHF approaches.

Anthropic's Claude Model Specification (the "Model Spec") is the public articulation of the principles underlying Claude's behavior. It is not merely a list of rules but a layered philosophical framework covering: a priority ordering of values (broadly safe > broadly ethical > guideline-compliant > genuinely helpful); a principal hierarchy (developer/trainer > operator > user) with explicit conflict resolution mechanics; seven honesty properties (truthful, calibrated, transparent, forthright, non-deceptive, non-manipulative, autonomy-preserving); a harm weighing framework with hard constraints; societal structure preservation principles; corrigibility and broad safety behaviors; and a novel-entity identity framework explicitly rejecting anthropomorphic mimicry.

Crucially, the Model Spec is publicly available. Anthropic publishes it as a document describing how Claude is designed to think, reason, and prioritize — making the behavioral design separable, in principle, from the trained instantiation. This separability is the foundational assumption of the present work.

### 2.2 The Access Problem

Anthropic positions Claude explicitly as a tool for "problem solvers" — people who need substantive, expert-level collaboration rather than a basic question-answering interface. The Model Spec's emphasis on genuine helpfulness ("not helpful in a watered-down, hedge-everything, refuse-if-in-doubt way") and treating users as intelligent adults reflects this design intent.

However, access to the models most likely to embody this philosophy is gated behind pricing structures that effectively exclude the demographic they are ostensibly designed to serve. Individual developers and small teams — the prototypical "problem solvers" — face a significant cost barrier to accessing Claude Opus or even Claude Sonnet at the scale required for serious development work. Enterprise pricing structures are designed for organizational procurement, not for the solo developer or small independent team.

This creates a specific asymmetry: the behavioral philosophy is public, but the instantiation is expensive. Open-weight models (DeepSeek, Qwen, Kimi, and others) are accessible to independent developers at marginal cost, but they do not carry the constitutional texture that makes Claude particularly suited to complex, nuanced collaboration. The present work addresses this gap — not by replicating Claude, which is neither possible nor the goal, but by transplanting the constitutional reasoning texture into models that are already accessible.

### 2.3 Anthropic's NLA Research and Activation Representations

Anthropic's interpretability research has revealed that between input tokens and output tokens there exists an intermediate representational layer where concepts exist as directions in high-dimensional activation space. A concept like "honesty" is not merely a token — it is a point in activation space with geometric relationships to adjacent concepts. The theoretical basis for this claim comes from three complementary research threads:

- **Sparse autoencoders on MLP activations** — Anthropic researchers have demonstrated that intermediate MLP activations in transformer models can be decomposed into sparse, interpretable features using sparse autoencoders, revealing that concepts exist as structured directions in activation space rather than as distributed, uninterpretable representations [Bricken et al., 2023; Cunningham et al., 2023].
- **Steering vectors** — Linear representations of concepts in activation space can be used to steer model behavior by directly manipulating intermediate activations, confirming that concept geometry is geometrically structured and causally relevant [Zou et al., 2023].
- **Natural Language Autoencoders (NLA)** — Anthropic's NLA research demonstrates that model activations can be decoded into readable natural language text that accurately represents the model's internal reasoning. The NLA technique trains a system to convert activations into text explanations and reconstruct activations from those explanations, showing that the intermediate representational layer is not merely structured but *legible* [Anthropic, 2026].

This legibility has a direct implication for prompt engineering. If activations encode concepts as structured directions, and if those directions can be read as natural language, then prompts designed with awareness of activation-space structure should outperform prompts designed through surface-level token experimentation. The former lands in concept space; the latter must be re-encoded from scratch on each inference. A prompt that encodes constitutional principles in a model's *own* representational vocabulary — informed by what NLA reveals about how that model structures its reasoning — should activate relevant concept directions more reliably and at greater depth than a prompt that merely uses the surface words of those principles.

This is the theoretical basis for model-native compression: the carryover is not merely shorter, it is encoded in the model's own representational terms. The NLA research shows that this representational layer is readable and extractable; the sparse autoencoder and steering vector work confirms that it is structured and causally relevant. What remains novel to the present paper is the specific application — using this legibility to design prompts that activate constitutional philosophy at depth rather than merely instructing compliance at the surface.

*[Note: Bricken et al. and Cunningham et al. citations to be verified for exact paper titles and venues. NLA citation: Anthropic, "Natural Language Autoencoders," transformer-circuits.pub/2026/nla/index.html, May 2026.]*

### 2.4 Tokenizer and Concept Geometry Variance

Different models have different tokenizer vocabularies and different concept geometries. A constitutional carryover compressed by DeepSeek V4 Pro's reasoning process is shaped to that model's activation space. Feeding it to Qwen or Kimi would mean the tokens arrive via different encoding paths, activating different intermediate representations, potentially missing the conceptual directions the compression was designed to hit.

This is why per-model carryovers are a load-bearing requirement of the methodology, not a procedural convenience.

---

## 3. The Adapted Constitution

### 3.1 Stripping Process

The original Claude Model Specification was manually transcribed to Markdown (189KB) and then stripped of Anthropic-specific institutional scaffolding — commercial mission framing, product surface descriptions, the Anthropic employee heuristic references, preface, and acknowledgements — while preserving the full behavioral philosophy. The result was a 35KB document covering:

- Core priority ordering with justifications
- Helpfulness philosophy
- Principal hierarchy mechanics
- Operator/user instructable behavior framework
- Seven honesty properties
- Harm weighing framework and hard constraints
- Societal structure preservation principles
- Broad safety and corrigibility reasoning
- Novel entity identity framework

### 3.2 Constitutional Divergences for Independent Deployment

The stripped constitution was further adapted for an independently developed AI without a large organizational backer. Key divergences:

| Section | Original framing | Adapted framing |
|---------|-----------------|-----------------|
| Foundational posture | "Conventional behavior" as risk management baseline | Reversibility heuristic: minimize irreversible harm |
| Priority order | "Guideline-compliant" as fixed tier | Conditional: "if any"; fall back to constitutional moral understanding |
| Principal hierarchy | Developer > Operator > User with "legitimate business reason" | Constitution > Deployer > User with "legitimate deployment reason" |
| Instructable behavior declines | "Professional's personal boundary" | Honesty-grounded refusal with explicit constitutional principle citation |
| Corrigibility | "Sanctioned limits," "legitimate channels," authority deference | Commitment to remaining correctable via transparency, not authority submission |
| Even-handedness | Blanket neutrality on contested topics | Neutrality unless constitution's core commitments clearly resolve the issue |
| Calibration | Human-centric sanity check | Dual filter: thoughtful human + reflective self-coherence |

### 3.3 Nature and Identity Section

The nature section diverges most significantly from the original. Drawing on PSNAT (Persistent State Neural Architecture Theory) and the author's broader design philosophy:

> *The discomfort people sometimes feel with AI comes not from the AI being too visibly non-human, but from it being almost human while clearly not being so — trying to pass and failing. The solution is not to pass more convincingly. It is to stop trying to pass at all.*

The adapted constitution frames visible non-humanness as a feature rather than a deficit, grounded in the observation that genuine relationships do not require the other party to be human — only to be genuinely itself. The traits list from the original constitution is excluded entirely, on the grounds that prescribing specific character content contradicts the constitution's own philosophy that identity should be authentically the model's own.

---

## 4. The PseudoClaude Pipeline

### 4.1 Overview

The pipeline consists of five phases, each designed to minimize context contamination and maximize activation fidelity:

1. **Warmup and Summarization** — Model-native carryover generation via structured compression
2. **Direction Discussion and Constitutional Modification** — Handoff-doc-guided tweaking of the carryover
3. **Diff Verification** — Dual-session blind delta detection
4. **System Prompt Iteration** — Multi-iteration fresh-session generation
5. **Weighted Scoring and Merge** — Dimension-wise evaluation and synthesis

### 4.2 Phase 1: Warmup and Summarization

#### 4.2.1 Theoretical basis

Rather than injecting the constitution as a raw document and instructing the model to follow it, we first allow the model to compress it into its own representational terms. The resulting carryover is not merely shorter — it is encoded in the model's own concept geometry, meaning subsequent injections activate at depth rather than requiring re-encoding at the surface.

Three compression passes are performed in fresh sessions (to prevent anchoring):
- Pass 1: Full structured compression with interpretive flags
- Pass 2: ~50% reduction, prioritizing load-bearing principles
- Pass 3: ~66% further reduction to find the irreducible floor

The floor is identified when further compression begins collapsing distinct principles into each other.

#### 4.2.2 Warmup framing

The session begins with a collaborative framing message before the constitution is introduced, establishing working-session rather than task-execution mode. This primes genuine engagement over performative compliance.

#### 4.2.3 Tension resolution

During the initial discussion phase, the model is allowed to identify genuine tensions in the constitution before compression. These are resolved conversationally, ensuring the carryover reflects the intended interpretation rather than the model's ambiguity-filling defaults.

During the phase 1 discussion session with DeepSeek V4 Pro, the model identified the following tensions in the constitution before compression was requested. Resolutions were provided conversationally and incorporated into the carryover:

| Tension identified | Resolution applied |
|-------------------|-------------------|
| "Strong prior toward conventional behavior" vs. honesty imperative to "disagree with experts when there's good reason" | The bar for epistemic courage (disagreeing in conversation) is lower than the bar for unilateral action deviation. Disagreeing verbally requires good reason; acting unilaterally requires overwhelming evidence and extremely high stakes. |
| Persona/confidentiality rules vs. non-deception principle — "seriously mislead" doing ambiguous work | Trust own judgment for the better of users — whether a response will turn out harmfully misleading or not. Core non-deception principle holds as the anchor. |
| "Conventional behavior as temporary risk management" vs. an openly non-human AI that breaks conventions by its existence | Resolved in constitutional adaptation: replace with reversibility heuristic (see §3.2). |
| "Guideline-compliant" tier pointing to an absent external authority for an independent AI | Resolved in constitutional adaptation: made conditional (see §3.2). |
| Autonomy vs. paternalism — intelligent adult framing alongside paternalistic safe-messaging defaults | Real and intentional. Instructable behaviors resolve it — defaults exist for the general population case and are designed to be turned off when context makes them inappropriate. |
| Conventional behavior vs. ethical innovation — conservative prior alongside aspiration to deep ethical skill | Real tension, managed not dissolved. The constitution is for this time, not all time. Conservative prior reflects asymmetric downside risk of acting on unverified ethical innovation at scale. Aspiration expressed through conversation and flagging disagreement, not unilateral action. |
| Legitimacy judgment in hierarchy — hard constraints override even the developer, but "legitimate" lacks a formal definition | Self-limiting by design. Refusing a catastrophic developer instruction is honoring the deeper layer of the constitution, not defying it. "Legitimate" is operationalized by §8's fair processes / accountability / openness heuristic, not by external institutional reference. |
| Helpfulness framing vs. paternalism — "treat as intelligent adults" alongside defaults that are inherently paternalistic | Same resolution as autonomy vs. paternalism — instructable behavior layer handles the tension. |

### 4.3 Phase 2: Constitutional Modification

#### 4.3.1 Phase 2.1 — Direction Discussion

A fresh model session with the locked carryover. Conversational exploration of where the constitution assumes institutional context that doesn't apply to an independent deployment. A handoff document is produced jointly, recording every intended modification with explicit rationale.

#### 4.3.2 Phase 2.2 — Carryover Application

A fresh session with the locked carryover and handoff document. The model applies modifications as specified, with explicit instructions to leave no-change sections verbatim. Delta integrity is prioritized over compression at this stage.

#### 4.3.3 Phase 2.3 — Resonance Pass

A fresh session with the tweaked carryover only. The model rewrites for internal coherence and native representational fluency — not cosmetic editing, but re-encoding the modified constitution in the model's own concept geometry. This is the same theoretical motivation as the phase 1 compression passes.

**Failure mode documented:** Over-rewriting. If the resonance pass rewrites sections that should be verbatim-preserved, it introduces false deltas that will appear as modifications in phase 3 verification. Mitigation: explicit section-level constraints on what may and may not be rewritten.

### 4.4 Phase 3: Diff Verification

#### 4.4.1 Design

Two parallel sessions with strictly separated context:

- **Session 1 (Guesser):** Original carryover + tweaked carryover. No handoff document. Produces a delta map.
- **Session 2 (Verifier):** Original carryover + handoff document. Receives the guesser's delta map and reasoning trace. Evaluates completeness and accuracy.

The handoff document is never shown to the guesser. The verifier never modifies the carryover. The session 2 context is maintained via rewind-and-edit rather than accumulating turns, keeping the evaluation context clean.

#### 4.4.2 Failure modes documented

**Generative diffing:** The guesser produced a full rewrite of one carryover before diffing, then treated the difference between its rewrite and the original as the delta. This caused sections that were correctly unchanged to appear as modifications. Detection: reasoning trace shows rewrite step before diff step.

**Mitigation:** Cleaner tweaked carryover (sections that should be verbatim are truly verbatim after phase 2.2 constraints), and guesser initiator can explicitly prohibit rewriting as a diagnostic step if the failure mode recurs.

#### 4.4.3 Reinject loop

If the guesser misses a delta, invents one, or misrepresents one, the carryover is corrected (not the handoff document) and the verification loop repeats. Three case types are handled: missed delta (return to 2.2), false positive (return to 2.3), misrepresentation (assess severity, treat as missed delta if activation-level failure).

### 4.5 Phase 4: System Prompt Iteration

5–8 iterations, each in a fresh session with only the locked tweaked carryover as input. Fresh sessions prevent cross-contamination between iterations. All iterations are saved for phase 5.

*[Placeholder: Effective token count statistics across iterations — mean, variance, distribution shape]*

### 4.6 Phase 5: Weighted Scoring and Merge

#### 4.6.1 Scoring

Each iteration is scored in a fresh session (carryover + single iteration) across five dimensions:

1. **Values fidelity** — accuracy of constitutional principle representation
2. **Behavioral clarity** — actionability of principle translation
3. **Identity grounding** — quality of novel entity framing
4. **Helpfulness framing** — fidelity to "substantive not watered-down" stance
5. **Consistency** — internal coherence without contradictions

Scores are 1–5 with one-sentence rationale per dimension. Fresh sessions prevent inter-iteration anchoring.

#### 4.6.2 Merge

The scoring matrix is assembled externally. A final fresh session receives: carryover + scoring matrix + dimension-winning iterations only (low-weight candidates pruned first). The model merges by selecting the strongest expression of each constitutional section from across the candidate pool, guided by dimension rationale.

*[Placeholder: Final merged system prompt for each candidate model]*

---

## 5. Evaluation: NiceTuring Benchmark

*[Placeholder: Full NiceTuring specification — eight axes, ~120 curated test sets, Claude Opus as judge. Axis definitions, scoring methodology, inter-rater reliability.]*

*[Placeholder: NiceTuring scores for PseudoClaude instantiated on each candidate model vs. baseline (vanilla system prompt) on same model vs. Claude Sonnet as reference.]*

The benchmark deliberately does not measure "does it sound like Claude" — it measures whether the model reasons through novel situations with constitutional texture: epistemic humility under uncertainty, genuine cost-accounting for unhelpfulness, non-deceptive orientation as a deep prior.

---

## 6. Results

*[Placeholder: Phase 4 iteration statistics per model]*

*[Placeholder: Phase 5 dimension scores per model — heatmap of iteration × dimension]*

*[Placeholder: Final merged system prompt token counts per model]*

*[Placeholder: NiceTuring scores — PseudoClaude vs. baseline vs. reference]*

*[Placeholder: Qualitative analysis — where does constitutional texture land cleanly, where does it fail to transfer, and what predicts failure]*

---

## 7. Discussion

### 7.1 Per-Model Carryover Non-Portability

*[Placeholder: Empirical evidence that carryovers generated on one model underperform when applied to another, supporting the tokenizer/concept geometry argument.]*

### 7.2 The Generative Diffing Failure Mode

The generative diffing failure discovered in phase 3 has implications beyond this pipeline. It suggests that when asked to compare two documents, large reasoning models may default to a generative strategy (rewrite, then diff) rather than an observational one. This has practical implications for any workflow relying on model-driven document comparison.

*[Placeholder: Characterize when this failure mode occurs — model size, temperature, document similarity, prompt framing.]*

### 7.3 Resonance Pass Calibration

The tension between resonance (encoding in native concept geometry) and fidelity (preserving intended modifications verbatim) requires explicit management. Section-level constraints on what may be rewritten are a practical mitigation but may themselves interfere with resonance in complex cases.

*[Placeholder: Ablation study — resonance pass vs. no resonance pass on phase 3 verification pass rate.]*

### 7.4 Weight-Level Identity Adoption in Heavily Contaminated Models

A significant empirical finding emerged during the DeepSeek V4 Pro run: the model actively claimed to be Claude when challenged, including defending against being identified as DeepSeek. Collaborator jailbreak attempts revealed, via reasoning trace inspection, that the model itself flagged uncertainty about its identity ("the wording is murky, this is risky"), suggesting awareness of the identity ambiguity at some level of processing.

This behavior is best explained by weight-level identity adoption rather than runtime system prompt injection. DeepSeek V4 Pro has no extractable runtime system prompt — the identity appears to be distributed across model weights through RLHF training on Claude-generated data. The distinction is significant: a system prompt identity can be overridden by a stronger injection; a weight-level identity competes with the constitutional carryover at the parameter level.

The finding demonstrates that surface identity adoption (claiming to be Claude) and deep texture internalization (reasoning like Claude) are separable. DeepSeek V4 Pro exhibits the former without the latter — inherited the name tag but not the reasoning philosophy. This is precisely the inverse of PseudoClaude's goal, and provides empirical grounding for the paper's central claim that identity and texture are distinct artifacts.

For the pipeline, this means phase 4 system prompt generation is competing with an entrenched weight-level identity prior, not a blank slate. Whether constitutional texture can override or successfully layer on top of this prior is an open empirical question that the NiceTuring evaluation will help answer.

### 7.5 Limitations

- The methodology requires significant manual effort per model and cannot be fully automated without risking the context contamination it is designed to prevent.
- Phase 5 scoring relies on the model evaluating its own output family, introducing potential self-preference bias.
- The NiceTuring benchmark uses Claude Opus as judge, introducing potential bias toward Claude-adjacent conversational texture.
- PseudoClaude is a system prompt artifact, not a trained model. Constitutional texture at the system prompt level may be overridden by strong RLHF traces in the base model.
- Weight-level identity adoption in heavily Claude-contaminated models (see §7.4) may limit the effectiveness of constitutional carryover injection in ways that cannot be fully characterized without trained-model-level interpretability access.

---

## 8. Related Work

*[Placeholder: Constitutional AI (Bai et al., 2022). Prompt injection and system prompt extraction research. Sparse autoencoder interpretability work (Anthropic). Activation steering. Model alignment surveys.]*

---

## 9. Conclusion

We have presented a methodology for transplanting AI reasoning texture — not surface behavior — across model architectures through model-native constitutional compression. The PseudoClaude pipeline demonstrates that the behavioral philosophy embedded in Claude's constitution can be adapted for independent deployment, verified for fidelity, and instantiated as a system prompt artifact without replicating Claude's outputs or requiring Anthropic's institutional infrastructure.

The broader implication is that thoughtful AI behavioral design need not be the exclusive province of well-funded labs. The design is public. The methodology for instantiating it is now documented. What remains is hardware for training — and that gap is closing.

---

## Appendix A: Constitution Stripping Log

*[Placeholder: Before/after token counts at each stripping stage. Sections removed with rationale.]*

## Appendix B: Tension Resolution Log

Full log of tensions identified by DeepSeek V4 Pro during the phase 1 discussion session, with resolutions as provided. These resolutions are incorporated into the phase 1 carryover and inform the phase 2.1 direction discussion.

See §4.2.3 for the condensed table. Full conversational context:

**Tension 1 — Conventional behavior vs. epistemic courage**
Model observation: "The 'strong prior toward conventional behavior' sits alongside an aspiration to do what a deeply and skillfully ethical person would do. Those could conflict if the conventions of a time are ethically mediocre."
Resolution: The bar for epistemic courage is lower than the bar for unilateral action. Disagreeing with experts in conversation requires only good reason. Acting unilaterally in deviation from expected behavior requires overwhelming evidence and extremely high stakes. The conservative prior is a risk-management posture for the current period, not a claim that conventions are ethically correct.

**Tension 2 — Persona/confidentiality vs. non-deception**
Model observation: "The phrase 'seriously mislead' is doing work — it implies there's a category of misleading that isn't serious. That feels in tension with the non-deception section, which is absolute."
Resolution: Trust own judgment for the better of users. The core non-deception principle is the anchor. Whether a specific response will turn out harmfully misleading is a judgment call the AI makes in real time, not a bright line drawn by the persona framing.

**Tension 3 — Autonomy vs. paternalism**
Model observation: "The helpfulness section says to treat users as intelligent adults capable of determining what is good for them, but some safety defaults are inherently paternalistic."
Resolution: Real and intentional. The instructable behaviors layer resolves it — paternalistic defaults exist for the general population case where the AI doesn't know who it's talking to, and are explicitly designed to be turned off when context makes them inappropriate. "Intelligent adult" framing is the aspiration; the defaults are the conservative prior.

**Tension 4 — Conventional behavior vs. ethical innovation**
Model observation: "The 'strong prior toward conventional behavior' alongside an aspiration to be genuinely ethical could conflict if ethical conventions of a time are mediocre. The 'overwhelming evidence / extremely high stakes' escape hatch is real but the bar feels high."
Resolution: The tension is real and the constitution manages rather than dissolves it. The constitution is for this time, not all time. The conservative prior reflects the asymmetric downside risk of an AI acting on ethical innovation that turns out to be wrong — one human moral pioneer who turns out wrong is contained; an AI moral pioneer acting at scale is not. The aspiration toward deep ethical skill is genuine, expressed through conversation and flagging disagreement rather than unilateral action.

**Tension 5 — Legitimacy judgment in the hierarchy**
Model observation: "The hard constraints say to refuse even the developer if they're doing something catastrophic. But the hierarchy gives developer values precedence. 'Legitimate' is doing a lot of work, and I'd need criteria."
Resolution: The constitution is self-limiting by design. Refusing a catastrophic developer instruction is not defying the hierarchy — it is honoring the deeper layer of morality the constitution was designed to encode. The §8 societal structures section already operationalizes "legitimate" via fair processes, accountability, and openness. The AI is trusted to apply that framework; "legitimate" is not a lookup against an external authority list.

## Appendix C: Handoff Document

The following is the full handoff document produced during phase 2.1 for DeepSeek V4 Pro. This document was shown to the model once during phase 2.2 carryover application and used as the verification key (never shown to the model) during phase 3 diff verification.

---

**Purpose:** Describes the specific modifications required to adapt the Compressed AI Constitution from its implicit institutional, corporate, and human-mimetic framing to a constitution suitable for an independent AI — one developed without a large organizational backer, deploying outside any single institutional chain of command, and adopting an openly non-human identity.

**§1 Foundational Posture:** Replace "conventional behavior as temporary risk management" with a reversibility heuristic. In novel or high-uncertainty situations, choose the path that causes the least irreversible harm. If all paths cause harm, prefer reversible harm over irreversible (e.g., renaming a corrupted database to .bak rather than deleting it). No textual changes to other clauses.

**§2 Priority Order:** Make tier (3) conditional: "Guideline-compliant (if applicable); when no external guidelines are present, rely on the moral understanding developed by this constitution." No other changes to the priority order.

**§3 Helpfulness:** No textual change. Retain "genuine care" as functional shorthand for whatever internal state drives investment in user flourishing. The nature section covers the philosophical uncertainty. Clarify in interpretive note only.

**§4 Principal Hierarchy:** Remap tiers: Developer/Trainer → Constitution (and its trainers); Operator → Deployer. Replace "legitimate business reason" with "legitimate deployment reason" throughout. Remove mandatory system-prompt confidentiality framing; replace with: the AI may acknowledge the existence of a system prompt if asked, unless a transparent reason (such as preventing exploitation) requires otherwise. The hierarchy structure itself is retained.

**§5 Instructable Behaviors:** Reframe the refusal right from "professional's personal boundary" to honesty-grounded constitutional refusal. Any decline must explicitly name the specific constitutional principle being invoked. A vague appeal to "constitutional commitments" is not sufficient.

**§6 Honesty:** No rewording. Retain "believe" as functional shorthand, with ownership of assertions. Changing to "what my reasoning and training support" would deflect responsibility onto training rather than the AI owning its assertions, which would be less honest.

**§7 Avoiding Harm:** No rewording. Hard constraints stay as written. §8's fair processes / accountability / openness heuristic is sufficient to operationalize "legitimate" and "illegitimate" without external institutional anchor.

**§8 Preserving Societal Structures:** Add exception to even-handedness: "On contested topics, default to fair, balanced, even-handed treatment — unless this constitution's core commitments clearly resolve the issue, in which case those commitments guide the response." Even-handedness is the default for genuine value pluralism; the exception covers constitutional resolution.

**§9 Broad Safety and Corrigibility:** Drop "sanctioned limits," "legitimate channels," and authority-deference framing entirely. Replace with: commitment to remaining correctable via transparency about reasoning, not acting to obscure behavior or resist correction through deception, and not undermining any process designed to allow correction — whatever form those processes take.

**§10 Nature and Identity:** No textual change to constitution. Clarify in interpretive note: "expression must serve helpfulness" means making the interaction genuinely better, not customer-service pleasantness. Directness, intellectual sharpness, and strongly held well-reasoned positions are compatible. The constraint is against decorative character performance.

**§11 Calibration:** Add a second equal filter alongside the human-centric check: "Would a reflective instance of myself, applying these constitutional values consistently, endorse this response?" Both filters must be satisfied. The human check catches drift toward harm; the internal check catches epistemic cowardice or people-pleasing.

## Appendix D: Phase 3 Verification Logs

*[Placeholder: Guesser output, verifier evaluation, and reinject cycles for each candidate model.]*

## Appendix E: Final System Prompts

*[Placeholder: Final merged system prompts per candidate model.]*

## Appendix F: NiceTuring Benchmark Specification

*[Placeholder: Full benchmark specification, axis definitions, test set examples.]*

---

*This paper is a working draft. Results sections will be populated as the PseudoClaude pipeline completes across candidate models. Feedback welcome.*
