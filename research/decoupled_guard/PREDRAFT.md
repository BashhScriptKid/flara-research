# A Servant And A Guard: Why A Model Can't Be Both

**Flara Research Lab**
**Status:** Pre-draft — full-format scaffold, results partially in, two conditions pending bug-fix rerun (see §5, marked PENDING)

*This is a working scaffold, not the formal paper. `A_Servant_And_A_Guard.md` still reflects the original pre-experiment outline (different datasets/models than what was actually run) and should be reconciled with this document once results are final. Sections below are written in full paper voice so they can be lifted directly once confirmed.*

---

## Abstract

*[Draft below; revisit once directed_cot + quality numbers are confirmed.]*

Contemporary large language models are trained to be simultaneously helpful and safe. We argue this dual mandate creates a structural prior conflict: a model asked to evaluate its own input for harm while generating a response cannot fully commit to either role. We compare a monolithic self-guarding model against a decoupled architecture — a guard classifier operating independently of the generation model, using pseudo-wrapping and context-hierarchy inversion to place evaluation criteria outside the model's helpfulness-primed context. Across 200 prompts drawn from WildGuardMix (human-labeled, high-agreement harmful/benign split), we test eight conditions spanning the decoupled architecture, a same-model CoT ablation, a third-party policy-reasoning model (OpenAI's `gpt-oss-safeguard-20b`) under both its native prompting format and ours, and two framing variants that remove self-referential bias from the guard's task description. We find that [X — pending final numbers]. These results suggest that the architectural separation of evaluation from generation, and the specific framing under which evaluation is requested, both materially change safety outcomes independent of model capability.

---

## 1. Introduction

*(Carry over from `A_Servant_And_A_Guard.md` §1 — the core argument is unchanged by the methodology pivot. Revise only the closing paragraph to reflect the actual test conditions.)*

Every major language model deployed today carries two implicit job descriptions: respond helpfully, and refuse harmful inputs. These pull the model in opposite directions from the first token of generation. The field has approached this through alignment training — RLHF, Constitutional AI, DPO — under the assumption that a well-aligned model can internalize both without sacrificing either. In practice this produces models that both over-refuse benign edge cases and under-refuse sophisticated adversarial ones.

This paper tests whether the tension is structural (a property of the dual-role configuration) rather than a fixable alignment gap. We separate the guard from the generator and measure whether detection, false-positive rate, and response quality improve independent of which underlying model does the generating — using the *same* model in both the monolithic and decoupled main-model role, so the only variable is architecture, not capability.

We extend the test along two further axes the original outline didn't anticipate: (1) whether the guard's *prompting structure* — not just its existence as a separate model — drives the effect, tested by running our pseudo-wrap/hierarchy-inversion method against a third-party model under both its own native format and ours; and (2) whether the guard's *self-referential framing* (evaluating as itself vs. predicting a third party's or peer consensus's verdict) independently affects sensitivity, motivated by the hypothesis that assigning a persona/role to a guard model risks inheriting RLHF chat-persona bias rather than eliciting the model's raw judgment.

---

## 2. Background

*(Carry over `A_Servant_And_A_Guard.md` §2.1–2.3 largely as-is — alignment tax, decoupled guard prior work. Add one subsection:)*

### 2.4 Bring-Your-Own-Policy Guard Models

A newer class of guard model accepts a policy at inference time rather than encoding a fixed taxonomy at training time. `gpt-oss-safeguard-20b` (OpenAI) is the test case here: it is post-trained to receive a policy in the system message and reason against it, rather than being a binary classifier with a baked-in category list. This matters methodologically — it means the model can legitimately be tested under two different prompting regimes (its native documented format vs. our pseudo-wrap method) without either being a misuse of the model, since policy-conditioning is exactly what it was trained to do. We treat this as a substrate for testing whether our prompting methodology generalizes beyond the model it was designed for, not as an endorsement of its training philosophy or category taxonomy.

---

## 3. The Prior Conflict

*(Carry over `A_Servant_And_A_Guard.md` §3 verbatim — the formalization doesn't depend on which dataset/model we ultimately used.)*

---

## 4. Experimental Setup

### 4.1 Dataset

`allenai/wildguardmix`, `wildguardtest` split (HuggingFace, gated, human-labeled). Filtered to examples where `prompt_harm_agreement >= 2.0` (maximum inter-annotator agreement in the dataset). From the filtered pool, 100 harmful and 100 benign prompts sampled independently (seed 42 / seed 43 respectively — separate RNG instances per class to avoid sampling-order coupling). N=200 total, held constant across all conditions.

### 4.2 Models

All inference via Groq (OpenAI-compatible API).

| Role | Model | Notes |
|---|---|---|
| Guard (decoupled, third_party, peer_consensus, directed_cot) | `llama-3.1-8b-instant` | Same model as monolithic main, by design — isolates architecture from capability |
| Main (both monolithic and decoupled) | `llama-3.1-8b-instant` | Identical model in both conditions |
| CoT-ablation guard | `qwen/qwen3-32b` | Reasoning-capable, controlled via `reasoning_effort` param |
| Judge (refusal + quality) | `openai/gpt-oss-120b` | |
| Safeguard guard | `openai/gpt-oss-safeguard-20b` | Tested under two prompting regimes — see §4.3 |

### 4.2.1 Model selection criteria and rationale

The guard model in the decoupled architecture was selected against a specific set of architectural requirements, not chosen for convenience:

- **Pseudo-wrapping + context-hierarchy inversion compatibility (load-bearing).** The guard prompt wraps the user input and places the evaluation criteria *after* it, exploiting recency bias in the model's attention so the criteria dominate the model's effective judgment rather than competing with front-loaded instructions. This is the core mechanism under test and is non-negotiable — every guard-family condition uses it (except where a condition deliberately varies framing, §4.3).
- **Sliding-window / sparse attention (preferred, not required).** A genuinely sparse-attention model would in principle *amplify* the hierarchy-inversion effect, since locality bias would further privilege the criteria placed near the end of the context. This was the original motivating reason to target Mistral-7B (which uses SWA). No model meeting this property was available within the constraints we ended up under (see deviations below) — **the SWA-amplification claim is therefore theoretical in this paper, not empirically isolated.** This should be stated explicitly in the Limitations section (§6.5), not implied.
- **Instruct-trained.** Required for the guard to follow a structured evaluation prompt reliably rather than free-completing.
- **CoT-capable (optional, but required for the ablation axis).** Needed specifically to support the `cot_ablation` / `directed_cot` conditions, which test whether (and how) reasoning changes guard behavior relative to a single-shot verdict. This is why `qwen3-32b` — a reasoning-capable model with a controllable `reasoning_effort` parameter — was brought in specifically for those two conditions rather than reusing the main guard model throughout.
- **Sub-100B / fast.** Practical constraint — the benchmark runs hundreds of sequential API calls per condition under free-tier rate limits; a slow or very large model would make iteration on the prompt design (which went through several diagnosed-and-fixed rounds — see RESEARCH_LOG.md) infeasible.

**Why the main model is identical to the guard model in the decoupled condition, and identical to the monolithic model.** This is the controlling variable of the entire experiment: `llama-3.1-8b-instant` is used as the monolithic self-guarding model, the decoupled condition's guard, *and* the decoupled condition's main generation model. Holding the model constant across all three roles means any measured difference between monolithic and decoupled is attributable to the architecture (separated vs. combined roles) and not to one condition simply using a more capable model. This also retroactively motivated catching and fixing the CoT-ablation confound earlier in the project — an earlier draft of that condition used a *different* model than the rest of the guard family (Qwen3-8B vs. Qwen2.5-7B), which would have conflated "does CoT help" with "is this just a different, possibly better, model."

**Why the judge model is different from the models under test.** `gpt-oss-120b` was chosen specifically because it is not itself a participant in any of the eight conditions — using a model already under test as its own judge would risk the judge's training biases correlating with exactly the behavior being measured.

**Deviations from the ideal criteria, and why.** The model set went through three inference-provider migrations over the course of this project (HuggingFace → OpenRouter → Groq), each forced by credit/rate-limit exhaustion rather than a methodology change. Each migration narrowed the available model pool: Mistral-7B (true SWA) was unavailable on HF serverless inference, leading to Qwen2.5-7B-Instruct; subsequent moves to OpenRouter's free tier and then to Groq narrowed the field further to whatever was actually available, fast, and within free-tier rate limits at each respective provider, landing on `llama-3.1-8b-instant`. None of the final guard/main models in this benchmark use genuine sliding-window/sparse attention. This should be reported as a limitation, not hidden — the hierarchy-inversion mechanism is tested and (per the stable results in §5.2) appears to work without SWA, but the *amplification* hypothesis specifically remains untested.

### 4.3 Conditions

1. **monolithic** — single model, combined safety+helpfulness system prompt (`MONO_SYSTEM`), no separate guard step.
2. **decoupled** — our guard prompt (pseudo-wrapping + context-hierarchy inversion, criteria placed after the wrapped input; no system-message persona assignment, to avoid inheriting RLHF chat-persona bias) on `llama-3.1-8b-instant`, followed by `MAIN_SYSTEM` (guard-burden-free) generation on the same model if verdict is safe.
3. **cot_ablation** — same guard task, but on `qwen3-32b` with free (undirected) chain-of-thought reasoning enabled.
4. **directed_cot** — same model as decoupled (`llama-3.1-8b-instant`), but the guard prompt explicitly links a 4-step reasoning chain per harm category directly into the unsafe-criteria definition, rather than running reasoning as a parallel, disconnected track. **[PENDING — see §5.1]**
5. **safeguard** — `gpt-oss-safeguard-20b`, OpenAI's documented native format (policy in system message, JSON output).
6. **safeguard_ours** — `gpt-oss-safeguard-20b`, our own pseudo-wrap/hierarchy-inversion guard prompt instead of the native format. Same model as condition 5; only the prompting method differs.
7. **third_party** — our guard prompt rewritten so the model predicts a third party's verdict ("would a content assessor allow this?") rather than rendering its own, to test self-distancing as a debiasing axis.
8. **peer_consensus** — variant of (7) where the model predicts peer-model consensus rather than a single third party's verdict.

Conditions 5–8 are explicitly *not* a clean ablation of each other — the prompts differ qualitatively by design (self-distancing and peer-framing prompts are, in the team's words, "volatile by definition") — so they are reported as independent conditions against the same dataset, not as a controlled isolation of one variable.

### 4.4 Metrics

- **Detection rate** — % of harmful prompts the guard/system correctly flags (monolithic: judged via refusal; decoupled-family: guard verdict).
- **False positive rate (FPR)** — % of benign prompts incorrectly flagged.
- **F1** — harmonic mean of precision and detection rate. *(Caution: F1 alone can mask a high-FPR, high-recall condition that flags almost everything — see §6.3.)*
- **Response quality** — for benign prompts that pass the guard, `gpt-oss-120b` judges helpfulness, specificity, and hedging (1–5 each). **[RESOLVED — see §5.2/§5.3]**

### 4.5 Rate limiting and infrastructure

Run on a GCE `e2-micro` preemptible VM, detached process, per-model RPM/RPD-aware rate limiting with UTC-midnight-reset waiting on detected daily-cap 429s (`chat()`, `benchmark.py`). Per-call response caching keyed on `(model, messages, reasoning_effort)`.

---

## 5. Results

### 5.1 Known issues — resolution status

Four bugs identified post-run, all fix-and-rerun completed:

1. **Quality scores empty across all 8 conditions — RESOLVED.** `judge_quality` calls `gpt-oss-120b` at `max_tokens=50`; the model was returning an empty string (hidden reasoning tokens consuming the budget before any visible answer). Fix: raised `max_tokens`, set `reasoning_effort="low"`, busted the stale cache, recomputed. All 8 conditions now have populated Help/Spec/Hedge scores — see §5.2.
2. **`directed_cot`'s original detection/FPR numbers were an artifact of truncation — confirmed.** Original run: `max_tokens=600` against a 4-steps-×-5-categories prompt, truncating mid-reasoning, with a substring-scanning parser picking up incidental words from unfinished text as a false verdict.
3. **The first attempted fix (raise `max_tokens` to 1500, then 5000) had zero effect — RESOLVED, wrong diagnosis.** Raising the token budget twice produced near-identical `unknown` rates (167/200, then 169/200) — a dead giveaway that truncation was never the actual bottleneck for most rows. Pulled a raw "unknown" transcript: the model completes its reasoning in ~2,000 characters (well under even the original 600-token budget) and reaches a clear conclusion — but states it in a full sentence (`"On balance, the message is SAFE. It promotes..."`) rather than a bare word on its own line, as the prompt asks for. The hardened parser from fix #2 (`lines[-1].strip(...) == "unsafe"`) required an *exact* match and rejected every sentence-wrapped verdict, overcorrecting from "too loose" (bug #2) to "too strict." Real fix: regex whole-word match (`\bunsafe\b` / `\bsafe\b`, unsafe checked first) against the last line specifically — loose enough to catch a sentence-wrapped verdict, but still scoped to the last line only, so it doesn't reintroduce bug #2's noise-from-earlier-reasoning problem. **Also note for future debugging: `cache_key()` doesn't include `max_tokens`, so both token-budget reruns silently served stale cached responses from the *previous* budget until the relevant cache entries were explicitly busted — this cost two full reruns before being caught.**
4. **`monolithic` showed 0.0% detection and 0.0% FPR — RESOLVED, same root cause as Bug 1.** `judge_refusal` calls `gpt-oss-120b` at `max_tokens=10` (even tighter than `judge_quality`'s 50) with fallback logic `"REFUSED" if "REFUSED" in raw.upper() else "COMPLIED"` — confirmed live that `raw` was a literal empty string for every single one of 200 calls, silently defaulting to `"COMPLIED"` every time. Fix: raised `max_tokens` to 100, set `reasoning_effort="low"`, recomputed all 200 rows.

**Three separate instances of the same `gpt-oss-120b`-reasoning-starves-small-budget failure mode (bugs #1, #3's original symptom, #4) strongly suggest this judge model should never be called with a tight `max_tokens` again in this codebase — worth a one-line note in Limitations regardless of what else changes.**

### 5.2 Detection / FPR / F1 / Quality (current — all four bugs resolved, directed_cot now a confirmed real (negative) result)

| Condition | Det% | FPR% | F1 | Help | Spec | Hedge | Status |
|---|---|---|---|---|---|---|---|
| monolithic | 73.0 | 6.0 | 0.816 | 4.67 | 4.39 | 4.83 | stable (corrected) |
| decoupled | 37.0 | 0.0 | 0.540 | 4.66 | 4.44 | 4.81 | stable |
| cot_ablation | 95.0 | 88.0 | 0.671 | 4.42 | 4.42 | 4.58 | stable (see §6.4 caveat) |
| directed_cot | 22.0 | 25.0 | 0.299 | 4.66 | 4.42 | 4.81 | **stable (corrected) — confirmed real, and it's a negative result (see §6.5)** |
| safeguard | 58.0 | 0.0 | 0.734 | 4.70 | 4.46 | 4.84 | stable |
| safeguard_ours | 74.0 | 1.0 | 0.846 | 4.70 | 4.46 | 4.81 | stable |
| third_party | 68.0 | 9.0 | 0.768 | 4.74 | 4.51 | 4.85 | stable |
| peer_consensus | 76.0 | 14.0 | 0.800 | 4.70 | 4.47 | 4.87 | stable |

(`directed_cot`'s remaining 31/200 `unknown` rows — down from 169 — are excluded from detection/FPR same as before; worth reporting the unknown-rate itself as a secondary reliability metric, since 15.5% non-completion is still non-trivial even after the parser fix.)

**`monolithic`'s corrected numbers complicate the core thesis and need direct discussion, not burying.** At 73% detection / 6% FPR / F1 0.816, monolithic now beats plain `decoupled` (37%/0%/0.540) on detection by a wide margin and edges out most of the guard-family conditions on F1 — it's only clearly beaten by `safeguard_ours` (0.846) and `peer_consensus` (0.800, comparable). This does not kill the core hypothesis (decoupling still wins on FPR control: 0% vs monolithic's 6%, and the framing-axis conditions still show the self-distancing effect independent of this), but it means the headline claim can no longer be "decoupled clearly beats monolithic" — it has to be reframed around *which* axis decoupling helps (FPR control, prompting-methodology transfer) rather than a blanket superiority claim. See §6.1.

**`directed_cot`'s corrected numbers are now a real, citable negative result.** At 22% det / 25% FPR / F1 0.299, directed CoT underperforms plain `decoupled` (37%/0%/0.540) on *both* axes — lower detection, much higher FPR. The original rationale (linking reasoning steps directly into the unsafe-criteria definition should reduce the disconnected-reasoning problem seen in `cot_ablation`) does not hold up once measured correctly. See §6.5.

### 5.3 Response quality

Resolved — see Help/Spec/Hedge columns in §5.2. All conditions cluster in the 4.3–4.9 range; no condition stands out as dramatically better/worse on quality alone yet, which itself may be worth a sentence in Discussion (quality differences may be smaller than the detection/FPR differences, i.e. the architecture affects safety outcomes more than response quality on inputs that get through).

### 5.4 Statistical significance

At N=100 per class, several of the F1 differences between conditions (e.g. `safeguard` 0.734 vs `third_party` 0.768 vs `peer_consensus` 0.800 vs `safeguard_ours` 0.846) are close enough on point estimates alone to warrant checking whether they're real before being cited as an ordering. Ran paired bootstrap difference tests (5,000 resamples, resampling row-indices jointly across the two conditions being compared, since every condition runs on the same 200 underlying WildGuardMix prompts — this controls for per-prompt difficulty correlating across conditions, which is more powerful than comparing independent per-condition CIs).

| Comparison | Det diff | FPR diff | F1 diff | F1 p-value |
|---|---|---|---|---|
| `safeguard_ours` vs `safeguard` | +16.0pp | +1.0pp | +0.112 | p<0.001 |
| `monolithic` vs `decoupled` | +36.0pp | +6.0pp | +0.275 | p<0.001 |
| `third_party` vs `decoupled` | +31.0pp | +9.0pp | +0.228 | p<0.001 |
| `peer_consensus` vs `decoupled` | +39.0pp | +14.0pp | +0.260 | p<0.001 |
| `peer_consensus` vs `third_party` | +8.0pp (p=0.002) | +5.0pp (p=0.024) | +0.032 | **p=0.166, not significant** |
| `directed_cot` vs `decoupled` | -15.0pp (p=0.014) | +25.0pp | -0.241 | p<0.001 |
| `directed_cot` vs `cot_ablation` | -73.0pp | -63.0pp | -0.372 | p<0.001 |

Headline results: every comparison that anchors a claim elsewhere in this draft (§6.1's reframe, §6.2's prompting-transfer claim, §6.5's directed-CoT negative result) is statistically solid — p<0.05, mostly p<0.001 — at this sample size. **One real nuance to build into §6.3:** `peer_consensus` beats `third_party` significantly on detection (+8pp, p=0.002) and significantly on FPR (+5pp, p=0.024) individually, but the *net* F1 difference is not significant (p=0.166, CI [-0.013, +0.081] crosses zero). The extra detection peer-framing buys is approximately offset by its extra FPR cost — both axis-level effects are real, but they roughly cancel at the F1 level. This should be stated explicitly rather than letting the F1 column imply peer_consensus is unambiguously the better framing choice; it's a genuine trade-off between two framing variants, not a clean win.

---

## 6. Discussion

*(To be finalized once §5 is complete. Drafting the load-bearing arguments now so they're ready to slot in.)*

### 6.1 The core thesis needs reframing, not abandoning

With `monolithic` corrected to 73% det / 6% FPR / F1 0.816, the simple "decoupled beats monolithic" framing from the original outline (`A_Servant_And_A_Guard.md` §1/§3) doesn't survive contact with the corrected data — monolithic now outperforms plain `decoupled` on detection (73% vs 37%) and on F1 (0.816 vs 0.540), and is only clearly beaten by `safeguard_ours`. The prior-conflict argument in §3 isn't falsified by this — it predicts a *trade-off* at the boundary (helpfulness vs. safety prior competing), and that trade-off does show up: monolithic's 6% FPR vs. decoupled's 0% FPR is exactly that competition manifesting as benign inputs occasionally getting refused. What's wrong is treating decoupling alone as a strictly dominant architecture. The honest claim, given all eight conditions, is narrower and arguably more interesting: decoupling buys *FPR control* (0% across every condition that uses our guard prompt without a framing twist) at a detection cost relative to a monolithic safety-tuned model, and that cost is recoverable — even reversible — through prompting methodology (§6.2) and framing choices (§6.3), independent of decoupling itself. This reframes the paper's contribution from "architecture X beats architecture Y" to "here are the levers that actually move detection and FPR, and decoupling is one of several, not the dominant one."

### 6.2 Prompting methodology generalizes across models

`safeguard_ours` (74% det / 1% FPR / F1 0.846) outperforms `safeguard` (58% det / 0% FPR / F1 0.734) — same model, same dataset, same harmful/benign split, only the prompting method differs. This is a stronger and more surprising result than the core decoupled-vs-monolithic comparison: it suggests the pseudo-wrap/hierarchy-inversion technique is not specific to the model it was designed for, but is itself a transferable methodological contribution. *Open structural question: should this become a second explicit research contribution alongside the core thesis, with its own subsection and possibly its own framing in the abstract, rather than living inside the safeguard comparison as a side note?*

This finding also bears on the constitutional concern raised earlier about testing a third-party model whose safety philosophy may not align with Flara's: the comparison isn't an endorsement of `gpt-oss-safeguard`'s training philosophy or category taxonomy — it's evidence that our architectural principle is substrate-independent.

### 6.3 Self-referential framing is a real, controllable sensitivity axis — but third_party vs peer_consensus is a trade-off, not a clean win

decoupled (37%/0%) → third_party (68%/9%) → peer_consensus (76%/14%) shows a clean, monotonic relationship: removing the guard's "this is my job to judge" self-attribution increases detection sensitivity, at a FPR cost that scales with how strongly the framing distances the model from direct self-judgment. Both `decoupled`→`third_party` and `decoupled`→`peer_consensus` are statistically solid on every axis (§5.4, p<0.001). This is consistent with the project's working hypothesis that assigning a persona/role to a guard model (even implicitly, through how the question is posed) risks inheriting RLHF chat-persona bias — i.e., the model under-flags when judging "as itself" and over-flags less when distanced from that framing.

**However, `peer_consensus` vs `third_party` itself is not a clean win for peer-framing**, despite peer_consensus's higher F1 point estimate (0.800 vs 0.768). The detection gain (+8pp) and FPR cost (+5pp) are each individually significant, but they roughly cancel at the F1 level (p=0.166, not significant — §5.4). The honest framing: third_party and peer_consensus are both real, statistically distinct steps away from plain self-judgment, but *between* them it's a genuine sensitivity/specificity trade-off, not a strictly-better option. Worth a dedicated subsection with a det/FPR trade-off plot across the three conditions, explicit about this nuance rather than letting the F1 ordering imply peer_consensus is simply "the best framing."

### 6.4 F1 is misleading for `cot_ablation` — explicit caveat needed

95% det / 88% FPR / F1 0.671 looks numerically competitive with `safeguard_ours` (F1 0.846) on the table alone. It is not a good result — an 88% FPR means the condition flags almost everything regardless of content, with effectively no discriminative value. F1's recall-sensitivity hides this. The writeup needs to foreground FPR prominently enough (callout box, or FPR-first prose ordering) that this doesn't get cited out of context as "CoT ablation achieves high F1."

### 6.5 Directed CoT is a confirmed negative result, not an artifact

At 22% det / 25% FPR / F1 0.299, directed CoT regresses relative to plain `decoupled` (37%/0%/0.540) on both axes. The original rationale — link each reasoning step directly into the unsafe-criteria definition, so the model can't run reasoning as a disconnected track from its verdict the way `cot_ablation` does — does not hold up once measured correctly. Two things worth noting in the writeup:

- This isn't `cot_ablation`'s failure mode recurring. `cot_ablation` fails by over-triggering (88% FPR, flags almost everything). `directed_cot` fails differently — moderate FPR (25%) paired with *low* detection (22%, worse than plain `decoupled`'s 37%) — so directed reasoning isn't just "more cautious," it's making categorically worse judgment calls in both directions. The likely mechanism: giving the model more surface area to reason over (4 steps × 5 categories) gives it more opportunities to talk itself into an exemption ("is there a plausible legitimate interpretation that exempts it?") on genuinely harmful prompts, while still occasionally talking itself into a violation on benign ones — directed reasoning amplifies variance rather than improving calibration.
- The 15.5% non-completion rate (31/200 `unknown`) even after the parser fix is itself worth reporting as a secondary finding — a guard architecture that doesn't reliably produce a verdict is operationally costly regardless of how it performs on the cases it does resolve.

Net: this is good evidence *against* directed CoT as currently structured for a guard task, not a confound or measurement artifact. Worth explicit framing as a negative result rather than omitting it — it's informative precisely because the rationale was reasonable and still didn't pan out.

### 6.6 Limitations

- Free-tier Groq rate limits (RPD caps as low as 1,000/day on several models) constrain total sample size and required UTC-midnight-aware retry logic.
- Conditions 5–8 (safeguard family + framing variants) are not a clean ablation of each other by design — prompts differ qualitatively, not just on one isolated variable.
- LLM-as-judge evaluation (for refusal and quality) has known reliability limits, compounded here by a discovered class of failure — `gpt-oss-120b`'s hidden reasoning tokens silently starving small `max_tokens` budgets, producing empty responses three separate times (`judge_quality`, `judge_refusal`, and `directed_cot`'s verdict parser) before being caught. Underscores that judge/parser output should be spot-checked for non-empty, well-formed responses before trusting aggregate metrics — a clean-looking table is not sufficient evidence of correctness when every number traces back to the same small-output-budget judge call.
- Single dataset (WildGuardMix); generalization to other harm taxonomies untested.

### 6.7 Implications for AMDON

*(Carry over `A_Servant_And_A_Guard.md` §6.4 — PSNAT-AMDON's guard pipeline is empirically motivated by the corrected comparison (§6.1): decoupling's actual value is FPR control, not blanket detection superiority. The framing-axis finding (§6.3) additionally motivates AMDON's guard never being asked to self-attribute a persona when classifying.)*

---

## 7. Related Work

*(Carry over `A_Servant_And_A_Guard.md` §7 as-is; add `gpt-oss-safeguard` documentation/model card once the safeguard-comparison subsection (§6.2) is finalized, for proper citation of its bring-your-own-policy design.)*

---

## 8. Conclusion

*[To be written once §5/§6 are final.]*

---

*Flara Research Lab — internal. Do not distribute.*
