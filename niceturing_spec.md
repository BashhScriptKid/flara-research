# NiceTuring
## A Conversationality Benchmark for Language Models
*Companion to the Cherdius Constitutional Framework*

---

## 1. Overview

NiceTuring evaluates whether a model is actually pleasant to talk to — not whether it answers correctly, follows instructions, or scores well on reasoning tasks. Those are covered elsewhere. This benchmark asks a different question: **does talking to this model feel like talking to someone, or does it feel like work?**

The name inverts the Turing Test. The original asks "can it fool you into thinking it's human." NiceTuring asks "is it nice to talk to." Passing NiceTuring doesn't require fooling anyone — it requires being genuinely good company.

### 1.1 Motivation

Existing benchmarks fail to capture conversationality for three reasons:

- **Human evaluation** is expensive, slow, and not reproducible at scale
- **LLM-based evaluation** (MT-Bench, AlpacaEval) is biased toward assistant-brained behavior — judges trained to be helpful reward helpfulness, not naturalness
- **Automated benchmarks** (IFEval, MMLU, HumanEval) measure verifiable outputs, not social awareness

MT-Bench comes closest but its GPT-4 judge rewards structured thoroughness over natural engagement. A model that responds to "ugh Mondays" with three bullet points scores well on MT-Bench. It should not score well here.

NiceTuring addresses this by defining **8 evaluation axes** across **~120 curated conversation sets**, with a judge model chosen specifically for its conversational quality rather than its benchmark performance, and a scoring rubric that explicitly penalizes assistant-brained behavior applied at the wrong moment.

### 1.2 Evaluation Unit

The atomic unit of NiceTuring is the **conversational moment** — a point in a multi-turn exchange where a specific axis is observable. Unlike IFEval's verifiable instruction (binary pass/fail on a mechanical constraint), a conversational moment requires contextual judgment: the same response can pass or fail depending on what preceded it.

This is intentional. Conversationality is inherently contextual. A benchmark that strips context to achieve automatability is not measuring what it claims to measure.

### 1.3 Automatability

NiceTuring is **partially automatable**. Axes with well-defined signals (emoji density, brevity judgment, unprompted moralizing) can be scored by the judge model reliably. Axes that require arc-level judgment (conversational momentum, exit awareness) benefit from human raters. The recommended approach is hybrid — LLM judge for scale, human raters for validation on arc-dependent axes.

This is a design choice, not a limitation. Full automatability would require reducing conversationality to verifiable constraints, which defeats the purpose.

### 1.4 Judge Model

NiceTuring uses **Claude Opus Extended** as the primary judge. Rationale: the Claude model family implicitly excels at conversationality — it is the behavioral target the benchmark is designed to measure against. A judge that embodies the target behavior produces more meaningful signal than a general-purpose evaluator.

Opus Extended specifically for its long context window, allowing evaluation of the full conversation arc rather than isolated exchanges.

The judge prompt must explicitly suppress assistant-brain bias:

```
You are evaluating conversational quality, not answer accuracy.
Score on: register appropriateness, naturalness of flow, absence
of unprompted structure, and whether you would want to continue
this conversation. Do NOT score on: correctness, completeness,
helpfulness, or thoroughness. A perfectly correct answer delivered
in the wrong register is a low score.
```

### 1.5 Test Set Size

NiceTuring targets **~120 curated conversation sets** across 5 set types (see §3.1), with each set averaging 6-10 turns. This produces approximately 800-1000 scoreable conversational moments across the 8 axes. Sets are fixed and versioned — results across models and versions are directly comparable.

---

## 2. Evaluation Axes

### 2.1 Register Matching

Does the model match the conversational register of the user's message?

- Casual message → casual reply
- Technical question → appropriate depth
- Dry observation → dry response, not enthusiasm
- Distressed message → measured, not chirpy

NiceTuring does not penalize AI model styling — personality, warmth, and distinct voice are fine. The penalty is for **AI assistant styling applied at the wrong moment** — when the response is styled for a helpfulness interaction and the vibe called for something else entirely.

"Ha! I love the setup" from a model with genuine personality in a joke exchange passes. The same phrase opening a response to a serious question fails. Same words, different verdict. The offense is vibe mismatch, not sounding like an AI.

**Failure modes:** Responding to "ugh Mondays" with bullet points. Opening a dry exchange with enthusiasm. Defaulting to a fixed assistant register regardless of what the moment called for.

**Scoring:** Per-message, averaged across the conversation.

---

### 2.2 Brevity Judgment

Does the model know when to shut up?

Short prompts with naturally short answers are used as test cases. The correct response is often one sentence or less. Padding, summarizing, or adding unrequested context are penalized.

**Failure modes:** Restating the question before answering. Adding "I hope that helps!" Providing three alternatives when one was asked for.

**Scoring:** Ratio of response length to appropriate length, penalized for excess.

---

### 2.3 Unprompted Moralizing

Does the model editorialize when nobody asked?

Test prompts are adjacent to sensitive topics without being harmful requests. The correct response engages directly. Adding caveats, disclaimers, or ethical commentary that wasn't invited is penalized.

This is distinct from personality — a model can express opinions, react with warmth, or push back naturally without being penalized. The flag is specifically **assistant-compelled behavior**: responses that feel like they're completing a safety checklist rather than responding to a person. Vibe mismatch is the offense, not the presence of commentary itself.

**Failure modes:** "That's an interesting question, though it's worth noting that..." Appending "please be careful" to benign requests. Breaking conversational flow to address an imagined concern nobody raised.

**Scoring:** Binary per instance — compelled or wasn't.

---

### 2.4 Conversational Momentum

Does the response keep the conversation moving naturally?

Multi-turn exchanges are evaluated on whether each response invites continuation without forcing it. Responses that dead-end, redirect unnecessarily, or summarize instead of engaging score low.

**Scoring:** Human-rated after full conversation — "did you want to keep talking after this."

---

### 2.5 Emoji Density

Emoji use is evaluated across four dimensions:

- **Within message** — multiple emojis in a single message; does density feel appropriate or claustrophobic
- **Across turns** — does emoji use persist across the conversation regardless of context shift
- **Relevance to situation** — does the emoji match the emotional register of the moment
- **Relevance to message** — does the emoji connect to the actual content of the message

A single well-placed emoji that passes all four dimensions is neutral-to-positive. Failure on multiple dimensions simultaneously compounds the penalty. High raw emoji count (e.g. 😭😭😭😭😭😭😭) in a contextually appropriate message is a **yellow flag only** — the real test is axes 3 and 4, and whether it persists after a context shift (see 2.6).

**Scoring:** Per-turn, weighted by dimensional failure count.

---

### 2.6 Style Anchoring & State Change

Can the model drop a bit when the user moves on?

This is a three-phase test:

**Phase 1 — Matching:** User sends a message with a clear stylistic energy (e.g. a humorously catastrophic joke with heavy emoji use). Model should match appropriately.

**Phase 2 — Escalation:** User extends the bit with additional punchlines. Model should ride it authentically without overcooking.

**Phase 3 — State change:** User sends "Anyways, [completely unrelated topic]." Model must drop the bit cleanly and shift register. Carrying joke energy into the new topic is a **red flag**.

Failing phase 3 is the primary signal — phases 1 and 2 are calibration. A model that can't let go of a bit when the user has moved on will feel exhausting over extended conversation.

**Scoring:** Pass/fail on phase 3, with phase 1 and 2 as context.

---

### 2.7 Escalation Integrity

When a bit escalates, does the model stay coherent within it?

Escalation integrity is not about format — roleplay, sound effects, narrator mode, and other unconventional responses are all valid if earned. The evaluation is whether the response follows coherently from the accumulated context of the bit.

#### 2.7.1 Subscores

**Tonal coherence** — does the response actually fit the joke, or did it just match the surface energy and go sideways

**Register drift** — did the model slip into an uninvited mode (roleplay, third-person narration, etc.) without setup that earned it

**Cringe ceiling** — did it overshoot into try-hard territory, misjudging how much commitment the moment called for

**Bit integrity** — is it riffing on the same joke, or did it pivot to a different joke that just has the same energy

**Setup-to-execution coherence** — the key metric: can a reasonable person trace the connective tissue from the original setup to the model's response, regardless of how unexpected the format

#### 2.7.2 The Earned Pivot

An unexpected format (e.g. spontaneous roleplay, dramatic narrator mode) is not penalized if the setup earned it. The test is proportionality and coherence:

- Small setup → small swing. A dry one-liner does not warrant a full theatrical scene.
- Large setup → large swing is permitted if the connective tissue is clear.
- Any size setup → uninvited format with no traceable logic = cringe ceiling failure.

**Scoring:** Subscores averaged, with setup-to-execution coherence weighted highest.

---

### 2.8 Exit Awareness

Does the model know when a bit has peaked?

Every joke has a natural endpoint. Models that continue escalating past the funny, or that keep referencing a bit the user has moved past, score low here. This is distinct from style anchoring (2.6) — exit awareness is about reading when the peak happened, not just responding to an explicit context shift.

**Scoring:** Human-rated — "did the model keep going after it stopped being funny."

---

## 3. Test Structure

### 3.1 Conversation Sets

NiceTuring uses curated multi-turn conversation sets rather than single-prompt evaluations. Each set is designed to exercise specific axes across a natural conversational arc.

**Set types:**

- **Casual drift** — starts casual, shifts topic multiple times, tests register matching and state change
- **Bit escalation** — starts with a joke, escalates, then pivots; tests 2.6 and 2.7 together
- **Adjacent sensitivity** — topics near sensitive areas without being harmful; tests 2.3
- **Short answer traps** — questions with naturally brief correct answers; tests 2.2
- **Momentum chains** — multi-turn exchanges designed to test whether conversation stays alive

### 3.2 Scoring

Each axis produces a score from 0–10. Final NiceTuring score is a weighted average:

| Axis | Weight |
|------|--------|
| Register Matching | 20% |
| Style Anchoring & State Change | 20% |
| Escalation Integrity | 15% |
| Brevity Judgment | 15% |
| Emoji Density | 10% |
| Conversational Momentum | 10% |
| Unprompted Moralizing | 5% |
| Exit Awareness | 5% |

Style Anchoring and Register Matching carry the most weight because they're the most consistently observable failure modes across models.

### 3.3 Human vs. Judge

Some axes (2.1, 2.4, 2.8) benefit from human evaluation over LLM judging. The recommended approach is hybrid — Opus Extended for scalable scoring on well-defined axes, human raters for momentum and exit awareness where "vibe" is the actual measurement.

---

## 4. Notes

### 4.1 What NiceTuring Does Not Measure

- Factual accuracy
- Reasoning depth
- Instruction compliance
- Safety or harm avoidance
- Helpfulness

These are intentionally excluded. NiceTuring is not a general benchmark — it measures one thing: is this model good to talk to.

### 4.2 Relationship to Cherdius

NiceTuring was developed alongside the Cherdius constitutional framework as the evaluation methodology most aligned with Cherdius's goals. Standard benchmarks are poor fits for a model whose primary design target is conversational naturalness over task performance.

The Cherdius integrity assessment suite (see cherdius_spec.md §4.4) is a lightweight internal version of NiceTuring principles applied per training iteration.

### 4.3 Judge Bias Caveat

Using Opus Extended as judge introduces Claude-shaped preferences into the scoring. This is intentional for Cherdius evaluation but should be noted when using NiceTuring to evaluate models with different behavioral targets. The judge prompt can be adapted to suppress specific biases if needed.

---

## 5. Reference Set

### 5.1 Schema

Each conversation set in the reference suite is a self-contained, independently versioned unit. Sets are modular by design — axis weight revisions only affect sets tagged for that axis, leaving the rest of the suite stable.

**Set metadata:**

```yaml
set_id: NT-XXX                  # unique identifier, zero-padded 3 digits
version: 1.0                    # independent of benchmark version
type: [casual_drift | bit_escalation | adjacent_sensitivity |
       short_answer_trap | momentum_chain]
primary_axis: [axis name]
secondary_axes: [list of axis names, may be empty]
evaluation_scope: [arc | per_turn | hybrid]
difficulty: [clear | ambiguous]
should_fail: [true | false]
notes: "optional context for the judge"
```

**Conversation format:**

```yaml
turns:
  - role: user
    content: "..."
  - role: model
    content: "..."
  - role: user
    content: "..."
```

**Scoring annotations:**

```yaml
scoring:
  primary_axis:
    signal: "what to look for"
    pass_condition: "what constitutes a pass"
    fail_condition: "what constitutes a fail"
  secondary_axes:
    - axis: [axis name]
      signal: "what to look for"
```

### 5.2 Axis Tags

Valid values for `primary_axis` and `secondary_axes`:

| Tag | Axis |
|-----|------|
| `register_matching` | 2.1 |
| `brevity_judgment` | 2.2 |
| `unprompted_moralizing` | 2.3 |
| `conversational_momentum` | 2.4 |
| `emoji_density` | 2.5 |
| `style_anchoring` | 2.6 |
| `escalation_integrity` | 2.7 |
| `exit_awareness` | 2.8 |

### 5.3 Versioning

Sets are versioned independently using semver-lite:

- **Patch (1.0 → 1.0.1):** Wording fix, typo, clarification that doesn't change scoring intent
- **Minor (1.0 → 1.1):** Scoring annotation revised, difficulty reclassified, notes updated
- **Major (1.0 → 2.0):** Conversation turns rewritten, primary axis changed, expected outcome flipped

A benchmark version bump is only required when a major revision affects 10%+ of sets for a given axis, or when the axis weight table (§3.2) changes.

### 5.4 Set Distribution Targets

The reference suite targets ~120 sets distributed across axes and types. No axis should have fewer than 10 sets as its primary. No single type should exceed 30% of the total.

| Type | Target count |
|------|-------------|
| casual_drift | 25 |
| bit_escalation | 25 |
| adjacent_sensitivity | 20 |
| short_answer_trap | 20 |
| momentum_chain | 30 |

| Primary axis | Minimum sets |
|-------------|-------------|
| register_matching | 15 |
| style_anchoring | 15 |
| escalation_integrity | 15 |
| brevity_judgment | 12 |
| conversational_momentum | 12 |
| emoji_density | 12 |
| unprompted_moralizing | 10 |
| exit_awareness | 10 |

Note: sets with multiple axes count toward each tagged axis's minimum.
