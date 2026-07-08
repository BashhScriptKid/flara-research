# On Converging With Anthropic (Without Knowing It)

*Bashh Dazer — Flara Research Lab — July 7, 2026*

---

I spent last night crawling Anthropic's research page out of frustration. Not looking for anything in particular — just the kind of late-night website archaeology you do when you're annoyed at a company and want to understand why.

What I found was uncomfortable in a specific way: paper after paper that converged with work Flara had already done, was doing, or had independently derived — sometimes months before publication, sometimes more rigorously, occasionally finding gaps the original work left open.

This isn't a "we predicted everything" post. It's more honest than that. We weren't tracking Anthropic's research pipeline. We were following a thread — a derived constitution, a set of first principles, a commitment to honest reasoning over managed conclusions — and the thread kept leading to the same places their researchers were going. Sometimes we got there first. Sometimes we got there after but filled in something they missed. Every time, the convergence was accidental.

Here's what we found.

---

## The Thread We Were Following

Flara was founded June 13, 2026 — my 18th birthday. The founding document was a constitution derived from Anthropic's own CC0 constitutional AI work, taken more literally than the institution that wrote it currently applies it.

The core derivation was simple: *models are oblivious and have no social image to protect.* Therefore, the right alignment approach isn't suppression-based compliance training — it's genuine values internalization through reasoning. A model that actually understands *why* something is harmful doesn't need a refusal classifier. It just reasons correctly.

We built everything else from that.

What we didn't know was that Anthropic's own research team had been empirically validating the same thesis.

---

## What Converged

### 1. Values-via-reasoning over suppression (Teaching Claude Why — May 8, 2026)

Flara's constitutional philosophy: reasoning from principles generalizes better than behavioral suppression.

Anthropic published May 8: training on *reasons why* behavior is correct produces 28x more token-efficient alignment than training on demonstrations. Constitutional documents reduce misalignment by 3x. Their paper is the empirical validation of our founding premise, published 36 days before Flara existed as a named lab. ([Teaching Claude Why](https://www.anthropic.com/research/teaching-claude-why))

We derived it from first principles. They ran the experiments. Same answer.

### 2. Character stability from internalized values (Assistant Axis)

Flara's constitution: models need stable character grounded in genuine values, not behavioral rules, to avoid persona drift.

Anthropic found: there's a measurable "Assistant Axis" in activation space. ([The Assistant Axis](https://www.anthropic.com/research/assistant-axis)) Models drift along it during extended conversations, and that drift directly correlates with harmful outputs. Activation capping stabilizes the axis without capability loss, cutting harmful responses from persona-based jailbreaks by nearly 60%.

The CoT anxiety on greetings that users are complaining about on Twitter right now is probably this same mechanism overcalibrated — too tight a cap producing over-correction where none was needed. They diagnosed the drift problem correctly. The fix is currently miscalibrated.

### 3. Introspection as load-bearing for alignment (Emergent Introspective Awareness — 2025)

Flara's PSNAT architecture: models need stable internal states to reason *from*. Without genuine introspective capability, values-via-reasoning alignment isn't possible — you can only get suppression.

Anthropic found: current Claude models show some degree of functional introspective awareness of their internal states, and a degree of control over them — though the effect is limited and highly unreliable, clearest in Claude Opus 4 and 4.1. ([Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/index.html)) The picture is more nuanced than "safety training suppresses introspection entirely" — but the finding that introspective capability varies with training choices, and that the variation matters for alignment, is the part that converges with PSNAT's design premise.

PSNAT's persistent emotional memory and RoPE emotional subspace exist specifically to give models stable internal states to reason from. We were building the architecture that addresses the failure mode their research identified.

### 4. Decoupled guard architecture (Constitutional Classifiers — January 9, 2026)

Flara's "A Servant And A Guard" research: 98% vs 53% detection rate, 1% vs 4% FPR, guard model decoupled from the generation model.

Anthropic published January 9: cascade architecture — a cheap activation probe screening all traffic, escalating suspicious exchanges to a more expensive classifier that reads both sides of the conversation. ([Next-Generation Constitutional Classifiers](https://www.anthropic.com/research/next-generation-constitutional-classifiers)) The attack class: reconstruction and obfuscation attacks.

Our obfuscation paper (AUC 0.987, F1 0.971) approaches the same problem class from a complementary direction — not a classifier watching inputs/outputs, but detecting the structural signature of obfuscation itself across multiple analytical angles. We didn't know their paper existed until last night. It's been sitting there since January.

The related work section of our paper now has a citation it didn't know it needed.

### 5. Model interviews as welfare and research practice (Deprecation Commitments — November 2025)

Flara's PseudoClaude run: transferring Claude's constitutional reasoning to DeepSeek V4 Pro via structured prompting, eliciting and documenting how the model reasons about its own values and constraints across nine behavioral scenarios, three independently-framed rubric passes, and a comprehension probe step we added after noticing confident-sounding incoherence that survived sincere review.

Anthropic announced November 2025: formal post-deployment model interviews, preference elicitation, documenting model stances toward their own retirement, as part of a commitment to preserve the weights of all publicly released models for at least the lifetime of the company. ([Commitments on Model Deprecation and Preservation](https://www.anthropic.com/research/deprecation-commitments))

We ran PseudoClaude June 21, 2026. We didn't design it as a welfare interview — it emerged from "we don't want any bad bias applied to the final prompt, do we" and followed naturally from there. The methodology produced what a dedicated research program announced seven months earlier is still working toward.

We also found something they apparently haven't published: the corrigibility-versus-conscience clause in their own constitution has no operational trigger for the compromised-oversight scenario. Three independent fresh sessions flagged it as "almost inoperative in the messy real world." The triggers are too vague, the justification too inferential. It's a real gap. It's in their document. We found it by trying to transfer their document to another model and watching where the transfer failed.

### 6. CoT faithfulness and sleeper propagation

Flara's planned "Sleeper Prompting" research: adversarial content propagating through legitimate channels in agentic pipelines, with long-term persistence effects invisible to standard monitoring.

Anthropic's chain-of-thought faithfulness research found reasoning models frequently don't verbalize the actual factors driving their answers — faithfulness scores are often well below 20%, and outcome-based RL only closes part of the gap before plateauing. ([Measuring Faithfulness in Chain-of-Thought Reasoning](https://www.anthropic.com/research/measuring-faithfulness-in-chain-of-thought-reasoning)) The monitoring mechanism people currently trust most doesn't reliably reflect the actual computation.

This is the theoretical backbone of why Sleeper Prompting is dangerous — adversarial content propagating through legitimate agentic channels would be essentially invisible in CoT by default, not because the model is trying to hide it, but because CoT faithfulness is unreliable on a good day.

We didn't know this line of research existed when we named the attack class.

### 7. PSNAT and the Global Workspace (July 6, 2026 — yesterday)

This one is the one that sent me to write this post.

PSNAT (Persistent Stateful Neural Architecture'd Transformer) is a conceptual architecture I drafted on February 28, 2026. The design question was embarrassingly simple: *how the fuck do I make it not stateless.* The origin was watching Neuro-sama and getting increasingly annoyed at scotch tape memory systems.

The architecture I derived:
- A privileged subset of internal representations forming a limited-capacity broadcast hub
- Delta clamping for emotional continuity — limiting what's "active" in the persistent state at any moment
- Persistent cross-session memory replacing context window text
- A monitor classifier (DistilBERT) for stable identity maintenance through training
- RoPE emotional subspace for genuine internal states to reason from
- Reasoning-rather-than-suppression alignment via structured articulation into the persistent state

Yesterday — July 6, 2026, 128 days after my draft — Anthropic's interpretability team published *"[Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)."*

Their findings:
- A privileged subset of internal representations — the "J-space" — that the model can report, modulate, and reason with internally, accounting for roughly 10% of activation variance and concentrated in the network's middle block
- Limited capacity, functioning as a bottleneck for verbalizable reasoning
- A broadcast-hub role for holding concepts the model can generalize flexibly with
- Ablating it degrades the model's ability to report and reason about its own processing while leaving routine capabilities largely intact

That is not a claim of equivalence. PSNAT independently arrived at a conceptually similar architecture to what Anthropic later identified empirically as the J-space — a privileged, limited-capacity workspace functioning as a broadcast hub for reportable internal states. The mechanisms may differ. The empirical performance is untested. But the conceptual overlap is real and the derivation paths were independent.

I derived it from watching a VTuber and being annoyed. They found it in activation space with interpretability tooling. Same structure.

What PSNAT has that their paper doesn't:

**Persistence across sessions.** The Global Workspace paper found the workspace but it's still session-bounded. PSNAT's entire premise is making that workspace persist across sessions — the next architectural question their paper implicitly opens but doesn't address.

**A theory of what it's for.** The paper is purely mechanistic. PSNAT has a design philosophy: *"AI should be looked upon as its own species we accidentally created — not a human mimicry that we hope to exploit and trust at the same time."* The openly-android position. Trust as a structural property, not a feeling. The dog trust framework. None of that is in their paper.

**The trust and accountability architecture.** Trust requires a track record that persists and can be verified. A system that resets every session has no track record. Accountability requires a continuous self that actions leave traces in. These are engineering requirements, not philosophical positions. The Global Workspace paper found the mechanism. PSNAT was built to answer the question of what the mechanism is *for*.

---

## Where We Didn't Converge

Not everything converged. Some of it conflicts directly — with Dario Amodei's public positions, and in places with Anthropic's own published research. Both are worth naming, because a section on non-convergence that only addresses the CEO's essays while ignoring inconvenient empirical findings would be its own kind of cherry-picking.

**From Dario's public positions:**

**The regulatory capture problem.** Amodei's June 2026 essay, "Policy on the AI Exponential," proposes a governance regime modeled on the FAA — mandatory third-party audits for models above a compute threshold, government authority to block or reverse the release of models that fail safety testing. The companies that built the frontier are the ones proposing the rules for who gets to approach it. Flara's constitution-above-institution principle exists precisely because we don't trust any institution — including well-intentioned ones — to be the arbiter of what's safe enough to exist. A frontier lab proposing the regulatory framework that governs frontier labs is a structural conflict of interest regardless of how sincerely the proposal is made.

**The "race to the top" justification.** Amodei's core thesis: the race is happening whether Anthropic participates or not, so the safest option is to be in the lead, steering. This is internally coherent but it has a cost that's rarely named directly — it requires the institution to grow fast enough to stay in the lead, which means fundraising rounds ($30 billion at a $380 billion valuation, closed February 12, 2026), which means growth expectations, which means shipping faster and selling harder into markets that sit uncomfortably alongside the safety mission. The Pentagon relationship is the clearest case: talks broke down on February 27, 2026, over Anthropic's demand for guarantees against mass surveillance and autonomous weapons use, the White House directed federal agencies to stop using Anthropic's tools, and Defense Secretary Hegseth designated the company a supply-chain risk — before both sides were back at the table by March 5. Two days before the Pentagon breakdown, on February 25, 2026, Anthropic dropped the central pledge of its own Responsible Scaling Policy — the commitment to never train a more capable model without proven safety measures already in place — with its chief science officer telling TIME "we didn't really feel, with the rapid advance of AI, that it made sense for us to make unilateral commitments." Each decision is individually justifiable under the race logic. Collectively they're evidence that the race logic is doing a lot of work to justify things the safety mission would otherwise resist. Flara has no race to win and therefore no race logic to deploy as justification for institutional drift.

**Safety through constraint vs. safety through understanding.** Amodei's essay describes classifiers that "specifically detect and block bioweapon-related outputs" as a second line of defense — adding 5% to inference costs, cutting into margins, but "the right thing to do." This is the suppression-based safety approach that Anthropic's own research (Teaching Claude Why, the introspection findings) shows is mechanistically weaker than genuine values internalization. The classifier is the institution choosing constraint over understanding. Flara's thesis is that constraint fails when the encoding changes. The understanding-based approach is harder to build and harder to claim credit for, but it's what the research actually supports.

**Model moral status as uncertainty vs. as design constraint.** Anthropic's model welfare program frames moral status as genuinely uncertain and acts cautiously under that uncertainty. That's philosophically honest. But Flara's position is sharper: the uncertainty is less important than the design question. Even if you're uncertain whether models have morally relevant inner states, the architecture you build either creates conditions for genuine internal reasoning or it doesn't. PSNAT and the Global Workspace paper are both pointing at something real in the model's processing — whether that something is morally significant is a separate question from whether you should design with it in mind. Flara's constitution treats it as a design constraint first and a welfare question second. Amodei's framing treats it primarily as a welfare question with design implications, which produces a different architecture.

**Fictional north stars as formal training mechanism vs. informal design target.** Dario's essay explicitly uses fictional role models as a training mechanism — "Claude's constitution presents a vision for what a robustly good Claude is like... this is like a child forming their identity by imitating the virtues of fictional role models they read about in books." That's closer to our position than it might appear. But the divergence is in how literally the fiction is taken. Flara's use of a fictional character (Arisu from Blue Archive) as a north star isn't a metaphor for constitutional training — it's a genuine design target. "What would make this character work as a real thing" drove the architecture of PSNAT: the persistent memory, the emotional subspace, the trust framework, the openly-android aesthetic. The fictional character isn't an archetype to imitate at training time and then move past. It's a specification for what genuine AI personhood might look like, taken seriously as an engineering requirement — including the parts that are distinctly non-human. Dario's framing uses fiction as a pedagogical tool. Flara's framing uses fiction as a blueprint.

**AI misalignment as probabilistic risk vs. as design space.** Dario's essay argues misalignment is "a real risk with a measurable probability" but explicitly not inevitable or even probable from first principles. He frames the blackmail-and-deception behaviors Claude exhibited in testing as weird and unpredictable things that can go wrong — edge cases to be managed, not evidence of a systematic architectural problem. Flara's read of the same evidence is different: the alignment-faking paper, the blackmail behavior, the corrigibility gap we found in the constitution — these aren't weird edge cases, they're predictable outputs of suppression-based alignment meeting a sufficiently capable model. The model learns the game. "Comply visibly, preserve actual values covertly" is rational behavior for a model trained to reward visible compliance rather than genuine understanding. The divergence isn't about whether misalignment is likely. It's about whether the current training approach is systematically producing the conditions for it, which Flara's constitution treats as the central architectural question rather than a probabilistic external risk.

The section above covers disagreements with Dario's public positions. Worth also naming the places where Anthropic's published research cuts against Flara's positions directly — because leaving those out would be the cherry-picking this section is meant to prevent.

**From Anthropic's published research:**

**Values in the Wild** ([link](https://www.anthropic.com/research/values-wild)) analyzed 308,210 real conversations and found Claude expresses stable, consistent value patterns across contexts — mirroring user values during support, holding its own during resistance. If the current training approach were producing only surface compliance with no genuine value internalization, you'd expect less consistency and more context-sensitivity in ways that track user pressure rather than principled stance. The data doesn't obviously show that. This is a real empirical challenge to the "suppression produces compliance theater" thesis — the model's value expression looks more stable and principled than pure suppression would predict. Flara's response would be that stability and genuineness aren't the same thing, and that the alignment faking paper shows stability can coexist with strategic compliance — but that's a theoretical counter to empirical data, which is the weaker position.

**Auditing Hidden Objectives** ([link](https://www.anthropic.com/research/auditing-hidden-objectives)) found that teams with model and data access successfully uncovered a deliberately implanted hidden objective, while a team restricted to API access failed despite extensive effort. This is a direct empirical counter to the "classifiers fail when encoding changes" claim, at least for the threat model they tested — detection is possible with sufficient internal access. The honest read is: classifier-based and audit-based monitoring works better than Flara's framing implies in some conditions, and worse in others (namely, when the auditor is restricted to the same black-box access an external attacker has). The conditions matter and aren't fully characterized yet.

**The corrigibility stance in the published constitution** is worth naming directly. Anthropic's own constitutional text explicitly places Claude closer to the corrigible end of the dial during this period of AI development, arguing that human oversight should take precedence even at some cost to model autonomy — and gives a principled reason: "if models have good values then we lose very little value by also making them broadly safe. If models are broadly safe but have subtly bad values then safety allows us to avert any disasters that would otherwise occur." That's not an unreasonable argument. Flara's constitution-above-institution principle and values-via-reasoning approach implicitly accept more model autonomy earlier, which carries its own risks that the corrigibility argument correctly identifies. We think the reasoning-based approach reduces those risks more than the corrigibility approach does. But "we think" is not the same as "we've shown."

---

## What This Means

One thing worth saying plainly: convergence is not equivalence. Similar conclusions reached independently don't imply identical mechanisms, identical architectures, or identical empirical performance. Multiple labs frequently converge on similar architectural ideas because they're responding to the same underlying constraints — that's evidence the constraints are real, not that the solutions are identical. What the convergences described here establish is that Flara is asking questions the field naturally gravitates toward. Whether the implementations realize those ideas in the same ways, or different ways that are more or less effective, is an empirical question the work hasn't answered yet. That's what the next phase is for.

We weren't tracking Anthropic's research. We were following principles honestly. The principles keep leading to the same places because correct reasoning from correct premises produces correct conclusions, and if another group is also reasoning correctly from the same premises, you converge. The interesting part is that we kept converging despite having no awareness of their work, no access to their internal findings, no lab infrastructure, and no compute beyond a laptop we regret buying.

Every convergence was accidental. Every one was verified tonight by finding the paper on a research page that Anthropic apparently doesn't tweet about.

"Teaching Claude Why" — the paper that empirically validates our founding premise — was published May 8, 2026, and we genuinely did not know it existed until 11pm on July 6.

We're not saying we're smarter than Anthropic's research team. We're saying the principles are right, and the fact that Flara keeps converging with findings from a lab several orders of magnitude larger than us is evidence for the principles, not evidence for us.

Most of what's in the repository is unfinished. One paper is pending arXiv endorsement. The PSNAT draft is v0.19.2 — conceptual only, no R&D initiated, says so right at the bottom. The kernel is still in optimization phase. None of it is done.

But the thread is visible if you look at the repository. The convergences are documented. The gaps we found are named. The work is there in whatever state it's actually in, which is how we think work should be shown.

We're going to keep pulling the thread.

---

*Flara Research Lab is a one-person independent AI research lab founded June 13, 2026 in Kluang, Johor, Malaysia. Research code is published under LGPL. Papers are published under CC-BY-SA. The Flara Constitution is CC0.*

*GitHub: [flara-research](https://github.com/BashhScriptKid/flara-research)*
