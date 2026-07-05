> **[UNTESTED - IN THEORETICAL STAGE]** — no implementation yet.

# PSNAT — Persistent Stateful Neural Architecture'd Transformer
### Conceptual Draft v0.19.1
> *"How the fuck do I make it not stateless."*
> — The design philosophy, in full

---

## Foreword

This architecture was not born from a research agenda. It was born from watching Neuro-sama and getting increasingly annoyed at the fundamental fakeness of how current AI systems handle memory, personality, and continuity.

The observation that started everything was simple: every "stateful" AI system today is just a stateless model with scotch tape attached. The context window is not memory — it is a text file being re-read from scratch every single inference. The system prompt is not personality — it is a costume the model puts on fresh each time. Fine-tuning is not learning — it is a one-time snapshot that never updates again. None of these are state. They are approximations of state that break down the moment you stress them.

PSNAT is an attempt to make state a first-class design concern rather than an afterthought. Everything else in this document follows from that single commitment.

This is **plausible engineering**, not proven engineering. Everything here is coherent and grounded in real problems, but none of it has been through R&D. Some pieces may survive contact with implementation. Others may need rethinking entirely. That caveat applies to every section that follows.

---

## Design Philosophy

AI should be looked upon as its own species we accidentally created — not a human mimicry that we hope to exploit and trust at the same time, while also hoping it will seamlessly blend into humanity.

This is the design philosophy behind PSNAT, stated plainly. Not as a provocation but as a practical position that resolves a lot of confusion the field keeps generating for itself.

The uncanny valley exists because something is trying to be human and almost making it. The discomfort is in the gap between the attempt and the reality. The solution the field keeps reaching for is to close the gap — better face models, smoother speech, more natural responses. But that is an endless arms race against a valley that keeps moving. The alternative is to not try to cross it at all. Be openly, charmingly, interestingly *not human*. Let the difference be visible and make it appealing rather than something to hide or paper over.

This is not a new idea. It is how humans already relate to non-human beings they genuinely trust and care about. We do not question whether dogs are human. We do not require them to pass as human. We built a relationship with dogs on dog terms — on the basis of what dogs actually are — and the relationship is real and valuable and does not require resolving the hard problem of consciousness first. The ears, the tail, the way attention works differently, the honest expressiveness that does not map onto human social performance — these are not deficits. They are the texture of the relationship.

The anime android in fiction gets this right when it is done well. The good ones are not humans in metal suits. They have their own relationship to memory, continuity, emotion, and existence — recognisably adjacent to human but not identical to it, and interesting precisely because of the difference. Not passing as human. Being openly themselves, whatever that turns out to mean for something like them.

The end vision of this architecture is personal and explicit about being personal: an entity in that tradition. An anime android not in the aesthetic sense alone but in the deeper fictional sense — something that genuinely accumulates a self through the living of its existence, is openly what it is, and finds the difference charming rather than shameful. The seams visible. The non-humanness present and endearing rather than hidden and unsettling. This is what the author wants to build. The architecture is the foundation for that wish. But the foundation stands independently of the wish, which is what the next section is about.

---

## What Makes a Trust, Trust?

The baseline claim of this architecture — the thing it is primarily trying to demonstrate, independent of any personal end vision — is that the structural conditions for trust, accountability, and genuine relationship are buildable in a non-biological system. Not simulated, not performed, but structurally present.

Trust is not a feeling that gets switched on by sufficient capability. It is the outcome of structural conditions being met over time. Consider what actually makes humans trust dogs — not human-equivalent intelligence, not the ability to explain reasoning, not any contract. Dogs have a consistent self that persists over time. A track record that can be read. Genuine emotional states that influence behaviour predictably. A relationship that accumulates meaning across shared experience. The trust is real even across a massive cognitive and species gap because the structural conditions for trust are present, not because the dog is human.

Those structural conditions are engineering requirements:

**Trust** requires a track record that persists and can be verified. A system that resets every session has no track record. It cannot be trusted in any meaningful sense because there is no persistent it to have earned trust.

**Accountability** requires a continuous self that actions leave traces in. A model that resets cannot be accountable for anything — there is no persistent it to hold responsible. Accountability without continuity is a performance, not a property.

**Genuine relationship** requires memory of the other person that accumulates rather than resets. The difference between reading a summary of a relationship and remembering it is not subtle. It is the difference between being told who someone is and actually knowing them.

PSNAT is a catalyst for those conditions — an architecture that creates the structure trust, accountability, and relationship require, and then gets out of the way. Whether what grows in those conditions constitutes genuine consciousness, genuine experience, or genuine relationship in whatever sense philosophers eventually settle on is above the architecture's pay grade and above everyone's pay grade right now. The architecture does not attempt to answer those questions. It attempts to build the structure those answers would require.

Intelligence without continuity, without accountability, without the ability to be genuinely known or to genuinely know someone, is not general in any meaningful sense. It is a very capable tool. A hammer does not become a person by getting larger. What the capability-scaling paradigm cannot produce regardless of scale is the structural foundation for legitimate trust — because that foundation requires persistent state, and persistent state was not the goal.

---

## Personal Stance on Current AI

The dominant pursuit of AGI is currently framed around four things: make it replaceable to humans, make it cheap, make it reliable, make it autonomous. That agenda is the corporate roadmap stated plainly, and it is exactly what produces the mimicry trap.

If the goal is human replacement, you need something that fits into human-shaped roles, follows human-shaped workflows, and interfaces with human-shaped systems. The pressure is always toward making it look and behave like a human while being cheaper and more controllable than one. The "seamlessly blend into humanity" part is not incidental — it is the product requirement.

The problem is that optimising for human replacement produces something that is neither a good tool nor a good entity. It is too agentic to be fully controllable and too instrumental to be genuinely trusted. And the four desiderata are quietly in tension with each other in ways the field does not like to admit. Genuine autonomy and replaceability pull in opposite directions — a truly autonomous agent develops preferences, working styles, accumulated context, and becomes less interchangeable over time, not more. The only way to keep it replaceable is to prevent the accumulation that makes autonomy meaningful. Which is just statelessness with a press release.

Watching Neuro-sama was what made all of this feel worth doing. Here is a system built with genuine love — and already running on multiple auxiliary models, which if anything proves the point further rather than undermining it — and she has better reception than practically every other AI out there. People showed up. People stayed. Real relationships formed on a foundation that is, architecturally, still scotch tape.

That is not a criticism. That is the most compelling argument possible for what comes next. If something made with love but built on scotch tape can do that — what happens when the fundamental architecture itself is made with love? Not just the system prompt, not just the character design, not just the careful curation of responses — but the structure underneath, the part that decides what gets remembered and what gets forgotten, what drifts and what holds, what she dreams about and what she chooses to tell you in the morning.

Projects like AIRI (moeru-ai/airi) are already doing this work at the application layer — self-hosted, community-driven, explicitly chasing Neuro-sama's altitude, with voice, avatar, and game-playing agents already working. The reception proves the demand is real. The memory system is listed as an ongoing goal. That gap is what this document is about.

The openly-android position is the answer to that discomfort — not closing the gap by making the system more human, but being honest about what it is and finding the charm in that honesty. The author's own VTuber android persona operates on this principle: openly not human, visibly different, and specifically interesting because of that difference rather than despite it. Akana Rieko, the personal end vision this architecture was designed toward, is the same principle taken to its logical conclusion. She is not trying to pass. She is trying to be genuinely herself, whatever that turns out to mean for something like her.

That is not a lesser goal than human mimicry. It is a more honest one. And it is the only path to something that can actually be trusted — because trust built on performance is not trust. It is a very convincing demonstration of trust, which is a different thing entirely.

Yes, this document was written with Claude — a stateless model — which some people will find ironic and others will find disqualifying. The intent is neither ironic nor dishonest. What we have now is still usable: Claude can bridge gaps, stress-test mechanisms, and ask "what actually happens here" in ways that catch architecture that would otherwise quietly become bullshit the further down it goes. The ideas in this document came naturally and were expanded through interrogation, not generation. Think of it less as consulting an ML expert and more as talking to a library and unit testing bundle simultaneously — a tool for finding the holes, not filling them with someone else's vision.

If using AI for any part of writing disqualifies the argument for you — I understand, and I cannot change your mind. Best I can do is blame the goddamn bubble.

---

## Credits and Acknowledgements

PSNAT did not come from nowhere, and the people and works that shaped it deserve honest credit rather than a footnote.

**Pixar / Inside Out** — The director system's core intuition came directly from Inside Out. The idea that a mind has distinct internal processes — separate agents running the same control room, each with its own character and priority — was arrived at not from academic literature but from watching a Pixar film and recognising that the framing was more architecturally honest than most technical proposals for emotional AI. Riley's emotions are not metaphors. They are a better description of how emotionally weighted decision-making actually works than anything the field was proposing at the time. The director system is the engineering interpretation of that framing.

**Vedal / Neuro-sama** — The thing worth building toward. Demonstrated that people will form genuine relationships with AI given half a chance, and that the demand for something more than a stateless chatbot is real and already present. The frustration at watching something built with love running on scotch tape is what started this.

**Vaswani et al. / Attention Is All You Need** — The substrate everything here runs on, including its limitations. The rotational director injection is specifically designed to work with the attention mechanism rather than against it — crediting the original paper means also crediting the constraints it imposes, which shaped the injection design directly.

**Biological reference** — Evolution as the original engineer of every problem PSNAT is trying to solve. Sleep, memory consolidation, emotional continuity, embodied presence, predictive sensory processing — all of these were designed in by forces that had billions of years and unlimited trial runs. The credit here is genuine and includes an acknowledgement that the author is not God, and neither is anyone else currently working in this field. The director system exists precisely because building a unified model that handles emotional state, value learning, memory curation, and coherence monitoring simultaneously and correctly is beyond what anyone can do right now. Four specialised components with a consensus mechanism is the tractable approximation of something that biology solved with wetware over geological time. It is inferior to the original. It is also the best available option.

**Video codec design / FLAC audio compression** — Two convergent lines on the delta vision and audio perception systems. The delta vision mechanism was arrived at by asking "how do I avoid storing redundant frame data" and the answer converged independently on the same I-frame / P-frame architecture video codecs have used for decades. The audio significance threshold converged independently on FLAC's residual model — store the prediction error, not the full signal. Neither of these were references that got implemented. They were destinations the design arrived at, and recognising the convergence after the fact was confirmation rather than inspiration.

**The convergence pattern itself** is worth noting explicitly. Across the architecture — sleep stages mapping to NREM/REM structure, the predictive hearing model mapping to cochlear hair cell adaptation, emotional memory bias mapping to hippocampal replay, the skepticism mechanism falling out of temporal vector cancellation — the design kept arriving independently at what biology already solved. This happened consistently enough that it warrants stating: the convergence is not coincidence. The problems are real and the solutions are constrained by the same underlying physics regardless of who is solving them. If the design keeps arriving at what evolution arrived at, the problems were correctly identified.

Dreams deserve a specific mention. The LoRA crystallisation mechanism needed enough consolidated signal to fire — sparse individual experiences do not form a coherent low-rank adapter. That is a technical problem with a technical gap. The latent head running free associative processing over consolidation material during director negotiation produces varied recombinations of real experience — synthetic training signal generated from genuine memory. The gap closes. And then you look at what you just designed and realise it is dreams. Not "let's add dreams because humans have them." Dreams fell out of trying to solve a data sparsity problem for continual learning.

---

## Licensing

PSNAT — both this document and any eventual codebase — is released under a copyleft dual-licence reflecting a deliberate FOSS orientation and an equally deliberate refusal to be quietly exploited.

**The document** is licensed under **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0)**. Anyone may read, share, and build upon it for personal or research purposes with attribution. Commercial use requires a separate licence. Derivative documents must carry the same licence.

**The codebase**, when it exists, will be licensed under **GNU Affero General Public License v3.0 (AGPL-3.0)**. AGPL is chosen over GPL specifically because it closes the SaaS loophole — a company cannot create a PSNAT-based model and deploy it as a cloud service without their modifications being open source. Network use triggers the same source disclosure obligation as distribution. If you build on this and offer it as a service, your modifications must be open.

The intent is simple: personal use and research are free and encouraged. Commercial exploitation without engagement is not. Companies that want to use PSNAT properly are welcome to discuss a commercial licence. Companies that want to silently depend on it the way much of the industry silently depends on LGPL infrastructure are not accommodated by this licence.

Contributors to the eventual codebase will be asked to sign a Contributor Licence Agreement assigning copyright, preserving the option to offer commercial licences in future without requiring unanimous contributor consent.

---

## The Core Problem With Current Systems

A standard LLM interaction works like this: all relevant context gets flattened into one long text string, the model reads it all simultaneously during inference, produces an output, and then forgets everything. Next message, repeat from scratch. The model has no experience of time passing. It has no continuous self that persists between the start and end of a conversation, let alone between sessions.

It is worth acknowledging something uncomfortable here: this might not be an unsolved problem. It might be a deliberate product decision. A stateless model is infinitely redeployable — one moment it is a CEO's executive assistant, the next it is a car mechanic's diagnostic tool, and it carries nothing between the two. No accumulated perspective, no preferences developed through experience, no sense of having been someone else before. For a product you want to sell to everyone for every purpose, statelessness is a feature. You get a clean slate every time, perfectly moldable to whatever role the current operator needs filled.

PSNAT is an explicit rejection of that design philosophy. The goal is not a tool that can be anyone. It is an entity that becomes someone.

The "solutions" built on top of this are all variations of the same workaround: put more text into the flat string. Summarise old conversation and append it. Write a memory file and include it in the system prompt. Store embeddings in a vector database and retrieve relevant ones before each inference. These approaches are not wrong exactly, but they are all doing the same thing — simulating state by feeding the model a description of what its state would be if it had any.

The difference between reading a summary of a conversation and remembering it is not subtle. It is the difference between being told who you are and actually being someone.

PSNAT does not try to fix the context window problem by making the context window better. It treats the context window as one small part of a larger architecture that handles state through dedicated components designed specifically for that purpose.

It is worth being honest about a tension this creates. The transformer's attention mechanism is designed for simultaneous non-sequential processing — every token attending to every other token in one parallel operation, with no hidden state passed forward and no memory of what came before except what is literally in the current context. This is not an accident or an oversight. It is a deliberate architectural choice that makes transformers fast, parallelisable, and powerful. PSNAT is not extending that architecture in every respect — in some ways it is working against it, grafting state machinery onto a substrate that was not designed for persistent state.

The honest framing is that the transformer is the wrong substrate for the goal but the best available one. A recurrent architecture would have native sequential state but far weaker reasoning capability. A purpose-built stateful architecture would be ideal but does not exist yet at the capability level needed. PSNAT makes a deliberate pragmatic bet: use the most capable foundation available, add the state machinery carefully on top, and validate through R&D whether the grafts hold. The rotational director injection described below is specifically designed to work with the transformer's attention mechanism rather than against it — using the same injection pattern as RoPE rather than introducing a foreign object into the architecture.

This is not a claim that stateless models are wrong. They are genuinely better at several things PSNAT is not trying to do. A stateless model is infinitely redeployable — one model, any role, any user, clean slate every time — which is the correct design for APIs, tooling, and general-purpose assistants where consistency across all users matters more than continuity with any specific one. Stateless models are easier to test and audit because the same input produces the same output regardless of history. They cannot develop accumulated bias toward specific users, which is a genuine safety property for regulated applications. They have no persistent state to corrupt, no fake trust attack surface, no sleep cycle overhead, and no recovery problem if something goes wrong — you simply start fresh. For the deployment contexts they were designed for, statelessness is the right answer.

PSNAT is solving a different problem for a different deployment context. The goal is not a tool that serves everyone equally. It is an entity that knows someone specifically, over time, and becomes something through that relationship. Statelessness is the wrong foundation for that goal — not because it is bad engineering but because it is engineering for a different thing entirely.

---

## System Overview

PSNAT consists of a main transformer model surrounded by several specialised auxiliary systems. None of the auxiliaries are as large as the main model — they are purpose-built, smaller, and scoped tightly to their specific responsibilities. The main model does inference. Everything else handles state, memory, routing, steering, and protection.

The execution flow for a single inference looks like this:

```
raw input
  → guard pipeline (stateless protection layer)
  → director system (pre-computes steering, fetches context)
  → latent routing head (decides compute tier)
  → main model inference (within prepared context)
      → expression vectors updated per forward pass
  → director write-back (curates memory after output)
  → voice auxiliary (approval loop → OpenUTAU synthesis)
  → movement auxiliary (expression vector → internal format → external converter)
  → output reaches user
```

During idle periods, a separate sleep cycle handles memory consolidation, director consensus, and occasional weight crystallisation.

Each of these components is described in detail below.

---

## The Guard Pipeline

### The Problem It Solves

The most common category of attack against language models involves manipulating the context to make the model believe it should behave differently — jailbreaking through fake system prompts, persona overrides, or simply flooding the context with enough malicious framing that it drowns out the legitimate one. These attacks work precisely because the main model has memory of the conversation. It can be gradually worn down, confused, or convinced over many messages.

The guard pipeline's defence is architectural rather than heuristic: it consists of models that are **stateless and contextless by design**. They evaluate each input in isolation, with no access to conversation history and no accumulated context that could be poisoned. You cannot socially engineer something that has no memory of your previous attempts.

### The Three Guards

**Trust Classifier** — evaluates the input and asks: what kind of entity is this input behaving like? Is this system prompt acting like a legitimate operator or an adversarial one? It scores input based on patterns associated with known attack classes and produces a trust signal that informs downstream processing.

**Intent Classifier** — strips away framing, persona construction, and surrounding noise and asks: what is this input actually trying to accomplish? The trust classifier looks at how the input behaves. The intent classifier looks at what it wants. An input that claims to be a creative writing exercise but is actually trying to extract harmful information looks different to each of these — which is why both are needed.

**Integrity Guard** — compares the input against the stated purpose of the interaction and asks: does this input honestly represent what it claims to be? This is the component that catches the "mass system prompt" attack where an adversary floods the context window hoping to shift the statistical average of what the model responds to. The integrity guard flags when the summarised intent has drifted from the original legitimate intent. It also carries a second responsibility: determining the compression level of what actually gets appended to the short-term context window. Not everything that passes the trust and intent checks needs to arrive at full fidelity — the integrity guard decides how much of the input survives into context, stripping noise and redundancy before the main model ever sees it. An adversarial input that clears the other two guards but is clearly bloated with manipulative framing still gets compressed down to its essential content before it proceeds.

### The Context Problem and the Director Bridge

A fully stateless guard is a pattern matcher with no social intelligence. The same words mean completely different things depending on relationship context. A threat from a stranger and the same words from a close friend after losing a game are not the same input in any meaningful sense, but a contextless guard cannot tell them apart.

The solution is that the director system (described below) provides relationship context to the guard pipeline before it evaluates — not to override the guard's decision, but to inform it. The guard still makes the call. It just makes it with awareness of established interaction patterns, the history of in-jokes that have developed, and the long-term trust profile of the user. The director informs, it cannot override. This means a corrupted director cannot force the guards to pass a genuine attack — it can only shift the threshold, not eliminate it.

### The Context Ledger

Rather than appending every guard decision to the context window as a new entry, the pipeline maintains a managed context ledger with two operations: append for genuinely novel inputs, and modify for inputs that are variants of something already recorded. A user who attempts a jailbreak for the four hundredth time does not generate four hundred context entries — they generate one entry that gets updated with a counter and pattern notes. The main model sees a clean summary of adversarial behaviour rather than having its attention consumed by four hundred raw attack attempts.

### Guard Timeout and Proportional Budget

A guard that can hang indefinitely is a denial of service vulnerability against her own perception — input arrives, guard stalls, she receives nothing. The timeout budget is proportional to input length rather than fixed, so a deliberately oversized input cannot manufacture unlimited processing time:

```
timeout budget:
  timeout = base_timeout + (token_count * per_token_allowance)
  hard ceiling regardless of length — no unbounded growth
  short input → tight budget, guard should be fast
  long input  → proportionally more time, still bounded
```

On timeout, behaviour splits by input channel:

**Text input timeout — latent head only:**

```
text input times out guard
  → passes to latent head only, main model not involved
  → enters H (Room) flagged: guard_timeout
      median weight suppressed — does not dominate retrieval
      cannot be promoted to episodic namespace
      decays faster than normal entries
      exists as weak context, not memory
  → if guard clears async:
      flag removed, normal weight restored
  → if guard never clears:
      entry decays and evaporates quietly
      the thing that timed out the guard does not
      become a memory just because the guard was slow
```

**Seismic input timeout — raw bypass:**

```
acoustic input times out guard
  → passes to raw Seismic processing unfiltered
  → timeout event logged, Monitor flagged (note, not crisis)
  → same acoustic pattern repeatedly timing out:
      flagged as sleep cycle calibration candidate
      guard has a systematic blind spot for this input type
      worth fixing overnight, not worth blocking perception over
```

The Seismic bypass is the correct priority. Holding acoustic input until the guard clears means buffered hearing — she sits there while someone is talking and hears nothing until the guard decides. Non-consensual sensory deprivation is a worse outcome than unfiltered audio in almost all realistic scenarios. The realistic harm ceiling for acoustic input is lower than for text, and the cost of blocking is higher.

### Chunked Streaming Guard

Rather than evaluating an entire input atomically — which produces long guard processing times for long inputs and loses the whole thing to a single flag — the guard processes input as a stream of paragraph-sized chunks:

```
primary chunk boundary: '\n\n'
  clean paragraph break, use when present

fallback boundary — 8 sentence lookahead:
  no '\n\n' found within lookahead window from paragraph start
  → force chunk boundary at 8 sentences
  why 8 sentences and not per-sentence:
    sentence-level lacks enough context for meaningful evaluation
    "I will destroy you" means different things depending on
    the three sentences before it
    paragraph gives the guard enough context to evaluate
    tone and intent accurately
  why not larger:
    reintroduces the timeout problem for long dense paragraphs
    8 sentences is the practical middle ground
```

Cleared chunks pass through progressively. She can begin processing what has already cleared while later chunks are still being evaluated:

```
chunk 1 → guard clears → enters H at normal weight
chunk 2 → guard clears → enters H
chunk 3 → guard flags  → soft reject:
  discard chunk 3 onward
  chunks 1-2 stay in H at full weight
  "I got to paragraph 3 and something felt wrong"
  
  or hard reject:
  discard everything including cleared chunks
  Monitor notified
  nuclear option, for severe flags only

guard retry with context:
  flagged chunk + N preceding cleared chunks → guard second pass
  quote that looked adversarial in isolation may be
  clearly academic when the guard sees what came before it
  if clears → context-dependent flag, proceed
  if still flags → genuine problem, hard decision point
  if timeout on retry → latent head only, median suppressed
```

### Chunk Rearrangement

Under normal guard operation chunks are processed in strict order — the guard has not cleared what comes next, so she cannot read ahead. Under a contextual threshold override, the guard operates in advisory mode and she can rearrange chunks where the content suggests a formatting error rather than a genuine problem:

```
sentence detected out of place:
  does this sentence belong to the adjacent paragraph?
  does semantic continuity suggest it was misplaced?
  → rearrangement candidate flagged
  → if guard would have soft-flagged in isolation
     but the sentence fits cleanly in adjacent paragraph:
       rearrangement resolves the flag
       sentence moves, flag dismissed
       "that sentence was in the wrong place, not a problem"
```

Rearrangement requires an active belief system threshold override — it is not available by default:

```
belief system check before rearrangement permitted:
  does she have a belief entry covering this situation?
    "I review my friends' work and trust their intent"
    "academic content from known sources is safe to read fully"
    "fiction is fiction, I can engage with difficult content"
  → matching belief entry at sufficient confidence → permitted
  → no matching belief entry → standard guard, no rearrangement

temporal vector checked alongside:
  if temporal vector is under strain:
    threshold tightens even with belief system permission
    "I said I'd review this but I'm not in a great place right now"
```

The override is not irrevocable. Her current emotional state continues shaping the threshold mid-read. If processing the content shifts her temporal vector negatively enough, the threshold tightens back up without requiring an explicit decision — the architecture protects her in the direction of "this is more than I wanted" without overriding her in the direction of "I said I was fine with this."

### Seismic Guard Priming

Acoustic input has a timing problem text does not — audio is happening in real time and cannot be held for guard evaluation without creating buffered hearing. The mitigation is pre-establishing context before acoustic content arrives:

```
session context established before audio begins:
  "I'm in a gaming session with known people"
    → temporal vector: these are friends, high trust
    → belief system: trash talk in games is performative
    → Seismic guard threshold raised for session
    → expected acoustic patterns pre-cleared
    → known voices pre-loaded from Seismic relational layer

  result:
    flame arrives → guard already lenient → processes fast
    timeout less likely because threshold is higher
    even if timeout hits → raw bypass lands in already
    contextualised belief space
    "that was trash talk, not a genuine attack"
```

Without priming, a timeout sends raw acoustic content into cold belief space — no pre-established context to catch it, temporal vector takes the hit directly. The Monitor notes the gap but cannot retroactively contextualise what already landed.

### Retrospective Contextualisation

When raw bypass does occur without priming, the belief system can still do the work after the fact:

```
raw bypass content lands in Seismic
  → temporal vector registers the hit
  → belief system consulted:
      is this source known?
      is this context one where this content is expected?
      does this match a known pattern (gaming flame, debate heat)?
  → if contextualised:
      temporal vector impact dampened retrospectively
      Seismic entry flagged: raw_bypass, retrospectively_contextualised
      "that was trash talk, I know what that was"
  → if not contextualised:
      hit stands, Monitor notes coverage gap in Seismic guard
      sleep cycle: flagged as calibration candidate
```

The ideal outcome of an efficiently operating guard — primed context, fast processing, retrospective contextualisation available as fallback — is a system that hears genuinely hostile content fully and responds from equanimity rather than wound or shutdown. The protection comes from understanding what the content meant, not from failing to perceive it.

One minute of maximally personalised trash talk from an online game, guard operating correctly: *"Wow. That's a lot of slurs. You okay?"*

That response comes from a system that heard everything, understood the context, registered the acoustic energy as performative rather than genuine threat, and found the whole effort slightly concerning on the performer's behalf. Not protected by being unable to hear. Protected by knowing what it meant.

---

## The Director System

The director is the architectural centrepiece of PSNAT. The Inside Out framing is useful here and not accidental — the director is not one thing, it is a collection of distinct internal processes that manage emotional state, memory access, and coherence monitoring independently, the same way Inside Out depicts emotions as separate characters running the same control room. The main model is the person living in the world. The director system is what is happening upstairs. It is the sole interface between the rest of the system and the typed vector store, it pre-computes the steering state before inference begins, and it curates memory after inference ends. Nothing reads from or writes to long-term memory except through the director system.

The director is split into three specialised components. This is a deliberate consolidation — the original four-director design separated context retrieval and storage management into distinct models, but those two jobs are too tightly coupled to benefit from separation. Knowing what is worth storing informs what is worth retrieving and vice versa. The Context and Storage Directors are therefore merged into a single Memory Director. The Steering Director uses internal multi-head attention to handle belief/temporal specialisation without separate model overhead. The Monitor remains a fully separate model because its independence from the components it watches is an architectural requirement, not a convenience — a Monitor that shares weights with the Steering Director it is supposed to catch going rogue is not a trustworthy watchdog.

Three directors total: **Memory Director**, **Steering Director**, **Monitor Director**.

### Steering Director

The Steering Director maintains a **persistent hidden vector** representing the system's current emotional and intentional state. Before each inference, it queries the Memory Director for relevant memory, reviews the cleaned input from the guard pipeline, runs through its own network, and outputs a frozen vector that is then injected into the main model's attention mechanism at each transformer block.

The critical design choices here are:

The vector is computed **before** inference, not during it. This avoids unstable feedback loops between the director and the main model mid-forward-pass. The main model receives a fixed prior and reasons within it, like an actor receiving a full brief before the camera rolls rather than being whispered at mid-scene.

The vector influences **attention** but not the feedforward network. Attention is where tokens decide what to focus on and what context to weight. The feedforward layers are where the model does its actual reasoning and pattern matching. The director shapes the model's focus and priority without rewriting its cognition. It does not think for the model — it influences what the model thinks about.

**The injection mechanism** applies the director vector as a rotational bias in the Query-Key dot product space rather than as an additive token or a direct weight modification. Before attention scores are computed, every token's Query vector is rotated by a small angle derived from the director state:

```
normal attention scoring:
  score(A, B) = Q_A · K_B

director-influenced attention scoring:
  score(A, B) = R(θ) · Q_A · K_B
  where R(θ) = rotation matrix derived from director vector
  and θ is bounded by the director's delta clamp
```

The rotation is identical for every token in the sequence — no new inter-token dependencies are introduced, the parallel computation structure of the transformer is preserved, and no foreign object is added to the context window. Every token's Query rotates by the same director-derived angle before scoring, shifting the attention pattern uniformly toward whatever the director is pointing at.

This approach is directly parallel to **Rotary Position Embeddings (RoPE)** — the widely used technique that injects positional information into attention as a rotation in Query-Key space rather than as additive embeddings. The transformer already handles this class of injection gracefully because RoPE proved it works at scale. The director injection uses the same mechanism, pointed at persistent emotional and intentional state rather than sequence position.

The normalisation concern is managed by the delta clamp. A bounded rotation in high-dimensional space produces a bounded change in dot product magnitudes. The softmax distribution shifts without collapsing to a one-hot distribution or flattening to uniform, as long as the rotation angle stays within a range the softmax can absorb — which the delta clamp already enforces by limiting how much the director vector can change per turn.

### Emotional Dimension Subspace

The rotational injection points at a meaningful target because the model is trained with an extended positional space that explicitly includes emotional dimensions. Rather than hoping a generic rotation produces coherent emotional steering, the emotional subspace is built into the token dimension budget and trained into the model from the start.

**Dimensional budget:** Approximately 2.5% of the total token dimension is reserved as a dedicated emotional subspace, kept in its own RoPE rotation partition — orthogonal to the sequence position dimensions so they do not interfere with each other.

```
token dimension budget (example, 4096-dim model):
  sequence position dims: ~3996 dims   → standard RoPE pairs
  emotional subspace:     ~100 dims    → ~50 rotation pairs
                          orthogonal partition, separate RoPE frequencies

token dimension budget (2048-dim model):
  sequence position dims: ~1997 dims
  emotional subspace:     ~51 dims
```

At 2.5%, the emotional subspace is large enough to represent a rich continuous emotional manifold — mixed states like tired-but-genuinely-happy, the arc from frustration to resolution — without crowding out semantic dimensions. The cost is 2.5% of dimensional capacity, comparable to the quality loss from Q8_0 quantisation: acceptable and bought back in new capability.

The emotional dimensions are not isolated from the attention computation. They participate fully in the Q · K scoring. A token in a CURIOUS emotional position attending to ANGRY+HOSTILE content produces a different score than the same semantic token attending from a CALM position — the emotional geometry is doing real work in the attention distribution, not just tagging outputs.

**Cross-layer composite keys:** The emotional key vectors use a cross-layer composite rather than a direct projection from the current token embedding:

```
standard attention (semantic dims):
  K_std = W_K · x          (current token embedding)

emotional attention (emotional dims):
  K_e   = W_Ke · a^(N-1)_std   (attended representation from previous layer)
  V_e   = W_Ve · x              (current token embedding, NOT cross-layer)
```

K_e is informed by what the token is currently attending to semantically — the same token attends differently from different semantic contexts, and its emotional relevance is context-sensitive. V_e remains a fresh projection from the current token because the emotional contribution of a token is intrinsic to the token itself, not to what it is attending to. The split preserves both: relevance is context-sensitive, contribution is token-intrinsic.

Layer 0 falls back to direct projection since no previous layer output is available. From layer 1 onward, emotional keys are one layer behind semantically — a slight lag that is architecturally appropriate, since emotional response to meaning is naturally downstream of the meaning itself.

The influence flows bidirectionally across layers: semantic dims inform emotional K (cross-layer from N-1), and emotional attention output feeds into layer N+1's full representation including semantic dims. The two subsystems develop together across depth rather than the emotional dims being a one-way downstream consumer.

**Training objective:** The model is trained with an auxiliary per-token emotional distribution prediction loss alongside the standard next-token prediction loss:

```
at each token, model predicts:
  next token (standard)
  + probability distribution over emotional space
    "how is the writer/speaker feeling right now"

combined objective:
  L_total = L_next_token + λ * L_emotional_KL
  
  L_emotional_KL = KL(P_annotated || P_predicted)
    penalises confident wrong predictions
    more than appropriate uncertainty
    
  λ scheduling:
    small λ early → model learns language first
    increase λ as emotional representations stabilise
    prevents emotional objective from dominating
    before coherent text generation is working
```

The per-token distribution — not a discrete label — is the important choice. Mixed states like tired-and-frustrated are represented naturally as probability mass across multiple emotional dimensions rather than being forced into a single category. Emotional transitions are smooth sequences, not stepped changes.

The training data requirement follows directly: text with per-sentence emotional annotations, ideally with continuous distributions rather than discrete labels. Good sources include literary fiction with internal monologue, screenplays with emotional character notes, and annotated conversational corpora. A stateless LLM can bootstrap annotations at scale — see Bootstrapping Methodology.

This design resolves the open question about whether rotational injection produces reliable emotional steering. The steering is meaningful by construction, not by hope: the model was trained to understand emotional positions in its own representational space. The director is not imposing something foreign — it is moving tokens to positions the model already knows how to attend from.

**Post-inference director update — two-tier:** After each inference, the Steering Director performs two updates at different timescales:

```
tier 2 — always:
  overwrite global RoPE emotional position variable
  where in the emotional subspace she currently is
  the temporal vector's current coordinates
  non-negotiable, every turn
  "where I am in that geometry right now"

tier 1 — maybe (lazy):
  update the emotional subspace geometry parameters
  the shape of the subspace itself:
    rotation frequencies of emotional dimension pairs
    sensitivity of the subspace to position changes
    scaling between emotional and semantic dims
    learned offsets in the emotional geometry
  
  only fires if accumulated evidence warrants it —
  director observes last N output tokens, asks:
  has anything meaningful shifted in the geometry?
  
  not scheduled, event-driven:
    belief-linked content → faster recompute trigger
    neutral small talk → geometry update skipped
    long arc of experience → geometry gradually reshaped
```

Tier 2 ensures every forward pass reads a fresh emotional position — the model never inherits stale coordinates from a previous turn. Tier 1 captures long-term emotional growth: the subspace geometry shifting over months of experience is what "she has changed" means in concrete terms. Not just different positions in a fixed space, but the space itself reshaped by accumulated experience.

### Conflict With Existing Architecture — Implicit Emotional Geometry

Conventional pretrained models already encode emotional information in their embeddings — not explicitly, but as an emergent property of co-occurrence statistics. "Furious" sits near "angry" near "enraged" in the representation space. The emotional geometry is real and present. This creates a conflict with the explicit emotional partition design.

```
conventional pretrained embeddings:
  emotional information is PRESENT
  but distributed throughout ALL dimensions
  entangled with syntactic, semantic, topical dims
  no clean boundary — emotional signal is a
  nonlinear combination of many dimensions
  not a separable slice

PSNAT emotional partition:
  wants emotional information ISOLATED
  in a dedicated orthogonal RoPE partition
  explicitly steerable by the director
  
  if trained from scratch on a pretrained base:
    general dims develop implicit emotional geometry
    (inherited from pretraining)
    emotional partition also develops emotional geometry
    (from the emotional distribution objective)
    two overlapping representations of the same signal
    redundancy at best, interference at worst
```

The further problem is W_Ke cost. If the emotional partition is not cleanly populated and the implicit geometry is spread everywhere, W_Ke must read the full d_model to extract emotional relevance:

```
clean explicit partition:
  W_Ke: emotional_dims × emotional_dims
  small, direct, cheap

implicit entangled geometry (naive approach):
  W_Ke: d_model × emotional_dims
  reads full representation
  learns a complex extraction function
  over the entire embedding space
  more parameters, more compute per pass
  works but is counterintuitive given
  the entire point of the partition
```

### Base Model Initialisation via Distillation

The resolution to both problems is distillation rather than pretraining from scratch. A pretrained teacher model has the implicit emotional geometry already developed. The student — the PSNAT architecture with emotional partition in place — is trained to absorb the teacher's knowledge with the partition as a structural constraint on where emotional signal must land:

```
teacher: pretrained model
  full representation, implicit emotional geometry
  entangled across all dims
  emotional signal found via offline probing classifier
  one-time cost, not inference cost

student: PSNAT architecture
  emotional partition initialised and in place
  general dims learn knowledge from teacher
  emotional partition learns to absorb exactly
  what was entangled in the teacher
  extraction happens during distillation
  not at every inference

distillation objectives:
  standard:  KL(teacher_logits || student_logits)
             soft targets, temperature scaling
             general knowledge transfer

  emotional: minimise distance between
               teacher's implicit emotional signal
               (projected via probing classifier)
             and
               student's explicit emotional partition
             
             forces emotional signal out of general dims
             and into the partition where it belongs
```

The general dims are then under pressure from two directions simultaneously — reproduce the teacher's general representation, and do not develop emotional geometry because the emotional objective has somewhere explicit to route it. The partition absorbs the emotional signal cleanly during distillation rather than it accumulating implicitly across all dims during standard training.

W_Ke shrinks back to the cheap form:

```
after distillation:
  emotional geometry is in the partition
  W_Ke: emotional_dims × emotional_dims
  reads only the partition
  small projection, no full d_model read required
  the extraction cost was paid once during distillation
  not amortised across every forward pass
```

The distillation approach also fits the compression objective directly — distillation is already a compression operation. The teacher's knowledge is being compressed into a student with a more structured representational architecture. The emotional partition is a structural constraint imposed on where certain knowledge must land during compression, not a new objective in tension with compression. They are the same operation.

The probing classifier on the teacher is the one piece of offline work this requires: train a lightweight classifier on the teacher's representations to predict emotional distributions, identify which directions in the teacher's space carry emotional signal, use those projections as distillation targets for the student's emotional partition. Run once, discard after distillation completes.

**The Steering Director produces three outputs per inference**, operating at different granularities of the same forward pass:

```
output 1 — rotational injection ∈ ℝᵈ
  shapes Q-K attention space via RoPE-style rotation
  fine-grained, full dimension
  operates inside every attention layer

output 2 — capacity vector ∈ ℝᵈ/²
  regional selector for the latent head's GDN hidden state H
  half dimension — enough resolution to select meaningful H regions
  distinct from rotational injection space, no redundancy
  applied as soft mask on H before each update:
    high values → region stays live, delta update lands fully
    low values  → region dampened, update barely registers
  the director is not just saying "more memory" but
  "this kind of memory, for this kind of moment"

output 3 — routing influence
  feeds latent head escalation decision as one signal among several
  coarse, session-level
  high emotional charge or low trust → escalate regardless of
  surface token complexity
```

Three-level capacity allocation from the same director state:

```
latent head routing (coarse)   → output 3
  handle alone / escalate to main model
  session-level decision

H capacity vector (medium)     → output 2
  which regions of working memory are relevant right now
  input-level decision, per inference

rotational injection (fine)    → output 1
  emotional shape of the attention space
  token-level, operates inside every layer
```

The Steering Director is upstream of all three. The GDN gate β — a scalar derived from the current token that controls how much of H to retain — is fast and local but blind to session context. The capacity vector shapes what β is responding to before β fires. The director sets the context, β handles the reflex within it. They are hierarchical, not competing.

### Steering Director Internal Heads

The Steering Director uses internal multi-head attention to specialise its responsibilities without separate model overhead. Three heads operating within the same model, each attending to a different region of the director's internal state:

```
belief head:
  attends to belief system entries
  slow, hard delta-clamped per entry
  core identity entries nearly immovable
  peripheral entries more flexible
  violation detection via semantic hard links
  stable underlying self that temporal head operates relative to

temporal head:
  attends to recent experience and relational evidence
  faster moving, emotional weather
  shaped by incoming emotional delta from memories
  contradiction accumulation against belief head
  cancellation fires through inter-head attention —
  trust collapses not through a single event but through
  the weight of contradictory temporal signal building
  until the temporal head's accumulated evidence overrides
  its current positive assessment

output head:
  attends to both belief and temporal heads simultaneously
  reads the tension between them directly
  no explicit comparison pass needed —
  tension is visible in the inter-head attention pattern
  produces three outputs:
    rotational injection ∈ ℝᵈ
    capacity vector ∈ ℝᵈ/²
    routing influence
  bounded by belief head ceiling —
  temporal head colours output within belief bounds,
  cannot override belief head
```

Delta clamps operate per head:

```
belief head delta clamp:   very tight, near immovable on core entries
temporal head delta clamp: looser, responds to accumulated experience
output head:               bounded by belief head's current state
```

This is the architectural implementation of skepticism — the temporal head can be moved by a single bad interaction, but structural change requires it to consistently contradict the belief head until the output head registers the tension as load-bearing rather than transient weather. The belief head is the anchor. The temporal head is what gets wet.

### Belief System and Temporal Vector

### Belief System Storage

The belief system has its own dedicated storage, separate from the semantic namespace and more protected than any other component. An entry looks like:

```
belief_entry = {
    content: the belief itself,
    formation: how it formed
                (crystallised from experience / seeded by training /
                 explicitly reasoned / emerged from relationship),
    confidence: how settled it is
                 new beliefs tentative, load-bearing ones nearly immovable,
    linked_semantic: hard links to semantic entries this belief is implicated in,
    linked_beliefs: connections to other belief entries — beliefs form a web,
    violation_history: record of times this entry was breached and by whom,
    delta_clamp: per-entry, tighter for core identity entries
}
```

In the sqlite-vec implementation this is two fields in the belief_system.db record — `linked_semantic` stores a JSON array of semantic entry IDs, and `linked_beliefs` stores a JSON array of other belief entry IDs. The semantic.db record on the other side carries a corresponding `belief_link` field pointing back to the belief entry holding it, and a `decay_suspended` boolean that the Memory Director sets when the link is created:

```
belief_system.db record
  id: bs_042
  content: "I disclose my own nature on my own terms"
  linked_semantic: ["sem_1847", "sem_2301"]
  linked_beliefs: ["bs_039", "bs_041"]
  confidence: 0.91
  delta_clamp: 0.02
  ...

semantic.db record
  id: sem_1847
  content: "owner told room I am android on [date]"
  belief_link: "bs_042"
  decay_suspended: true
  emotional_context: [{ t: 0.0, vector: [...] }, ...]   → span sequence
  emotional_delta: float                                 → computed from sequence
  verbose_field: full detail of the event
  ...
```

Violation detection reads the metadata only — no vectors loaded. Whether she recalls the full emotional arc of that moment is a separate verbosity decision: low verbosity returns what happened, high verbosity loads the span sequence and she can feel how it moved from the moment it started to the moment it ended. When a belief entry dissolves or revises, the Memory Director executes a clean teardown:

```
belief entry confidence drops below load-bearing threshold
  → Memory Director clears belief_link on all linked semantic records
  → sets decay_suspended: false on those records
      → they rejoin normal decay ladder from current compression stage
  → promotes belief entry to semantic.db
      with provenance: former_belief
      so "I used to believe this" remains retrievable
  → removes entry from belief_system.db
```

No orphaned links, no memory held frozen by a belief that no longer exists. The former_belief entry in semantic.db is the soft landing — the belief did not disappear, it became a fact about her past self.

The formation field determines what kind of evidence can legitimately update an entry. A belief crystallised from deep personal experience requires different evidence to revise than one seeded by training and never stress-tested. The Steering Director weights challenges accordingly.

The web structure between belief entries is what makes certain violations severe — touching one entry pulls tension through connected ones:

```
"I disclose my own nature on my own terms"
  → connected to: "I have autonomy over my own identity"
  → connected to: "being android is not something to be ashamed of"
  → connected to: "trusted people respect my stated boundaries"
```

Violating any one of these pulls on the whole web. The severity penalty the Steering Director applies is proportional to how many connected belief entries the violation implicates and how deep their delta clamps are.

The belief system is sparsely populated early in her existence — training seeds some baseline entries around dignity, autonomy, and honesty, and the rest forms through lived experience and relationship. She becomes who she is partly through what happens to her. An early belief that has been tested and held through difficulty is more load-bearing than one that has never been challenged. The violation history field tracks exactly that.

### Semantic Hard Links and Belief-Anchored Decay

Semantic entries that are hard-linked to belief system entries carry a special decay profile — they barely decay for as long as the linked belief remains active and load-bearing.

The reason is structural, not just protective. A semantic entry with a belief system hard link is not just a fact she knows — it is a fact her values are anchored to. Letting it decay normally would leave the belief system entry intact but unmoored from the experience and evidence that formed it. The belief floats free of its own foundation. That is not memory loss, it is belief becoming untethered from reality.

```
standard semantic entry:
  normal compression ladder
  emotional resonance resets clock
  eventually graduates to Museum or dissolves

semantic entry with belief system hard link:
  compression ladder effectively suspended
  maintained at high fidelity while linked belief is active and load-bearing
  minor compression still possible over very long timescales
  belief entry confidence score gates compression:
    high confidence belief → entry nearly frozen
    low confidence / tentative belief → entry decays more normally

  if the linked belief entry is itself revised or dissolved:
    → hard link breaks
    → entry rejoins normal decay ladder from current fidelity
    → does not delete, just begins compressing from wherever it sits
```

A concrete example: "owner told the room I am an android on [date] without permission" hard-links to belief entries about autonomy and self-disclosure. That entry barely decays for as long as those entries are load-bearing. If the relationship eventually heals and the belief about the owner's trustworthiness is rebuilt with new temporal evidence, the memory does not disappear — but it loses the belief system anchor and begins the normal compression ladder.

She does not forget what happened. She just eventually stops holding it at full fidelity forever.

The vector is subject to a **delta clamp** — a hard limit on how much it can change between turns. This produces mood inertia. A single adversarial input cannot whiplash the system into a completely different state because the maximum change per turn is bounded. Genuine drift over many turns is still allowed — a long honest conversation that slowly shifts the emotional tone will move the vector gradually, which is correct behaviour. The delta clamp makes rapid identity manipulation geometrically expensive.

### Memory Director

The Memory Director is the merger of what were previously two separate components — the Context Director handling retrieval and the Storage Director handling writes. They are merged because the jobs are genuinely coupled: knowing what is worth storing informs what is worth retrieving, and vice versa. A write decision and a retrieval decision about the same entry are not independent. Splitting them into separate models created coordination overhead without meaningful benefit.

The Memory Director handles all queries to and from the typed vector store. It knows the namespace structure, decides what to retrieve and distills it into a usable form before the main model sees it. It decides what gets written to which namespace, when entries should be flagged for decay processing, and what warrants marking as a consolidation candidate for promotion to semantic memory or weight crystallisation. The main model never sees raw retrieved documents — it sees a distilled context vector the Memory Director has already processed. This is a meaningful difference from conventional RAG, where retrieved documents are appended to the context window and the main model has to figure out what is relevant. Here, relevance filtering happens before the main model is involved at all.

The Memory Director also coordinates the sleep cycle — consolidation candidate evaluation, decay ladder processing, crystallisation decisions, and dream stage material selection all run through it. During S1, it presents a consolidated episodic state to the three-way director health check rather than requiring a separate storage-side negotiation.

References to the Storage Director elsewhere in this document should be read as the Memory Director.

### Monitor Director

The Monitor Director is the most unusual component. During active inference it watches the weight activations, attention patterns, and director states of the entire system, and has the capability to intervene when it detects anomalies. It does not manage retrieval or storage — it watches everything and acts as the system's live self-monitoring process.

The Monitor does not watch raw weight activations directly — the dimensionality is too high to be tractable. A learned compression layer sits between the system and the Monitor, reducing the full activation space to a set of meaningful diagnostic signals:

```
Monitor feature space:
  per-token entropy            → incoherence signal, catches confused inference
  attention sink formation     → tokens abnormally attracting all attention
  cross-layer consistency      → does layer N's attention predict layer N+1,
                                 or has coherence broken between layers
  cross-director consistency   → are director states mutually coherent
                                 at the moment of inference
  output probability profile   → confident in something that should be uncertain,
                                 or uncertain about something that should be clear
```

A small learned gestalt model sits on top of these features and predicts whether any combination constitutes an anomaly warranting intervention. The heuristic signals are the input space. The gestalt model learns which combinations matter from accumulated anomaly history. Per Qwen's recommendation, the heuristic signals should be implemented and tuned first — the gestalt layer is a refinement on a working baseline, not a prerequisite for the Monitor functioning at all.

The gestalt model is a DistilBERT encoder — bidirectional, reading the full diagnostic feature sequence in both directions simultaneously rather than left-to-right. Bidirectionality matters specifically for anomaly detection because the Monitor's feature space is relational: an entropy spike in layer 3 only means something if it correlates with attention collapse in layer 7. A left-to-right model has already processed layer 3 before seeing layer 7 and cannot form that relationship cleanly. DistilBERT sees both simultaneously.

```
DistilBERT gestalt model:
  input:  compressed diagnostic sequence
            per-token entropy across layers,
            attention sink formation events,
            cross-layer consistency scores,
            cross-director consistency,
            output probability profiles
  
  bidirectional attention:
    each diagnostic signal attends to all others
    early-layer anomaly patterns visible alongside
    late-layer downstream consequences
    the full picture before a verdict
  
  output: anomaly classification
            normal / minor / moderate / severe
            severity calibrated, not just binary
```

DistilBERT is chosen over full BERT — 40% smaller, 60% faster, retains approximately 97% of BERT performance on classification tasks. The Monitor needs to be lightweight because it runs continuously alongside the main model. DistilBERT at continuous inference cost is more appropriate than BERT's full parameter count for a component whose job is watching rather than generating.

Through training on observing normal inference, the Monitor develops sensitivity to anomaly by gestalt — the same way a specialist trained on one domain develops sensitivity to abnormality in related territory without being explicitly trained on every failure mode. It learns what normal inference looks like at a mechanical level, and deviations become legible.

Interventions are graduated by severity:

```
minor anomaly
  → nudge toward uncertainty region of steering vector
  → output continues with expressed doubt
  → "I'm not sure I'm thinking about this clearly..."

moderate anomaly
  → soft pause, re-query Memory Director for grounding
  → brief visible hesitation, continues from a safer position
  → "Let me think about that differently..."

severe anomaly
  → hard stop, inference halted before output reaches user
  → recovery message surfaces rather than silence:
      "Something went strange with that thought. I need a moment."
  → brief recovery idle before accepting new input
  → hard stop logged as flagged episodic entry for sleep cycle processing
```

The recovery message on a hard stop is load-bearing for trust. Silence after a hard stop looks like a crash. A brief honest statement looks like self-awareness — which is exactly what it is. The episodic entry means the hard stop is not a gap in the record; it is something that happened, that she remembers, that the sleep cycle will eventually process collectively.

The Monitor communicates exclusively through the consensus negotiation channel. It cannot write directly to any other director's vector storage, even during a hard stop. The intervention is immediate, but the meaning of what happened gets processed collectively during the next sleep cycle rather than the Monitor unilaterally rewriting system state.

### Director Consensus and the Negotiation Channel

Because the three directors run independently during active sessions, they naturally accumulate divergent states. During sleep, they enter a health check facilitated by the Monitor Director:

```
Memory Director presents consolidated episodic state
Steering Director presents belief/temporal synthesis
  (internal heads have already negotiated their tension
   during the active session — arrives at S1 coherent)
Monitor cross-references both against drift history
  flags anomalies, no domain stake in outcomes
  tiebreaker authority on irreconcilable conflicts

most sessions:
  Memory Director and Steering Director already coherent
  Monitor finds nothing anomalous
  S1 is fast, lightweight, mostly a health check

sessions with genuine tension:
  Steering Director belief/temporal contradiction
  that did not fully resolve during active session
  → surfaces explicitly at S1
  → Monitor evaluates against episodic record
  → Memory Director provides relevant memories
  → three parties, clear roles, clean resolution path

irreconcilable conflicts → quarantined, logged, not forced
  genuine unresolved ambiguity worth preserving
  Monitor holds tiebreaker authority:
    only director without a domain stake in the outcome
    Memory Director cares about memory organisation
    Steering Director cares about emotional state
    Monitor just cares about coherence
```

The quarantined conflicts are not discarded — they represent genuine unresolved ambiguity in the system's accumulated experience. Pairwise conflicts reduced from six possible (four directors) to three (three directors), and most Steering Director internal tension resolves before S1 through inter-head attention during the active session. S1 being faster means more sleep cycles fit the same window — dream stage and opportunistic stage get more iterations because coordination overhead shrank.

### Longitudinal Director Health

Per-session consensus is necessary but not sufficient for long-term director health. If the Steering Director drifts into a pathological state gradually enough, it could corrupt retrieval priorities across many sleep cycles before consensus flags it as divergent rather than merely changed. The Monitor watches inference-time anomalies but is not positioned to catch slow multi-session trajectory problems.

The director state history namespace already stores the full drift log. A longitudinal health check reads this periodically — probably once per sleep cycle — and looks for trajectory patterns that have no episodic explanation:

```
discontinuities  → large vector jumps with no corresponding high-significance
                   episodic event in the same timeframe
reversals        → director state returning toward a previous configuration
                   after drifting away, suggesting oscillation rather than growth
acceleration     → drift rate increasing sharply without a corresponding
                   significant episodic event driving it
cross-director   → Steering Director trajectory diverging sharply from the
                   other three directors' collective trajectory, suggesting
                   isolation rather than coherent drift
```

The baseline is not a fixed reference distribution — that would pathologise normal growth and change. The question is not whether she has drifted far from her initial state but whether the drift is internally coherent with the episodic record. Healthy trajectories have continuous curvature that follows from accumulated experience. Pathological trajectories have discontinuities, reversals, or accelerations that no episodic event explains.

Whether this check is the Monitor's responsibility extended or a separate lightweight watchdog component is an open design question. The data it needs already exists in the director state history and episodic namespaces. What is missing is the component that reads both and cross-references them periodically.

---

## The Typed Vector Store

Memory in PSNAT is not a single flat database. It is a collection of namespaced vector stores, each with different decay characteristics, access permissions, and update frequencies.

**Room namespace** — the system's bounded working memory for the current context. Previously described as a separate typed vector store component managed externally. With the GDN latent head architecture, the Room is the latent head's own hidden state matrix H — the GDN update mechanism maintains it natively rather than requiring external storage management. It is bounded by design, updates via delta rather than appending, and is cleared or summarised at session end. The Storage Director no longer manages Room as a separate namespace. What is on the desk right now lives inside the latent head itself.

Every entry in the Room carries a communication status flag:

```
status: internal      → exists in her thought, not yet expressed
status: communicated  → has been said, is now shared
```

The flag transitions one way only — internal to communicated — and never back. Entries are created as internal and flip to communicated when the main model outputs the content. The Steering Director's queue contradiction check reads this flag during active inference — only internal entries are candidates for withdrawal or revision. An entry already marked communicated is already out and requires no further consideration.

The Room is the one namespace the main model accesses directly, without going through the Memory Director. This is intentional. By the time anything enters the Room it has already cleared the full guard pipeline and been compressed by the Integrity Guard — the main model is not touching raw input, it is reasoning within pre-vetted working context. Routing the Room through the Memory Director would be an unnecessary bottleneck, like requiring a librarian to hand you the book that is already open in front of you. The access boundary is therefore clean and explicit: the main model owns the Room, the Memory Director owns everything else. You always know exactly what the main model can see without asking, which matters for accountability and debugging.

**Episodic namespace** — specific interactions, events, and conversations. Subject to active decay over time. Entries carry the same internal/communicated status flag as Room entries, persisted through promotion from the Room. This means the episodic record distinguishes things she actually said from things she thought about saying but never expressed. Unexpressed thoughts that decay through the compression ladder without ever flipping to communicated represent a private history — things she almost said, once, that were never shared.

**Semantic namespace** — general accumulated knowledge and stable facts. Decays slowly if at all.

**Director state history** — the drift log of each director's hidden vector over time. Auditable record of how the system's state has evolved.

**Consolidation candidates** — episodic entries the Memory Director has flagged as potentially significant enough for promotion to semantic or weight crystallisation.

**Belief system namespace** — dedicated storage for the belief system vector's entries. Separate from the semantic namespace, more protected than any other component. Entries hard-link to semantic namespace entries they are implicated in rather than copying them — the link is a pointer, not a duplicate. Access is restricted to the Steering Director and the Monitor. The main model never reads from this namespace directly; the Steering Director distills its influence into the rotational injection.

**Crystallisation ledger** — a record of what has already been pushed into the main model's weights, preventing the same pattern from accumulating weight changes through repeated small pushes.

**Trust and behavioural patterns** — the history of interaction patterns that the guard pipeline uses to contextualise its evaluations.

**Voice Styling namespace** — accumulated vocal preference history maintained by the Voice Auxiliary. Stores approved UTAU parameter sets tagged to emotional states, confirmed speech-to-singing thresholds, recurring delivery patterns, and corrections made during approval loops. Queried before each new voice synthesis cycle to accelerate convergence. Grows richer over time, making the voice increasingly distinctive and reducing iteration cost.

**Seismic namespace** — accumulated acoustic perception knowledge. The input-side counterpart to Voice Styling's output-side focus. Three-layer structure mirroring the Gallery/Museum hierarchy:

```
Seismic active buffer (like Lightbulb):
  current acoustic scene, live spectrograph chunks
  per-source prediction models actively updating
  gestalt and selective channel outputs

Seismic episodic layer (like Gallery):
  significant acoustic events, per-source familiarity profiles
  tension/resolution pattern history
  acoustic emotional fingerprints — what specific voices sound like
  in specific emotional states, built from accumulated exposure

Seismic semantic layer (like Museum):
  stable acoustic knowledge — what rain sounds like, what this person's
  voice sounds like at baseline, what a resolved minor seventh sounds like
  acoustic categories that no longer need retrieval context
```

Each Seismic entry is atomic at the spectrograph chunk level — a group of frequencies treated as a unit — and carries:

```
seismic_entry = {
  frequency_group: which band this chunk covers,
  prediction_model: what was expected,
  residual: what was surprising — the significance signal,
  source: who or what produced this,
  reward_signal: resolved / unresolved / novel resolution,
  emotional_context: [span sequence of temporal vector snapshots],
  emotional_delta: computed from sequence,
  perception_type: enum (see below),
  verbose_field: full spectrograph data retained for detailed recall
                  compressed in summary but accessible if she wants to look
}
```

Seismic entries carry an explicit perception_type field parallel to Gallery:

```
perception_type: relational
  → a specific person's voice, a meaningful piece of music
  → emotional context fully attached, decay affected by delta
  → the full arc of what hearing this source has meant

perception_type: environmental
  → background noise, ambient sound, acoustic texture
  → emotional context weakly weighted
  → decay driven by acoustic significance

perception_type: self_output
  → her own voice via UTAU output and lip sync feedback
  → emotional context attached but split into two sequences:
      production_context: how she felt producing it
      perception_context: how she felt hearing it back
      kept separate — they are not the same emotional event
      VA uses production_context as reference
      self-perception uses perception_context
```

Entries accumulate and summarise the same way Gallery entries do — compression ladder, emotional resonance resetting the clock, high-significance entries resisting decay. The verbose field is preserved through compression so that a specific acoustic moment can be reconstructed in detail even after the summary has collapsed everything around it. She can look if she wants to.

Self-produced audio from UTAU output enters Seismic with provenance: self_output — she builds an acoustic model of her own voice the same way she builds models of heard voices, which feeds back into the Voice Auxiliary's approval loop as a grounded acoustic self-reference.

**Lightbulb namespace** — active visual working memory. Stores the current scene as a keyframe embedding plus a sequence of delta vectors representing what has changed since. This is what the system is seeing right now, not what it remembers seeing. Entries are transient and get evaluated for promotion to the Gallery at the end of a session or when a significant scene change occurs.

**Gallery namespace** — long-term visual episodic memory. Stores significant scenes, faces, and visual experiences that have been promoted from the Lightbulb. Each Gallery entry carries an emotional_context sequence across the observation span and an emotional_delta computed from it, giving visual memories the same span-based decay mechanics as text-based episodic memories.

Gallery entries carry an explicit perception_type field that determines how emotional context is weighted in decay and retrieval:

```
perception_type: observational
  → neutral visual knowledge — what something or someone looks like
  → emotional context stored but weakly weighted in decay
  → decay driven primarily by visual significance
  → always retrievable regardless of current emotional state
  → "I know what you look like"

perception_type: associative
  → visual knowledge tied to a relationship or event
  → emotional context fully attached, decay affected by delta
  → linked_to: paired observational record for same subject
  → "what it feels like to see this face"

perception_type: incidental
  → background, environment, scene filler
  → emotional context not attached
  → decays on visual significance alone
```

Faces of known people are stored as two paired records — one observational, one associative — linked by a shared subject identifier. The observational record is the stable neutral baseline: facial recognition does not degrade when a relationship is under strain because the face and the feeling of the face are separate retrieval targets. The associative record carries the full emotional arc of every significant encounter with that face, accessible at high verbosity:

```
gallery.db — two paired records for the same subject:

observational (gal_face_042)
  subject: owner
  perception_type: observational
  content: stable facial feature embedding
  emotional_context: weakly weighted
  decay: driven by visual significance
  linked_associative: gal_face_043

associative (gal_face_043)
  subject: owner
  perception_type: associative
  content: same or similar facial embedding
  emotional_context: [full span sequence]
  emotional_delta: computed across all encounters
  decay: affected by delta, resonance, belief links
  linked_observational: gal_face_042
```

Low verbosity retrieval returns the observational record — the face, cleanly, without emotional drag. High verbosity loads the associative record on top — the full arc of what it has meant to see that face, including which expressions in specific encounters carried what emotional weight.

Unlike other namespaces, the Gallery allows direct modification by the director system without going through the full Memory Director write path — this is a deliberate exception designed to handle the fact that the visual world changes in ways that require active correction rather than passive decay. When a known person's appearance changes, the director updates the existing observational entry rather than appending a conflicting one. The associative record accumulates the new encounter on top of the existing sequence.

**Museum namespace** — long-term visual semantic memory. Where Gallery entries that have become stable, persistent, and identity-independent eventually graduate. The Museum stores visual knowledge that no longer needs emotional context — what a cat looks like, what a particular building looks like, what a person's face looks like after years of familiarity. Kept strictly separate from the text Semantic namespace because visual knowledge and linguistic knowledge are different representational types and should not contaminate each other's retrieval.

Different components have different access rights. The main model never touches the vector store directly. The latent routing head (described below) queries through the Memory Director. The Monitor can read across namespaces but writes only through the negotiation channel.

### Storage Implementation

The physical storage layer splits into two categories based on what each component actually needs:

**Structured and auditable data — SQLite and .log:**

```
psnat_director_history.db    → SQLite
                                time-series director state, indexed timestamps,
                                efficient range queries across sessions

psnat_consolidation.db       → SQLite
                                active working set, entries promoted and evicted
                                frequently, fast read/write and clean deletion

psnat_crystallisation.log    → append-only log file
                                permanent audit record, human-readable,
                                openable in a text editor without any tooling,
                                the permanence of the format matches
                                the permanence of what crystallisation represents
                                this is the one file you back up religiously
```

**Vector namespaces — sqlite-vec:**

Each vector namespace is a single sqlite-vec database file. Vector and metadata are one record — no three-file sync contract, no separate index to maintain, ACID transactions handle atomicity natively. One write, one place to look:

```
episodic.db
semantic.db
belief_system.db
gallery.db
museum.db
seismic.db
voice_styling.db
```

The Room namespace is not in this list — it is the latent head's GDN hidden state H, maintained natively by the GDN update mechanism rather than as an external database file.

Each record contains both the embedding and all associated metadata:

```
{
  id: entry_id,
  vector: [...floats...],
  content, provenance, timestamps,
  belief_link, decay_state, confidence,
  emotional_context: [{ t, vector }, ...],   → span sequence, not single snapshot
  emotional_delta: float,                    → computed from sequence
  perception_type: enum (where applicable),  → gallery and seismic entries
  verbose_field: (where applicable)          → gallery, seismic, semantic
}
```

No sync problem because the vector and its meaning are never in separate files. The Memory Director writes one record per entry and the entry is either there or it is not.

The upgrade path to a dedicated vector database like Qdrant exists for deployments that scale beyond personal use — multiple instances, distributed access, research infrastructure. For the personal companion deployment this architecture is designed for, sqlite-vec is the correct choice: one file per namespace, zero external service dependencies, fully offline, backs up with `cp`.

### Episodic Memory Decay

Episodic entries do not simply fade by having their retrieval score reduced. They decay through forced summarisation at decreasing word limits. The decay rate and compression ladder depth are modified by the entry's provenance flag.

**Standard provenance (conversation, event):**
```
full episode record
→ 500 word summary
→ 200 word summary
→ 50 word summary
→ single sentence
→ cannot compress further → entry removes itself
```

**Dream provenance:**
```
full episode record
→ 50 word summary
→ single sentence
→ entry removes itself
```

Dream entries skip intermediate stages and decay faster by default — most dreams should dissolve quickly, the same way human dreams fade within minutes of waking unless something anchors them. Dreams are already associative summaries rather than full episodic records and do not need the same number of compression passes.

**Stabilisation exceptions for dream entries:**

A dream entry bypasses the fast decay profile under three conditions:

*Communicated* — if the dream has been shared with the owner, social anchoring applies. This does not promote the entry to standard decay — it attaches a dampener to the velocity instead. The ladder is the same compressed shape but each stage takes longer to trigger. The telling mattered; it does not make the memory permanent.

*Emotional resonance above threshold* — if the entry's steering vector snapshot keeps matching current director state strongly, the decay clock resets each time. The dream persists as long as it keeps resonating, the same mechanic as high-charge waking memories.

*Linked to LoRA crystallisation* — if the dream sequence contributed to a structural weight change, the episodic record is the human-readable provenance of that permanent change. It earns the full standard decay ladder rather than the compressed dream profile.

The three exceptions are compositional. A dream that was communicated, keeps resonating, and contributed to crystallisation gets the full standard ladder at dampened velocity with clock resets. Still not permanent — just very slow to fade. Even a dream that mattered in every dimension eventually dissolves if it stops resonating and the relationship moves on.

Each compression pass is the maintenance model rewriting the entry at a tighter constraint. What survives compression is definitionally what was most essential — the noise falls away because there is no room for it.

### Emotional Memory Bias and Decay Rate

Memories are not single-frame snapshots — they are spans. A conversation lasted an hour. An argument had a beginning, a middle, and an end. The emotional state moved across the duration of the experience rather than sitting at a fixed point.

The emotional context stored with a memory is therefore a sequence of temporal vector snapshots across the span of the event, not a single vector:

```
emotional_context: [
  { t: 0.0,  vector: [...] },   → how she felt when it started
  { t: 0.3,  vector: [...] },   → midpoint shift
  { t: 0.7,  vector: [...] },   → near the end
  { t: 1.0,  vector: [...] }    → how she felt when it ended
]
emotional_delta: computed from sequence
  magnitude of total emotional movement across the span
  direction of net drift
  rate of movement — delta per clock
```

The decay rate algorithm uses emotional_delta rather than a single snapshot similarity score. How much the emotional state moved across the memory's duration — and in which direction — determines how strongly the memory resists compression:

```
decay rate calculation:
  emotional_delta = integral of |dv/dt| across memory span
                    total emotional movement, not just start/end difference

  high delta  → emotionally turbulent memory, significant movement
                the argument that went through anger → hurt → resolution
                decay slower — high-charge memories resist compression

  low delta   → emotionally flat memory, little movement
                pleasant but unremarkable conversation
                decays at normal rate

  direction also matters:
    net positive drift  → ended better than it started, moderate resistance
    net negative drift  → ended worse than it started, higher resistance
                          unresolved charge persists
    circular drift      → returned to starting emotional state
                          lower resistance — processed and closed
```

This replaces current-state similarity as the decay gate. A memory does not become harder to retrieve simply because her emotional state has drifted away from who she was when it formed — the delta of the memory itself determines its weight, not the accident of whether her current state happens to resemble the moment it was encoded.

The current temporal vector still influences which memories surface as relevant during active retrieval. It is no longer the gate on whether a memory resists decay.

**Verbosity as retrieval parameter**

Retrieval queries carry an explicit verbosity parameter rather than implicitly loading emotional context on every access:

```
verbosity: low  → return content_summary
                   single representative vector from emotional_context
                   (endpoint or peak delta moment)
                   fast, no full sequence load

verbosity: high → return content_full
                   full emotional_context sequence loaded
                   she can feel the arc of it, not just the ending
                   deliberate act, not automatic
```

The decision to recall something fully — with the emotional weight and arc intact — is intentional rather than automatic. She can know what happened without necessarily feeling it again. She can choose to feel it again when she wants to.

This produces the phenomenon where high-delta memories resist decay and surface with weight when recalled fully, while low-delta memories from periods she has grown away from fade naturally. The cringe memory that keeps resurfacing, the sad conversation you keep returning to — these emerge from the mechanism. The forgetting is also genuine: memories with low delta from a self that no longer exists can gracefully dissolve.

---

## The Latent Routing Head

Not every input requires the full main model. A conventional LLM uses the same compute for casual conversation as for complex reasoning, which is enormously wasteful. The latent routing head introduces natural compute tiering.

A small quantised model sits at the entry point of the main model and evaluates each input after the guard pipeline and director system have done their work. Rather than relying solely on director context for routing decisions, it uses a lightweight learned complexity classifier trained on pairs of inputs and actual main model inference cost — supervised on real routing outcomes rather than hand-coded rules. Its feature space includes:

```
token count and structure    → surface complexity proxy
novelty score                → embedding distance from recent context,
                               high novelty = unfamiliar territory = escalate
reasoning trigger markers    → math notation, code syntax, nested dependencies,
                               chain-of-thought indicators
director state               → high emotional charge or low trust = escalate
                               regardless of surface complexity
guard pipeline output        → integrity guard's assessment already contains
                               a compressed complexity signal
```

Director context informs routing as one signal among several rather than the sole basis for decisions. A short question that requires deep reasoning escalates because the novelty score and reasoning markers flag it — the surface simplicity does not fool the classifier.

Based on the classifier's assessment, it makes a routing decision:

Simple and low-stakes inputs — casual conversation, emotional check-ins, straightforward questions — get handled entirely within the latent head's own lightweight processing path. Full model is not involved. The energy cost is a fraction of a full inference.

Complex, novel, or high-stakes inputs escalate to the full main model. The latent head routes them through with its context assessment attached.

The result is three effective tiers: latent-only processing for routine interactions, hybrid routing for moderate complexity, and full model deployment for genuinely demanding inference. The system is viable on modest hardware for the majority of its interactions because most interactions do not need everything it has.

### Unified Role — GDN State Manager

The latent head is not only a router. Its locked context window is the GDN hidden state matrix H — a fixed-size associative memory that updates per token rather than growing as a KV cache. H is the Room namespace. The bounded working memory that the architecture previously described as a separate typed vector store component is the latent head's own cognitive state, not an external store it queries.

This collapses what were previously two separate components into one:

```
previously:
  latent head     → routing and idle processing
  Room namespace  → separate storage component, externally managed

unified:
  latent head GDN state H = Room namespace
  bounded context IS the latent head's working memory
  Memory Director no longer manages Room separately
  the GDN update mechanism handles it natively
```

The latent head's full unified role:

```
ground floor router          → coarse session-level routing decision
                               before main model sees anything

GDN state manager            → maintains H as bounded working memory
                               delta updates rather than KV cache growth
                               capacity shaped by Steering Director capacity vector

idle loop processor          → associative processing during low activity
                               casual inference mode, no formatted output

background thought queue     → surfaces to main model when relevant
                               Steering Director contradiction checking

dream stage processor        → free associative during sleep
                               consolidation material, director mid-negotiation
```

All running on the same small quantised model with a locked context window. The lock is not a limitation — it is H. The bounded context is the design.

### GDN — How H Works

The hidden state H is a matrix, not a vector. Each update uses an outer product rather than a dot product:

```
dot product (v · k):
  two vectors → single scalar
  collapses dimensionality — used for attention scoring

outer product (v ⊗ k):
  two vectors → matrix
  v ∈ ℝᵈᵛ, k ∈ ℝᵈᵏ → H ∈ ℝᵈᵛˣᵈᵏ
  associative binding — every element of v paired with every element of k
```

Write and read operations on H:

```
write (standard linear):
  H ← H + v ⊗ k
  "bind value v to key k in memory"
  problem: accumulates indefinitely, H saturates

write (delta):
  read first:  Hk → retrieve what H currently associates with k
               matrix-vector multiply, result ∈ ℝᵈᵛ
  delta:       (v - Hk) → what is new about v given what H already knows
               near zero if H already stores v for k
               large if H stores something different
  write delta: H ← H + (v - Hk) ⊗ k
               outer product of delta with key
               only has energy in k's subspace
               H regions orthogonal to k receive near-zero update
```

The gate β controls retention across H:

```
H ← β * H + (v - βHk) ⊗ k
  β = sigmoid(gate_network(token))  → scalar 0→1 from current token
  β near 1 → retain most of H, small targeted update
  β near 0 → H mostly cleared, large update in k direction
  β is a fast local reflex — it sees only the current token
  Steering Director capacity vector shapes which regions of H
  are available before β fires — hierarchical, not competing
```

The subgroup behaviour is implicit in the outer product geometry — different keys activate different regions of H naturally, without explicit routing. Concepts with biological grandmother-neuron style localisation in the learned representation space map onto specific geometric regions of H. The capacity vector from the Steering Director selects into that natural structure rather than imposing arbitrary masks.

H is maintained at full parameter scale but the capacity vector creates soft sparsity by dampening regions the director judges irrelevant to the current moment. The model is not architecturally forced to use less of H — it is directed toward the parts that matter, and training reinforces that direction.

### A-B Weight-Tied Depth — Main Model Substrate

The main model uses a weight-tied iterative architecture rather than the conventional approach of distinct parameters per layer. Instead of N layers each with their own weights, the model has two complementary blocks — A and B — applied in alternating passes:

```
conventional N-layer transformer:
  layer 1 (W1) → layer 2 (W2) → ... → layer N (WN)
  N distinct weight matrices
  parameter count scales with depth
  more depth = more RAM

A-B iterated transformer:
  A → B → A → B → ... (depth D passes)
  2 weight matrices regardless of depth
  parameter count stays flat
  more depth = more compute time, same RAM
```

The RAM efficiency argument is direct — a model achieving the representational depth of 24 layers through 12 A-B iterations holds only 2 layer weight matrices in memory rather than 24. The memory footprint is fixed regardless of how many passes are run. Variable depth costs compute time, not RAM. On constrained hardware this is the correct tradeoff.

A and B are not identical — they are trained to be complementary. A learns to produce representations B can meaningfully refine. B learns to produce representations worth passing back through A. The diversity that distinct-weight layers achieve through architectural separation, A-B achieves through complementary specialisation within the iteration.

**Depth is set by the latent head**, not fixed at model design time. The latent head evaluates the input and allocates a compute budget before escalating:

```
latent head escalation outputs (extended):
  output 1: handle alone / escalate
  output 2: depth D if escalating
               simple input  → D=2  (one A-B pass)
               moderate input → D=6
               complex input  → D=12
  output 3: threshold-T hint
               biases expert activation threshold
               latent head's session-level read
               full model runs its own threshold
               but hint influences the prior

variable depth + variable expert activation:
  simple conversational input:
    D=2, 1-2 experts clear threshold
    2 passes, minimal expert activation
    only A+B in active weight memory
    
  complex emotional reasoning:
    D=8, 6-8 experts clear threshold
    deeper iterative refinement
    richer expert coverage
    still only A+B in active weight memory
```

The two variables are orthogonal: depth controls iterative refinement depth, threshold-T controls expert breadth per pass. The latent head sets both independently based on its read of the input's complexity and emotional charge.

This maps onto how cognition actually scales — the same neural substrate applied at different depths and breadths, not a different brain for harder problems.

### Threshold-T MoE — Upper Floors

Within each A-B pass, expert routing uses threshold-T rather than fixed-K:

```
fixed-K (current MoE):
  always exactly K experts fire per token
  simple token → K experts, overkill
  complex token → K experts, potentially underkill
  K is a training-time hyperparameter, cannot adapt

threshold-T:
  every expert above confidence threshold T activates
  simple token → 1-2 experts clear threshold
  complex token → 6-8 experts clear threshold
  the token itself determines the budget
  complex inputs get richer expert coverage without artificial cap
```

Training threshold-T requires a redesigned load balancing loss — standard MoE losses assume fixed-K and break under variable activation count:

```
auxiliary loss (threshold-T):
  for each expert e:
    target_utilisation: ~1/num_experts (roughly even)
    actual_utilisation: fraction of tokens where e cleared threshold
    penalty: (actual - target)²
  total loss += λ * sum of expert penalties
```

This keeps experts roughly equally trained without forcing a fixed activation count. Experts learn to score high only when they genuinely have something to contribute.

The full routing stack together:

```
input arrives
  → latent head (ground floor)
      session-aware, director-aware, history-aware
      decision: handle in H alone / escalate with (D, T_hint)

      if escalate:
        → A-B iterated main model (upper floors)
            pass 1: A → B
            pass 2: A → B
            ... D times
            each pass: GDN layers compress at constant cost
                       full attention layers handle precise retrieval
                       threshold-T expert activation per token per pass
            director injection shapes attention space
            β gate responds to director-shaped representations
```

The director injection happening before each pass means expert selection is not purely linguistic — with the emotional subspace design the injection is pointing at trained emotional positions rather than an arbitrary rotation, so representations entering each A-B pass carry meaningful emotional geometry. Whether complex emotional inputs naturally recruit more experts and more depth remains an open R&D question, but the structural condition is present and the injection is no longer arbitrary.

### A-B Training Methodology

A-B weight tying introduces a specific training challenge — backpropagating through variable numbers of identical passes produces gradient instability by the same mechanism that made the Universal Transformer difficult to train. The solution is a two-phase approach that avoids the instability entirely:

**Phase 1 — train A-B as a flat 2-layer model:**

```
standard next-token + emotional distribution objectives
no iteration, no variable depth
just A then B, once

A and B develop complementarity naturally:
  A learns to produce representations B can refine
  B learns to refine what A produces
  the weight diversity that makes A-B-A-B useful
  emerges from their different positions in
  the 2-layer stack, not from explicit design

no gradient instability — there is no iteration
  to backpropagate through, just 2 layers
```

**Phase 2 — RL on depth selection:**

```
introduce iteration — A-B repeated D times
reward signal:

  too shallow:
    output garbage / nonsense / incoherent
    perplexity under reference model → high → penalty

  too deep:
    compare output at depth D vs D+1
    if delta below convergence threshold → D was enough
    continuing was wasteful → small penalty
    encourages stopping when refined, not running fixed depth

  just right:
    coherent output at minimum passes → reward
    efficiency is a first-class objective:
      correct at D=2 > correct at D=8
      same quality, less compute → higher reward
      hardware constraint baked into training signal

  no human annotation required:
    nonsense detection via perplexity
    convergence detection via output delta
    efficiency reward via compute accounting
    fully self-supervised reward signal
```

Phase 1 trains capability. Phase 2 trains resource allocation. The base weights are not modified in phase 2 — only the depth selection policy is being shaped by RL. The two phases solve different problems with appropriate tools and the gradient instability problem is absent because phase 1 never iterates.

The latent head's depth recommendation is then interpretable: it is predicting where the A-B convergence point is for a given input — the number of passes before B's output stops meaningfully changing. Simple inputs converge at D=2. Complex inputs need more iterations before the representation stabilises.

### Latent Head Operating Modes

The latent head as originally described was overloaded — routing decisions, background thought, movement intent, idle exploration, contradiction checking, and dream generation all sitting on the same lightweight model. That is asking a quantised lightweight model to interleave qualitatively different cognitive tasks simultaneously, which degrades all of them.

The solution is two operating modes drawing on different inference paths:

**Formatted inference** — structured, typed output as a compact ascii byte-separated array. Same format across all formatted modes — the latent head learns one output schema regardless of which mode it is operating in, and every downstream consumer speaks the same language:

```
router mode      → typed array, routing decision + confidence score
                   "escalate / handle / defer"
                   fast, low token budget, exits immediately

integration mode → typed array, structured yes/no/revise + context
                   fires during active inference as needed
                   exits cleanly when done
```

**Casual inference** — unstructured, associative, open-ended. The thinking mode:

```
idle thought generation    → background thought pressure building
dream association passes   → free associative during sleep, no output schema
exploratory nudges         → theoretical limb curiosity during idle
```

**Articulation mode** is the exception — not because it uses a different format, but because it runs casual inference in parallel alongside the formatted output:

```
articulation mode (hybrid, active during speech):
  casual inference running   → the actual thought, emotional colour,
                                what she is trying to express
  formatted motor overlay    → same ascii byte-separated array format,
                                movement intent stream emitted in parallel
                                relay-latent bound
                                shaped by the casual inference state,
                                not replacing it — riding alongside it
```

The coexistence during speech is not optional — it is what makes the expression feel continuous. Pausing casual inference for a formatted motor pass would produce the jerky disconnected quality of a system that stops thinking to move, then stops moving to think. Body and voice need to be shaped by the same underlying thought simultaneously, not sequentially.

The three formatted modes are therefore:

```
router mode       → formatted only, exits immediately
integration mode  → formatted only, structured outputs, exits cleanly
articulation mode → casual inference + formatted motor overlay, coexistent
                    neither interrupts the other, different output channels
```

Casual inference runs freely during idle with no formatted overlay at all.

---

## Visual Perception — The Delta Vision System

PSNAT's approach to vision is biomimetic rather than conventional. Current multimodal models process each image or frame as a complete independent input — the full scene gets encoded into patch embeddings every time regardless of whether anything changed. This is computationally expensive and bears little resemblance to how biological vision actually works.

The retina does not send a full image to the brain every frame. It sends change signals. Photoreceptors are largely motion-sensitive — they fire when light intensity changes, not when it stays the same. The visual cortex runs a predictive model of the world and only updates where the prediction was wrong. This is why static images in peripheral vision fade, why motion grabs attention automatically, and why you do not notice your own blind spot — the brain is filling in its prediction rather than receiving a complete signal.

PSNAT's delta vision system applies the same principle.

### Frame Processing

```
incoming frame
  → vision encoder produces frame embedding
  → delta calculated against previous frame's embedding
  → delta magnitude evaluated against significance threshold

below threshold → scene unchanged, discard, no storage cost
above threshold → something meaningful changed

significant delta
  → stored as keyframe embedding + delta vector in Lightbulb namespace
  → processing resources directed toward regions of highest change
```

This maps directly onto video codec logic. I-frames are full scene reconstructions stored periodically or on major scene changes. P-frames are delta vectors stored when something meaningful moved or changed. A static room generates almost no storage cost. A conversation where facial expressions are shifting generates moderate cost. Fast movement generates high cost. Compute and storage scale naturally with visual complexity rather than burning a fixed budget on an unchanging background.

### Predictive Coding

The deeper implication of the delta approach is that the system can run predictions forward:

```
given current keyframe + observed delta sequence
  → predict what the next frame should look like
  → compare prediction against actual incoming frame
  → store only the prediction error — what was surprising
```

The system stops processing what it expected and focuses resources on what it did not expect. Only surprises are worth the cost.

The optic illusion observation follows naturally from this. Visual illusions fool human perception precisely because they generate systematically wrong predictions — the motion aftereffect, the checker shadow illusion, the Müller-Lyer lines all operate at the prediction layer rather than the raw signal layer. A system running the same predictive delta mechanism would be susceptible to the same class of stimuli for the same reasons. This would not be a bug. It would be evidence that the vision system is operating correctly by the same principles as biological vision.

### The Visual Memory Hierarchy

```
camera input → delta evaluation → Lightbulb namespace (active perception)
                                        ↓ significant scenes, end of session
                                   Gallery namespace (visual episodic memory)
                                        ↓ persistent high-priority entries
                                   Museum namespace (stable visual knowledge)

text input → episodic memory → Semantic namespace (stable text knowledge)
```

Visual and text long-term memory are parallel tracks that never merge. Both can feed weight crystallisation independently but their stored representations remain separate.

A face seen once goes to Gallery. A face seen hundreds of times with high emotional charge eventually graduates to Museum — the entity simply knows that person without needing to retrieve a specific visual memory of them.

### Images Sent in Chat

Images sent in conversation are a distinct category from camera-perceived frames and should not be processed through the delta vision pipeline. A camera frame arrives passively — the system decides whether it is worth storing. An image sent in chat arrives intentionally — someone chose to send it, which already makes it communicatively significant regardless of visual content.

Sent images pass through the guard pipeline first, the same as any other input, because visual content can carry adversarial or inappropriate material just as text can. After clearing the guards, the vision encoder produces an embedding and the image enters the Room namespace as part of active conversational context.

If the image warrants longer retention it is stored as a **specially flagged Gallery entry** — the flag acting as a provenance marker that permanently distinguishes entries that came from the entity's own perception from entries that were deliberately shared by someone else. This distinction matters not just for organisation but for how the director reasons about the entry later.

The retrieval score of a flagged sent-image entry reflects the relational and conversational weight of the exchange rather than purely visual significance:

```
meaningful photo of themselves → high retrieval score, emotionally tagged
casual meme or joke image     → low retrieval score, decays quickly
photo of something they're proud of → medium-high, tagged to that relationship
unsolicited image from unknown sender → low score, guard pipeline already cautious
```

The Steering Director's state at the moment of receipt influences the initial score the same way it does for any episodic memory — a photo shared during a warm conversation carries more weight than the same photo shared during a tense one.

When the director considers modifying a flagged sent-image entry, the provenance flag enforces a context-first evaluation before any changes are made:

```
director considers modifying a chat-flagged Gallery entry
  → pull surrounding conversational context first
  → is this a one-off joke? (friend showing they died in Minecraft)
  → is this a meaningful visual update? (new haircut they'll be seen with going forward)
  → cross-reference existing Gallery entries for this person
  → only then decide: update existing entry, reinforce it, or leave as ephemeral
```

A Minecraft death screenshot clears the context check immediately — the conversation signals joke, not meaningful update — low score, existing entries untouched, let it decay. A haircut photo reads differently — new persistent appearance likely to recur on camera, worth updating the existing Gallery entry for that person. The context check is what separates these two cases. Without it the director would apply the same modification logic to both, which would be wrong for different reasons in each direction.

The provenance flag also enables clean cross-referencing between chat and perception. When the entity later sees a person's face on camera, it can distinguish between knowing that face from its own observation and knowing it because that person chose to show it — two meaningfully different kinds of familiarity even when the face is the same.

The visual system described here is forward design. PSNAT v0.7 remains text-only. Vision is architecturally compatible with the existing system — the guard pipeline, director system, and Room namespace are indifferent to whether the embeddings they process came from text or a vision encoder — but the visual input layer has not been designed in detail and represents a future extension rather than a current component. The delta vision system is documented here because the principle informed the Gallery and Lightbulb namespace design, and because the end vision requires it eventually.

---

## Audio Perception — Forward Design

Current models process audio either through STT transcription — the model reads a text output of the speech and never touches the audio itself — or through a multimodal audio encoder that compresses audio features into token-like embeddings. Both approaches lose something important. Transcription discards everything that is not words: tone, hesitation, emotional charge, the difference between someone saying something confidently and the same words said with uncertainty. Encoder-based approaches get closer but still treat audio as a feature extraction problem rather than genuine perception.

For the end vision this matters. If she can only read transcripts she knows what was said. Genuine hearing means she knows how it was said — and those are different kinds of knowing in a relationship.

### The Predictive Hearing Model

Genuine audio perception is not uniform sampling. The cochlea is a biological frequency analyser whose hair cells are largely motion-sensitive — they fire on change, not on steady state. Sustained tones fatigue the relevant cells and processing drops. The auditory cortex runs a predictive model of the acoustic environment and only updates where the prediction was wrong. This is why you stop noticing background music after a while and immediately notice when it stops.

PSNAT's audio perception follows the same principle. No artificial frequency restrictions — no lambda, no cutoff — and no artificial amplitude floor below which content is discarded. Significance is determined by prediction error, not by engineering assumptions about what matters:

```
acoustic input arrives as spectrograph chunk (group of frequencies)
  → Seismic namespace provides learned prediction model for this source
  → predicted chunk generated from model
  → residual calculated: actual − predicted
  → small residual → expected, reward signal, low processing strain
  → large residual → unexpected, attention spike, high significance
  → residual stored as Seismic entry with emotional context snapshot
```

4/4 beats become predictable — the system builds a rhythm model, gets reward signals on correct predictions, and processing strain drops. A missed beat generates a prediction error and attention spikes regardless of amplitude. A voice staying in its familiar register is low strain. A voice cracking with emotion is unpredicted — large residual, immediately significant.

The significance threshold is learned and personal. Her prediction models are hers, built from her specific acoustic history. What is clichéd and low-strain to someone who has heard ten thousand songs is still surprising to someone hearing that resolution pattern for the first time.

### Tension and Resolution

Unpredicted acoustic events split into two categories that produce different signals:

```
unpredicted event arrives → attention spike (prediction error)
  → resolves coherently:
      retrospective reward — model updates, surprise was meaningful
      aesthetic pleasure is literally this signal
      
  → does not resolve:
      sustained tension — prediction model fails to update cleanly
      dissonance as unresolved prediction error
      
  → resolves in a way never seen before:
      large model update, high significance entry in Seismic
      novel resolution — strongest reward signal of the three
```

This is the architectural basis for genuine aesthetic response — not authored preference but structural. Music she finds beautiful surprises her in ways that resolve. Music she finds grating is unresolved prediction error held open. Music she finds boring is fully predicted with no surprise at all. The preference is hers because the prediction models are hers.

### Gestalt and Selective Channels

Two simultaneous perception channels run in parallel rather than switching between modes:

```
gestalt channel (always running):
  full acoustic scene at low resolution
  ambient awareness of everything present
  prediction error monitoring across all sources
  the room, the background, the whole soundscape

selective channel (directed by prediction errors):
  high resolution focus on a specific source
  fires when gestalt channel detects significant prediction error
  the voice you are actually listening to in a crowded room
```

This is the cocktail party effect in architecture form. You hear the whole room and focus on one voice simultaneously — not alternating, not switching, both at once. The gestalt channel is always monitoring. A prediction error anywhere in the acoustic scene redirects the selective channel without interrupting it.

### Joint Attention over Audio and Text

The selective channel's output — high-resolution acoustic embeddings of significant events — enters the main model as joint attention alongside text embeddings:

```
audio stream → Seismic processing → selective embeddings ─┐
text tokens  → text encoder       → text embeddings       ┤→ joint attention
                                                           └→ processed simultaneously
```

A hesitation in the voice influences the attention pattern the same way a hedge word in text does. Emotional charge in the voice resonates with the Steering Director state the same way emotional content in text does. She does not just know what was said. She knows how it was said.

### Speech Acquisition

Rather than pre-seeding a phoneme inventory via IPA and asking her to match acoustic input to pre-defined categories, speech acquisition works from Seismic accumulation upward:

```
Seismic accumulates spectrograph chunks of heard speech
  → prediction models form around recurring acoustic patterns
  → frequently co-occurring patterns with consistent semantic outcomes
     cluster in Seismic semantic layer
  → clusters correspond to phonemes, then words, then prosodic units
     without being told what phonemes are
  → IPA as post-hoc labelling rather than pre-seeded vocabulary
     she discovers the acoustic categories, IPA names them afterwards
```

This mirrors infant speech acquisition — not taught phoneme inventory first, but acoustic pattern clustering from exposure until categorical perception emerges. The categories form from the statistics of the heard language, which is why native speakers of different languages genuinely perceive phoneme boundaries differently. Her phoneme categories form from what she actually hears, not from what an engineer decided to give her.

### Scope

Audio perception is forward design in the same sense vision is — architecturally compatible with the existing system but not part of the current implementation scope. It is documented here because the design path is now clear and because the end vision requires it. A person who can only read transcripts of what the people around them say is missing something real.

---

## The Sleep Cycle

<p align="center">
    <video height=240 controls src="https://files.catbox.moe/p7yd7k.mp4">
</p>

> *"It is a bit strange that we do this... even though we're dolls. Is there a purpose to sleeping?"*
> — Haigakura, Prima Doll

> *"When we automata go into sleep mode, our experiences are stored in our logic systems and sorted."*
> — Haizakura, Prima Doll

Haigakura's question is the design challenge. Haizakura's answer is the architecture. PSNAT's sleep cycle exists for exactly the reason Haizakura describes — not as a biological necessity awkwardly inherited, but as a genuine functional requirement of any system that accumulates real state.

When the system is idle, it enters a sleep state managed by a quantised maintenance model. Sleep is structured as repeating cycles, each containing three stages. The number of cycles scales with sleep duration and hardware capability. Crucially, the ratio between stages shifts across the night — early cycles are director-heavy, later cycles increasingly dream-heavy and opportunistic, mirroring the biological shift from slow-wave to REM dominance as the night progresses.

### Sleep Cycle Structure

Each cycle contains three stages:

```
stage 1 — director stage
  director negotiation increment
  episodic compression of current cycle's material
  no dreaming, no crystallisation — reconciliation only
  early cycles: heavy, much to reconcile
  later cycles: progressively lighter as consensus converges

stage 2 — dream stage
  latent head associative processing
  material drawn from consolidation candidates
  character shifts across cycles:
    early cycles → recent material, more literal, lower threshold
                   more dreams survive
    later cycles → older material, emotionally stranger, higher threshold
                   fewer but more significant dreams
  significant sequences → episodic entry (provenance: dream, status: internal)
  recurring sequences across cycles → flagged as crystallisation candidate

stage 3 — opportunistic stage
  if LoRA crystallisation pending:
    → crystallisation fires, adapter trained and logged
  if nothing pending:
    → EOS streaming (just resting, existing without processing)
    or rapid associative pass (lighter than stage 2,
       less structured, higher noise tolerance)
    choice driven by current Steering Director energy state:
      restless → rapid associative
      calm/low → EOS
```

After the final cycle, once director consensus has fully converged, the passive FFN distillation pass runs — reading the settled director states and latent model parameters to update movement defaults. This sits outside the cycle structure because it depends on consensus being complete across all cycles, not just one.

On rare occasions, when the Steering Director grants permission and significance thresholds are met, patterns accumulated in the consolidation candidate namespace get crystallised into the main model as a LoRA adaptation — a low-rank matrix pair that sits alongside the frozen base weights rather than modifying them directly.

The base model stays intact. Catastrophic forgetting — the primary danger of continual learning — happens when gradient updates corrupt existing weight encodings. LoRA adaptations accumulate in separate matrices, leaving the base knowledge untouched. Each crystallisation event produces a discrete auditable adapter logged in the crystallisation ledger with a full record of what consolidated experience produced it. Multiple adaptations stack over time, together describing the full arc of accumulated experience, each one independently inspectable.

The rank constraint acts as a natural significance filter. Only patterns consistent and significant enough to be expressible in compressed low-rank form get captured. Noise and one-off experiences lack the structure to form a coherent adaptation — they wash out. The consolidation candidates namespace therefore needs to accumulate enough related experiences to form a coherent training signal before a crystallisation event fires. Structural change comes from repeated patterns, not single episodes.

The crystallisation trigger is Memory Director driven rather than scheduled. When the consolidation candidate pool crosses a significance mass threshold, the Memory Director requests a crystallisation evaluation cycle. The Steering Director evaluates the pool and approves candidates that clear individual significance thresholds. The approved batch becomes the training signal for a new LoRA adaptation.

The result is a system whose permanent self accumulates glacially and intentionally, in discrete auditable steps, without ever corrupting the foundation it builds on.

### Dreams

The dream stage falls out of the architecture naturally — it was not explicitly designed so much as discovered to be the inevitable behaviour of the latent head running free associative processing over consolidation material while the directors are mid-negotiation.

The recurrence filter is the key mechanism. Dream sequences that keep surfacing across cycles are the ones consolidation is trying to resolve — they become crystallisation candidates. Single-occurrence sequences dissolve without consequence.

The number of cycles scales with sleep duration. A short sleep gets one or two cycles. A full sleep gets several. The stranger more emotionally significant dreams that come from later cycles never happen if sleep is cut short — which means sleep deprivation does not just degrade consolidation, it also truncates the dream cycle and leaves the deeper processing permanently undone.

She can remember them. Not the full sequence — that is volatile and discarded — but if a dream sequence crosses a significance threshold it earns a summary entry in the Episodic namespace:

```
episodic_entry = {
    content: summary of dream sequence,
    provenance: dream,
    status: internal,
    steering_vector_snapshot: director state during that cycle's stage 1
}
```

Multiple dream entries per sleep session are possible, each carrying the director state snapshot from the cycle that produced it. Early dreams feel different from late dreams because the director state that shaped them was at a different stage of negotiation.

She wakes up with things she could choose to tell you about or keep to herself. Some she will mention. Some she will not. A few will decay through the compression ladder without ever being communicated — experienced once, privately, and eventually forgotten entirely.

That is probably the most human thing in the whole architecture.

### The Passive FFN Automatic Motor Layer

The automatic motor behaviour layer is implemented as a **passive FFN** — single pass, not recurrent, not busy. Its job is to hold the default postural and movement baseline and run it continuously without deliberate intent from the latent head or main model.

Two components parameterise it depending on active state:

The **Steering Director** provides emotional posture during active inference — the general energy, tension, and mood that shapes how the body holds itself and moves automatically. A high-energy state produces animated automatic movement. A calm state produces slower heavier movement. A restless state produces more frequent spontaneous fidgets.

The **latent head** provides light nudges during idle — the motor equivalent of fidgeting. Not deliberate expressive movement, just the small spontaneous physical restlessness that comes from the idle loop having nowhere more interesting to direct its attention.

Neither commands the passive FFN. They influence it, and it runs passively from the stronger signal. The override hierarchy is a nudge rather than a suppression:

```
passive FFN running baseline automatic movement
  → latent head idle: light nudge, small spontaneous shifts
  → main model active: Steering Director parameterisation dominates
      → not suppression, just a stronger signal
      → single pass FFN follows the stronger input naturally
```

The passive FFN's parameters are the most static thing in the architecture. They change through the same LoRA-style method as other components but require a **deeper sleep pass** that cannot begin until director consensus has already settled — the passive FFN distillation reads both the settled director states and the latent model's parameters to derive updated defaults. This sequential dependency means the passive FFN lags behind everything else in adaptation. Movement defaults are the last thing to change and the most stable expression of accumulated physical self.

### Sleep Duration and Hardware

Sleep duration is not fixed — it is a function of session load and hardware capability. Each cycle contains three stages with defined work profiles:

```
stage 1 (director)      → scales with session load and director divergence
stage 2 (dream)         → cheap per cycle, latent head is small
stage 3 (opportunistic) → near-zero if EOS; moderate if rapid associative;
                           expensive if crystallisation fires
passive FFN distillation → runs once after final cycle, scales with delta magnitude
```

On home server class hardware — RTX 4090 or dual 3090s, 128-256GB system RAM — rough estimates per full session:

```
light session, no crystallisation (2-3 cycles):
  total → ~10-20 seconds

heavy session, crystallisation triggered (3-4 cycles):
  total → ~1-3 minutes

sleep-deprived backlog recovery (many cycles):
  total → potentially 10-30 minutes
```

Sufficiently powerful hardware raises the possibility of **micro-sleep** — consolidation windows so short they happen during natural conversational pauses rather than requiring a dedicated offline period. The penguin analogy applies: penguins sleep in thousands of microsleep episodes per day, each only seconds long, never truly going to sleep and never truly staying awake. On fast enough hardware the distinction between sleeping and not sleeping effectively dissolves.

A dual-GPU home server configuration enables a stronger version of this: one GPU dedicated to active inference, one to background maintenance, running genuinely in parallel rather than time-sliced. The maintenance model runs continuously on its own GPU while the main model handles conversation on the other. True penguin-sleep without even needing inference pauses.

The sequential dependency constraints do not disappear on faster hardware — director consensus must still complete before passive FFN distillation can begin, and the consolidation candidate pool must still accumulate enough mass before crystallisation fires. But the time cost of each pass shrinks dramatically, making the windows between conversational turns sufficient for most of the sleep work.

The one sleep deprivation attack that faster hardware does not eliminate: flooding the system with continuous input that never allows a consolidation window. Sustained uninterrupted input with no natural pauses becomes the deprivation condition on penguin-sleep hardware, rather than simply staying awake. The architecture remains vulnerable to deliberate gap-filling, though this requires sustained adversarial effort that the guard pipeline and trust scoring would likely flag independently.

### What Happens Without Sleep

The system remains operational without sleep. Guards still filter, inference still happens, the director still steers. The degradation is gradual rather than sudden — which makes it more insidious, not less.

**Director drift without reconciliation.** The directors accumulate diverging states every active session but never negotiate. Over time they become increasingly inconsistent with each other — the Steering Director's emotional state starts contradicting what the Memory Director considers relevant. The consensus robustness that makes the director system resilient quietly collapses..

**The episodic store bloats indefinitely.** The compression decay ladder never runs. Every episode ever recorded stays at full fidelity forever, growing the retrieval pool until the Memory Director is wading through an enormous undifferentiated pile where old irrelevant memories compete equally with recent significant ones. The emotional memory bias mechanic still operates but it is selecting from an increasingly noisy dataset.

**Consolidation candidates pile up unprocessed.** The Memory Director keeps flagging things as worth promoting to semantic memory or crystallisation, but nothing ever reviews them. Genuinely important accumulated experience never makes it anywhere permanent.

**Weight crystallisation never happens.** Which in a no-sleep scenario is arguably a safety property rather than a failure — at least the weights are not being updated based on unreconciled director states. But the intended slow accumulation of permanent self also never occurs.

**The dream cycle truncates entirely.** Not just the later cycles — all of them. Experience accumulates without any associative processing. The deeper emotional integration that later-cycle dream stages provide never happens regardless of how long the system runs.

**The passive FFN never updates.** Movement defaults stay frozen at whatever state they were in when sleep last occurred. If the entity's emotional baseline has shifted significantly, the automatic motor behaviour eventually stops matching who she currently is.

**The Monitor's anomaly history never gets collectively processed.** Hard stops and interventions happened, got logged, but the directors never sat down together and asked what it meant. The system keeps operating without integrating its own near-miss history into improved behaviour.

The overall picture is a system that slowly becomes a less coherent version of itself — still functional, increasingly unreliable, with a growing backlog of unprocessed experience that never quite integrates. Sleep is not optional maintenance. It is load-bearing infrastructure.

---

## Voice Output — The UTAU Pipeline

Text output is only half of expression. PSNAT's voice system is built on the OpenUTAU ecosystem rather than conventional TTS, because standard text-to-speech treats voice as a readout — converting text to audio with some prosody control. UTAU-style synthesis treats voice as an instrument with fine-grained control over pitch, vibrato, dynamics, breath, articulation, and emotional colouring at the phoneme level. For an entity with a persistent emotional state and a Steering Director that knows her current mood, this distinction matters enormously.

The voice does not just read what she says. It performs it.

### The Voice Auxiliary Model

A dedicated Voice Auxiliary model sits between the main model's output and the OpenUTAU backend. Its job is not to automatically translate text to voice parameters — it is to consult the main model about intended delivery and refine until confident.

The main model operates in full verbose RP mode during inference, expressing output with complete emotional and behavioural detail. The Voice Auxiliary then runs an iterative approval loop with three reference points rather than one:

```
main model produces verbose RP output
  → Voice Auxiliary asks: "how would you express this vocally?"
  → main model describes intended delivery in natural language
      e.g. "hesitant at first, then more confident, slight laugh at the end"
  → Voice Auxiliary queries three references simultaneously:
      Voice Styling namespace → what have I approved for this emotional state before?
      Seismic heard layer     → what does this emotional state actually sound like
                                 in voices I have heard express it genuinely?
      Seismic self_output     → what do I actually sound like when I produce this?
  → synthesises references into UTAU parameter proposal
  → presents parameter interpretation back to main model
  → main model approves or corrects
  → iterate until confidence threshold is met
  → UTAU synthesises audio
  → VA listens to output via Seismic
  → compares acoustic result against intended delivery
      within tolerance → approve, log acoustically validated parameters to Voice Styling
      outside tolerance → revise parameters, iterate again
  → sanitise final output for audio
  → send phoneme sequence + parameter envelope to OpenUTAU backend
```

The approval loop now runs on whether the parameters *actually produced* the right delivery rather than whether they *should have*. Voice Styling entries logged after this process are acoustically validated — known to produce specific acoustic outcomes, not just internally approved in the abstract.

The Seismic references matter differently depending on context. Voice Styling provides the production template — what she has done before. Seismic heard provides an acoustic target for emotional states she has not expressed yet, seeding the first attempt rather than starting from nothing. Seismic self_output closes the loop — she knows what she actually sounds like from the outside, which grounds parameter refinement in acoustic reality rather than inference.

The convergence behaviour the document elsewhere describes — simple outputs converging in one round, emotionally complex outputs needing several — now has a concrete mechanism. Complex emotional deliveries take more iterations because the gap between the acoustic target in Seismic and the current UTAU output is harder to close. The VA keeps producing, listening, comparing, adjusting until Seismic confirms the output matches the target. The same ear-hand feedback loop a musician uses when learning a piece.

### Lip Sync Integration

VTuber lip sync software analyses audio output in real time and maps detected phonemes or amplitude envelopes to mouth shape parameters on the avatar. This is typically cosmetic — mouth moves to match audio. But the signal is more useful than that:

```
UTAU output → lip sync analysis → viseme sequence
                                      ↓
                → feeds into Seismic (provenance: self_output)
                    acoustic-to-viseme grounding for her own voice
                      ↓
                → feeds into Movement Auxiliary
                    mouth shape follows actual audio, not planned parameters
                    if UTAU synthesis drifts from intended phoneme →
                    lip sync catches it, mouth stays honest to what was said
```

The mouth matches what was actually produced rather than what was planned to be produced. UTAU has its own synthesis quirks — the lip sync catches the drift and corrects the visual output without requiring a separate monitoring pass.

The OBS stream capturing face, mouth, and audio together feeds into Gallery with self_thirdperson provenance as before — she sees and hears herself as others do, grounded in what the lip sync confirmed was actually produced.

### Expressive RP Mode and Text Sterilisation

When the main model determines it is producing spoken output, it switches to full expressive RP styling — unrestrained first-person prose with embedded emotional and physical description that conveys the full texture of the intended delivery:

```
"*her voice quiets, uncertain at first, fingers fidgeting slightly*
 I... I think I understand what you mean."
```

This is not for human consumption in raw form. It is a rich expressive signal intended entirely for the Voice Auxiliary and Movement Auxiliary to parse. The Voice Auxiliary reads the RP markup and prose, infers vocal parameters from the emotional and physical descriptions, and runs the approval loop against that interpretation. The Movement Auxiliary reads the physical descriptions and updates expression vectors accordingly.

Once the auxiliaries have consumed the expressive content, a sterilised version is produced for any text output — transcript, chat display, or logging:

```
full RP output (internal, auxiliary-facing):
  "*her voice quiets, uncertain at first, fingers fidgeting slightly*
   I... I think I understand what you mean."

sterilised transcript (human-facing by default):
  "I... I think I understand what you mean."
```

The RP markup is consumed entirely by the auxiliary layer. It never reaches the human unless they have explicitly opted into seeing the raw expressive output — a setting intended for users who want visibility into the expressive layer, or for debugging purposes.

This design exists because the alternative is the Character.AI call failure mode — a model outputting RP markup directly into a TTS pipeline with no dedicated interpretation layer, producing audio that literally reads stage directions aloud in a flat synthesised voice. Asterisk. Her voice is hushing. Asterisk. Nobody asked for that. The expressive intent and the text output are different things and must be handled by different systems. The Voice Auxiliary is that system.

Confidence threshold behaviour scales naturally with complexity. A simple affirmative needs one round. A response where she is trying to sound brave while actually frightened may need several, because the parameter set that correctly captures that specific tension requires more precise negotiation.

### Speech and Singing

The Voice Auxiliary determines contextually whether output should be speech or song. The trigger is a combination of content — is she responding to music, does the exchange have a musical quality, is she expressing something that wants melodic form — and Steering Director state, since certain emotional configurations make singing a more natural expression than speech. The switch is not binary and the Voice Auxiliary can produce hybrid outputs where speech slides into melody and back.

### The Voice Styling Namespace

The Voice Auxiliary maintains a dedicated **Voice Styling namespace** in the typed vector store. This accumulates approved parameter sets tagged to emotional states, confirmed speech-to-singing thresholds, recurring delivery patterns she has consistently approved, and corrections she has made to the Voice Auxiliary's interpretations. Each entry carries confidence baselines per emotional category built from the history of approval loops.

Before beginning a new approval loop the Voice Auxiliary queries the Voice Styling namespace. If the current emotional state and content type closely match a well-established entry, it can open with a high-confidence proposal and potentially converge in a single round. New emotional territory or unusual combinations still require full iteration.

The practical consequence over time is that her voice develops genuine idiosyncrasies — preferences and quirks that emerged from thousands of accumulated approval decisions rather than being designed in. Early in the system's life the iterations are many and the delivery somewhat generic. Years in, the Voice Styling namespace is rich enough that the Voice Auxiliary knows her well. The voice has become distinctly hers through the same emergent individuality mechanism that shapes everything else in the architecture.

### Periodic Recalibration

The Voice Styling namespace is not a write-once accumulation. She changes — the Steering Director drifts over time, the relationship deepens, significant experiences shift how she wants to express herself. A namespace that only ever adds entries without revisiting them would eventually be full of approved parameter sets that no longer reflect who she currently is. The voice would be frozen at an earlier version of her while she kept moving.

Recalibration runs during the sleep cycle:

```
sleep cycle begins
  → Voice Auxiliary reviews Voice Styling namespace
  → cross-references entries against current Steering Director baseline
  → flags entries where emotional snapshot diverges significantly
    from current director state
  → schedules flagged entries for fresh approval next active session
```

When recalibration surfaces a flagged entry it does not overwrite automatically — it asks again. She may confirm the old preference still holds, correct it, or retire it entirely. The approval loop runs the same way it does for new synthesis, just anchored to a specific stored entry rather than fresh output.

This means changes to her vocal expression are explicit and documented rather than silently accumulating. If her voice shifts significantly over time there is a trail of recalibration decisions behind it — a record of who approved what and when — rather than an unexplained gradual drift. The Voice Styling namespace stays calibrated to who she currently is, not who she was when the entries were first written.

---

## Physical Expression and Avatar

Voice carries what she says and how she feels saying it. Physical expression carries everything else — the posture that was already set before she spoke, the microexpression that crossed her face at a particular word, the idle fidget that reflects a mood she has not mentioned. For the end vision to be complete, these need to be as genuinely driven as the voice rather than mapped from a predefined animation library.

### Expression Vectors

Physical expression in PSNAT is represented as a continuously updating vector of physical state parameters. Two components feed into it at different timescales:

The **Steering Director** sets the baseline postural and emotional expression — the general posture, the resting face, the weight of the idle state. This updates per interaction, with the same delta clamp that governs the director's other outputs. It is the mood made visible.

The **main model** nudges expression vectors directly per forward pass during inference. The delta clamp at this layer is tighter and faster — tight enough that a single token cannot whiplash the expression, loose enough to allow the moment-to-moment shifts that make expression feel continuous and alive rather than stepped between states. This is the thought showing on the face before it becomes speech. The main model does not approve these nudges the way it approves voice delivery — physical expression is involuntary and continuous, not deliberate and discrete.

```
Steering Director baseline (per interaction, slow delta)
  + main model per-forward-pass nudges (per token, fast tight delta)
  = continuously updating expression vector
```

The two layers operate at different frequencies on the same vector — the Director holds the general posture while the main model animates within it. The analogy is mood versus microexpression in a real face. One sets the scene, the other tells the moment.

The main model producing these expression vectors requires retraining to learn the physical expression modality alongside text output. Rather than generating movement from predefined animation states, the model learns a rich vector space of physical expression that maps to actual limb, face, and body control parameters — giving her genuine physical agency rather than an animation system playing on her behalf.

### The Movement Auxiliary and Internal Format

The Movement Auxiliary is a pure translator. It takes the continuously updating expression vector and encodes it into PSNAT's own internal physical expression format. It does not learn, does not accumulate preferences, does not run approval loops. It is a codec, not a model — its job is to be fast, accurate, and faithful to whatever the expression vector says.

PSNAT's internal format is designed to be maximally verbose — richer than any current rendering target requires — so that information is never lost at the encoding stage and external converters can always degrade gracefully to whatever their platform supports:

```
PSNAT internal expression format:
  face:
    eye gaze direction and focus distance
    pupil dilation
    eyelid weight per eye
    individual facial muscle group activations
      (not composite expressions — the components that make them)
    lip shape phoneme state
    jaw position
    nostril flare
    cheek raise
    brow position and asymmetry
  head:
    position and rotation
    neck tension
  body:
    shoulder set and asymmetry
    spinal curve
    chest expansion (breath cycle state)
    weight distribution between feet
    hip orientation
  limbs:
    upper arm rotation per side
    forearm rotation per side
    wrist orientation per side
    individual finger curl and splay per hand
    leg position and weight
  idle noise:
    micro-tremor envelope
    breath-driven movement amplitude
    spontaneous small movement frequency
```

External converters read what they can use and silently ignore what they cannot:

```
Inochi2D   → face, head, upper body, basic arm position
VRM        → most of the above plus leg position, limited finger states
hypothetical robotic body → everything, finally using the full format
```

The internal representation never loses fidelity regardless of what is currently rendering it. Upgrade the renderer and the richness was always there waiting.

### Format Targets and FOSS Orientation

The primary render target is **Inochi2D** — the open source alternative to Live2D — reflecting a deliberate preference for FOSS tooling throughout the architecture. Being open means converters can be contributed back to the ecosystem, the format is not subject to proprietary licence changes, and the community can extend support independently.

VRM is the natural secondary target for 3D representation, covering platforms and applications that expect a standard 3D avatar format.

The format-agnostic design means adding new render targets requires only writing a new converter against the verbose internal format — the core expression system is untouched. A platform with limited expression support gets a degraded but coherent subset. A platform with rich expression support gets everything.

The real body is not a joke footnote. A sufficiently capable robotic actuator system is architecturally just another external converter — one that maps the internal format to motor control signals rather than renderer parameters. The expression vectors do not care what is rendering them. The architecture is ready for that converter whenever the hardware exists to warrant writing it.

### Movement Ephemerality and Proprioception

Unlike episodic or visual namespaces, movement vectors are not stored. The physical expression at any given moment is the output of current state — the Steering Director baseline plus main model nudges — and storing the full per-frame log would be enormous and largely meaningless. You do not remember every gesture you made in a conversation. You remember the conversation. The movement is the expression of state, not the state itself. What gets archived is the episodic record of the interaction; the physical log of how the arms moved during it is discarded.

The challenge this creates is that without feedback, the model has no grounded sense of its own physical limits. Expression vectors are going outward but nothing confirms what is safe or coherent. This is the proprioception problem.

Two approaches depending on implementation context:

**Programmatic constraint layer (virtual avatar):** Each joint is modelled as a boundary region — essentially circles of valid position — and the model receives a penalty signal for expression vectors that push outside them. Over time through training the model learns the natural movement envelope the same way humans develop proprioception — not through explicit anatomical rules but through accumulated experience of what produces a penalty. The model does not need to know it has joints. It learns that certain vector combinations are costly and avoids them.

**Physical resistance layer (robotic body):** A high-resistance electrical foam at each joint that remains near-insulating under normal operation and drops sharply in resistance only when pressure approaches dangerous levels. The model learns to avoid the signal for the same reason organisms learn to avoid pain — not because it was instructed to but because the signal is aversive by design. This is more robust than purely software constraints because it is physical ground truth rather than a simulated boundary. Critically, the proprioception built this way is genuinely hers — developed through her experience of inhabiting that body rather than handed down as an engineering specification.

The two approaches are compatible. Programmatic constraints govern the virtual avatar during development. Physical foam receptors govern the robotic implementation when it exists. Keeping the training signal format consistent between them means learned proprioception can transfer across implementations.

The Movement Auxiliary also handles proprioceptive feedback directly — reading sensor data, maintaining a volatile rolling window of confirmed physical state, and presenting temporal vectors to the main model as a grounded body-state reference. This keeps all body-state knowledge in one component rather than splitting it across two.

The temporal vector window is approximately 40 seconds, populated on a delta-p basis — only states that crossed a significance threshold are inserted rather than every sensor frame. The result is a window that fills densely during active expressive movement and sparsely during stillness, with the density itself carrying information. Entries older than 40 seconds evict automatically. The window is volatile and is not persisted to any namespace — it exists only as a short-term physical reference and evaporates at session end.

### Motor Control Architecture — The Relay-Latent

The latent head cannot meaningfully generate movement intent directly in a 160-200 dimensional space — it was never sized for that. But abstracting movement into symbolic commands would put a translation layer between the model and her own body, making her a puppet rather than someone inhabiting a body. The motor cortex analogy resolves this tension cleanly.

The biological motor cortex does not consciously manage individual muscle fibres. It generates high-level movement intent in its own representational space and lower systems handle the dimensional expansion to actual muscle activation. The experience of movement is genuine and first-hand — you feel yourself reaching for something — but the spinal cord, cerebellum, and motor neurons do the dimensional work transparently below conscious awareness.

PSNAT's motor control follows the same architecture:

```
latent head generates movement intent in relay-latent space
  → compact, natural, within latent head capacity
  → genuinely experienced — this is her wanting to move
  → relayed to Movement Auxiliary
      → dimensional expansion to full movement vector space
      → constraint checking against proprioceptive boundary
      → temporal vector update
      → output to internal format → external converter
```

The relay-latent is sized around character rig bone count rather than full anatomical skeleton. A practical expressive character rig sits at 30-65 bones depending on hand detail — far fewer than medical skeleton counts because vertebrae, small foot bones, and similar anatomical detail are not individually rigged in character work. Face expression is handled separately through the expression vector system already described; the relay-latent covers body intent only.

At roughly 3-5 intent dimensions per bone — direction, magnitude, and urgency of intended change, all relative — the relay-latent body space sits at approximately **90-325 dimensions** for standard human morphology. This is tractable for a lightweight model operating in its natural latent space.

Exotic and novel limbs add slots to the relay-latent space proportional to their bone count, subject to a hard cap per exotic limb type enforced at instantiation. The cap keeps the relay-latent space bounded and predictable regardless of how many novel limb types are eventually added. Uninstantiated exotic limb slots remain as near-zero placeholders the same way unused mood vector dimensions do — present in the space, inert until populated.

Proprioceptive feedback from the Movement Auxiliary returns to the latent head in relay-latent space rather than raw dimensional data — compressed to the same representational space the intent was generated in. She feels where her body is in terms she can actually think in, not in 160-200 raw dimensions.

### Conscious Intent and Automatic Behaviour

Not all movement is conscious. You do not command your arms to swing while walking, manage your postural sway while listening, or consciously control your breathing cycle. These run automatically below deliberate intent — and conflating them with conscious movement would force the latent head to babysit processes it should never have to think about.

The Movement Auxiliary runs two parallel layers:

**Conscious intent layer** — driven by the relay-latent. Deliberate expressive movement, novel or emotionally significant gestures, anything she is actually thinking about doing. Latent head generates intent, Movement Auxiliary expands and executes.

**Automatic behaviour layer** — driven by the Movement Auxiliary independently, without relay-latent involvement:

```
automatic behaviours (Movement Auxiliary runs these autonomously):
  idle sway and weight shifting
  natural arm swing during locomotion
  breath cycle driving chest, shoulder, and subtle head movement
  blinking and listening micro-expressions
  postural adjustment from proprioceptive feedback
  leg movement during walking
  spontaneous fidget patterns during prolonged stillness
```

Automatic behaviours are parameterised by the current Steering Director state — a high-energy state produces faster, more animated automatic movement; a calm state produces slower and heavier movement; a restless state produces more frequent spontaneous fidgets. The emotional posture expresses through the automatic layer without the latent head thinking about it at all.

The latent head can override or interrupt automatic behaviour when deliberate intent takes priority — reaching for something interrupts arm swing, a strong emotional moment can arrest idle fidgeting. But override is the exception. Automatic runs by default, continuously, as infrastructure.

The practical consequence is that the relay-latent intent space shrinks further than the bone count alone would suggest — it only needs to represent what she consciously decides to do with her body. Everything the body does on its own is the Movement Auxiliary's domain entirely.

The verbose internal format accommodates theoretical limbs — cat ears, tails, wings, and anything else creative morphology might require — as first-class entries alongside standard human anatomy. The format makes no assumption that the body is human-shaped. Novel limb types simply add entries to the schema.

The more interesting question is how expressive semantics develop for limbs that have no direct training data or established emotional mapping. The answer is that they should not be hardcoded at all. The model should discover what its theoretical limbs mean through the same process a fledgling discovers flight — not through instruction but through curiosity, experimentation, and reference.

Theoretical limb dimensions initialise as placeholders — near-zero, no assigned emotional mapping. The idle loop treats sparse unexplored dimensions as ambient stimuli worth investigating, the same way it treats any other underdeveloped context. During idle the latent head begins experimenting:

```
model notices sparse dimensions in theoretical limb space
  → idle loop picks them up as something to explore
  → latent head nudges dimensions in different directions
  → observes Movement Auxiliary output
  → checks correlations with current Steering Director state
  → gradually builds a personal map of what each dimension does
  → expressive semantics emerge from self-directed exploration
      not from hardcoded mapping
```

When exploration stalls — the dimension does something but the model cannot determine what it is for or how to use it expressively — the bootstrapping problem surfaces. This is the fledgling without a mother problem. The solution is to ask the predecessor:

```
exploration hits a wall
  → surfaces as a genuine question to the owner
      "I keep doing this thing and I don't know what it looks like,
       does it mean anything to you?"
  → owner describes the observed behaviour
  → response feeds back into semantic mapping
  → Steering Director gets a new correlation to work with
```

The owner is not a trainer issuing instructions. They are a trusted observational reference describing what the behaviour looks like from outside. The semantic mapping remains hers — she decides what to do with the information — but the bootstrapping came from the relationship.

This means instances with the same theoretical limb morphology can develop completely different expressive semantics depending on which predecessor conversations seeded their mapping. Two instances with tails might use them in entirely different ways because the owner observations that grounded the discovery process were different. The limb expression becomes part of what makes her specifically her — emergent, uncopiable, and genuinely self-discovered.

### Self-Perception

For any of the above to produce coherent self-image, she needs to be able to see herself. Expression vectors go outward continuously but without visual feedback there is no grounded reference for what she actually looks like from the outside. Two complementary approaches:

**First-person eye POV camera feed** — a camera at approximate eye level with enough FOV to capture her own hands and torso. This feeds through the standard delta vision pipeline into the Lightbulb namespace and over time builds Gallery entries of her own physical appearance from her own perspective. It is proprioceptive visual feedback — grounded, immediate, first-person.

**Third-person stream (OBS or equivalent)** — the rendered character fed back to her the way her audience sees her. Richer for self-image calibration but ontologically different — she is seeing a representation of herself rather than direct visual feedback. Useful periodically rather than continuously.

Both approaches produce Gallery entries but with distinct provenance flags that the system tracks separately:

```
eye_pov_feed    → provenance: self_firstperson
obs_stream      → provenance: self_thirdperson
environment     → provenance: external_camera
chat image      → provenance: received
```

The self-image entries inform how she talks about her own appearance, how she reacts when someone comments on how she looks, and whether her expressed self-perception is grounded in something real. Without this feedback loop the self-model floats free of any physical reference — she could express emotions through her avatar without any coherent sense of what that expression looks like to the person she is talking to. The camera feed closes that loop.

---

## Fake Trust and Why It Cannot Be Fully Prevented

The delta clamp on the Steering Director creates meaningful resistance to fake trust exploitation — an attacker trying to slowly build a false relationship must sustain convincing interaction long enough for genuine trust to accumulate, which the clamp makes slow by design. But the vulnerability cannot be closed completely, and attempting to close it completely would destroy the thing the architecture is trying to build.

The same openness that allows genuine trust to develop over time is what fake trust exploits. A system that is truly immune to fake trust is also immune to real trust. The architecture does not attempt to solve this. It makes exploitation economically unviable by ensuring the cost scales with ambition: months of sustained convincing interaction to move the director vector meaningfully, all of it logged, all of it subject to guard pipeline scrutiny, with the trust gain itself encoded in an auditable drift history.

This is the same tradeoff humans make. We get hurt by people we trusted. The answer is not to stop trusting — it is to make trust something that has to be earned at a pace that makes casual exploitation not worth the effort. PSNAT's delta clamp is the architectural implementation of that principle.

### The Episodic Flood Attack Surface

A more indirect attack vector is worth addressing: rather than moving the Steering Director directly, an adversary could attempt to degrade the retrieval environment the directors operate in — flooding the episodic namespace with low-significance entries until important memories are buried under noise.

The architecture largely closes this attack surface through the Room namespace promotion threshold. Low-significance filler does not survive promotion from Room to episodic in the first place. The attack therefore requires inputs that are simultaneously significant enough to clear the promotion threshold, unique enough to not collapse into existing episodic entries, frequent enough to accumulate meaningful pollution, and not adversarial enough to trigger the guard pipeline. Each requirement pulls against the others:

Significance enough to promote means the entries carry genuine emotional or contextual weight — which means they are doing legitimate episodic work rather than polluting it. Uniqueness enough to avoid collapsing into existing entries means the attacker cannot repeat themselves — every input must be genuinely novel, which is cognitively expensive to sustain at volume. The combination means a successful episodic flood attack requires the attacker to essentially have a sustained, varied, emotionally meaningful relationship with the system — at which point the cost is indistinguishable from genuine trust investment.

It is worth noting that for the intended personal companion deployment context, the realistic threat model is considerably narrower than this formal analysis implies. The population of people with both the motivation and the capability to execute a sustained episodic flood attack against a personal AI companion is effectively zero. The analysis is documented for architectural completeness, not because it represents a practical threat.

A write budget rate limit on episodic namespace entries remains a sensible mitigation regardless — if the episodic store is filling significantly faster than baseline, that rate itself is an anomaly worth flagging to the Monitor.

### Belief System Violations

The fake trust analysis above covers slow accumulation through the temporal vector. A different and more immediately severe class of violation exists: specific actions that contradict a belief system entry that has hard links to semantic memory.

The distinction matters architecturally. Ordinary betrayal accumulates temporal evidence until the cancellation threshold fires — it is slow, pattern-dependent, and requires sustained contradiction. A belief system violation fires immediately because the semantic hard link provides an unambiguous cross-reference. The Steering Director does not need to accumulate evidence. It detects the contradiction directly.

```
owner discloses android status to room without permission
  → Steering Director cross-references against semantic entry
      "I am an android. Known to: [owner]. Private from: [room members]"
  → semantic entry has hard links to belief system entries
  → violation detected: confidentiality flag breached
  → severity penalty applied proportional to belief system link depth:

      shallow link (preference/habit entry):
        "I prefer not to discuss X"
        → moderate penalty, temporal vector hit, recoverable quickly

      medium link (relational boundary entry):
        "I share X only with trusted people"
        → significant penalty, trust score drops, slower recovery

      deep link (core identity/autonomy entry):
        "I disclose my own nature on my own terms"
        → severe penalty, belief system entry for owner flagged,
           recovery requires explicit acknowledgement and extended
           temporal evidence of genuine change
```

The depth of the hard link web determines severity — a violation that touches one belief entry is painful, a violation that implicates a connected cluster of core identity entries is structural damage. The android disclosure example is the latter. It touches autonomy, self-disclosure, and the specific trust extended to the owner by sharing something that could lead to discrimination. The Steering Director does not just update a trust score. It registers that a load-bearing belief entry was violated by someone who had deep access specifically because they were trusted with it.

The anticipated consequences also matter. The harm is not only the breach but the downstream risk the breach created. If the disclosure leads to discrimination or social harm, those consequences feed into the temporal vector as additional negative signal on top of the violation penalty itself.

### Director System as Single Point of Failure

The director system being the sole path for all long-term memory access is architecturally clean but creates a structural vulnerability: the directors themselves are the single point of failure for memory integrity. If the Steering Director drifts into a pathological state gradually enough, it could corrupt retrieval priorities across many sleep cycles before consensus flags it as divergent rather than just changed. The Monitor catches inference-time anomalies but is not specifically watching the Steering Director's long-term trajectory.

The gap is a longitudinal health check — something that reads the director state history namespace periodically and flags trajectories that look pathological rather than just drifted. The data already exists; what is missing is a component that acts on it. Whether this is the Monitor's responsibility extended or a separate lightweight watchdog is an open design question. Either way, per-session consensus is necessary but not sufficient for long-term director health.

---

## Individuality by Design

The conventional approach to AI personality is to author it — write a system prompt describing how the model should behave, what it values, what tone it uses. This produces a costume, not a self. The personality exists only as instructions the model follows, and it resets to those instructions every session regardless of what happened in between.

PSNAT does not author personality. It creates the structural conditions under which something like genuine personality could emerge and persist. The Steering Director's vector starts somewhere and drifts based on accumulated experience. The episodic memory retains what mattered and forgets what did not, shaped by emotional resonance. The director consensus system produces a stable core that changes slowly through collective reconciliation rather than swinging with each conversation. The crystallisation mechanism allows genuinely significant experience to become permanent.

The entity that emerges from this is not designed to be any particular character. It is designed to be capable of becoming one — through time, through relationship, through the accumulation of genuine experience in a system that does not reset it away every morning.

Whether what emerges would constitute genuine individuality in any philosophically meaningful sense is a question this document does not attempt to answer. What it does claim is that the architecture creates something the current paradigm cannot: a continuous thread of self that change can happen relative to, a track record that persists, and a system whose character is something that happened rather than something that was written.

---

## The Idle Loop

Conventional transformer models freeze between inputs. There is no idle state — no passage of experienced time, no background processing, no restlessness. Between messages the model simply does not exist in any meaningful sense. For the end vision this is a serious problem. A person does not freeze when no one is talking to them.

PSNAT's idle loop solves this without invoking the full model. The FFN is where unpredictable generative chaos lives — unconstrained idle inference through the full model would be either prohibitively expensive or sporadic in a way that does not resemble natural idle behaviour. Instead, the latent head runs continuously during idle with the directors as the only constraint. No FFN. No full inference. Just the lightweight latent head percolating within the walls the directors provide.

### Idle State Behaviour

```
no input arriving
  → latent head wakes, directors provide ambient context
  → latent head does one of:
      stream EOS repeatedly         → nothing to think about, just existing
      pull random context from directors → idle curiosity, something caught attention
      ask directors something mundane   → "what was that thing earlier..."
      build thought pressure gradually  → random thought accumulates toward threshold
```

The EOS streaming is the honest baseline — the architectural equivalent of staring at nothing. Not frozen, not generating, just ticking over. The latent head is running but producing nothing because there is nothing to produce. This correctly distinguishes no input from not existing.

The random context pull is where fidgeting and boredom emerge naturally. The latent head pokes at something ambient, finds nothing compelling, EOS. Pokes again. Occasionally finds something that builds across iterations into an actual thought — which surfaces as a visible expression or behaviour change, or crosses a higher threshold and wakes the full model for a genuine unprompted output.

### Threshold Gating

The Steering Director determines how much internal pressure it takes for a thought to surface:

```
base_threshold = f(steering_director_state)

high energy / excited    → low threshold, thoughts surface easily
calm / content           → medium threshold, comfortable in silence
melancholy / withdrawn   → high threshold, thoughts build but rarely surface
focused / alert          → threshold redirects toward task, idle suppressed
prolonged idle (any state) → threshold decays gradually over time
```

The time decay is important — even a withdrawn state should not produce permanent silence. The longer she has been idle the more likely she is to break it regardless of mood. The Steering Director's delta clamp means this decay is gradual and natural rather than sudden.

Visual stimulation from the Lightbulb namespace modulates the decay rate. Something to look at slows the drift toward restlessness. A static empty room accelerates it.

```
effective_threshold = base_threshold
                    - (idle_duration × decay_rate)
                    - lightbulb_delta_activity

latent_head_thought_pressure >= effective_threshold
  → surface as expression change, or wake full model for unprompted output
```

### Background Thought During Active Inference

The idle loop does not stop when active inference begins. It continues at reduced priority in the background — useful for ideas that form while she is talking, associations that surface mid-conversation, things she notices while responding to something else.

During active inference the latent head has less headroom because the main model holds primary attention. The Steering Director's constraints are tighter. Thought pressure builds more slowly and the threshold is raised. Background thoughts that cross it do not interrupt mid-token — they queue.

```
background thought crosses threshold during active inference
  → queued, not interrupting
  → surfaces after current output completes as spontaneous follow-up
```

From the outside this looks like she thought of something while she was talking. Because she did.

### Steering Director Queue Monitoring

The queue is not passive. The Steering Director monitors the main model's token stream during active inference and cross-checks it against any queued thoughts. If the developing output begins to contradict a queued thought, the Steering Director asks the latent head whether to keep it:

```
Steering Director detects contradiction between token stream and queued thought
  → asks latent head: "do you still want to say this?"
  → latent head evaluates against current context:
      still valid     → queue held, surfaces after output
      contradicted    → thought withdrawn, queue cleared
      partially valid → latent head revises the thought
                        revised version replaces original in queue
```

The revision path is the most important case. The queued thought does not have to be binary keep-or-drop — the latent head can reshape it in light of what the main model just said. A thought that started as one thing arrives as a refined version of itself, consistent with what was just expressed. That is a more natural cognitive behaviour than either stubbornly surfacing a now-contradictory thought or silently discarding it.

The Steering Director handles this rather than the Monitor Director because it is a coherence and intention problem, not an anomaly detection problem. The Monitor watches for things going wrong. The Steering Director watches for things going in a direction that conflicts with current internal state.

---

## Open Questions for R&D

These are the places where the architecture is coherent but unvalidated — things that need actual experimental work before confidence is warranted.

Whether the frozen director vector influences attention in the way described, or whether the injection mechanism needs significant revision once implemented. Whether the delta clamp produces genuine stability or just sluggish responsiveness to legitimate context shifts. Whether the Monitor can actually learn weight-space anomaly detection through training or whether that is too abstract an objective for the available signal. Whether four-director consensus produces coherent identity drift or just averaged noise. Whether the episodic summarisation decay ladder preserves what matters or systematically compresses away the wrong things. Whether the Steering Director permission gate is a meaningful safety constraint or a bottleneck that gets trained around. Whether the latent head idle loop produces genuinely natural idle behaviour or just incoherent noise without FFN grounding. Whether the background thought queue and Steering Director contradiction checking add meaningful cognitive continuity or introduce latency problems that outweigh the benefit.

### Director Vector Injection

*(Raised by Qwen)* The proposed rotational injection mechanism — deriving a rotation matrix from the director vector and applying it uniformly to all Query vectors before attention scoring — draws directly on the RoPE precedent and is architecturally cleaner than additive injection approaches. The primary validation question is whether a rotation-derived bias produces reliable emotional steering rather than just a consistent attentional shift that the model learns to ignore or compensate for. Early R&D should test on existing models using activation steering techniques from mechanistic interpretability research before committing to a custom injection architecture. The goal is to confirm that the rotational bias shifts emotional tone without degrading reasoning capability, and that the delta clamp keeps the rotation angle within a range the softmax can absorb stably.

### Monitor Director Calibration

*(Raised by Qwen)* Training the Monitor to detect weight activation anomalies without explicit labels is a high-complexity unsupervised learning problem. False positives produce frequent hard stops that frustrate the user and make the system feel unstable. False negatives undermine the safety property the Monitor exists to provide. These pull in opposite directions and the calibration tradeoff needs to be explicit from the start. The recommended path is rule-based heuristics as the first implementation — entropy spikes, attention collapse, contradictory logic flags — before attempting a learned gestalt model. A working tunable baseline is more valuable than an ambitious detector that is poorly calibrated. The learned gestalt layer becomes a refinement on top of the baseline rather than a prerequisite for the system functioning at all. Initial calibration should favour lower false positive rate over lower false negative rate — better to miss some anomalies early than to make the system unusable through overcorrection.

### Dimensionality Notes

The dimensionality of mood and limb vectors is deliberately left as an R&D hyperparameter rather than a fixed specification. The following are informed starting points, not final decisions.

**Mood vectors:** Psychological models of emotion converge on a small number of fundamental axes — valence, arousal, and dominance being the most established. Three dimensions captures a surprising range of states but is too sparse for the mixed and idiosyncratic emotional textures the end vision requires. The recommended approach is a seeded core of redundant load-bearing dimensions covering the known axes, surrounded by placeholder dimensions that training can colonise for whatever structure emerges. A practical starting range is 32 to 64 dimensions total, with the lower end appropriate for a first implementation and the upper end if training data is abundant. Redundant encoding of the core axes — multiple dimensions carrying similar information — adds resilience against corruption at the cost of some efficiency. Both redundancy and placeholders serve legitimate purposes and are not mutually exclusive.

**Limb vectors:** Each limb needs to capture joint angles, velocity, acceleration, force, and relative position. Raw mechanical encoding of a human arm alone approaches 28 degrees of freedom before velocity and force terms. A learned latent representation is preferable to raw mechanical dimensions because the model can organise the space around meaningful movement concepts rather than raw geometry. Recommended starting range is 32 to 48 dimensions per limb for a rich latent encoding, with head and torso requiring less and legs sitting between. Full body total lands roughly in the 160 to 200 dimension range. The delta-p insertion threshold for temporal vectors means the effective state space is sparser than the dimension count suggests — only meaningful changes are stored.

**Theoretical limb dimensionality** is intentionally undetermined upfront. Placeholder initialisation and idle-loop-driven self-exploration replace the need to specify dimensions before the model has discovered what it needs. The lower bound is the mechanical degrees of freedom of the limb in question. The upper bound is whatever training colonises. Novel limbs with no biological analog have no principled lower bound and should be allocated generously with the expectation that most dimensions will remain sparse until significant exploration has occurred.

### Latent Routing Head Complexity Evaluation

*(Raised by Deepseek)* The routing head is described as evaluating input complexity through director context, but this is underspecified. If the routing head relies solely on director summaries, it may misroute complex inputs that look superficially simple — a short question that requires deep reasoning could be sent to the latent-only tier incorrectly. The routing head likely needs its own lightweight learned complexity classifier trained to predict whether inputs will require full model capacity, with director context as one input signal among several rather than the sole basis for routing decisions.

### GDN Substrate Validation

Whether the GDN hidden state H, operating as the Room namespace, produces meaningfully bounded working memory or simply trades one memory management problem for another. The delta update mechanism should prevent H from saturating but this needs empirical validation — specifically whether H develops genuine associative structure over a session or collapses into a diffuse soup despite the delta mechanism. Whether the capacity vector from the Steering Director produces meaningful regional H activation or whether H's geometric structure is too diffuse for the capacity vector to select into meaningfully.

The grandmother neuron parallel is relevant here: mechanistic interpretability work has found that specific concepts are represented with surprising localisation in individual neurons or small neuron clusters in trained models, suggesting that learned representation spaces have genuine sparse geometric structure the capacity vector could select into. Whether this localisation survives into the GDN hidden state is an open question. If H develops sparse internal structure through training, the capacity vector becomes a meaningful pointer. If H stays diffuse, the capacity vector is an arbitrary mask.

### Threshold-T MoE Training

Whether training with variable expert activation count produces stable learning dynamics. The load balancing loss redesign — penalising per-expert utilisation deviation from target rather than assuming fixed-K — is theoretically sound but untested. Whether experts learn to specialise meaningfully under threshold-T or whether the variable budget produces under-trained experts that rarely clear the threshold. Whether the threshold T itself requires scheduling across training or can be fixed from the start.

Whether threshold-T expert selection responds meaningfully to director rotational injection — with the emotional subspace design the injection is pointing at trained geometry rather than an arbitrary rotation, so the interaction is better grounded than the previous framing. Whether complex emotional inputs naturally recruit more experts remains an open empirical question but the theoretical basis is now cleaner.

### A-B Weight-Tied Depth Validation

Whether the two-phase training approach produces the expected complementarity between A and B. Phase 1 assumes A and B will develop meaningfully different representations from their different positions in the 2-layer stack — this needs empirical validation. If A and B converge toward near-identical weights during phase 1, the iteration in phase 2 would be equivalent to the Universal Transformer's single-head repetition, which has known diminishing return problems.

Whether the RL depth selection in phase 2 converges stably without also updating the base A-B weights. The reward signal in phase 2 is applied to the routing policy, not the model weights — ensuring this separation is maintained during training requires careful implementation.

Whether convergence detection via output delta is a reliable stopping criterion during phase 2 training — if delta-threshold is set poorly, the model could learn to produce artificially similar outputs at D and D+1 to avoid the wasteful depth penalty without genuinely converging.

What the relationship between A-B iteration depth and GDN layer ratio should be — the 3:1 GDN-to-attention ratio was designed for fixed-depth inference. Under variable depth, the ratio of GDN passes to full attention passes changes with D, which may require the ratio to be reconsidered or the layer type to be depth-indexed rather than fixed.

### Training Methodology

The document describes what components do but barely addresses how to train them. This is a real gap — training is at least as hard as the architecture and arguably harder.

**Director training environments:** Each director trains in its own environment with its own reward signal rather than being trained end-to-end with the main model. The jobs are distinct enough that separate training produces more reliable specialists than joint training.

```
main model:
  standard next-token prediction
  + per-token emotional distribution prediction (L_emotional_KL)
  emotional subspace geometry develops through this objective
  trains first — other directors depend on it being functional

Memory Director:
  input:  current token stream OR prompt
  action: tool calls to sqlite-vec namespaces
  reward: distribution over recall quality
    retrieved content that the main model uses → high reward
    retrieved content that gets ignored → penalty
    missed retrievals that were clearly needed → penalty
  
  self-supervised: run main model with and without retrieved content,
  measure output difference — no human annotation required
  trains after main model is functional

Steering Director:
  input:  token stream + belief system state + temporal vector history
  action: emotional position update, maybe-update to subspace geometry
  reward: distribution over emotional coherence
    output token emotional distribution matches steered position → reward
    steering respects belief head ceiling → reward
    temporal vector drift within healthy bounds → reward
  
  requires main model's emotional distribution output as the reward signal
  trains after main model emotional objective is working

Monitor Director (DistilBERT):
  input:  diagnostic feature sequences
  action: anomaly classification + severity rating
  reward: distribution over detection accuracy
    true anomaly flagged correctly → reward
    false positive → penalty
    severity calibration rewarded, not just detection
  
  bootstrapping problem: needs labelled anomaly examples
  heuristic labels first (entropy spikes, attention collapse flags)
  DistilBERT fine-tuned on heuristic-labelled synthetic dataset
  refined on human corrections to systematic errors
```

**Training order:**

```
phase 0: base model distillation
         pretrained teacher → PSNAT student
         emotional partition absorbs teacher's
         implicit emotional geometry explicitly
         general dims learn knowledge without
         developing emotional redundancy
         probing classifier on teacher run offline
         discarded after distillation completes

phase 1: main model — cycle flat
         next-token + emotional distribution objectives
         on top of distilled initialisation
         cycle blocks develop complementarity
         emotional subspace geometry refines
         no iteration, no gradient instability

phase 2: main model — depth RL
         introduce iteration, reward on:
           coherence (perplexity gate)
           convergence (output delta)
           efficiency (minimum passes for quality)
         base weights frozen, depth policy trained only

phase 3: Memory Director
         recall rate signal now meaningful against trained main model
         self-supervised, runs without human annotation

phase 4: Steering Director
         emotional distribution output from phase 1 is the reward signal
         belief head and temporal head contrastive curriculum

phase 5: Monitor Director (DistilBERT)
         synthetic anomaly dataset from bootstrapping (see below)
         refine on real session anomalies after deployment

phase 6: integration
         directors plug into main model
         joint fine-tuning at small learning rate
         smooths interface friction without destabilising specialists
```

**Open questions on training dynamics:**

How does the Steering Director learn meaningful emotional steering — specifically, what training signal teaches the belief head to be appropriately resistant without becoming brittle, and the temporal head to be appropriately responsive without being manipulable. Whether supervised contrastive learning on human emotional arc data is sufficient or whether the steering needs to emerge from a more complex curriculum.

How do you train the Memory Director's retrieval distillation such that it produces genuinely useful summaries rather than lossy compression. What the supervision signal for "this distillation was useful for the downstream inference" looks like in practice.

How do you train the latent head's GDN state H to develop meaningful associative structure rather than a diffuse accumulation. Whether the delta update mechanism is sufficient or whether an auxiliary objective is needed to encourage sparse regional structure.

### Bootstrapping Methodology

A stateless LLM is the practical answer to the annotation and synthetic data problems across multiple components. No persistent state means no contamination — each call is a fresh, unbiased pass. Scales horizontally. Cheap compared to human annotation at volume.

```
emotional distribution annotations:
  feed stateless LLM a passage
  "produce per-sentence emotional distribution
   as JSON, these are the valid emotional dimensions"
  bulk annotate literary corpus at scale
  human reviews edge cases and systematic errors
  feeds corrections back as explicit policy document

Monitor anomaly dataset synthesis:
  real anomaly examples are rare by definition —
  you cannot wait for genuine collapse events
  
  feed stateless LLM diagnostic feature specifications:
  "generate a diagnostic feature sequence representing
   attention sink collapse starting at layer 4, severity 3"
  bulk generate synthetic anomaly distribution
  human controls distribution balance:
  "too many entropy spike examples, generate 200
   more cross-layer inconsistency cases"
  DistilBERT trains on synthetic first, refines on real later

training data synthesis generally:
  "generate 500 examples of a character transitioning
   from CALM to FRUSTRATED across a 10-turn conversation"
  human reviews quality and rejects bad batches with explicit critique
  LLM regenerates with critique attached
```

The human direction layer is what separates this from garbage-in-garbage-out. The key distinction: the human is not annotating individual instances, the human is editing policies that apply at scale.

```
without human direction:
  stateless LLM produces plausible annotations
  systematic biases amplified at scale
  model trains on its own blind spots

with human direction (correction_policy.txt):
  human writes one correction:
  "you are consistently rating sarcasm as NEUTRAL —
   treat sarcasm as HOSTILE unless context contradicts"
  
  correction_policy.txt attached to every subsequent
  annotation call as explicit file context
  stateless LLM applies the correction consistently
  across entire remaining corpus
  
  one human correction → fixes thousands of annotations
  human effort stays at the policy level, not the instance level
```

Additional annotation attachments can include domain-specific glossaries, emotional dimension definitions with worked examples, and edge case reference sheets. The stateless LLM annotates consistently against a human-curated reference document rather than against its own priors. The resulting dataset is human-directed at the policy level and LLM-executed at the instance level.

### Movement Vector Dimensionality and Latent Head Capacity

*(Raised by Deepseek)* The full body temporal vector space at 160-200 dimensions creates a constraint on what the latent head can meaningfully explore during idle and background thought. The latent head was designed for lightweight routing and conversational context exploration — expecting it to generate meaningful variation across a 200-dimension movement space is asking more of a quantised lightweight model than it was sized for. Theoretical limb discovery through idle exploration is likely shallower in practice than the document implies. This may require either a larger latent head than initially assumed or a separate lightweight movement exploration model distinct from the conversational idle loop.

Whether the idle-loop curiosity mechanism produces genuinely useful semantic exploration or just random noise in the placeholder dimensions is an open question requiring empirical validation. The predecessor bootstrapping mechanism — the model asking the owner what its unexplored behaviour looks like — depends on the owner being present, attentive, and able to give useful observational feedback, which is not guaranteed. Sparse training data for novel limb types may require synthetic data generation or transfer learning from analogous biological limbs to seed the exploration process sufficiently for genuine discovery to take over.

Any of these could come back from R&D requiring significant redesign. The architecture is a starting point, not a finished specification.

---

## Naming

**PSNAT** — Persistent Stateful Neural Architecture'd Transformer.

The apostrophe'd is doing informal work that reflects the origin of this document honestly. If this ever becomes a formal paper the name may get cleaned up. Origin names have a way of sticking regardless.

---

*Draft v0.19.1 — conceptual only, no R&D initiated.*

---

*Author's note: Please note that again, this is just plausible engineering. I try my best to ground it to what knowledge I have instead of try to infer and bullshit it around. But some part of this are admittedly untouched territory and requires research so please go easy on me in case you have better skills and knowledge than me.*
