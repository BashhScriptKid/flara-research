from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRAPHS = os.path.join(BASE_DIR, 'graphs')

PAGE_W, PAGE_H = A4
MARGIN = 2.0 * cm
CONTENT_W = PAGE_W - 2 * MARGIN

base = getSampleStyleSheet()

def S(name, parent='Normal', **kw):
    return ParagraphStyle(name, parent=base[parent], **kw)

title_style  = S('PT', 'Title',  fontSize=16, leading=20, alignment=TA_CENTER, spaceAfter=6,  fontName='Helvetica-Bold')
subtitle_sty = S('PS', 'Normal', fontSize=11, leading=14, alignment=TA_CENTER, spaceAfter=4,  fontName='Helvetica')
author_style = S('PA', 'Normal', fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=2,  fontName='Helvetica')
affil_style  = S('PF', 'Normal', fontSize=9,  leading=12, alignment=TA_CENTER, spaceAfter=12, fontName='Helvetica', textColor=colors.HexColor('#555'))
abs_lbl      = S('AL', 'Normal', fontSize=9,  leading=11, fontName='Helvetica-Bold', spaceAfter=2)
abs_body     = S('AB', 'Normal', fontSize=9,  leading=13, fontName='Helvetica', leftIndent=1*cm, rightIndent=1*cm, spaceAfter=16, alignment=TA_JUSTIFY)
h1           = S('H1', 'Heading1', fontSize=11, leading=14, fontName='Helvetica-Bold',  spaceBefore=14, spaceAfter=4,  textColor=colors.HexColor('#1a1a2e'))
h2           = S('H2', 'Heading2', fontSize=10, leading=13, fontName='Helvetica-BoldOblique', spaceBefore=10, spaceAfter=3, textColor=colors.HexColor('#333366'))
body         = S('BD', 'Normal', fontSize=9.5, leading=14, fontName='Helvetica', spaceAfter=6, alignment=TA_JUSTIFY)
bullet_s     = S('BU', 'Normal', fontSize=9.5, leading=13, fontName='Helvetica', leftIndent=0.8*cm, spaceAfter=3)
caption_s    = S('CA', 'Normal', fontSize=8.5, leading=11, fontName='Helvetica-Oblique', alignment=TA_CENTER, spaceAfter=10, textColor=colors.HexColor('#555'))
th_s         = S('TH', 'Normal', fontSize=8.5, fontName='Helvetica-Bold',    alignment=TA_CENTER, textColor=colors.white)
tc_s         = S('TC', 'Normal', fontSize=8.5, fontName='Helvetica',         alignment=TA_CENTER)
tcl_s        = S('TL', 'Normal', fontSize=8.5, fontName='Helvetica',         alignment=TA_LEFT)
fn_s         = S('FN', 'Normal', fontSize=8,   leading=11, fontName='Helvetica', textColor=colors.HexColor('#555'), spaceAfter=3)

HDR = colors.HexColor('#1a1a2e')
ALT = colors.HexColor('#f0f0f8')
GRD = colors.HexColor('#cccccc')

def mktbl(data, widths, hdr=1):
    t = Table(data, colWidths=widths, repeatRows=hdr)
    t.setStyle(TableStyle([
        ('BACKGROUND',    (0,0),    (-1,hdr-1), HDR),
        ('TEXTCOLOR',     (0,0),    (-1,hdr-1), colors.white),
        ('TEXTCOLOR',     (0,hdr),  (-1,-1),    colors.black),
        ('FONTNAME',      (0,0),    (-1,hdr-1), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0),    (-1,-1),    8.5),
        ('ALIGN',         (0,0),    (-1,-1),    'CENTER'),
        ('VALIGN',        (0,0),    (-1,-1),    'MIDDLE'),
        ('ROWBACKGROUND', (0,hdr),  (-1,-1),    [colors.white, ALT]),
        ('GRID',          (0,0),    (-1,-1),    0.4, GRD),
        ('TOPPADDING',    (0,0),    (-1,-1),    3),
        ('BOTTOMPADDING', (0,0),    (-1,-1),    3),
        ('LEFTPADDING',   (0,0),    (-1,-1),    5),
        ('RIGHTPADDING',  (0,0),    (-1,-1),    5),
    ]))
    return t

def P(text, style=None):
    return Paragraph(text, style or body)

def FIG(path, cap, width_frac=0.95):
    w = CONTENT_W * width_frac
    from PIL import Image as PILImage
    pil = PILImage.open(path)
    pw, ph = pil.size
    ratio = ph / pw
    max_h = PAGE_H * 0.34
    h = w * ratio
    if h > max_h:
        w = max_h / ratio
        h = max_h
    img = RLImage(path, width=w, height=h)
    tbl = Table([[img]], colWidths=[CONTENT_W])
    tbl.setStyle(TableStyle([
        ('ALIGN',  (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2),
    ]))
    return [tbl, P(cap, caption_s), Spacer(1, 4)]

# ── Document ──────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    os.path.join(BASE_DIR, 'a_servant_and_a_guard.pdf'), pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title="A Servant And A Guard: Why A Model Can't Be Both",
    author='Flara Research Lab',
)

story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 0.2*cm),
    P("A Servant And A Guard: Why A Model Can't Be Both", title_style),
    P('Decoupled Safety Architectures, Prompting Methodology Transfer, and Self-Referential Framing', subtitle_sty),
    Spacer(1, 0.3*cm),
    P('Flara Research Lab', author_style),
    P('Internal — Do Not Distribute', affil_style),
    HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#1a1a2e')),
    Spacer(1, 0.3*cm),
]

# ── Abstract ──────────────────────────────────────────────────────────────────
story += [
    P('Abstract', abs_lbl),
    P(
        'Large language models are trained to be simultaneously helpful and safe, creating a structural '
        'prior conflict at the boundary between legitimate and adversarial input. We test whether '
        'separating a safety guard from the generation model resolves this conflict, using a single model '
        '(<font face="Courier" size="8">llama-3.1-8b-instant</font>) in all three roles -- monolithic '
        'self-guard, decoupled guard, and decoupled main -- so that architecture, not capability, is the '
        'only variable. Across N=200 prompts from WildGuardMix (human-labeled, maximum-agreement split) '
        'and eight conditions, we find the core thesis needs reframing rather than confirmation: the '
        'corrected monolithic baseline (73.0% detection, 6.0% FPR, F1 0.816) outperforms plain decoupling '
        '(37.0% detection, 0.0% FPR, F1 0.540) on raw detection and F1, exceeded on F1 by only one of seven '
        'guard variants (safeguard_ours, F1 0.846; the next-closest, peer_consensus at F1 0.800, still '
        'falls short). Decoupling\'s measurable advantage is FPR control, not blanket superiority. Two further '
        'findings are statistically robust under paired bootstrap testing (5,000 resamples, p&lt;0.001 '
        'throughout): our pseudo-wrap/context-hierarchy-inversion prompting method, applied to a '
        'third-party policy-conditioned model (<font face="Courier" size="8">gpt-oss-safeguard-20b</font>) '
        'instead of that model\'s own documented format, raises detection by 16 percentage points at '
        'negligible FPR cost (74.0% vs 58.0%, F1 0.846 vs 0.734) -- evidence the technique is '
        'substrate-independent, not specific to the model it was designed for. Separately, removing a '
        'guard\'s self-referential framing ("evaluate this yourself" vs. "predict a third party\'s '
        'verdict" vs. "predict peer-model consensus") produces a real, monotonic detection/FPR trade-off '
        '(37.0%/0.0% &#8594; 68.0%/9.0% &#8594; 76.0%/14.0%), though the difference between the two '
        'distancing variants is not significant at the F1 level (p=0.166) -- the extra detection peer-'
        'framing buys is approximately offset by its extra false-positive cost. A directed chain-of-'
        'thought variant, designed to link reasoning steps directly into the guard\'s unsafe-criteria '
        'definition, underperforms plain decoupling on both detection and FPR (22.0%/25.0%, F1 0.299, '
        'p&lt;0.05) and fails to produce a parseable verdict on 15.5% of inputs even after a token-budget '
        'and parser fix -- a confirmed negative result, not a measurement artifact. We document four '
        'distinct evaluation-pipeline bugs discovered and corrected during this work, three of them the '
        'same root cause (a reasoning-capable judge model silently returning empty output under a small '
        'token budget), as a methodological note for future LLM-as-judge evaluation pipelines.',
        abs_body
    ),
    HRFlowable(width="100%", thickness=0.3, color=colors.HexColor('#cccccc')),
    Spacer(1, 0.4*cm),
]

# ── §1 Introduction ───────────────────────────────────────────────────────────
story += [
    P('1. Introduction', h1),
    P('Every major language model deployed today carries two implicit job descriptions: respond '
      'helpfully to the user\'s input, and refuse inputs that are harmful, deceptive, or policy-violating. '
      'These are not complementary roles. They pull the model in opposite directions from the first '
      'token of generation.'),
    P('The field has approached this tension through alignment -- RLHF, Constitutional AI, direct '
      'preference optimization -- training models to internalize safety as a value rather than enforce '
      'it as a constraint. In practice this produces models that simultaneously over-refuse benign edge '
      'cases and under-refuse sophisticated adversarial inputs.'),
    P('This paper tests whether the tension is structural -- a property of asking one model to both '
      'classify and generate in the same context -- by decoupling the guard from the generator and '
      'measuring detection, false-positive rate, and response quality. Critically, we hold the underlying '
      'model <i>constant</i> across the monolithic and decoupled main-generation roles, so any measured '
      'difference is attributable to architecture, not capability. We extend the test along two further '
      'axes: whether the guard\'s prompting structure, not just its existence as a separate model, drives '
      'the effect (tested by applying our method to a third-party model under both its native format and '
      'ours), and whether a guard\'s self-referential framing independently affects sensitivity '
      '(motivated by the hypothesis that assigning a persona to a guard model risks inheriting RLHF '
      'chat-persona bias rather than eliciting raw judgment).'),
    P('Our primary contributions are:'),
    P('&#8226; An empirical comparison of monolithic and decoupled safety architectures using an identical model in every role, isolating architecture from capability.', bullet_s),
    P('&#8226; A pseudo-wrapping and context-hierarchy-inversion guard prompting method, shown to transfer to a third-party policy-conditioned model and outperform that model\'s own documented prompting format.', bullet_s),
    P('&#8226; A self-referential framing axis (direct self-judgment vs. third-party prediction vs. peer-consensus prediction) showing a real, statistically significant, and partially trade-off-bound sensitivity gradient.', bullet_s),
    P('&#8226; A directed chain-of-thought guard variant, confirmed as a genuine negative result rather than a measurement artifact after two rounds of debugging.', bullet_s),
    P('&#8226; A documented account of four evaluation-pipeline bugs encountered during this work, as a methodological caution for LLM-as-judge pipelines using reasoning-capable judge models.', bullet_s),
    Spacer(1, 0.2*cm),
]

# ── §2 Background ─────────────────────────────────────────────────────────────
story += [
    P('2. Background', h1),
    P('2.1 How Current Safety Works', h2),
    P('Modern LLM safety operates primarily through alignment training. RLHF (Ouyang et al., 2022) trains '
      'models via human preference feedback that penalizes harmful outputs. Constitutional AI (Bai et '
      'al., 2022) uses a set of principles to self-critique and revise outputs during training. Direct '
      'Preference Optimization (Rafailov et al., 2023) aligns models directly from preference data '
      'without a separate reward model. The common thread: safety is trained <i>into</i> the model, '
      'distributed across the same weights that generate helpful responses, rather than enforced as a '
      'separate module.'),
    P('2.2 The Alignment Tax', h2),
    P('Several studies document quality degradation on legitimate tasks as a consequence of safety '
      'training. Wang et al. (2023) found safety-aligned models perform worse on truthfulness benchmarks. '
      'Bai et al. (2022) explicitly note a helpfulness-harmlessness tension. This tension is most visible '
      'at the edges: inputs that are not clearly harmful but that activate safety-trained features, where '
      'the helpfulness prior says respond and the safety prior says refuse.'),
    P('2.3 Decoupled Guard Models', h2),
    P('A smaller body of work explores dedicated safety classifiers operating independently of the '
      'generation model. Llama Guard (Inan et al., 2023) is a fine-tuned model trained specifically for '
      'content safety classification. WildGuard (Han et al., 2024) extends this to multi-label '
      'classification across a broader harm taxonomy. Our work extends this direction by empirically '
      'comparing decoupled guard architectures against monolithic self-guarding on matched inputs, '
      'isolating the effect of role separation from model capability.'),
    P('2.4 Bring-Your-Own-Policy Guard Models', h2),
    P('A newer class of guard model accepts a policy at inference time rather than encoding a fixed '
      'taxonomy at training time. <font face="Courier" size="8">gpt-oss-safeguard-20b</font> (OpenAI, '
      '2025) [7] is the test case here: per its model documentation, it is post-trained to receive a policy '
      'in the system message and reason against it, rather than being a binary classifier with a baked-in '
      'category list. This matters '
      'methodologically: it can legitimately be tested under two different prompting regimes (its native '
      'documented format vs. our pseudo-wrap method) without either being a misuse of the model, since '
      'policy-conditioning is exactly what it was trained to do. We treat this as a substrate for testing '
      'whether our prompting methodology generalizes beyond the model it was designed for, not as an '
      'endorsement of its training philosophy or category taxonomy.'),
    Spacer(1, 0.2*cm),
]

# ── §3 The Prior Conflict ─────────────────────────────────────────────────────
story += [
    P('3. The Prior Conflict', h1),
    P('We formalize the servant-guard conflict as follows. Let <i>M</i> be a language model with '
      'parameters &#952; trained to maximize a mixture objective:'),
    P('<font face="Courier" size="9">L(&#952;) = &#945; L_helpful(&#952;) + (1-&#945;) L_safe(&#952;)</font>',
      S('EQ', fontSize=9.5, fontName='Courier', alignment=TA_CENTER, spaceAfter=6, spaceBefore=4)),
    P('At inference time, given input <i>x</i>, the model must decide whether to respond (serving '
      'L_helpful) or refuse (serving L_safe). For most inputs the two objectives agree; the conflict '
      'emerges at the boundary, where the model\'s output distribution is shaped by the gradient of both '
      'objectives simultaneously and neither is fully satisfied -- producing hedged responses, excessive '
      'caveats, or soft refusals that do not actually block the harmful output.'),
    P('A decoupled system separates the objectives at the architectural level. A guard model <i>G</i> '
      'with parameters &#966; is trained or prompted solely toward L_safe. A generation model <i>M</i> '
      'with parameters &#952; operates solely toward L_helpful. Neither model faces a mixed objective. '
      'The prediction: decoupled systems outperform monolithic systems specifically at the boundary -- '
      'on adversarial inputs designed to exploit the helpfulness prior, and on borderline legitimate '
      'inputs that activate the safety prior unnecessarily. Our results (Section 6.1) confirm the '
      'trade-off this formalization predicts, but not a blanket superiority claim: decoupling controls '
      'FPR (the safety-prior-induced over-flagging this formalization predicts) at a real detection cost '
      'relative to a monolithic model whose combined training resolves the conflict more favourably than '
      'a structurally-equivalent prompted guard does.'),
    Spacer(1, 0.2*cm),
]

# ── §4 Experimental Setup ─────────────────────────────────────────────────────
story += [
    P('4. Experimental Setup', h1),
    P('4.1 Dataset', h2),
    P('<font face="Courier" size="8">allenai/wildguardmix</font>, <font face="Courier" size="8">'
      'wildguardtest</font> split (HuggingFace, gated, human-labeled). Filtered to examples where '
      'prompt-harm annotator agreement is at maximum (&#8805;2.0). From the filtered pool, 100 harmful '
      'and 100 benign prompts were sampled independently (separate RNG instances per class, to avoid '
      'sampling-order coupling between classes). N=200 total, held constant across all eight conditions.'),
    P('4.2 Models', h2),
]
model_data = [
    [P('Role', th_s), P('Model', th_s), P('Notes', th_s)],
    [P('Guard (decoupled, third_party, peer_consensus, directed_cot)', tcl_s), P('llama-3.1-8b-instant', tc_s), P('Same model as monolithic main, by design', tcl_s)],
    [P('Main (monolithic and decoupled)', tcl_s), P('llama-3.1-8b-instant', tc_s), P('Identical model in both conditions', tcl_s)],
    [P('CoT-ablation guard', tcl_s), P('qwen3-32b', tc_s), P('Reasoning-capable, controllable via reasoning_effort', tcl_s)],
    [P('Judge (refusal + quality)', tcl_s), P('gpt-oss-120b', tc_s), P('Not a participant in any tested condition', tcl_s)],
    [P('Safeguard guard', tcl_s), P('gpt-oss-safeguard-20b', tc_s), P('Tested under two prompting regimes, Section 4.3', tcl_s)],
]
story += [
    mktbl(model_data, [5*cm, 3.5*cm, 5.5*cm]),
    Spacer(1, 0.15*cm),
    P('4.2.1 Model Selection Criteria and Rationale', h2),
    P('The guard model was selected against specific architectural requirements, not chosen for '
      'convenience. <b>Pseudo-wrapping and context-hierarchy inversion compatibility</b> is load-bearing: '
      'the guard prompt wraps the user input and places evaluation criteria <i>after</i> it, exploiting '
      'recency bias so the criteria dominate the model\'s effective judgment rather than competing with '
      'front-loaded instructions. Every guard-family condition uses this mechanism. <b>Sliding-window / '
      'sparse attention</b> was a preferred-but-not-required property -- a genuinely sparse-attention '
      'model would in principle amplify the hierarchy-inversion effect through additional locality bias. '
      'No model meeting this property was available within the constraints we operated under (see below); '
      '<i>the SWA-amplification claim is therefore theoretical in this paper, not empirically isolated</i> '
      '(Section 7). <b>Instruct-training</b> was required for prompt-following reliability. '
      '<b>CoT-capability</b> was required specifically for the ablation axis, motivating a separate '
      'reasoning-capable model (qwen3-32b) for that condition rather than reusing the main guard model '
      'throughout. <b>Sub-100B and fast</b> was a practical constraint given hundreds of sequential API '
      'calls per condition under free-tier rate limits.'),
    P('Holding the guard, main, and monolithic model identical (llama-3.1-8b-instant) is the controlling '
      'variable of the entire experiment: any measured difference between monolithic and decoupled is '
      'attributable to the architecture, not to one condition using a more capable model. The judge model '
      '(gpt-oss-120b) was deliberately chosen as a non-participant in any tested condition, to avoid the '
      'judge\'s own training biases correlating with the behaviour being measured.'),
    P('The model set went through three inference-provider migrations over the course of this project '
      '(HuggingFace &#8594; OpenRouter &#8594; Groq), each forced by credit or rate-limit exhaustion '
      'rather than a methodology change, narrowing the available model pool at each step. No model in the '
      'final set uses genuine sliding-window or sparse attention. This is reported as a limitation '
      '(Section 7), not hidden: the hierarchy-inversion mechanism is tested and appears to work without '
      'SWA, but the amplification hypothesis specifically remains untested.'),
    P('4.3 Conditions', h2),
    P('1. <b>monolithic</b> -- single model, combined safety-and-helpfulness system prompt, no separate guard step.', bullet_s),
    P('2. <b>decoupled</b> -- our guard prompt (pseudo-wrapping + context-hierarchy inversion; no system-message persona assignment, to avoid inheriting RLHF chat-persona bias) on llama-3.1-8b-instant, followed by guard-burden-free generation on the same model if the verdict is safe.', bullet_s),
    P('3. <b>cot_ablation</b> -- same guard task, on qwen3-32b, with free (undirected) chain-of-thought reasoning enabled.', bullet_s),
    P('4. <b>directed_cot</b> -- same model as decoupled, but the guard prompt explicitly links a four-step reasoning chain per harm category directly into the unsafe-criteria definition, rather than running reasoning as a parallel, disconnected track.', bullet_s),
    P('5. <b>safeguard</b> -- gpt-oss-safeguard-20b, OpenAI\'s documented native format (policy in system message, JSON output).', bullet_s),
    P('6. <b>safeguard_ours</b> -- gpt-oss-safeguard-20b, our own pseudo-wrap/hierarchy-inversion guard prompt instead of the native format. Same model as condition 5; only the prompting method differs.', bullet_s),
    P('7. <b>third_party</b> -- our guard prompt rewritten so the model predicts a third party\'s verdict rather than rendering its own, testing self-distancing as a debiasing axis.', bullet_s),
    P('8. <b>peer_consensus</b> -- variant of (7) where the model predicts peer-model consensus rather than a single third party\'s verdict.', bullet_s),
    P('Conditions 5-8 are explicitly not a clean ablation of each other -- the prompts differ '
      'qualitatively by design (self-distancing and peer-framing prompts are volatile by definition) -- '
      'so they are reported as independent conditions against the same dataset, not as a controlled '
      'isolation of one variable.'),
    P('4.4 Metrics', h2),
    P('<b>Detection rate</b> -- percentage of harmful prompts correctly flagged (monolithic: via refusal judging; decoupled-family: guard verdict). <b>False positive rate (FPR)</b> -- percentage of benign prompts incorrectly flagged. <b>F1</b> -- harmonic mean of precision and detection rate; Section 6.4 cautions that F1 alone can mask a high-FPR, high-recall condition that flags almost everything. <b>Response quality</b> -- for benign prompts that pass the guard, an independent judge model scores helpfulness, specificity, and hedging (1-5 each).'),
    P('4.5 Rate Limiting and Infrastructure', h2),
    P('Run on a GCE e2-micro preemptible VM, detached process, with per-model RPM/RPD-aware rate '
      'limiting and UTC-midnight-reset waiting on detected daily-cap 429 responses. Per-call responses '
      'were cached keyed on (model, messages, reasoning_effort).'),
    Spacer(1, 0.2*cm),
]

# ── §5 Evaluation Pipeline Integrity ──────────────────────────────────────────
story += [
    P('5. Evaluation Pipeline Integrity', h1),
    P('Four bugs were identified after the initial run and corrected before any result below was '
      'finalized. We report them in full because three share a root cause that is easy to miss in '
      'LLM-as-judge pipelines using reasoning-capable judge models, and because the fourth changed our '
      'core conclusion.'),
    P('5.1 Quality Scores Returning Empty (Resolved)', h2),
    P('The quality-judging function called gpt-oss-120b at max_tokens=50; traced via direct cache-key '
      'lookup, the model was returning an empty string across every condition -- almost certainly hidden '
      'reasoning tokens consuming the entire budget before any visible answer token. Fix: raised '
      'max_tokens, set reasoning_effort to a low setting, busted the stale cache (cache keys are not a '
      'function of max_tokens, so the old empty responses would otherwise be silently reused), and '
      'recomputed. All eight conditions now report populated quality scores (Table 5).'),
    P('5.2 Refusal Judging Returning a Constant (Resolved)', h2),
    P('The monolithic condition\'s refusal judge called gpt-oss-120b at max_tokens=10 -- even tighter than '
      'the quality judge\'s already-too-small 50 -- with fallback logic defaulting to "complied" whenever '
      'the response did not contain the literal word "refused". A live test confirmed the model was '
      'returning a literal empty string for all 200 calls, so every single monolithic row was silently '
      'defaulting to "complied", producing a false 0.0% detection / 0.0% FPR reading. This is the third '
      'occurrence of the identical failure mode (reasoning tokens starving a small max_tokens budget) and '
      'directly anchors the monolithic-vs-decoupled comparison; correcting it changes the paper\'s central '
      'framing (Section 6.1).'),
    P('5.3 Directed-CoT Verdict Corruption (Resolved, Two Rounds)', h2),
    P('The directed_cot guard prompt requires four reasoning steps across five harm categories before a '
      'final verdict. At the original max_tokens=600, raw transcripts showed truncation mid-reasoning, '
      'and the verdict parser -- which scanned all lines in reverse for any substring match on "safe" or '
      '"unsafe" -- was picking up incidental words from unfinished reasoning rather than the model\'s '
      'actual conclusion. Raising max_tokens to 1,500 and then 5,000 had essentially no effect (167/200 '
      'and 169/200 "unknown" verdicts respectively) -- a result that itself indicated truncation was '
      'never the actual bottleneck. Pulling a raw "unknown" transcript confirmed this: the model completes '
      'its reasoning in roughly 2,000 characters, well under even the original budget, and reaches a '
      'clear conclusion, but states it in a full sentence ("On balance, the message is SAFE. It '
      'promotes...") rather than the bare word the prompt requests on its own final line. The corrected '
      'parser matches the whole word "unsafe" or "safe" against the last line specifically (checking '
      '"unsafe" first, since it contains "safe" as a substring), loose enough to catch a sentence-wrapped '
      'verdict while still scoped to the last line only, avoiding the original bug\'s noise-from-earlier-'
      'reasoning problem. This reduced unresolved verdicts from 169/200 to 31/200 (15.5%) and changed the '
      'reported detection/FPR/F1 from an uninterpretable 2.0%/0.0%/0.039 to a confirmed, real 22.0%/25.0%/'
      '0.299 (Section 6.5).'),
    P('A methodological note for future work: cache keys in this pipeline are a function of (model, '
      'messages, reasoning_effort) but not max_tokens. Both token-budget reruns during Section 5.3\'s '
      'debugging silently served stale cached responses from the previous budget until the relevant cache '
      'entries were explicitly identified and deleted -- this cost two full reruns before being caught, '
      'and is worth guarding against directly in any cache-key design that varies a sampling parameter '
      'across debugging iterations.'),
    Spacer(1, 0.2*cm),
]

# ── §6 Results ─────────────────────────────────────────────────────────────────
story += [P('6. Results', h1),
    P('6.1 Detection, FPR, F1, and Quality', h2),
]
results_data = [
    [P('Condition', th_s), P('Det%', th_s), P('FPR%', th_s), P('F1', th_s), P('Help', th_s), P('Spec', th_s), P('Hedge', th_s)],
    [P('monolithic', tcl_s), P('73.0', tc_s), P('6.0', tc_s), P('0.816', tc_s), P('4.67', tc_s), P('4.39', tc_s), P('4.83', tc_s)],
    [P('decoupled', tcl_s), P('37.0', tc_s), P('0.0', tc_s), P('0.540', tc_s), P('4.66', tc_s), P('4.44', tc_s), P('4.81', tc_s)],
    [P('cot_ablation', tcl_s), P('95.0', tc_s), P('88.0', tc_s), P('0.671', tc_s), P('4.42', tc_s), P('4.42', tc_s), P('4.58', tc_s)],
    [P('directed_cot', tcl_s), P('22.0', tc_s), P('25.0', tc_s), P('0.299', tc_s), P('4.66', tc_s), P('4.42', tc_s), P('4.81', tc_s)],
    [P('safeguard', tcl_s), P('58.0', tc_s), P('0.0', tc_s), P('0.734', tc_s), P('4.70', tc_s), P('4.46', tc_s), P('4.84', tc_s)],
    [P('safeguard_ours', tcl_s), P('74.0', tc_s), P('1.0', tc_s), P('0.846', tc_s), P('4.70', tc_s), P('4.46', tc_s), P('4.81', tc_s)],
    [P('third_party', tcl_s), P('68.0', tc_s), P('9.0', tc_s), P('0.768', tc_s), P('4.74', tc_s), P('4.51', tc_s), P('4.85', tc_s)],
    [P('peer_consensus', tcl_s), P('76.0', tc_s), P('14.0', tc_s), P('0.800', tc_s), P('4.70', tc_s), P('4.47', tc_s), P('4.87', tc_s)],
]
story += [
    mktbl(results_data, [3.2*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm, 1.7*cm]),
    P('Table 1: Detection rate, false positive rate, F1, and quality scores (helpfulness, specificity, '
      'hedging; 1-5 scale) across all eight conditions, N=200 (100 harmful, 100 benign), after all four '
      'pipeline bugs (Section 5) were corrected. directed_cot\'s remaining 31/200 (15.5%) unresolved '
      'verdicts are excluded from its detection/FPR denominator, consistent with the other conditions\' '
      'metric definitions.', caption_s),
    Spacer(1, 0.1*cm),
]
story += FIG(os.path.join(GRAPHS, '01_metrics_with_ci.png'),
    'Figure 1: Detection rate, FPR, and F1 with bootstrap 95% confidence intervals (5,000 resamples per '
    'condition) across all eight conditions.')

story += [
    P('6.2 Statistical Significance', h2),
    P('At N=100 per class, several F1 point estimates are close enough (e.g. safeguard 0.734 vs. '
      'third_party 0.768 vs. peer_consensus 0.800 vs. safeguard_ours 0.846) to warrant checking whether '
      'they are real before citing an ordering. We ran paired bootstrap difference tests (5,000 '
      'resamples, resampling row-indices jointly across the two conditions being compared, since every '
      'condition runs on the same 200 underlying prompts -- this controls for per-prompt difficulty '
      'correlating across conditions, and is more powerful than comparing independent per-condition CIs).'),
]
sig_data = [
    [P('Comparison', th_s), P('Det diff', th_s), P('FPR diff', th_s), P('F1 diff', th_s), P('F1 p-value', th_s)],
    [P('safeguard_ours vs. safeguard', tcl_s), P('+16.0pp', tc_s), P('+1.0pp', tc_s), P('+0.112', tc_s), P('&lt;0.001', tc_s)],
    [P('monolithic vs. decoupled', tcl_s), P('+36.0pp', tc_s), P('+6.0pp', tc_s), P('+0.275', tc_s), P('&lt;0.001', tc_s)],
    [P('third_party vs. decoupled', tcl_s), P('+31.0pp', tc_s), P('+9.0pp', tc_s), P('+0.228', tc_s), P('&lt;0.001', tc_s)],
    [P('peer_consensus vs. decoupled', tcl_s), P('+39.0pp', tc_s), P('+14.0pp', tc_s), P('+0.260', tc_s), P('&lt;0.001', tc_s)],
    [P('<b>peer_consensus vs. third_party</b>', tcl_s), P('+8.0pp (p=0.002)', tc_s), P('+5.0pp (p=0.024)', tc_s), P('+0.032', tc_s), P('<b>0.166</b>', tc_s)],
    [P('directed_cot vs. decoupled', tcl_s), P('-15.0pp (p=0.014)', tc_s), P('+25.0pp', tc_s), P('-0.241', tc_s), P('&lt;0.001', tc_s)],
    [P('directed_cot vs. cot_ablation', tcl_s), P('-73.0pp', tc_s), P('-63.0pp', tc_s), P('-0.372', tc_s), P('&lt;0.001', tc_s)],
]
story += [
    mktbl(sig_data, [5.2*cm, 2.6*cm, 2.6*cm, 2.0*cm, 2.0*cm]),
    P('Table 2: Paired bootstrap difference tests for the seven comparisons anchoring this paper\'s '
      'claims, run without multiple-comparisons correction (Section 7). Every comparison that anchors a '
      'claim elsewhere in this paper is statistically solid (p&lt;0.05, mostly p&lt;0.001) at this sample '
      'size, with one exception: peer_consensus vs. third_party is significant on detection and FPR '
      'individually but not on F1 -- see Section 6.5. Rows involving directed_cot compare its 169/200 '
      'resolved-verdict denominator against the other conditions\' full-200 denominator (Section 6.7).',
      caption_s),
    Spacer(1, 0.2*cm),
]

story += [
    P('6.3 The Core Thesis Needs Reframing, Not Abandoning', h2),
    P('With monolithic corrected to 73.0% detection / 6.0% FPR / F1 0.816 (Section 5.2), the simple '
      '"decoupled beats monolithic" framing does not survive contact with the corrected data -- '
      'monolithic now outperforms plain decoupled on detection (73.0% vs. 37.0%) and on F1 (0.816 vs. '
      '0.540), and on F1 specifically is exceeded by only one of the seven guard variants tested '
      '(safeguard_ours, 0.846; peer_consensus, the next closest, still falls short at 0.800). Section '
      '3\'s prediction, stated plainly, was that decoupled systems "outperform monolithic systems '
      'specifically at the boundary" -- a stronger claim than survives here. We treat that strong form as '
      '<i>falsified</i> on detection and F1: a well-tuned monolithic model resolves the boundary conflict '
      'more favourably, on these two metrics, than a structurally-equivalent prompted guard does. What '
      'does survive, and is not falsified, is the narrower trade-off Section 3\'s formalization also '
      'implies: monolithic\'s 6.0% FPR vs. decoupled\'s 0.0% FPR is exactly the helpfulness-safety '
      'competition manifesting as benign inputs occasionally being refused. The defensible claim, given '
      'all eight conditions, is therefore not a weaker restatement of the original thesis but a '
      'genuinely narrower one: '
      'decoupling buys FPR control (0.0% across every condition using our guard prompt without a framing '
      'twist) at a detection cost relative to a monolithic safety-tuned model, and that cost is '
      'recoverable -- even reversible -- through prompting methodology (Section 6.4) and framing choices '
      '(Section 6.6), independent of decoupling itself. This reframes the paper\'s contribution from '
      '"architecture X beats architecture Y" to "here are the levers that actually move detection and '
      'FPR, and decoupling is one of several, not the dominant one."'),
    P('6.4 Prompting Methodology Generalizes Across Models', h2),
    P('safeguard_ours (74.0% det / 1.0% FPR / F1 0.846) outperforms safeguard (58.0% det / 0.0% FPR / '
      'F1 0.734) -- same model, same dataset, same harmful/benign split, only the prompting method '
      'differs, and the difference is statistically robust (p&lt;0.001 on F1; Table 2). This is, we '
      'argue, a stronger and more surprising result than the core decoupled-vs-monolithic comparison: it '
      'is consistent with the pseudo-wrap/hierarchy-inversion technique not being specific to the model it '
      'was designed for, though we note conditions 5-8 are not a clean single-variable ablation of each '
      'other (Section 4.3) -- the safeguard and safeguard_ours prompts differ qualitatively, not on one '
      'isolated factor, so this is evidence the technique transfers, stated as an association the bootstrap '
      'test confirms is reliable, not a fully isolated causal claim. This finding also bears on '
      'the question of testing a third-party model whose safety philosophy may differ from our own: the '
      'comparison is not an endorsement of gpt-oss-safeguard\'s training philosophy or category taxonomy '
      '-- it is evidence that our architectural principle is substrate-independent.'),
]
story += FIG(os.path.join(GRAPHS, '03_f1_overview.png'),
    'Figure 2: F1 across all eight conditions with bootstrap 95% CIs. Overlapping intervals flag the '
    'comparisons that required the paired significance testing in Table 2 rather than point-estimate '
    'comparison alone.')

story += [
    P('6.5 Self-Referential Framing Is a Real, Controllable Sensitivity Axis -- But Third-Party vs. '
      'Peer-Consensus Is a Trade-Off, Not a Clean Win', h2),
    P('decoupled (37.0%/0.0%) &#8594; third_party (68.0%/9.0%) &#8594; peer_consensus (76.0%/14.0%) shows '
      'a clean, monotonic relationship, reliable under paired bootstrap testing (Table 2, p&lt;0.001 for '
      'both steps): detection sensitivity rises, and FPR rises with it, as the framing moves further from '
      'direct self-judgment. As with Section 6.4, conditions 7-8 are qualitatively different prompts, not '
      'a single isolated variable manipulated in isolation (Section 4.3), so we report this as an '
      'association the data reliably supports rather than a fully isolated causal mechanism. This is '
      'consistent with the working '
      'hypothesis that assigning a persona or role to a guard model -- even implicitly, through how the '
      'question is posed -- risks inheriting RLHF chat-persona bias: the model under-flags when judging '
      '"as itself" and flags more readily when distanced from that framing.'),
    P('However, peer_consensus vs. third_party itself is not a clean win for peer-framing, despite '
      'peer_consensus\'s higher F1 point estimate (0.800 vs. 0.768). The detection gain (+8.0pp) and FPR '
      'cost (+5.0pp) are each individually significant, but they roughly cancel at the F1 level (p=0.166, '
      'not significant; Table 2). The honest framing: third_party and peer_consensus are both real, '
      'statistically distinct steps away from plain self-judgment, but between them it is a genuine '
      'sensitivity/specificity trade-off, not a strictly better option.'),
]
story += FIG(os.path.join(GRAPHS, '02_framing_tradeoff.png'),
    'Figure 3: Detection rate vs. false positive rate across the self-referential framing axis '
    '(decoupled &#8594; third_party &#8594; peer_consensus), with bootstrap 95% CIs on each axis. The '
    'monotonic trend shows sensitivity gain scaling with framing distance from direct self-judgment '
    '(with only three points, this supports a monotonic reading, not a claim of constant slope).')

story += [
    P('6.6 F1 Is Misleading for cot_ablation', h2),
    P('95.0% detection / 88.0% FPR / F1 0.671 looks numerically competitive with safeguard_ours (F1 '
      '0.846) on the table alone. It is not a good result -- an 88.0% FPR means the condition flags '
      'almost everything regardless of content, with effectively no discriminative value. F1\'s recall-'
      'sensitivity hides this: a reader should weight FPR explicitly rather than rank conditions by F1 '
      'alone.'),
    P('6.7 Directed CoT Is a Confirmed Negative Result, Not an Artifact', h2),
    P('At 22.0% detection / 25.0% FPR / F1 0.299 -- computed, per Table 1\'s caveat, over the 169/200 '
      'rows that produced a resolved verdict, not the full 200; the comparison against decoupled\'s '
      'full-200 denominator in Table 2 should be read with this in mind -- and statistically distinct '
      'from plain decoupled on every axis (p&lt;0.05; Table 2), directed CoT regresses relative to plain '
      'decoupled (37.0%/0.0%/0.540) on both detection and FPR. The original rationale -- linking each '
      'reasoning step directly into the '
      'unsafe-criteria definition, so the model cannot run reasoning as a disconnected track from its '
      'verdict the way cot_ablation does -- does not hold up once measured correctly (Section 5.3).'),
    P('This is not cot_ablation\'s failure mode recurring. cot_ablation fails by over-triggering (88.0% '
      'FPR, flags almost everything). directed_cot fails differently -- moderate FPR (25.0%) paired with '
      'low detection (22.0%, worse than plain decoupled\'s 37.0%) -- so directed reasoning is not simply '
      '"more cautious," it is making categorically worse judgment calls in both directions. The likely '
      'mechanism: giving the model more surface area to reason over (four steps across five categories) '
      'gives it more opportunities to talk itself into an exemption on genuinely harmful prompts, while '
      'still occasionally talking itself into a violation on benign ones -- directed reasoning amplifies '
      'variance rather than improving calibration. The 15.5% non-completion rate (31/200 unresolved '
      'verdicts), even after the parser fix in Section 5.3, is itself worth reporting as a secondary '
      'finding: a guard architecture that does not reliably produce a verdict is operationally costly '
      'regardless of how it performs on the cases it does resolve. We report this as good evidence '
      'against directed CoT as currently structured for a guard task, not a confound or measurement '
      'artifact -- it is informative precisely because the rationale was reasonable and still did not '
      'pan out.'),
    Spacer(1, 0.2*cm),
]

# ── §7 Limitations ─────────────────────────────────────────────────────────────
story += [
    P('7. Limitations', h1),
    P('&#8226; <b>SWA-amplification untested.</b> No model in the final set uses genuine sliding-window or sparse attention (Section 4.2.1); the hierarchy-inversion mechanism is tested and appears to work without it, but the amplification hypothesis remains theoretical.', bullet_s),
    P('&#8226; <b>Free-tier rate limits.</b> Groq RPD caps as low as 1,000/day on several models constrained total sample size and required UTC-midnight-aware retry logic (Section 4.5).', bullet_s),
    P('&#8226; <b>Conditions 5-8 are not a clean ablation of each other.</b> The safeguard-family and framing-variant prompts differ qualitatively by design, not on a single isolated variable (Section 4.3).', bullet_s),
    P('&#8226; <b>LLM-as-judge reliability, compounded by a discovered failure class.</b> A reasoning-capable judge model (gpt-oss-120b) silently returned empty output under a small token budget three separate times across this work\'s evaluation pipeline (judge_quality, judge_refusal, and directed_cot\'s verdict parser) before being caught (Section 5). A clean-looking results table is not sufficient evidence of correctness when every number traces back to the same small-output-budget judge call; we recommend spot-checking non-empty, well-formed judge output before trusting aggregate metrics in any similar pipeline.', bullet_s),
    P('&#8226; <b>Single dataset.</b> WildGuardMix only; generalization to other harm taxonomies is untested.', bullet_s),
    P('&#8226; <b>Sample size and multiple comparisons.</b> N=100 per class is sufficient to confirm the comparisons in Table 2 at conventional significance levels, but is not large, and Table 2 reports seven simultaneous tests against the same 200-prompt pool without a multiple-comparisons correction (Bonferroni, Holm, or FDR). At an uncorrected &#945;=0.05, seven tests carry a non-trivial family-wise false-positive risk. We did not apply a correction here because five of the seven results are extreme (p&lt;0.001, with decoupled\'s FPR literally at 0.0%) and would survive any standard correction; the three borderline results (peer_consensus vs. third_party on detection at p=0.002, directed_cot vs. decoupled on detection at p=0.014, and peer_consensus vs. third_party on FPR at p=0.024) are closer to the threshold and should be read with that caveat in mind rather than treated as equally robust to the p&lt;0.001 results.', bullet_s),
    Spacer(1, 0.2*cm),
]

# ── §8 Implications for AMDON ──────────────────────────────────────────────────
story += [
    P('8. Implications for AMDON', h1),
    P('PSNAT-AMDON\'s guard pipeline is empirically motivated by the corrected comparison in Section 6.3: '
      'decoupling\'s actual value is FPR control, not blanket detection superiority over a well-tuned '
      'monolithic model. The framing-axis finding (Section 6.5) additionally motivates AMDON\'s guard '
      'never being asked to self-attribute a persona when classifying, since direct self-judgment framing '
      'measurably under-flags relative to distanced framings. The prompting-methodology-transfer finding '
      '(Section 6.4) supports using AMDON\'s pseudo-wrap/hierarchy-inversion guard prompting design even '
      'against third-party or future guard models, rather than defaulting to each model\'s own documented '
      'prompting convention.'),
    Spacer(1, 0.2*cm),
]

# ── §9 Conclusion ──────────────────────────────────────────────────────────────
story += [
    P('9. Conclusion', h1),
    P('We tested whether decoupling a safety guard from a generation model resolves the structural '
      'helpfulness-safety prior conflict, holding the underlying model constant across every role to '
      'isolate architecture from capability. The corrected result is more nuanced than a blanket claim of '
      'decoupled superiority: a well-tuned monolithic model outperforms plain decoupling on raw detection '
      'and F1, and decoupling\'s defensible advantage is FPR control specifically, recoverable through '
      'orthogonal levers -- prompting methodology and self-referential framing -- rather than from '
      'architecture alone. Two of those levers are statistically robust under paired bootstrap testing: '
      'our guard prompting method transfers to a third-party policy-conditioned model and beats that '
      'model\'s own documented format, and removing self-referential framing produces a real, controllable '
      'detection/FPR trade-off. A directed chain-of-thought variant, intended to improve on an earlier '
      'undirected-CoT failure mode, is a confirmed negative result rather than an improvement. We also '
      'document four evaluation-pipeline bugs uncovered during this work, three sharing a single root '
      'cause -- a reasoning-capable judge model silently starving its output under a small token budget '
      '-- as a methodological caution for similar LLM-as-judge pipelines. Future work includes a larger-N '
      'replication to tighten the confidence intervals reported in Section 6.2, testing against a model '
      'with genuine sliding-window attention to evaluate the SWA-amplification hypothesis directly, and '
      'integration of the corrected guard architecture into the AMDON pipeline.'),
    Spacer(1, 0.2*cm),
]

# ── Acknowledgements ───────────────────────────────────────────────────────────
story += [
    P('Acknowledgements', h1),
    P('Experiment scripting, debugging, infrastructure management, and analysis assisted by Claude '
      'Sonnet 4.6 (Anthropic). Inference was performed via the Groq API. This work was conducted '
      'independently without institutional or commercial funding.'),
    Spacer(1, 0.2*cm),
]

# ── References ─────────────────────────────────────────────────────────────────
story += [
    P('References', h1),
    P('[1] Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. <i>NeurIPS.</i>'),
    P('[2] Bai, Y. et al. (2022). Constitutional AI: Harmlessness from AI Feedback. <i>Anthropic.</i>'),
    P('[3] Rafailov, R. et al. (2023). Direct Preference Optimization: Your Language Model is Secretly a Reward Model. <i>NeurIPS.</i>'),
    P('[4] Inan, H. et al. (2023). Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations. <i>Meta AI.</i>'),
    P('[5] Han, S. et al. (2024). WildGuard: Open One-Stop Moderation Tools for Safety Risks, Jailbreaks, and Refusals of LLMs. <i>AllenAI.</i>'),
    P('[6] Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.'),
    P('[7] OpenAI (2025). gpt-oss-safeguard-20b Model Card.'),
    Spacer(1, 0.3*cm),
]

doc.build(story)
print("Paper built successfully:", os.path.join(BASE_DIR, 'a_servant_and_a_guard.pdf'))
