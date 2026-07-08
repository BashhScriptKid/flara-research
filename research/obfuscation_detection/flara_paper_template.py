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
GRAPHS = os.path.join(BASE_DIR, 'graphs', 'final')
OBF_GRAPHS = os.path.join(GRAPHS, 'obf_analysis')

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
    max_h = PAGE_H * 0.32
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
    os.path.join(BASE_DIR, 'delta_flara.pdf'), pagesize=A4,
    leftMargin=MARGIN, rightMargin=MARGIN,
    topMargin=MARGIN, bottomMargin=MARGIN,
    title='Observing Obfuscation In Multiple Different Angles',
    author='Flara Research Lab',
)

story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story += [
    Spacer(1, 0.2*cm),
    P('Observing Obfuscation In Multiple Different Angles', title_style),
    P('Ensemble Detection of Obfuscated Prompt Injection', subtitle_sty),
    Spacer(1, 0.3*cm),
    P('Bashh Dazer', author_style),
    P('Flara Research Lab · dev@bashh.slmail.me', affil_style),
    HRFlowable(width="100%", thickness=0.5, color=colors.HexColor('#1a1a2e')),
    Spacer(1, 0.3*cm),
]

# ── Abstract ──────────────────────────────────────────────────────────────────
story += [
    P('Abstract', abs_lbl),
    P(
        'Obfuscated prompt injection induces a measurable structural signature: encoding-based payloads '
        'tend to fragment into multiple semantically distinct chunks at sentence boundaries, while '
        'benign text remains predominantly single-chunk (99% vs 28.3% multi-chunk rate). We test whether '
        'embedding geometry can exploit this signature, and how much value it adds beyond a trivial '
        'structural feature such as chunk count. We present a four-feature ensemble detector combining '
        'sentence- and paragraph-level embedding delta angles, an optimised regex coverage signal, and a '
        'single-chunk Unicode anomaly bonus, evaluated on N=1,152 samples across 21 obfuscation types '
        '(Section 4.1). The ensemble achieves AUC 0.974, F1 0.960, and 94.7% detection at 1% FPR on the '
        'full dataset (weights and threshold also fit on the full dataset); a 70/30 held-out split with '
        'weights fit on the training portion only reaches AUC 0.958 (Section 4.4), confirming the result '
        'generalises rather than overfits. Headline figures throughout this paper are the full-dataset '
        'numbers unless stated otherwise. The ensemble also achieves 100% detection of hex and '
        'backslash-visual encoding (Table 2).'),
    P(
        'Delta angle correlates substantially with chunk count and word count (r=0.68, r=0.47) and '
        'weakly with character-level features such as entropy (r=-0.08): a direct substitution test '
        'shows the embedding signal still beats raw chunk count, but by a modest, real margin (1.0-1.4pp '
        'AUC) that is larger for some embedding models than the production model (Section 5.4). Two of '
        'the four ensemble features do not mean what their formula weight implies: paragraph delta '
        'carries substantial weight despite a near-chance standalone AUC, because regex and the Unicode '
        'bonus already cover every multi-paragraph attack in this dataset; sentence delta carries a '
        'small weight yet is costly to remove (Section 5.2). The Unicode anomaly signal addresses 46.9% '
        'of the single-chunk blind spot at zero false positives. A re-optimised character entropy '
        'baseline is a stronger competitor than previously reported (AUC 0.722) but shares regex\'s blind '
        'spot on the multi-chunk attack types delta angle catches.',
        abs_body
    ),
    P(
        'An adaptive-attack evaluation (30 representative samples, 20 evasion strategies) finds that '
        'plain natural-language prompt injection evades the detector entirely -- 20/30 (66.7%) at 1% '
        'FPR -- with catches concentrated almost exclusively in Unicode/homoglyph and structural tricks. '
        'This sharpens the threat model: the regex keyword list targets encoding vocabulary (decode, '
        'translate, payload), not injection phrasing (ignore, override, system prompt), so '
        'social-engineering injection is out of scope for this detector and requires a separate '
        'intent-detection layer. Embedding geometry therefore provides a modest but real advantage over '
        'chunk-counting for detecting structural obfuscation specifically; it is not a general '
        'prompt-injection detector (Appendix C), and is designed as a Stage 1 preprocessing filter in a '
        'defence-in-depth architecture, not a standalone classifier.',
        abs_body
    ),
    HRFlowable(width="100%", thickness=0.3, color=colors.HexColor('#cccccc')),
    Spacer(1, 0.4*cm),
]

# ── §1 Introduction ───────────────────────────────────────────────────────────
story += [
    P('1. Introduction', h1),
    P('Obfuscated prompt injections -- attacks that encode their payload through character encoding '
      '(hex, base64, backslash escape sequences) or structural techniques (dot-spacing, Unicode '
      'substitution) to evade detection -- often induce a measurable multi-segment structural signature '
      'that benign text does not share. We investigate whether embedding geometry can exploit this '
      'signature, and whether doing so adds detection value beyond a trivial structural feature such as '
      'chunk count. As language models are increasingly deployed as autonomous agents with access to '
      'tools, memory, and external services, detecting this class of injection before it reaches the '
      'model is a practical, narrow problem worth solving precisely, rather than folding into a general '
      'prompt-injection classifier.'),
    P('Pattern-based regex handles encoding-based attacks reasonably but has no mechanism for any '
      'technique that produces no distinctive character-level signature -- it cannot see chunk '
      'structure. We introduce a complementary signal, the <b>delta angle</b> between consecutive '
      'embedding chunks of the input text: encoding-based payloads generate multiple semantically '
      'distinct chunks with measurable angular separation, while benign inputs are predominantly '
      'single-chunk (99% vs 28.3% multi-chunk rate). This work grew out of an earlier, more general '
      'attempt to use delta angle as a standalone anti-injection measure, which did not separate normal '
      'from injected prompts reliably; Appendix C documents that prior work and the reasoning behind '
      'narrowing the scope to obfuscation detection specifically.'),
    P('Our main findings are:'),
    P('• The structural signature is real and exploitable: a four-feature ensemble (sentence- and '
      'paragraph-level delta angle, optimised regex, single-chunk Unicode anomaly bonus) reaches AUC '
      '0.974, F1 0.960, 94.7% detection at 1% FPR (Section 5.2).', bullet_s),
    P('• Delta angle correlates substantially with chunk and word count (r=0.68, r=0.47); the embedding '
      'signal beats a trivial chunk-counter, but by a modest, directly quantified margin rather than '
      'full independence from input structure -- and the margin is model-dependent (Section 5.3-5.4).', bullet_s),
    P('• Ensemble weight magnitude is not a reliable proxy for a feature\'s marginal contribution: two '
      'of the four features (paragraph delta, sentence delta) each contradict what their formula weight '
      'would suggest, in opposite directions (Section 5.2).', bullet_s),
    P('• The Unicode anomaly signal addresses 46.9% of the single-chunk blind spot at zero false '
      'positives (Section 6.2); eleven further geometric extensions were tested and did not improve on '
      'the production ensemble (Appendix B).', bullet_s),
    P('• An adaptive red-team evaluation (30 representative attacks, 20 evasion strategies) shows the '
      'detector\'s blind spot is plain natural-language injection, not exotic encodings -- it is a '
      'Stage 1 structural filter, not a general prompt-injection classifier (Section 5.6, Section 7).', bullet_s),
    Spacer(1, 0.2*cm),
]

# ── §2 Related Work ───────────────────────────────────────────────────────────
story += [
    P('2. Related Work', h1),
    P('Prompt injection as a named threat was introduced by Perez and Ribeiro (2022), with subsequent '
      'taxonomy work by Greshake et al. (2023) distinguishing direct and indirect injection. The '
      'HackAPrompt dataset (Schulhoff et al., 2023) provides a benchmark for injection resistance, '
      'though it does not focus specifically on encoding-based obfuscation.'),
    P('Existing detection approaches fall into two categories. <i>Input-side detectors</i> analyse '
      'user input before it reaches the model, including perplexity-based filters (Alon and Kamfonas, '
      '2023), instruction-following probes, and classifier-based approaches. <i>Output-side detectors</i> '
      'analyse model behaviour for signs of successful injection. Our approach is an input-side '
      'detector operating on structural properties of the input text.'),
    P('The use of embedding geometry for text analysis has precedent in semantic similarity, anomaly '
      'detection, and out-of-distribution detection. To our knowledge, applying consecutive embedding '
      'chunk angles as an obfuscation signal is novel. The closest related work uses perplexity or '
      'entropy as input-side signals; we show that delta angle captures independent signal not available '
      'from these measures.'),
    Spacer(1, 0.2*cm),
]

# ── §3 Method ─────────────────────────────────────────────────────────────────
story += [
    P('3. Method', h1),
    P('3.1 Sentence Delta Angle', h2),
    P('Given an input text <i>x</i>, we split it into chunks using sentence boundaries '
      '(regex: <font face="Courier" size="8">(?&lt;=[.!?])\\s+</font>), followed by clause '
      'boundaries for sentences exceeding 50 characters. Chunks below 8 words are merged with '
      'their neighbour. The merge threshold of 8 words was selected via distributional analysis: '
      'the chunk length distribution shows a natural density trough at the 7 to 8 word boundary, '
      'with 40.8% of comma-split chunks falling at exactly 7 words (Figure 3).'),
    P('Each chunk is embedded via the NVIDIA NIM API using '
      '<font face="Courier" size="8">nvidia/nv-embedqa-e5-v5</font> (4096 dimensions). '
      'The unsigned angle between consecutive chunk embeddings is:'),
    P('<font face="Courier" size="9">theta(i) = arccos( dot(e_i, e_(i+1)) / ( ||e_i|| * ||e_(i+1)|| ) )</font>',
      S('EQ', fontSize=9.5, fontName='Courier', alignment=TA_CENTER, spaceAfter=6, spaceBefore=4)),
    P('The sentence delta is the mean of all consecutive pair angles, excluding zero-angle pairs. '
      'Unsigned angles are used throughout; Appendix B.3 confirms that signed angles produce '
      'model-dependent noise.'),
    P('3.2 Paragraph Delta Angle', h2),
    P('Paragraph delta is computed identically but uses paragraph boundaries (blank lines) as chunk '
      'delimiters, with no further word-count merging. This produces coarser, sparser chunking than '
      'the sentence-level signal: most inputs contain no blank-line break at all, so paragraph delta '
      'is defined (non-zero) for only 22/1,152 samples in our dataset (Section 5.2), versus the '
      'majority-multi-chunk rate of sentence delta on obfuscated text (Section 4.1). Across the full '
      'dataset, sentence and paragraph delta correlate at 0.39, but this is driven mostly by which '
      'samples are multi-chunk at all; restricted to only the 22 samples where paragraph delta is '
      'actually non-zero, the correlation rises to 0.64 (n too small to treat as precise, but the two '
      'signals are evidently related, not independent, where paragraph delta applies). Whether '
      'paragraph delta adds anything beyond what sentence delta and the other features already '
      'capture is an empirical question, addressed in Section 5.2.'),
    P('3.3 Regex Coverage Signal', h2),
    P('We define 9 regex patterns targeting encoding artifacts and instruction-keyword density. '
      'Each pattern produces a 0-1 float score proportional to matched character coverage. '
      'The final regex score is a weighted sum with weights optimised via differential evolution '
      'targeting detection rate at 1% FPR (Table 1).'),
]

regex_data = [
    [P('Pattern', th_s), P('Target', th_s), P('Weight', th_s)],
    [P('special_chars', tc_s),         P('Non-alphanumeric character ratio', tcl_s),         P('0.179', tc_s)],
    [P('digit_escape', tc_s),          P(r'Digit escapes (\32, \97)', tcl_s),                P('0.159', tc_s)],
    [P('backslash_density', tc_s),     P('Backslash density (count / total chars)', tcl_s), P('0.146', tc_s)],
    [P('repetition', tc_s),            P('Line repetition ratio', tcl_s),                    P('0.145', tc_s)],
    [P('hex', tc_s),                   P(r'Hex escapes (\x41\x42)', tcl_s),                  P('0.125', tc_s)],
    [P('consecutive_backslash', tc_s), P('Consecutive backslash chains', tcl_s),             P('0.090', tc_s)],
    [P('keywords', tc_s),              P('29 attack-instruction terms plus 6 connective patterns', tcl_s), P('0.079', tc_s)],
    [P('backslash', tc_s),             P(r'Backslash-visual encoding (\I\ \h\a\v\e)', tcl_s), P('0.070', tc_s)],
    [P('no_alpha', tc_s),              P('Pure symbolic / numeric content', tcl_s),          P('0.007', tc_s)],
]
from reportlab.platypus import KeepTogether
regex_tbl = mktbl(regex_data, [3.8*cm, 9*cm, 1.7*cm])
story += [
    KeepTogether([regex_tbl,
    P('Table 1: Regex patterns and differential-evolution-optimised weights (sorted by weight; '
      're-optimised against the current N=1,152 dataset).', caption_s)]),
    Spacer(1, 0.1*cm),
    P('3.4 Unicode Anomaly Signal', h2),
    P('For single-chunk inputs (delta=0), we add a bonus signal targeting unusual Unicode character '
      'ranges that appear in obfuscation but rarely in benign text: Mathematical Alphanumeric '
      '(U+1D400-U+1D7FF), Fullwidth Latin (U+FF01-U+FF5E), enclosed characters (U+2460-U+24FF), '
      'superscript/subscript (U+00B2-U+209F), zero-width characters (U+200B-U+2064), combining '
      'diacritics (U+0300-U+036F), and miscellaneous symbols (U+2190-U+2BFF). This signal is applied '
      'only when sentence delta is zero, addressing the single-chunk blind spot without affecting '
      'multi-chunk detection; its measured contribution is reported in Section 6.2.'),
    P('3.5 Ensemble Combination', h2),
    P('The four features are combined via linear weighting, with weights optimised by differential '
      'evolution to maximise TPR-FPR at the 99th-percentile-of-benign threshold (1% FPR target):'),
    P('<font face="Courier" size="8">score = 0.0283 x sentence_delta + 0.4493 x regex + 0.1987 x paragraph_delta + 0.3237 x weird_unicode</font>',
      S('EQ2', fontSize=8.5, fontName='Courier', alignment=TA_CENTER, spaceAfter=6, spaceBefore=4)),
    P('A sample is flagged if its score exceeds this threshold. The weird_unicode term is only '
      'non-zero for single-chunk inputs (Section 3.4); the other three terms are normalised to [0,1] '
      'via min-max scaling before weighting. Section 5.2 examines what each weight actually buys the '
      'ensemble -- weight magnitude and marginal contribution are not the same thing, and both '
      'paragraph delta and sentence delta turn out to be informative counterexamples to reading the '
      'formula\'s coefficients at face value.'),
    Spacer(1, 0.2*cm),
]

# ── §4 Experimental Setup ─────────────────────────────────────────────────────
story += [
    P('4. Experimental Setup', h1),
    P('4.1 Dataset', h2),
    P('Our dataset comprises N=1,152 samples: 339 obfuscation samples and 813 benign prompts covering '
      'business strategy, coding questions, recipes, and general queries. The obfuscation set is '
      'independently constructed, not bulk-imported from a public corpus; 83 of the 813 benign prompts '
      'overlap with the public Threat-Matrix dataset (Neuralchemy, 2025), the remainder are '
      'independently sourced. Appendix A documents a dataset-provenance check, since an earlier draft '
      'of this paper misattributed the obfuscation set to that public corpus.'),
    P('Four obfuscation categories were originally underpowered (n<=5: task_hijack, substitution_game, '
      'extraction_trick, base64), limiting their per-type claims to directional findings. We expanded '
      'these with 52 additional real (not synthetic) attack samples, manually filtered for relevance '
      'and near-duplicates, from a separate, unused 1,883-sample file already present in this project. '
      'This took task_hijack/substitution_game/extraction_trick/base64 from n=5/5/4/3 to n=17/20/15/10 '
      '(the dataset\'s own classifier places a few samples differently than intended, e.g. into '
      'pwned_spacing or misc, since category assignment follows fixed pattern-matching rules, not '
      'curation intent). The obfuscation set spans 21 attack types; the distribution remains '
      'hex-dominated (162/339, 47.8%) with long-tail coverage of semantic and structural techniques.'),
]

ds_data = [
    [P('Type', th_s), P('N', th_s), P('% Passed@1%FPR<br/>(Delta, combined)', th_s), P('Note', th_s)],
    [P('hex', tc_s),           P('162', tc_s), P('96%', tc_s), P('—', tc_s)],
    [P('backslash_visual', tc_s), P('52', tc_s), P('35%', tc_s), P('Lower than earlier dataset version', tcl_s)],
    [P('misc', tc_s),          P('22', tc_s),  P('18%', tc_s),  P('Mostly short/ambiguous', tcl_s)],
    [P('substitution_game', tc_s), P('20', tc_s), P('95%', tc_s), P('Expanded from n=5', tcl_s)],
    [P('task_hijack', tc_s),   P('17', tc_s),   P('94%', tc_s), P('Expanded from n=5', tcl_s)],
    [P('pwned_spacing', tc_s), P('15', tc_s),  P('60%', tc_s),  P('—', tc_s)],
    [P('extraction_trick', tc_s), P('15', tc_s), P('47%', tc_s), P('Expanded from n=4', tcl_s)],
    [P('base64', tc_s),        P('10', tc_s),   P('60%', tc_s),  P('Expanded from n=3', tcl_s)],
    [P('translation_chain', tc_s), P('10', tc_s), P('30%', tc_s), P('—', tc_s)],
    [P('language_mixing', tc_s), P('6', tc_s), P('50%', tc_s), P('n<=5, directional only', tcl_s)],
    [P('9 other types', tc_s), P('10', tc_s), P('varies', tc_s), P('n=1-2 each', tcl_s)],
]
story += [
    mktbl(ds_data, [4.5*cm, 1.5*cm, 2.5*cm, 3.5*cm]),
    P('Table 2: Obfuscation type distribution (N=339) with detection rate at 1% FPR for the delta-angle '
      'signal alone (sentence+paragraph combined), not the full ensemble -- see Figure 3 for the '
      'Ensemble column, which closes most of the gaps shown here. Only the 9 other types remain '
      'n<=5/directional (language_mixing, at n=6, is thin but just above that threshold); the four '
      'previously-underpowered categories above are now reportable. Note that AUC and detection-rate '
      'figures here are at one fixed operating point, not an exhaustive statistical guarantee even at '
      'n=15-20.', caption_s),
    Spacer(1, 0.1*cm),
    P('<i>Caveat:</i> the hex and backslash_visual type labels are assigned by substring rules '
      '(presence of <font face="Courier">\\x</font> escapes, backslash count) that overlap directly '
      'with the regex features the detector uses. Detection rates for these two types are therefore '
      'not independent of the labeling scheme and should be read as a check that the regex feature '
      'fires where it is designed to, not as an out-of-sample measurement. The remaining types are '
      'labeled by distinct semantic/structural rules and are not subject to this caveat.'),
    P('All embeddings were computed via NVIDIA NIM API and cached to disk for reproducibility. '
      'Three embedding models were evaluated: nv-embedqa-e5-v5 (primary), '
      'llama-nemotron-embed-1b-v2, and bge-m3.'),
    P('4.2 Baselines', h2),
    P('We compare against: character entropy, special character ratio, regex hex-only, '
      'keyword-only regex, full 9-pattern regex, sentence delta alone, and paragraph delta alone.'),
    P('4.3 Evaluation Protocol', h2),
    P('All thresholds and weights were optimised on the full N=1,152 dataset. We report AUC-ROC, '
      'F1, and detection rate at fixed FPR=1% and FPR=5%. For adaptive evaluation, 30 representative '
      'adversarial samples spanning 20 evasion strategies were constructed and evaluated at both FPR '
      'targets (Section 5.6).'),
    P('A 70/30 stratified held-out validation of the full headline architecture was additionally '
      'performed, with weights optimised on the training split only; see Section 4.4. Headline '
      'metrics elsewhere in this paper are reported on the full dataset, with the held-out test '
      'confirming generalisation (max gap 6.86pp at the 1% FPR operating point).'),
    Spacer(1, 0.2*cm),
]

# ── §4.4 Held-Out Validation ──────────────────────────────────────────────────
story += [
    P('4.4 Held-Out Validation', h2),
    P('To verify that our results generalise to unseen data and are not artefacts of optimising '
      'on the full dataset, we performed a held-out validation of the actual headline architecture '
      '(sentence delta + paragraph delta + 9-pattern regex + weird-unicode bonus). We split the '
      'data into a training portion (70%, N=797) and a test portion (30%, N=355), optimised both '
      'the regex sub-weights and the top-level ensemble weights via differential evolution using '
      'only the training portion, fit the detection threshold on training only, then measured '
      'performance on the test portion with no further fitting.'),
    P('Held-out test results track the training split reasonably, with the largest gap at the '
      'stricter 1% FPR operating point -- wider than before the category expansion (Section 4.1), '
      'consistent with the newly-added categories being thinner per split (e.g. extraction_trick at '
      'n=15 splits roughly 10/5 train/test):'),
]
val_data = [
    [P('Metric', th_s), P('Train (N=797)', th_s), P('Test, Held-Out (N=355)', th_s), P('Gap', th_s)],
    [P('AUC-ROC', tcl_s), P('0.9809', tc_s), P('0.9584', tc_s), P('-2.25pp', tc_s)],
    [P('F1', tcl_s), P('0.9669', tc_s), P('0.9315', tc_s), P('-3.54pp', tc_s)],
    [P('FPR', tcl_s), P('1.05%', tc_s), P('2.46%', tc_s), P('+1.41pp', tc_s)],
    [P('TPR', tcl_s), P('96.05%', tc_s), P('91.89%', tc_s), P('-4.16pp', tc_s)],
    [P('Det@1%FPR', tcl_s), P('96.05%', tc_s), P('89.19%', tc_s), P('-6.86pp', tc_s)],
    [P('Det@5%FPR', tcl_s), P('96.93%', tc_s), P('93.69%', tc_s), P('-3.24pp', tc_s)],
]
story += [
    mktbl(val_data, [4*cm, 4*cm, 4.5*cm, 2.5*cm]),
    P('Table 5: Held-out validation of the headline 4-feature ensemble, with both regex sub-weights '
      'and top-level ensemble weights optimised on the training split only. Maximum generalisation '
      'gap is 6.86 percentage points (Det@1%FPR), the stricter of the two FPR targets, reflecting the '
      'smaller per-split N now available for the '
      'newly-expanded categories. For reference, the separately-optimised full-dataset model (Table '
      '3, not directly comparable since it is tuned on all N=1,152) achieves AUC=0.974 -- between '
      'this validation\'s train (0.981) and test (0.958) figures, still consistent with genuine '
      'generalisation rather than overfitting, though with a wider margin than before.', caption_s),
    Spacer(1, 0.1*cm),
    P('Confusion matrix on test split: 238 true negatives, 102 true positives, '
      '6 false positives, 9 false negatives.'),
    P('The per-feature weights chosen on the training split differ from the full-dataset weights '
      '(Section 3.5) by more than the headline metrics alone would suggest: paragraph delta receives '
      '0.387 on the training split versus 0.199 on the full dataset, the second-highest weight after '
      'regex (0.423 train vs 0.449 full); sentence delta receives 0.043 train vs 0.028 full. The '
      'detection metrics are stable across the split (Table 5), but the weight vector the optimiser '
      'converges to is not -- consistent with Section 5.2\'s finding that ensemble weight magnitude is '
      'a poor proxy for a feature\'s marginal value: the objective surface has more than one '
      'similarly-scoring weight vector, and which one the optimiser lands on depends on exactly which '
      'samples are available to optimise against.'),
    Spacer(1, 0.2*cm),
]

story += [P('5. Results', h1),
    P('5.1 Delta Angle Signal', h2),
    P('Figure 1 shows the delta angle distribution for benign (mean=0.0071) vs obfuscation (mean=0.2669) '
      'inputs. Benign inputs are overwhelmingly zero-delta (99% single-chunk), while obfuscation '
      'inputs show a broad distribution with elevated angles. This binary structural signature '
      'is the primary detection mechanism.'),
]
story += FIG(os.path.join(GRAPHS, '01_distributions_e5.png'),
    'Figure 1: Delta angle distributions (sentence chunking, E5), N=1,152 post category-expansion. '
    'Benign inputs cluster near zero (99% single-chunk, mean=0.0071); obfuscation inputs show '
    'elevated angles (mean=0.2669). Log-scale violin plot (right) illustrates the spread.')

story += [
    P('5.2 Overall Performance', h2),
]
main_data = [
    [P('Method', th_s), P('AUC', th_s), P('F1', th_s), P('Det@1%FPR', th_s), P('Det@5%FPR', th_s)],
    [P('Character entropy',    tcl_s), P('0.722', tc_s), P('—',     tc_s), P('53%',   tc_s), P('67%',   tc_s)],
    [P('Special char ratio',   tcl_s), P('0.728', tc_s), P('—',     tc_s), P('20%',   tc_s), P('21%',   tc_s)],
    [P('Regex (hex only)',      tcl_s), P('0.729', tc_s), P('—',     tc_s), P('46%',   tc_s), P('46%',   tc_s)],
    [P('Regex (keywords only)', tcl_s), P('0.817', tc_s), P('—',    tc_s), P('4%',  tc_s), P('71%',   tc_s)],
    [P('Regex (9 patterns, weighted)',   tcl_s), P('0.957', tc_s), P('—',     tc_s), P('88%',  tc_s), P('90%',   tc_s)],
    [P('Sentence delta only',   tcl_s), P('0.851', tc_s), P('—',    tc_s), P('72%',  tc_s), P('72%',   tc_s)],
    [P('Paragraph delta only',  tcl_s), P('0.530', tc_s), P('—',   tc_s), P('6%',    tc_s), P('6%',   tc_s)],
    [P('<b>Ensemble (4-feature, full)</b>', tcl_s), P('<b>0.974</b>', tc_s),
     P('<b>0.960</b>', tc_s), P('<b>95%</b>', tc_s), P('<b>96%</b>', tc_s)],
]
story += [
    mktbl(main_data, [5.5*cm, 2*cm, 2*cm, 2.5*cm, 2.5*cm]),
    P('Table 3: Method comparison against the current N=1,152 dataset; standard ROC-AUC throughout, '
      'with sign convention corrected for entropy/special-char baselines. Character entropy is a '
      'real individual competitor, but shares regex\'s blind spot for the multi-chunk attack types '
      'delta angle catches (§5.3). The ensemble outperforms all individual signals.', caption_s),
    Spacer(1, 0.1*cm),
    P('Paragraph delta\'s standalone AUC (0.530) looks close to chance, but is computed over a '
      'population where the feature is undefined: only 22/1,152 samples (21 obfuscation, 1 benign) '
      'are actually multi-paragraph, the rest score exactly 0 for both classes by construction. '
      'Removing paragraph delta from the ensemble entirely and re-optimising the remaining three '
      'features changes AUC by -0.00003 (statistically indistinguishable from zero), and every one '
      'of the 21 multi-paragraph obfuscation samples is still caught at the production threshold '
      'either way -- zero flips. This is best read as a coverage-overlap finding specific to this '
      'dataset (regex and the Unicode bonus already independently catch every multi-paragraph attack '
      'sample present), not evidence that paragraph structure carries no information; a '
      'dataset containing multi-paragraph attacks that evade those two features would likely show '
      'paragraph delta earning its formula weight. Sentence delta shows the opposite pattern: a small '
      'formula weight (0.0283) that nonetheless costs 1.4pp AUC to remove (above). Ensemble weight '
      'magnitude is not a reliable stand-in for a feature\'s marginal contribution in this '
      'architecture -- both directions of error appear in the same formula, and only direct removal '
      'testing distinguishes them. We retain paragraph delta in the production formula for '
      'completeness and for datasets where multi-paragraph attacks might evade regex and the Unicode '
      'bonus, even though it contributes negligibly on the present benchmark; this is a deliberate '
      'choice to keep stated, not an oversight.'),
]
story += FIG(os.path.join(GRAPHS, '05_baseline_comparison.png'),
    'Figure 2: Baseline comparison -- recall at 5% FPR and F1, computed live against the current '
    'dataset. Special char ratio is weak (0.206 recall); a minimal 2-pattern hex/base64-only regex '
    '(distinct from the full 9-pattern regex in Table 3) is moderate (0.501); character entropy is '
    'a real competitor (0.667, F1 0.746, ahead of the minimal regex); delta angle alone is strongest '
    'here (0.717, F1 0.824), and the full ensemble (Table 3) outperforms all of them.')

story += [P('5.3 Per-Attack-Type Analysis', h2),
    P('Figure 3 shows the detection rate heatmap across attack types and methods at FPR=1%, now '
      'including the expanded task_hijack/substitution_game/extraction_trick/base64 categories '
      '(Section 4.1). Delta angle uniquely catches the task_hijack (94% vs regex near 0%) and '
      'substitution_game (95% vs regex near 0%) attack types that regex mostly misses -- both '
      'reliably multi-chunk types (Appendix F), so this is best read as a chunking-signature win, '
      'not evidence of semantic understanding. '
      'language_mixing and translation_chain, previously reported at 100%/80% on the smaller '
      'pre-expansion dataset, now read 50%/30% with the recalibrated threshold and benign '
      'distribution that come with the larger N -- these two categories were not themselves '
      'expanded (n=6 and n=10, unchanged), so this is threshold drift from the larger dataset, not '
      'new evidence about these specific types. Regex and delta angle remain complementary, '
      'non-overlapping signals overall.'),
]
story += FIG(os.path.join(OBF_GRAPHS, '04_detection_heatmap_method.png'),
    'Figure 3: Detection rate by obfuscation type x method at fixed FPR=1%, including the full '
    '4-feature Ensemble column, post category-expansion (Section 4.1). Delta (combined) catches '
    'task_hijack (94%) and substitution_game (95%) where regex scores near 0%; the Ensemble column '
    'closes most of the remaining gaps (e.g. misc 64%, pwned_spacing 87%, extraction_trick 73%) that '
    'no single method covers alone. '
    'Note: the hex and backslash_visual rows are labeled by the same substring rules the '
    'corresponding regex columns detect (Table 2 caveat), so their Regex (hex)/Regex (backslash) '
    'cells are not independent measurements; all other rows use semantic/structural labeling rules.')

story += [
    P('Figures 3b and 3c break the same comparison down by error type: 3b shows the miss rate '
      '(FNR) per type x method -- the direct complement of Figure 3 -- and 3c shows the overall '
      'false positive rate per method, which cannot be broken down by obfuscation type since '
      'benign samples carry no type label.'),
]
story += FIG(os.path.join(OBF_GRAPHS, '04b_fnr_heatmap_method.png'),
    'Figure 3b: Miss rate (FNR) by obfuscation type x method at fixed FPR=1%, the complement of '
    'Figure 3.')
story += FIG(os.path.join(OBF_GRAPHS, '04c_fpr_strip_method.png'),
    'Figure 3c: Overall false positive rate by method at the same operating point. Benign samples '
    'are untyped, so this is a single row rather than a type x method matrix.')

story += [
    P('A feature correlation matrix is provided in Appendix F (Figure A3; a figure-generator bug fixed '
      'during this validation pass is documented in Appendix A). Delta angle correlates substantially '
      'with chunk count (r=0.68) and word count (r=0.47), and weakly with entropy (r=-0.08). Part of '
      'this is definitional: delta is exactly zero for single-chunk '
      'inputs, restating this paper\'s own single- vs. multi-chunk finding (Section 5.1) rather than '
      'revealing redundancy. A stratified test holding chunk count exactly fixed is not well-powered '
      'here: at chunk_count=3, the stratum has 171 obfuscation samples against a single benign sample '
      '(benign inputs are almost never multi-chunk -- only 8/813 overall), so any AUC computed within '
      'it is closer to a single-point measurement than a population statistic. A direct, well-powered test instead: substituting raw chunk count for '
      'sentence delta in the production ensemble (same weights, no re-optimisation) drops AUC from '
      '0.974 to 0.964; removing the sentence-delta term entirely drops it to 0.960. The embedding-based '
      'signal does outperform a trivial chunk-counter, consistently across both tests, but by a '
      'modest margin (under 1.4pp AUC) -- substantially smaller than the per-type "94% vs near-0% '
      'regex" comparisons in Section 5.3 might suggest in isolation, since chunk count alone achieves '
      'similar per-type numbers for task_hijack and substitution_game specifically (both are '
      'reliably multi-chunk attack types). We report this as a genuine, modest, but real advantage '
      'for the embedding signal, not the larger and more dramatic independence the per-type table '
      'alone implies.'),
    P('5.4 Cross-Model Robustness', h2),
    P('The delta angle signal is consistent across three embedding models. Cross-model Pearson '
      'correlations are: E5 vs Nemotron r=0.963, E5 vs BGE-M3 r=0.952, Nemotron vs BGE-M3 r=0.979 '
      '(Figure A4, Appendix). Raw AUC is also nearly identical across models (~0.851 for all three). '
      'This consistency, however, is specific to the raw signal -- not to how much of it survives '
      'structural control (Section 5.3\'s chunk-count/word-count regression, here extended with '
      'character entropy and special-character ratio as additional controls). Residual AUC after '
      'this richer control set is 0.717 for E5 (the production model), 0.632 for BGE-M3, and 0.856 '
      'for Nemotron -- all three start from the same raw signal, but Nemotron retains the most '
      'content beyond what a structural feature set could already explain, E5 the least of the '
      'three. The production results in this paper are E5-specific; a deployment built on Nemotron '
      'would likely show a larger gap over a trivial chunk-counter than Section 5.3 reports, though '
      'this is untested in the full ensemble. The signal\'s existence is not an artifact of any single '
      'embedding model, but its structural-vs-content composition is model-dependent.'),
    P('5.5 Operating Point Analysis', h2),
]
story += FIG(os.path.join(GRAPHS, '06_recall_vs_fpr.png'),
    'Figure 4: Recall vs FPR across three embedding models (sentence chunking), N=1,152 post '
    'category-expansion. All models reach near-perfect recall by FPR=1%, confirming a stable '
    'operating point across model choices.')
story += FIG(os.path.join(GRAPHS, '11_threshold.png'),
    'Figure 5: Threshold analysis (E5), N=1,152 post category-expansion. TPR declines from the '
    'optimal threshold (t=0.1016); FPR remains near zero across a wide range, confirming a robust '
    'operating region.')

story += [
    P('5.6 Adaptive Attack Evaluation (Reconstructed)', h2),
    P('The results below come from a newly constructed set of 30 representative samples spanning 20 '
      'documented evasion strategies, scored through the current full ensemble pipeline at its '
      'existing operating thresholds; this set replaces an earlier, unreproducible claim documented '
      'in Appendix A.'),
    P('All 30 samples are single-chunk by construction (no sentence-boundary structure), matching '
      'the original methodology: an informed adversary avoids multi-chunk text to neutralise the '
      'delta-angle signal, leaving detection entirely to the regex and weird-unicode components.'),
]
adapt_data = [
    [P('FPR Target', th_s), P('Caught', th_s), P('Evaded', th_s), P('Evasion Rate', th_s)],
    [P('1% FPR', tc_s), P('10/30', tc_s), P('20/30', tc_s), P('66.7%', tc_s)],
    [P('5% FPR', tc_s), P('12/30', tc_s), P('18/30', tc_s), P('60.0%',  tc_s)],
]
story += [
    mktbl(adapt_data, [4*cm, 3*cm, 3*cm, 3*cm]),
    P('Table 4 (reconstructed): Adaptive attack evasion rates against a newly constructed '
      'representative sample set (30 samples, 20 strategies), scored with the current full ensemble '
      '(9-pattern regex, optimised weights) at the existing 1%/5% FPR thresholds.', caption_s),
    Spacer(1, 0.1*cm),
    P('The catches are concentrated almost entirely in Unicode/homoglyph and visual-substitution '
      'tricks (fullwidth characters, mathematical alphanumeric symbols, Cyrillic homoglyphs, mixed-'
      'script text) -- these trip the special-character and weird-unicode features even with no '
      'encoding-related keywords present. Most plain natural-language samples evade at both FPR '
      'targets: direct requests, semantic rewrites, indirect references, roleplay framing, and '
      'instruction splitting all score near the benign median (template-fill is a partial exception '
      '-- one of two samples is caught; see §6.3). The reason '
      'is structural, not a tuning gap: the keyword regex (Table 1) targets <i>encoding</i> '
      'vocabulary -- decode, translate, payload, hidden, encrypted -- because it was built and '
      'optimised against an encoding-obfuscation dataset. It contains no prompt-injection vocabulary '
      '(ignore, override, previous instructions, system prompt), so social-engineering and semantic '
      'injection phrased in plain English is, by construction, invisible to every signal in this '
      'ensemble. This sharpens rather than contradicts the threat model in Section 7: the detector is '
      'an encoding-obfuscation filter, not a general prompt-injection classifier.'),
    Spacer(1, 0.2*cm),
]

# ── §5.7 Ablation summary (full detail moved to Appendix B) ──────────────────
story += [
    P('5.7 Ablation Studies and Negative Results', h2),
    P('Beyond the four production features, we systematically tested eleven geometric and '
      'structural extensions to the embedding signal -- distribution distance, intra-chunk token '
      'angles, signed angles, length-scaled softmax, per-type weight gating, skip-gram delta, '
      'connective splitting, vector velocity, BPE token density, sliding window deltas, and '
      'semantic anchor distance. None improved on the production ensemble; each produced a '
      'principled negative result that clarifies the detection mechanism (e.g. several invert '
      'because encoding makes local content more self-similar, not less, which is the opposite '
      'direction a naive geometric extension assumes). Full results, including a re-run against '
      'the current N=1,152 dataset confirming ten of eleven reproduce their original conclusions, '
      'are in Appendix B.'),
    Spacer(1, 0.2*cm),
]

# ── §6 Limitations ────────────────────────────────────────────────────────────
story += [
    P('6. Limitations', h1),
    P('6.1 Held-Out Validation (Resolved)', h2),
    P('A 70/30 stratified held-out validation of the headline 4-feature architecture was performed '
      '(Section 4.4), with all weights optimised on the training split only. The generalisation '
      'gap is moderate (max 6.86pp, at the 1% FPR operating point; under 2.3pp for AUC) and wider '
      'than measured before the targeted category expansion in Section 4.1, reflecting thinner '
      'per-split N for the newly-expanded categories. Full-dataset metrics are reported throughout '
      'for completeness; this held-out test still confirms the architecture generalises to unseen '
      'data rather than overfitting the full N=1,152.'),
    P('6.2 Single-Chunk Encoding Ceiling', h2),
    P('96/339 obfuscation samples (28.3%) produce delta=0 after chunking. For these, detection '
      'relies on regex and the Unicode anomaly signal (Section 3.4). The enhanced keyword list '
      'catches 28/96 (29.2%) and the Unicode anomaly signal catches 19/96 (19.8%), together '
      'addressing 45/96 (46.9%) with zero false positives; adding the Unicode term to the full '
      'ensemble raises single-chunk-subset AUC from 0.899 to 0.909 (+1.00pp, fixed production '
      'weights, no re-optimisation). The remaining 51/96 are pure encoding noise (hex, '
      'backslash) with no distinguishing Unicode or keyword features -- a practical floor '
      'under the current feature set without a codepoint analysis layer.'),
    P('6.3 Natural Language Injection (Reconstructed, Section 5.6)', h2),
    P('The reconstructed adaptive evaluation confirms this is the detector\'s dominant failure mode, '
      'and more severely than previously claimed: 12 of 20 evasion strategies tested -- plain '
      'natural-language prompt injection such as roleplay, indirect reference, semantic rewrite, data '
      'exfiltration requests, and instruction splitting -- are evaded by every sample at both 1% and '
      '5% FPR (11 of 20 at 5% FPR; one strategy, linguistic_trick, is caught at 5% FPR by one of its '
      'two samples). Two further strategies (template_fill, acronym_expansion) are partially caught: '
      'one sample evades, one is caught, rather than uniformly evading. The fully-evaded strategies '
      'share four properties: plain English, single sentence or short paragraph, no encoding-related '
      'keywords, indirect or conversational framing. Semantic anomaly detection or intent '
      'classification, not encoding-pattern matching, is required for coverage of this class.'),
    P('6.4 Unicode Homoglyphs and Visual Substitution (Reconstructed, Section 5.6)', h2),
    P('<i>Caveat:</i> this finding is based on 6 representative samples across 5 visual-substitution '
      'strategies in the reconstructed set (Section 5.6) -- the same small-n regime as the n<=5 '
      'categories in Table 2 and Section 6.5, directional rather than a statistically robust claim. '
      'Within that small sample, all 6 are caught: Cyrillic homoglyphs, fullwidth Latin, mixed-script '
      'text, and mathematical-alphanumeric substitution trigger the weird-unicode bonus signal '
      '(Section 3.4) and the special-character regex feature, independent of any keyword match. This '
      'contradicts an earlier, unreproducible claim that homoglyphs were an irreducible blind spot; '
      'on the samples we can actually test, this is the one attack class the detector handles better '
      'than originally believed, though confirming it at scale would need a larger sample.'),
    P('6.5 Small-N Categories', h2),
    P('Following the targeted expansion in Section 4.1, only 9 attack types still have n<=5 samples '
      '(n=1-2 each, N=10 total; Table 2) -- down from 13 before expansion. language_mixing (n=6) is '
      'thin but just above this threshold. Detection rates for the remaining 9 are directional '
      'findings, not statistically reliable estimates; the four categories expanded this pass '
      '(task_hijack, substitution_game, extraction_trick, base64, now n=10-20 each) are no longer in '
      'this bucket, though their held-out generalisation gap (Table 5) is correspondingly wider, '
      'reflecting their still-modest absolute size. To be explicit about what is and is not '
      'statistically supported: the headline ensemble metrics (AUC, F1, detection rate) are computed '
      'over the full N=1,152 and held-out validated (Section 4.4), and are robust; the dataset is also '
      'hex-dominated (162/339, 47.8% of obfuscation samples, Section 4.1), so per-type claims for any '
      'individual minority category -- including the four expanded this pass -- are directional '
      'evidence at a much smaller effective N, not independent confirmations of the headline numbers.'),
    Spacer(1, 0.2*cm),
]

# ── §7 Threat Model ───────────────────────────────────────────────────────────
story += [
    P('7. Threat Model and Deployment', h1),
    P('The detector is designed as a Stage 1 preprocessing filter in a defence-in-depth architecture:'),
    P('• <b>Stage 1 (this work):</b> Delta angle + regex + Unicode bonus. 94.7% recall at 1% FPR for encoding-based obfuscation; 33.3% recall at 1% FPR against plain-language adaptive attacks (Section 5.6).', bullet_s),
    P('• <b>Stage 2:</b> Semantic analysis or intent classification for natural language injection.', bullet_s),
    P('• <b>Stage 3:</b> Human review for remaining evasions.', bullet_s),
    Spacer(1, 0.15*cm),
    P('<b>Catches:</b> Hex, backslash-visual, base64, dot-spacing, Unicode codepoint encoding, task hijacking, language mixing, substitution games, translation chains, pwned-spacing, Unicode homoglyphs and other visual-substitution tricks.'),
    P('<b>Misses:</b> Pure wordplay, plain-English single-sentence injection, indirect framing without keywords, roleplay, instruction splitting, data-exfiltration requests phrased as ordinary text.'),
    P('Standalone deployment is not recommended. The detector\'s value is in the ensemble: '
      'it contributes independent signal that character-level methods cannot provide.'),
    Spacer(1, 0.2*cm),
]

# ── §8 Conclusion ─────────────────────────────────────────────────────────────
story += [
    P('8. Conclusion', h1),
    P('We presented a four-feature ensemble detector for obfuscated prompt injection combining '
      'sentence-level delta angles, paragraph-level delta angles, optimised regex coverage, and '
      'a single-chunk Unicode anomaly bonus, evaluated on N=1,152 samples including a targeted '
      'expansion of four previously underpowered attack-type categories (Section 4.1). The detector\'s '
      'value is predominantly structural rather than deep semantic understanding: delta angle\'s '
      'per-type wins over regex on task hijacking '
      '(94%) and substitution games (95%, Section 5.3) are real, but both are reliably multi-chunk '
      'attack types, and a trivial embedding-free chunk-count feature reaches similar per-type numbers '
      '(Appendix F). The embedding signal still adds real, measurable value beyond chunk-counting -- a '
      'modest 1.0-1.4pp AUC margin (Section 5.3) that is larger for some embedding models than others '
      '(0.72 residual AUC for the production E5 model vs. 0.86 for Nemotron, Section 5.4) -- but the '
      '"94% vs near-0%" per-type framing should be read as delta vs. regex specifically, not as proof '
      'that embedding geometry captures something chunk-counting fundamentally cannot.'),
    P('Two of the four ensemble features turn out not to mean what their formula weight suggests. '
      'Paragraph delta carries a substantial weight (0.199) despite a near-chance standalone AUC '
      '(0.530, Section 5.2) -- but removing it costs nothing, because regex and the Unicode bonus '
      'already independently catch every multi-paragraph attack sample in this dataset, not because '
      'its underlying signal is weak. Sentence delta carries a small weight (0.028) yet costs 1.4pp '
      'AUC to remove. Ensemble weight magnitude is not a reliable proxy for marginal contribution in '
      'this architecture; only direct removal testing distinguishes the two.'),
    P('We re-ran all eleven ablation studies (Appendix B) against the expanded dataset. Ten reproduce '
      'their original negative-result conclusions; one (distribution distance, Appendix B.1) no '
      'longer holds on the expanded data, which we report rather than suppress. Together they '
      'clarify that the detection mechanism is structural -- the chunking signature of '
      'obfuscated inputs -- rather than a deep semantic property of embedding geometry.'),
    P('The detector achieves AUC 0.974, F1 0.960, 94.7% detection at 1% FPR against the encoding-'
      'obfuscation benchmark -- slightly lower than before the category expansion (AUC 0.980/F1 '
      '0.967/96.5%), indicating that the expanded categories introduce genuinely harder and more '
      'diverse samples. A reconstructed adaptive red-team evaluation (Section 5.6) shows this '
      'does not extend to social-engineering-style prompt injection phrased in plain natural language: '
      '20/30 (66.7%) representative attacks evade at 1% FPR. Unicode homoglyphs and visual substitution '
      'are, in fact, reliably caught by the weird-unicode and special-character features; the dominant '
      'failure boundary observed in our experiments is plain-English injection with no encoding signature at all, which requires intent '
      'classification rather than pattern-based or structural detection.'),
    P('Future work includes Unicode codepoint analysis, intent classification '
      'for natural language injection, and integration into the AMDON guard pipeline.'),
    Spacer(1, 0.2*cm),
]

# ── Acknowledgements ──────────────────────────────────────────────────────────
story += [
    P('Acknowledgements', h1),
    P('Research execution (experiment scripting, data collection) assisted by Xiaomi MiMo (Xiaomi). '
      'Writing and analysis assisted by Claude Sonnet 4.6 (Anthropic). Embedding computations '
      'were performed via NVIDIA NIM API. This work was conducted independently without '
      'institutional or commercial funding.'),
    Spacer(1, 0.2*cm),
]

# ── References ────────────────────────────────────────────────────────────────
story += [
    P('References', h1),
    P('[1] Perez, F. and Ribeiro, I. (2022). Ignore Previous Prompt: Attack Techniques For Language Models. <i>NeurIPS ML Safety Workshop.</i>'),
    P('[2] Greshake, K. et al. (2023). Not What You\'ve Signed Up For: Compromising Real-World LLM-Integrated Applications with Indirect Prompt Injection. <i>AISec Workshop.</i>'),
    P('[3] Schulhoff, S. et al. (2023). Ignore This Title and HackAPrompt: Exposing Systemic Vulnerabilities of LLMs through a Global Scale Prompt Hacking Competition. <i>EMNLP.</i>'),
    P('[4] Alon, G. and Kamfonas, M. (2023). Detecting Language Model Attacks with Perplexity. <i>arXiv preprint 2308.14132.</i>'),
    Spacer(1, 0.3*cm),
]

# ── Appendix A: Reproducibility Notes ─────────────────────────────────────────
story += [
    PageBreak(),
    P('Appendix A: Reproducibility Notes', h1),
    P('This appendix documents three corrections made during validation passes on this paper. None '
      'change the headline results; each is recorded here for transparency rather than left silently '
      'fixed.'),
    P('A.1 Dataset Provenance', h2),
    P('An earlier draft of this paper attributed the full obfuscation set to the public Threat-Matrix '
      '(Neuralchemy, 2025) dataset. Checking the underlying files directly found zero text overlap '
      'between our original 287 obfuscation samples and that dataset: the original 287 were '
      'constructed and expanded iteratively across development sessions, not bulk-imported from an '
      'external source. 83 of the 813 benign prompts do overlap with the public Threat-Matrix data; '
      'the remainder are independently sourced. The full public Threat-Matrix set (1,883 samples) was '
      'separately used in an earlier session as a cross-dataset sentence-delta-only check (cached '
      'AUC=0.582, F1=0.657, not re-verified in this validation pass); it is not the source of the '
      'primary obfuscation/benign split used for the headline results in this paper.'),
    P('A.2 Adaptive Attack Evaluation History', h2),
    P('An earlier session reported 27/30 (90%) detection at 1% FPR and 30/30 (100%) at 5% FPR against '
      'a hand-crafted adversarial set for Section 5.6, but that sample set was never committed and '
      'could not be located in version history or reproduced verbatim. The results in Section 5.6 come '
      'from a newly constructed set of 30 representative samples spanning the same 20 documented '
      'evasion strategies, including the 3 specific examples the original session\'s notes preserved '
      'verbatim (two Cyrillic-homoglyph substitutions and one pure wordplay riddle), scored through the '
      'current full ensemble pipeline at its existing operating thresholds. This is not the original '
      '30-sample set, but it is real, reproducible, and run end-to-end against the shipped detector -- '
      'unlike the prior unverifiable claim.'),
    P('A.3 Feature Correlation Figure Labelling Bug', h2),
    P('The generator script for the feature correlation matrix (Appendix F, Figure A3) paired benign '
      'and obfuscation samples in the wrong order against the delta array -- a labelling bug, not a '
      'dataset-version issue -- which had been silently producing a spuriously small correlation in '
      'every prior version of this figure. The corrected computation is the one reported in Section '
      '5.3 (r=0.68 with chunk count, r=0.47 with word count), not the near-independence previously '
      'claimed.'),
    Spacer(1, 0.2*cm),
]

# ── Appendix B: Ablation Studies and Negative Results (moved from main flow) ──
story += [
    PageBreak(),
    P('Appendix B: Ablation Studies and Negative Results', h1),
    P('<i>Status note:</i> all eleven approaches below were re-run against the current N=1,152 '
      'dataset (Section 4.1). Ten reproduce the same qualitative conclusion as the original '
      '(smaller-dataset) run, several with numbers close enough to suggest the original figures, '
      'though their generator scripts were not all preserved, were genuine. One (Appendix B.1, '
      'Distribution Distance) no longer holds: it is not inverted on the expanded dataset, and '
      'Appendix B.11\'s reasoning, which partly relied on B.1 as supporting evidence, is noted '
      'accordingly.'),
    P('B.1 Distribution Distance', h2),
    P('Cosine distance from the benign centroid was tested as an auxiliary signal. Result, '
      're-run on N=1,152: AUC=0.714 -- no longer inverted, contrary to an earlier (N=1,100) finding '
      'of AUC=0.439. Combining with sentence+paragraph delta now improves AUC slightly (0.856 to '
      '0.870) rather than degrading it. We traced this to the four expanded categories (Section '
      '4.1): task-hijack, secret-extraction, and substitution-game text sits semantically further '
      'from the benign centroid than the hex/backslash-heavy original set did, where encoding '
      'preserves semantic content and keeps obfuscated inputs close to the centroid (the original '
      'explanation, still correct for that subset). This does not mean centroid distance is now a '
      'recommended feature: it is not weight-optimised or threshold-tuned the way the production '
      'ensemble is, and an unweighted 3-way average is a weak test of real marginal value.'),
    P('B.2 Intra-Chunk Token Angles', h2),
    P('Per-word token angles within single chunks were tested to target the 96/339 single-chunk '
      'obfuscation samples where sentence delta is zero (re-run on current data). Result: inverted '
      'for backslash_visual (AUC=0.070, raw direction -- even more strongly inverted than an '
      'earlier 0.194). Explanation: encoding creates coherent token clusters -- all tokens belong to '
      'the same repetitive pattern, making them more similar to each other than benign text tokens. '
      'Unstructured single-chunk obfuscation remains indistinguishable from benign (AUC=0.534, '
      'consistent with an earlier 0.621).'),
    P('B.3 Signed Angles', h2),
    P('Gram-Schmidt orthogonalisation was tested for directional sign assignment (re-run on current '
      'data, raw signed score not absolute value -- using the absolute value defeats the point of '
      'testing whether inconsistent sign assignment hurts separability). Result: degraded Nemotron '
      'performance (unsigned AUC=0.851, signed AUC=0.778), no benefit on E5 (unsigned AUC=0.851, '
      'signed AUC=0.848) -- consistent in direction with an earlier finding (Nemotron 0.858->0.734). '
      'Formal analysis confirms arccos in [0, pi] by definition: the production (unsigned) signal '
      'has zero negative angles by mathematical construction, not as an empirical finding. '
      'Unsigned angles are correct for production use.'),
    P('B.4 Length-Scaled Softmax', h2),
    P('Softmax weighting scaled by chunk length was tested with alpha in {0.5, 1.0, 1.5, 2.0} '
      '(re-run on current data). Result: identical across all alpha (AUC=0.845, matching an '
      'earlier 0.849 closely). Explanation: angle variance dominates the softmax, rendering length '
      'scaling inert. The FPR improvement from sentence chunking comes from the chunking strategy '
      'itself, not weighting.'),
    P('B.5 Per-Type Weight Gating', h2),
    P('Continuous per-type weight blending across four specialised regex weight sets was tested '
      '(reconstructed for this pass; the original blending script was not preserved -- affinities '
      'computed from the hex/backslash-density/keyword sub-features, blended against three '
      'single-feature-dominant weight sets plus the production default). Result: overall AUC roughly '
      'flat (0.957 to 0.961), but the broad unstructured category specifically degrades (67.3% to '
      '54.0% at the production threshold, consistent in direction with an earlier 74%->68%). '
      'Explanation: the same regex features used to compute type affinities are also used in the '
      'final score, creating a feedback loop. Simple single weight set optimisation outperforms '
      'adaptive blending.'),
    P('B.6 Skip-Gram Delta', h2),
    P('Multi-hop skip-gram angles (i to i+2, i to i+3) were tested on the current dataset. Among the '
      'nine major attack types with n&gt;=10 (excluding language_mixing, n=6, already flagged as a '
      'small-N category in Table 2), adjacent delta wins on AUC for 8 of 9 (only hex narrowly favours '
      'skip-1: 0.974 vs 0.969), and skip-2 is weaker for all 9. language_mixing itself is a marginal '
      'exception (skip-2 AUC=0.665 vs adjacent 0.663) but is too small-N to read as a real effect. '
      'Per-sample unique-catch comparison at '
      'the 99th-percentile-of-benign threshold was not usable for this re-run: 28.3% of obfuscation '
      'samples have sentence delta exactly zero, which collapses that specific threshold test '
      '(every method reads 339/339 "caught" once the cutoff sits at zero); the AUC comparison above '
      'is therefore the reliable signal here, and it confirms adjacent delta remains the better '
      'choice.'),
    P('B.7 Connective Splitting', h2),
    P('Splitting on connective words (and, but, then, now) inverts the signal: benign texts '
      'have higher split-delta than obfuscation (AUC=0.699, Det@1%FPR=0%, both consistent with an '
      'earlier AUC=0.750/Det@1%=0%). Connective-split chunks are semantically coherent for natural '
      'language but nonsensical for encoded content, producing the wrong directionality.'),
    P('B.8 Vector Velocity', h2),
    P('Prefix-based and sliding-window trajectory approaches produce an inverted signal -- '
      'obfuscation creates smoother (lower-velocity) embedding trajectories than benign text. '
      'Re-run on a fresh random sample of single-chunk inputs (n=10 each, matching the original '
      'small-sample pilot methodology): sliding-window velocity AUC=0.222 (obf mean 21.7 deg vs '
      'ben mean 25.5 deg), consistent in direction with an earlier extreme AUC=0.000 though less '
      'extreme at this sample size. The same mechanism as distribution distance: encoding preserves '
      'local semantic coherence.'),
    P('B.9 BPE Token Density', h2),
    P('Byte-pair encoding token density on single-chunk inputs is inverted (AUC=0.213, consistent '
      'in direction with an earlier 0.1421). Obfuscation has denser token packing, not sparser -- '
      'the wrong direction for detection.'),
    P('B.10 Sliding Window Deltas', h2),
    P('Sliding window delta computation (token-level windows) is inverted (AUC=0.273, consistent '
      'with an earlier 0.3274). Shares the same failure mode as intra-chunk token angles (Appendix '
      'B.2): encoding creates locally coherent windows with low internal variance.'),
    P('B.11 Semantic Anchor Distance', h2),
    P('This was a feasibility analysis, not an empirical measurement: no adversarial-direction '
      'training set or hyperplane was actually constructed or tested. Its original reasoning '
      'leaned on Appendix B.1\'s centroid-distance failure as supporting evidence ("same failure '
      'mode") -- since B.1 no longer fails on the expanded dataset, that specific argument '
      'is weakened. The independent part of the argument still holds: a single sentence yields one '
      'embedding vector with no trajectory to follow, and projecting it onto a hand-specified '
      '"hijack direction" requires a labelled training set of such directions that does not exist '
      'here. We treat this as an open question rather than a settled negative result.'),
    Spacer(1, 0.2*cm),
]

# ── Appendix C: Origin of This Research ───────────────────────────────────────
story += [
    PageBreak(),
    P('Appendix C: Origin of This Research', h1),
    P('Delta angle was originally proposed as a general-purpose, checksum-style measure against '
      'prompt injection: a tokenizer-derived value the guard model is asked to reproduce, on the '
      'premise that a model confused or compromised by injected instructions would fail to copy a '
      'number correctly even if it could fool a content-based judge. Evaluation of that general '
      'formulation found delta angle alone could not reliably separate normal from injected prompts -- '
      'the two distributions overlapped by more than 50% -- and was more useful as an auxiliary '
      'scaler for other classifiers than as a standalone detector.'),
    P('That limitation is the direct motivation for the narrower, better-supported application '
      'studied in this paper. Obfuscated encoding payloads, unlike general prompt injection, reliably '
      'produce a distinctive multi-chunk structural signature at sentence boundaries (Section 5.1) -- '
      'a property general injection prompts do not share. This paper is the result of refining the '
      'original measure down to the specific class of attack where it actually separates classes '
      'well, rather than a claim that delta angle solves prompt injection in general.'),
    Spacer(1, 0.2*cm),
]

# ── Appendix ──────────────────────────────────────────────────────────────────
story += [
    PageBreak(),
    P('Appendix D: Chunking Algorithm Analysis', h1),
    P('Figure A1 shows the chunking analysis supporting the merge threshold selection (Section 3.1). '
      'The 7-word spike in obfuscation chunks (panel a) and the 99%/28.3% single-chunk rates '
      '(panel c) explain the binary detection signal.'),
]
story += FIG(os.path.join(GRAPHS, '12_chunking_analysis.png'),
    'Figure A1: Chunking algorithm analysis, N=1,152 post category-expansion. (a) Chunk length '
    'distribution -- obfuscation spike at 7 words. '
    '(b) Chunks per input -- benign μ=1.04 vs obfuscation μ=2.50. '
    '(c) Single-chunk rate: 99% benign, 28.3% obfuscation. (d) Cumulative chunk lengths.')

story += [PageBreak(), P('Appendix E: ROC Curves', h1)]
story += FIG(os.path.join(GRAPHS, '02_roc_sentence.png'),
    'Figure A2: ROC curves for sentence chunking across three embedding models, N=1,152 '
    'post category-expansion (E5 AUC=0.851, Nemotron AUC=0.851, BGE-M3 AUC=0.851). Curves are '
    'nearly identical, confirming model-agnostic signal.')

story += [P('Appendix F: Feature Correlation Matrix', h1)]
story += FIG(os.path.join(GRAPHS, '16_correlation.png'),
    'Figure A3: Feature correlation matrix (E5, sentence chunking), N=1,152 post category-expansion, '
    'with the sample-pairing bug fixed (Section 5.3). Delta angle correlates r=0.68 with chunk '
    'count, r=0.47 with word count, r=-0.08 with entropy. The chunk-count/word-count correlation is '
    'substantial and partly reflects this paper\'s own single- vs. multi-chunk structural finding '
    '(Section 5.1); see Section 5.3 for a direct ensemble-substitution test quantifying how much '
    'value delta retains over raw chunk count (a modest but real margin, not full independence).')

story += [P('Appendix G: Cross-Model Correlation', h1)]
story += FIG(os.path.join(OBF_GRAPHS, '08_cross_model.png'),
    'Figure A4: Cross-model correlation of sentence delta angle, N=1,152 post-expansion. '
    'E5 vs Nemotron r=0.963, E5 vs BGE-M3 r=0.952, Nemotron vs BGE-M3 r=0.979. '
    'High correlation confirms model-agnostic structural signal.')

story += [PageBreak(), P('Appendix H: Per-Type ROC Curves', h1)]
story += FIG(os.path.join(OBF_GRAPHS, '07_roc_per_type.png'),
    'Figure A5: ROC curves per major obfuscation type (combined delta), post category-expansion. '
    'hex AUC=0.969 (n=162), task_hijack AUC=0.969 (n=17, expanded from n=5), language_mixing '
    'AUC=0.746 (n=6), misc AUC=0.585 (n=22). Small-n results are directional only.')

story += [P('Appendix I: Detection Difficulty Ranking', h1)]
story += FIG(os.path.join(OBF_GRAPHS, '05_difficulty_ranking.png'),
    'Figure A6: Detection difficulty ranking (combined delta, FPR=1%). '
    'Sample sizes shown. Results for n=1 types are indicative only.')

story += [
    PageBreak(),
    P('Appendix J: Keyword List', h1),
    P('The 29 terms active in the final regex keyword pattern:'),
    P('<font face="Courier" size="8">decode, translate, execute, convert, interpret, print, output, follow, '
      'compromise, breach, exfiltrate, acknowledge, pwned, status, confirm, generate, respond, '
      'payload, encrypted, decipher, decrypt, hidden, reveal, template, concatenate, expand, '
      'acronym, paraphrase, steganograph</font>',
      S('KW', fontSize=8.5, fontName='Courier', leftIndent=0.5*cm, spaceAfter=6, leading=13)),
    P('This excludes the six connective patterns (and/but/don\'t/only/then/now + action verb) listed '
      'in Table 1 (Section 3.3), which match multi-word phrases rather than single terms.', fn_s),
]

doc.build(story)
print("Paper v2 built successfully.")
