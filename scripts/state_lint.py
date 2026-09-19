#!/usr/bin/env python
"""Lint a states/*.md or STATE.md file for the failure modes the lead has caught.
Deterministic, so it works even when the agent's attention has drifted. Exit 1 on any hit.

  python scripts/state_lint.py states/03-*.md          # one or more files
  python scripts/state_lint.py --all                   # every state + STATE.md

Rules (each traces to agent/FEEDBACK.md):
  jargon      project shorthand that means nothing to a reader who was not here
  numbering   'State 3' style numbering in titles, deks, nav links or prose
  title       H1 must be a plain sentence, no colon-jargon, ≤ 110 chars
  expectation a figure embedded in Status must be introduced by prose that says what to expect
  links       every ../evidence/ and ../REASONING.md#Dxx link must resolve
  anachronism (states only) a figure may not cite a measure later than the state (edit ORDER below)
"""
import re, sys, glob, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JARGON = {  # phrase -> what to write instead
    r'\bruler\b(?! and)': 'say what it is: "a distance measured on the objects\' 3-D shape"',
    r'\bthe control\b': '"the pretrained model, which never saw the training set"',
    r'\bby construction\b': 'say why: "because it is the same trial on all twelve points"',
    r'\bpooled row\b|\bon-category row\b': 'describe the selection: "all trials", "only the model trained on the trial\'s category"',
    r'\bbeats?\b(?=\s+(the|any|every))': 'not evidence; say what the pattern is consistent with',
    r'\bin beats\b|\bbeat\b(?=s? with)': 'write prose, not a list of beats',
    r'\bleaks?\b': 'say what is happening: "still partly reports which trials are hard"',
    r'\bthe form\b(?! of)': '"the way the estimate is computed"',
    r'\bthe space\b(?! it)': '"the feature space it is measured in"',
    r'\bwithin-trial design\b': '"comparing the twelve models on the same trial"',
    r'\bcausal result\b': 'say what moved what',
    r'\bd\(A,\s*B\)\b|\bd_AB\b': 'first use must say "how different the two objects in a trial are"',
    r'\bknn\b|\bk-?NN\b': '"the nearest fifty training objects"',
    r'\bε\b|\bepsilon\b': 'say "a fixed radius" and what it is',
}
NUMBERING = r'\bStates? \d\b'

def lint(path):
    txt = open(path).read(); lines = txt.split('\n'); hits = []
    ok = set(t.strip().lower() for m in re.finditer(r'<!--\s*lint:\s*ok\s+([^>]+)-->', txt) for t in m.group(1).split(','))
    h1 = next((l for l in lines if l.startswith('# ')), '')
    if not h1: hits.append((1, 'title', 'no H1'))
    if len(h1) > 112: hits.append((1, 'title', f'H1 is {len(h1)} chars; a plain sentence, not a compound'))
    if re.search(NUMBERING, txt): 
        for i, l in enumerate(lines, 1):
            if re.search(NUMBERING, l): hits.append((i, 'numbering', l.strip()[:80]))
    # jargon: first occurrence only unless it is also in the title/dek (lines 1-2)
    for pat, fix in JARGON.items():
        for i, l in enumerate(lines, 1):
            if l.startswith('![') or l.startswith('|') or l.startswith('<'): continue
            l = re.sub(r'\]\([^)]*\)', ']()', l)   # ignore link targets (filenames)
            mm = re.search(pat, l, flags=re.I)
            if mm:
                if mm.group(0).lower() not in ok: hits.append((i, 'jargon', f'"{mm.group(0)}" → {fix}  (silence with <!-- lint: ok {mm.group(0).lower()} --> once it is defined in this file)'))
                break
    # expectation: a figure must be preceded (within 3 non-blank lines) by prose containing a cue
    cues = r'should|expect|if .* then|would be flat|below|panel'
    for i, l in enumerate(lines):
        if l.startswith('!['):
            prev = [x for x in lines[max(0, i-4):i] if x.strip()]
            if 'figure' not in ok and not any(re.search(cues, x, flags=re.I) for x in prev):
                hits.append((i+1, 'expectation', 'figure not introduced by prose saying what to expect / what it shows'))
    # links
    for m in re.finditer(r'\]\((\.\./[^)#]+)(#[^)]+)?\)', txt):
        p = os.path.normpath(os.path.join(os.path.dirname(path), m.group(1)))
        if not os.path.exists(p): hits.append((txt[:m.start()].count('\n')+1, 'links', m.group(1)+' missing'))
        elif m.group(2) and m.group(2)[1:] not in open(p).read(): hits.append((txt[:m.start()].count('\n')+1, 'links', m.group(0)+' anchor missing'))
    return hits

files = sorted(glob.glob(f'{ROOT}/states/0*.md') + [f'{ROOT}/STATE.md']) if '--all' in sys.argv else [a for a in sys.argv[1:] if a.endswith('.md')]
bad = 0
for f in files:
    for ln, rule, msg in lint(f):
        print(f'{os.path.relpath(f, ROOT)}:{ln}: [{rule}] {msg}'); bad += 1
print('state_lint:', 'clean' if not bad else f'{bad} hits (see agent/FEEDBACK.md)')
sys.exit(1 if bad else 0)
