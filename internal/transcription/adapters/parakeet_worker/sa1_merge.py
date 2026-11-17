"""Pinned SA1 timestamp reconciliation. Standard library only; never performs ASR."""
import math
import numbers
import unicodedata
from functools import lru_cache

SR = 16000
MAX_EDGE = 1280

class Unmergeable(ValueError):
    pass

def normalize(text):
    text = unicodedata.normalize('NFKC', text).lower().replace("'", '').replace('\u2019', '')
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split())

def validate_chunks(chunks, total, maximum=30):
    if not chunks or total <= 0:
        raise Unmergeable('Empty timeline or chunk map')
    previous = 0
    for i, c in enumerate(chunks):
        fields = ['core_start_sample', 'core_end_sample', 'decode_start_sample', 'decode_end_sample']
        if any(type(c.get(k)) is not int for k in fields) or type(c.get('chunk')) is not int or c.get('chunk') != i:
            raise Unmergeable('Invalid sample bounds/chunk identity')
        a, b, x, y = (c[k] for k in fields)
        if not (a == previous and 0 <= x <= a < b <= y <= total and y-x <= int(maximum*SR)):
            raise Unmergeable('Timeline gap, invalid context or oversized decoded input')
        previous = b
    if previous != total:
        raise Unmergeable('Missing file ending')

def _samples(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real) or not math.isfinite(value):
        raise Unmergeable('Non-finite/non-numeric native timestamp')
    return math.floor(float(value) * SR + 0.5)

def _interval(record, length, corrections, identity):
    try:
        start, end = _samples(record['start']), _samples(record['end'])
        if record['start'] > record['end']:
            raise Unmergeable('Reversed native interval')
    except (KeyError, TypeError) as error:
        raise Unmergeable('Missing native timestamps') from error
    if start < -MAX_EDGE or end > length + MAX_EDGE:
        raise Unmergeable('Native timestamp exceeds 80ms edge tolerance')
    clipped = (max(0, min(length, start)), max(0, min(length, end)))
    if clipped != (start, end):
        corrections.append(dict(source=identity, original_samples=[start, end], clipped_samples=list(clipped)))
    return clipped, (start+end)//2

def _shift(c, raw, corrections):
    if not isinstance(raw.get('text'), str) or not isinstance(raw.get('words'), list):
        raise Unmergeable('Missing native full text or timestamped words')
    if not isinstance(raw.get('segments'), list):
        raise Unmergeable('Missing native segment timestamp list')
    try:
        surfaces = [w['word'] for w in raw['words']]
        if any(not isinstance(w, str) for w in surfaces):
            raise Unmergeable('Non-string native word surface')
        if normalize(' '.join(surfaces)) != normalize(raw['text']):
            raise Unmergeable('Native words do not represent full chunk text')
    except (KeyError, TypeError) as error:
        raise Unmergeable('Invalid native word records') from error
    offset = c['decode_start_sample']
    length = c['decode_end_sample'] - offset
    words = []
    last_raw_mid = -math.inf
    for index, w in enumerate(raw['words']):
        identity = [c['chunk'], index]
        (a, b), raw_mid = _interval(w, length, corrections, dict(chunk=c['chunk'], word=index))
        if raw_mid < last_raw_mid:
            raise Unmergeable('Native word midpoints regress; no reordering repair')
        last_raw_mid = raw_mid
        words.append(dict(id=identity, chunk=c['chunk'], index=index, word=w['word'], key=normalize(w['word']),
                          start_sample=offset+a, end_sample=offset+b, midpoint=offset+(a+b)//2, native=dict(w)))
    for index, seg in enumerate(raw['segments']):
        _interval(seg, length, corrections, dict(chunk=c['chunk'], segment=index))
    return words

def _intersects(w, low, high):
    if w['start_sample'] == w['end_sample']:
        return low <= w['midpoint'] < high
    return w['start_sample'] < high and w['end_sample'] > low

def _pair_metrics(a, b):
    if not a['key'] or a['key'] != b['key']:
        return None
    delta = abs(a['midpoint'] - b['midpoint'])
    if delta > 1280:
        return None
    da, db = a['end_sample']-a['start_sample'], b['end_sample']-b['start_sample']
    if da == 0 or db == 0:
        return dict(midpoint_delta_samples=delta, interval_IoU=1.0) if da == db == 0 and delta == 0 else None
    intersection = max(0, min(a['end_sample'], b['end_sample'])-max(a['start_sample'], b['start_sample']))
    union = max(a['end_sample'], b['end_sample'])-min(a['start_sample'], b['start_sample'])
    if 2*intersection < union:
        return None
    return dict(midpoint_delta_samples=delta, interval_IoU=intersection/union)

def ordered_pairs(left, right):
    """Max pairs, then min distance, then lexicographically smallest native-index pairs."""
    @lru_cache(None)
    def solve(i, j):
        if i == len(left) or j == len(right):
            return (0, 0, ())
        choices = [solve(i+1, j), solve(i, j+1)]
        metrics = _pair_metrics(left[i], right[j])
        if metrics is not None:
            n, distance, tail = solve(i+1, j+1)
            choices.append((n+1, distance+metrics['midpoint_delta_samples'], ((left[i]['index'], right[j]['index']),)+tail))
        return min(choices, key=lambda v: (-v[0], v[1], v[2]))
    return solve(0, 0)[2]

def _owner(point, chunks, total):
    for c in chunks:
        if c['core_start_sample'] <= point < c['core_end_sample'] or point == total == c['core_end_sample']:
            return c['chunk']
    raise Unmergeable('Word ownership outside source timeline')

def merge_chunks(chunks, raw_chunks, total, maximum=30):
    validate_chunks(chunks, total, maximum)
    if len(chunks) != len(raw_chunks):
        raise Unmergeable('Missing raw chunk result')
    corrections, decisions, audits = [], [], []
    shifted = [_shift(c, r, corrections) for c, r in zip(chunks, raw_chunks)]
    used, accepted = set(), []
    for a, b in zip(chunks, chunks[1:]):
        low = max(a['decode_start_sample'], b['decode_start_sample'])
        high = min(a['decode_end_sample'], b['decode_end_sample'])
        left = [w for w in shifted[a['chunk']] if _intersects(w, low, high)]
        right = [w for w in shifted[b['chunk']] if _intersects(w, low, high)]
        matches = ordered_pairs(left, right) if low < high else ()
        lm, rm = {w['index']: w for w in left}, {w['index']: w for w in right}
        eligible = [dict(left=x['id'], right=y['id'], **m) for x in left for y in right if (m := _pair_metrics(x, y)) is not None]
        audits.append(dict(boundary=b['core_start_sample'], overlap_samples=[low, high], eligible_pairs=eligible, chosen_pairs=[list(p) for p in matches]))
        for li, ri in matches:
            x, y = lm[li], rm[ri]
            if tuple(x['id']) in used or tuple(y['id']) in used:
                raise Unmergeable('A native word would belong to two duplicate pairs')
            point = (x['midpoint']+y['midpoint'])//2
            chosen, dropped = (x, y) if point < b['core_start_sample'] else (y, x)
            owner = a['chunk'] if point < b['core_start_sample'] else b['chunk']
            if _owner(point, chunks, total) != owner:
                raise Unmergeable('Duplicate pair ownership crosses a nonadjacent core')
            accepted.append(dict(**chosen, ownership_point=point, owner=owner))
            used.update([tuple(x['id']), tuple(y['id'])])
            decisions.append(dict(action='duplicate_pair', boundary=b['core_start_sample'], key=x['key'],
                                  left=x, right=y, chosen=chosen['id'], discarded=dropped['id'],
                                  ownership_point=point, owner=owner, **_pair_metrics(x, y)))
    for group in shifted:
        for w in group:
            if tuple(w['id']) in used:
                continue
            owner = _owner(w['midpoint'], chunks, total)
            keep = owner == w['chunk']
            if keep:
                accepted.append(dict(**w, ownership_point=w['midpoint'], owner=owner))
            decisions.append(dict(action='retained_own_core' if keep else 'rejected_context_only', source=w, owner=owner,
                                  core=[chunks[w['chunk']]['core_start_sample'], chunks[w['chunk']]['core_end_sample']]))
    accepted.sort(key=lambda w: (w['ownership_point'], w['chunk'], w['index']))
    text = ' '.join(w['word'] for w in accepted)
    exported, segments, cursor = [], [], 0
    for w in accepted:
        exported.append(dict(word=w['word'], start=w['start_sample']/SR, end=w['end_sample']/SR,
                             start_offset=cursor, end_offset=cursor+len(w['word'])))
        cursor += len(w['word'])+1
    cursor = 0
    for c in chunks:
        group = [w for w in accepted if w['owner'] == c['chunk']]
        if not group:
            continue
        surface = ' '.join(w['word'] for w in group)
        segments.append(dict(segment=surface, start=min(w['start_sample'] for w in group)/SR,
                             end=max(w['end_sample'] for w in group)/SR, start_offset=cursor, end_offset=cursor+len(surface)))
        cursor += len(surface)+1
    if normalize(' '.join(s['segment'] for s in segments)) != normalize(text):
        raise Unmergeable('Segment grouping changed exported word order')
    return dict(transcription=text, word_timestamps=exported, segment_timestamps=segments), dict(
        shifted_words=shifted, corrections=corrections, boundary_matches=audits, decisions=decisions,
        accepted_words=accepted, raw_concatenation=' '.join(r['text'] for r in raw_chunks))
