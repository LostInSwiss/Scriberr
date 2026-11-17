"""Pinned SA1-D1 pure postpass; verbatim functions from review source."""

def _edge(words,boundary,side):
    found=[w for w in words if w['key'] and w['end_sample']>w['start_sample'] and abs(w['midpoint']-boundary)<=16000]
    return found[-8:] if side=='left' else found[:8]

def _repeat(words):
    pairs=[(a['key'],b['key']) for a,b in zip(words,words[1:]) if b['index']==a['index']+1]
    return len(pairs)!=len(set(pairs))

def _occurrences(words,keys):
    n=len(keys)
    return sum(tuple(w['key'] for w in words[i:i+n])==keys and all(words[i+k+1]['index']==words[i+k]['index']+1 for k in range(n-1)) for i in range(len(words)-n+1))

def _anchor(words,a,b):return any(w['start_sample']<b+1280 and w['end_sample']>a-1280 for w in words)

def revise(chunks,audit):
    """Deletion-only pass over unchanged build2 accepted words; no reference access."""
    accepted={tuple(w['id']):w for w in audit['accepted_words']};removed=set();decisions=[];reviews=[]
    for i,(left,right) in enumerate(zip(chunks,chunks[1:])):
        boundary=right['core_start_sample'];row=dict(boundary_seconds=boundary/16000,type=left['right_boundary'],left_window='',right_window='',status='unchanged',matched_phrases=0,removed_words=0)
        if left['right_boundary']!='forced':row['status']='silence_boundary_unchanged';reviews.append(row);continue
        lw=_edge(audit['shifted_words'][i],boundary,'left');rw=_edge(audit['shifted_words'][i+1],boundary,'right')
        row['left_window']=' '.join(w['word'] for w in lw);row['right_window']=' '.join(w['word'] for w in rw)
        if _repeat(lw) or _repeat(rw):row['status']='ambiguous_repeated_bigram_abstain';reviews.append(row);continue
        low=max(left['decode_start_sample'],right['decode_start_sample']);high=min(left['decode_end_sample'],right['decode_end_sample'])
        candidates=[]
        for a in range(len(lw)):
            for b in range(len(rw)):
                for n in range(2,min(8,len(lw)-a,len(rw)-b)+1):
                    x,y=lw[a:a+n],rw[b:b+n];keys=tuple(w['key'] for w in x)
                    if keys!=tuple(w['key'] for w in y) or len(set(keys))<2:continue
                    if any(z[k+1]['index']!=z[k]['index']+1 for z in [x,y] for k in range(n-1)):continue
                    if _occurrences(lw,keys)!=1 or _occurrences(rw,keys)!=1:continue
                    shifts=[v['midpoint']-u['midpoint'] for u,v in zip(x,y)]
                    if max(map(abs,shifts))>12800 or max(shifts)-min(shifts)>3840:continue
                    if not _anchor(x,low,high) or not _anchor(y,low,high):continue
                    candidates.append((n,sum(map(abs,shifts)),x,y,shifts))
        candidates.sort(key=lambda c:(-c[0],c[1],tuple(w['index'] for w in c[2]),tuple(w['index'] for w in c[3])))
        used=set()
        for n,distance,x,y,shifts in candidates:
            identities={tuple(w['id']) for w in x+y}
            if identities&used:continue
            interior=lambda group,c:sum(min(w['midpoint']-c['decode_start_sample'],c['decode_end_sample']-w['midpoint']) for w in group)
            ls,rs=interior(x,left),interior(y,right);winner='left' if ls>=rs else 'right'
            dropped=[]
            for u,v in zip(x,y):
                uid,vid=tuple(u['id']),tuple(v['id'])
                if uid in accepted and vid in accepted:
                    losing=vid if winner=='left' else uid
                    if losing not in removed:removed.add(losing);dropped.append(list(losing))
            used|=identities
            decisions.append(dict(boundary_sample=boundary,left_ids=[w['id'] for w in x],right_ids=[w['id'] for w in y],phrase_keys=list(w['key'] for w in x),
                midpoint_shifts_samples=shifts,left_interior_sum_samples=ls,right_interior_sum_samples=rs,winner=winner,removed_native_ids=dropped))
            row['matched_phrases']+=1;row['removed_words']+=len(dropped)
        row['status']='phrase_match' if row['matched_phrases'] else 'no_unambiguous_phrase';reviews.append(row)
    kept=[w for w in audit['accepted_words'] if tuple(w['id']) not in removed]
    return kept,decisions,reviews

def export(kept,chunks):
    text=' '.join(w['word'] for w in kept);words=[];segments=[];cursor=0
    for w in kept:
        words.append(dict(word=w['word'],start=w['start_sample']/16000,end=w['end_sample']/16000,start_offset=cursor,end_offset=cursor+len(w['word'])))
        cursor+=len(w['word'])+1
    cursor=0
    for c in chunks:
        group=[w for w in kept if w['owner']==c['chunk']]
        if not group:continue
        surface=' '.join(w['word'] for w in group)
        segments.append(dict(segment=surface,start=min(w['start_sample'] for w in group)/16000,end=max(w['end_sample'] for w in group)/16000,start_offset=cursor,end_offset=cursor+len(surface)));cursor+=len(surface)+1
    assert ' '.join(s['segment'] for s in segments)==text
    return dict(transcription=text,word_timestamps=words,segment_timestamps=segments)
