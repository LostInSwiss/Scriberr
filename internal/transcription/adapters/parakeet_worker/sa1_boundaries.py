"""SA1 full-timeline boundary policy. Maximum includes context on both sides."""
import math

def validate_maximum(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 5 <= value <= 120:
        raise ValueError('Maximum chunk length must be finite and between 5 and 120 seconds')
    return value

def boundaries(n,speech,maximum=30):
    """Contiguous ownership, 0.3s safe context, 0.25s forced context; no audio deletion."""
    maximum = validate_maximum(maximum)
    limit = int(maximum*16000)
    budget = limit-9600
    search_min = min(320000, budget) if maximum >= 20.6 else budget*2//3
    if type(n) is not int or n <= 0:
        raise ValueError('Audio must have a nonempty integer sample timeline')
    speech=sorted(speech,key=lambda x:x['start'])
    end=0;gaps=[]
    for r in speech:
        assert 0<=r['start']<r['end']<=n and r['start']>=end
        if r['start']>end:gaps.append((end,r['start']))
        end=r['end']
    if end<n:gaps.append((end,n))
    cuts=[dict(sample=0,kind='file_start')];start=0
    while n-start>budget: # 29.4s ownership budget; decoded input <=30s
        low=start+search_min;deadline=start+budget # search20.0â€“29.4s
        candidates=[]
        for a,b in gaps:
            lo=max(a+4800,low);hi=min(b-4800,deadline)
            if lo<=hi:candidates.append(min(max((a+b)//2,lo),hi))
        cut=max(candidates) if candidates else deadline
        kind='silence' if candidates else 'forced'
        cuts.append(dict(sample=cut,kind=kind));start=cut
    cuts.append(dict(sample=n,kind='file_end'))
    chunks=[]
    for i,(a,b) in enumerate(zip(cuts,cuts[1:])):
        left=0 if i==0 else 4000 if a['kind']=='forced' else 4800
        right=0 if i==len(cuts)-2 else 4000 if b['kind']=='forced' else 4800
        begin=max(0,a['sample']-left);end=min(n,b['sample']+right)
        assert end-begin<=limit
        chunks.append(dict(chunk=i,core_start_sample=a['sample'],core_end_sample=b['sample'],decode_start_sample=begin,decode_end_sample=end,left_context_samples=a['sample']-begin,right_context_samples=end-b['sample'],right_boundary=b['kind']))
    assert chunks[0]['core_start_sample']==0 and chunks[-1]['core_end_sample']==n
    assert sum(c['core_end_sample']-c['core_start_sample'] for c in chunks)==n
    assert all(a['core_end_sample']==b['core_start_sample'] for a,b in zip(chunks,chunks[1:]))
    return chunks
