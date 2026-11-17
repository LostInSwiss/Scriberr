"""Portable pure SA1/D1 regressions. No ASR imports, VAD inference or model load."""
from pathlib import Path
import sys, unittest, json, math
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'internal/transcription/adapters/parakeet_worker'))
from sa1_merge import Unmergeable, merge_chunks, normalize, ordered_pairs, validate_chunks
from sa1_duplicates import revise
from sa1_boundaries import boundaries, validate_maximum
def two_chunks():
    return [dict(chunk=0, core_start_sample=0, core_end_sample=16000, decode_start_sample=0, decode_end_sample=20000), dict(chunk=1, core_start_sample=16000, core_end_sample=32000, decode_start_sample=12000, decode_end_sample=32000)]

def raw(c, entries):
    words = [dict(word=s, start=a - c['decode_start_sample'] / 16000, end=b - c['decode_start_sample'] / 16000) for s, a, b in entries]
    return dict(text=' '.join((w['word'] for w in words)), words=words, segments=[dict(segment=w['word'], start=w['start'], end=w['end']) for w in words])

def merged(left, right):
    c = two_chunks()
    return merge_chunks(c, [raw(c[0], left), raw(c[1], right)], 32000)

class MergerChecks(unittest.TestCase):

    def test_decode_offset_not_core_offset(self):
        result, _ = merged([], [('hello', 1.2, 1.3)])
        self.assertEqual(result['word_timestamps'][0]['start'], 1.2)
        self.assertEqual(result['word_timestamps'][0]['end'], 1.3)

    def test_duplicate_and_boundary_right_ownership(self):
        result, audit = merged([('same', 0.95, 1.05)], [('same', 0.95, 1.05)])
        self.assertEqual(result['transcription'], 'same')
        pair = next((d for d in audit['decisions'] if d['action'] == 'duplicate_pair'))
        self.assertEqual(pair['chosen'], [1, 0])
        self.assertEqual(pair['ownership_point'], 16000)

    def test_punctuation_match_keeps_selected_surface(self):
        result, _ = merged([('Hello,', 0.95, 1.05)], [('hello!', 0.95, 1.05)])
        self.assertEqual(result['transcription'], 'hello!')

    def test_true_repeated_no_no_is_preserved(self):
        entries = [('no', 0.8, 0.9), ('no', 1.1, 1.2)]
        result, audit = merged(entries, entries)
        self.assertEqual(result['transcription'], 'no no')
        self.assertEqual(sum((d['action'] == 'duplicate_pair' for d in audit['decisions'])), 2)

    def test_no_global_uniqueness_filter(self):
        result, _ = merged([('same', 0.2, 0.3)], [('same', 1.7, 1.8)])
        self.assertEqual(result['transcription'], 'same same')

    def test_context_only_rejected_and_logged(self):
        result, audit = merged([('unmatched', 1.1, 1.2)], [])
        self.assertEqual(result['transcription'], '')
        self.assertEqual(audit['raw_concatenation'].strip(), 'unmatched')
        self.assertEqual(audit['decisions'][0]['action'], 'rejected_context_only')

    def test_delta_time_gate(self):
        result, audit = merged([('same', 0.9, 1.0)], [('same', 1.1, 1.2)])
        self.assertEqual(result['transcription'], 'same same')
        self.assertEqual(audit['boundary_matches'][0]['chosen_pairs'], [])

    def test_IoU_gate_even_when_midpoints_close(self):
        result, audit = merged([('same', 0.94, 1.0)], [('same', 1.0, 1.06)])
        self.assertEqual(result['transcription'], 'same same')
        self.assertEqual(audit['boundary_matches'][0]['chosen_pairs'], [])

    def test_zero_duration_exact_pair(self):
        result, _ = merged([('point', 1.0, 1.0)], [('point', 1.0, 1.0)])
        self.assertEqual(result['transcription'], 'point')

    def test_zero_duration_vs_interval_does_not_pair(self):
        result, audit = merged([('point', 0.98, 1.0)], [('point', 1.0, 1.0)])
        self.assertEqual(result['transcription'], 'point point')
        self.assertEqual(audit['boundary_matches'][0]['chosen_pairs'], [])

    def test_final_file_endpoint_inclusive(self):
        result, _ = merged([], [('ending', 2.0, 2.0)])
        self.assertEqual(result['transcription'], 'ending')
        self.assertEqual(result['word_timestamps'][0]['end'], 2.0)

    def test_clips_only_within_80ms_and_records_corrections(self):
        c = [dict(chunk=0, core_start_sample=0, core_end_sample=16000, decode_start_sample=0, decode_end_sample=16000)]
        result, audit = merge_chunks(c, [raw(c[0], [('first', -0.04, 0.04), ('last', 0.95, 1.04)])], 16000)
        self.assertEqual(result['word_timestamps'][0]['start'], 0)
        self.assertEqual(result['word_timestamps'][-1]['end'], 1)
        self.assertEqual(len(audit['corrections']), 4)

    def test_excessive_edges_invalid_missing_or_reversed_timestamps(self):
        for value in [float('nan'), float('inf'), True, '0', None]:
            c = two_chunks()
            r = raw(c[0], [('bad', 0.1, 0.2)])
            r['words'][0]['start'] = value
            with self.subTest(value=value), self.assertRaises(Unmergeable):
                merge_chunks(c, [r, raw(c[1], [])], 32000)
        for entries in [[('bad', -0.081, 0.1)], [('bad', 0.3, 0.2)], [('bad', 1.3, 1.34)]]:
            with self.subTest(entries=entries), self.assertRaises(Unmergeable):
                merged(entries, [])
        c = two_chunks()
        r = raw(c[0], [('bad', 0.1, 0.2)])
        del r['words'][0]['start']
        with self.assertRaises(Unmergeable):
            merge_chunks(c, [r, raw(c[1], [])], 32000)

    def test_native_midpoint_regression_is_not_sorted_away(self):
        with self.assertRaises(Unmergeable):
            merged([('second', 0.4, 0.5), ('first', 0.2, 0.3)], [])

    def test_native_text_coverage_gate(self):
        c = two_chunks()
        r = raw(c[0], [('hello', 0.1, 0.2)])
        r['text'] = 'hello missing'
        with self.assertRaises(Unmergeable):
            merge_chunks(c, [r, raw(c[1], [])], 32000)
        for field in ['words', 'segments']:
            r = raw(c[0], [('hello', 0.1, 0.2)])
            r[field] = None
            with self.subTest(field=field), self.assertRaises(Unmergeable):
                merge_chunks(c, [r, raw(c[1], [])], 32000)

    def test_offsets_unicode_multiword_and_segment_text_consistency(self):
        result, _ = merged([('L’été', 0.2, 0.3), ('New York', 0.4, 0.5)], [('end.', 1.7, 1.8)])
        for w in result['word_timestamps']:
            self.assertEqual(result['transcription'][w['start_offset']:w['end_offset']], w['word'])
        for s in result['segment_timestamps']:
            self.assertEqual(result['transcription'][s['start_offset']:s['end_offset']], s['segment'])
        self.assertEqual(normalize(' '.join((s['segment'] for s in result['segment_timestamps']))), normalize(result['transcription']))

    def test_punctuation_only_keys_never_pair(self):
        _, audit = merged([('...', 0.95, 1.05)], [('...', 0.95, 1.05)])
        self.assertEqual(audit['boundary_matches'][0]['eligible_pairs'], [])

    def test_ordered_alignment_count_distance_lexicographic_ties(self):

        def word(index, mid):
            return dict(index=index, key='x', start_sample=mid - 200, end_sample=mid + 200, midpoint=mid)
        left = [word(0, 16000), word(1, 16000)]
        right = [word(0, 16000), word(1, 16000), word(2, 16000)]
        self.assertEqual(ordered_pairs(left, right), ((0, 0), (1, 1)))
        self.assertEqual(ordered_pairs(left[:1], [word(0, 15990), word(1, 16000)]), ((0, 1),))

    def test_invalid_core_coverage_and_overlong_decoded_input(self):
        for field, value in [('core_start_sample', 1), ('decode_start_sample', -1), ('decode_end_sample', 500000), ('core_end_sample', 0)]:
            c = two_chunks()
            c[0][field] = value
            with self.subTest(field=field), self.assertRaises(Unmergeable):
                validate_chunks(c, 32000)

def fixture(leftkeys, rightkeys, delta=6400, kind='forced', zero=False):
    chunks = [dict(chunk=0, core_start_sample=0, core_end_sample=32000, decode_start_sample=0, decode_end_sample=36000, right_boundary=kind), dict(chunk=1, core_start_sample=32000, core_end_sample=64000, decode_start_sample=28000, decode_end_sample=64000, right_boundary='file_end')]
    groups = []
    for side, keys in enumerate([leftkeys, rightkeys]):
        group = []
        for i, key in enumerate(keys):
            mid = 29600 + i * 1600 + (delta if side else 0)
            width = 0 if zero else 800
            group.append(dict(id=[side, i], chunk=side, index=i, key=key, word=key, start_sample=mid - width, end_sample=mid + width, midpoint=mid, owner=side, ownership_point=mid))
        groups.append(group)
    return (chunks, dict(shifted_words=groups, accepted_words=groups[0] + groups[1]))

class Checks(unittest.TestCase):

    def test_coherent_phrase_duplicate(self):
        c, a = fixture(['of', 'this', 'kind'], ['of', 'this', 'kind'])
        k, d, r = revise(c, a)
        self.assertEqual(len(k), 3)
        self.assertEqual(r[0]['removed_words'], 3)

    def test_single_repeated_word_protected(self):
        c, a = fixture(['no'], ['no'])
        self.assertEqual(len(revise(c, a)[0]), 2)

    def test_no_no_phrase_protected(self):
        c, a = fixture(['no', 'no'], ['no', 'no'])
        self.assertEqual(len(revise(c, a)[0]), 4)

    def test_repeated_phrase_ambiguity_abstains(self):
        c, a = fixture(['i', 'mean', 'i', 'mean'], ['i', 'mean', 'i', 'mean'])
        self.assertEqual(len(revise(c, a)[0]), 8)

    def test_inconsistent_timestamp_shifts_protected(self):
        c, a = fixture(['of', 'this'], ['of', 'this'])
        a['shifted_words'][1][1]['midpoint'] += 5000
        self.assertEqual(len(revise(c, a)[0]), 4)

    def test_distant_equal_phrase_protected(self):
        c, a = fixture(['of', 'this'], ['of', 'this'], delta=16000)
        self.assertEqual(len(revise(c, a)[0]), 4)

    def test_silence_cut_unchanged(self):
        c, a = fixture(['of', 'this'], ['of', 'this'], kind='silence')
        self.assertEqual(len(revise(c, a)[0]), 4)

    def test_zero_duration_is_not_phrase_anchor(self):
        c, a = fixture(['of', 'this'], ['of', 'this'], zero=True)
        self.assertEqual(len(revise(c, a)[0]), 4)

    def test_noncontiguous_native_indices_not_bridged(self):
        c, a = fixture(['of', 'this'], ['of', 'this'])
        a['shifted_words'][0][1]['index'] = 3
        self.assertEqual(len(revise(c, a)[0]), 4)

    def test_no_reference_or_content_mutation(self):
        c, a = fixture(['of', 'this'], ['of', 'this'])
        before = json.dumps(a)
        k, d, r = revise(c, a)
        self.assertEqual(json.dumps(a), before)
        indices = [next((i for i, original in enumerate(a['accepted_words']) if w is original)) for w in k]
        self.assertEqual(indices, sorted(indices))

    def test_does_not_drop_only_surviving_occurrence(self):
        c, a = fixture(['of', 'this'], ['of', 'this'])
        a['accepted_words'] = a['shifted_words'][1]
        self.assertEqual(len(revise(c, a)[0]), 2)

class ConfigurableBounds(unittest.TestCase):
    def test_complete_timeline_at_all_supported_sizes(self):
        for maximum in [5,5.1,12.75,20.6,30,60,120]:
            for n in [1,80000,6467280,16789675]:
                for speech in [[],[dict(start=0,end=n)]]:
                    chunks=boundaries(n,speech,maximum)
                    validate_chunks(chunks,n,maximum)
                    self.assertEqual(chunks[-1]['core_end_sample'],n)
                    self.assertTrue(all(c['decode_end_sample']-c['decode_start_sample']<=int(maximum*16000) for c in chunks))
    def test_invalid_maximum_values(self):
        for value in [None,True,'30',0,4.9,120.1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):validate_maximum(value)
    def test_default_forced_core_budget(self):
        n=6467280
        chunks=boundaries(n,[dict(start=0,end=n)])
        self.assertEqual(chunks[0]['core_end_sample'],470400)
        self.assertTrue(all(c['right_context_samples'] in [0,4000] for c in chunks))

if __name__ == '__main__': unittest.main()
