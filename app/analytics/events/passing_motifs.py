"""Source-ordered three-pass motifs; descriptive counts, not a tactical model.

Uses the five flow motifs of Gyarmati, Kwak and Rodriguez (arXiv:1409.0308).
Adds conservative StatsBomb possession, period and interruption boundaries.
"""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from app.file_io import write_json_atomic

PATTERNS = ('ABAB', 'ABAC', 'ABCA', 'ABCB', 'ABCD')


def motif_label(players):
    if len(players) != 4 or any(a == b for a, b in zip(players, players[1:])):
        raise ValueError('Invalid three-pass actors')
    letters = {}
    return ''.join(letters.setdefault(player, chr(65 + len(letters))) for player in players)


def point(value):
    if (isinstance(value, list) and len(value) >= 2
            and all(type(v) in (int, float) and math.isfinite(v) for v in value[:2])
            and 0 <= value[0] <= 120 and 0 <= value[1] <= 80):
        return value[:2]
    return None


def period_segment(period, minute):
    start, duration = {1: (0, 45), 2: (45, 45), 3: (90, 15), 4: (105, 15)}[period]
    return start // 15 + min(duration // 15 - 1, max(0, (minute - start) // 15))


def summarize(match, events):
    teams = (match['home_team'], match['away_team'])
    events = list(events)
    indices = [e.get('index') for e in events]
    if any(type(i) is not int or i <= 0 for i in indices) or len(set(indices)) != len(indices):
        raise ValueError('Invalid passing motif event order')
    audit = {team: dict.fromkeys(('passes', 'completed_passes', 'eligible_passes', 'chain_passes', 'chains', 'windows'), 0) for team in teams}
    chains, players, chain = [], {}, []
    context = None

    def flush():
        nonlocal chain
        if len(chain) >= 3:
            period, possession, team = context
            motifs = [motif_label([p['player'] for p in chain[i:i + 3]] + [chain[i + 2]['recipient']]) for i in range(len(chain) - 2)]
            chains.append({'id': f"{period}-{chain[0]['index']}", 'period': period,
                           'possession': possession, 'team': team, 'passes': chain, 'motifs': motifs})
            audit[team]['chains'] += 1; audit[team]['chain_passes'] += len(chain); audit[team]['windows'] += len(motifs)
        chain = []

    for event in sorted(events, key=lambda e: e['index']):
        period, possession = event.get('period'), event.get('possession')
        owner, team = event.get('possession_team', {}).get('name'), event.get('team', {}).get('name')
        current = (period, possession, owner)
        if current != context:
            flush(); context = current
        if type(period) is not int or period not in (1, 2, 3, 4) or team not in teams:
            flush(); continue
        kind = event.get('type', {}).get('name')
        if kind != 'Pass':
            # Pressure is not an on-ball touch. Carries/receipts may continue the
            # current receiver's control; other event types break conservatively.
            continued = (kind in ('Carry', 'Ball Receipt*') and team == owner
                         and (not chain or str(event.get('player', {}).get('id')) == chain[-1]['recipient'])
                         and not event.get('ball_receipt', {}).get('outcome'))
            if kind != 'Pressure' and not continued:
                flush()
            continue
        audit[team]['passes'] += 1
        detail = event.get('pass', {})
        if detail.get('outcome'):
            flush(); continue
        audit[team]['completed_passes'] += 1
        sender, receiver = event.get('player', {}), detail.get('recipient', {})
        clock = re.fullmatch(r'(\d{2}):([0-5]\d):([0-5]\d)\.(\d{3})', event.get('timestamp', ''))
        if (type(possession) is not int or possession <= 0 or owner != team
                or any(type(person.get('id')) is not int or person['id'] <= 0 for person in (sender, receiver))
                or sender['id'] == receiver['id'] or not clock
                or type(event.get('minute')) is not int or event['minute'] < 0
                or type(event.get('second')) is not int or not 0 <= event['second'] < 60):
            flush(); continue
        h, minute, second, millisecond = map(int, clock.groups())
        clock_ms = h * 3600000 + minute * 60000 + second * 1000 + millisecond
        if event['minute'] != clock_ms // 60000 + {1: 0, 2: 45, 3: 90, 4: 105}[period] or event['second'] != clock_ms // 1000 % 60:
            flush(); continue
        actor, recipient = str(sender['id']), str(receiver['id'])
        if chain and (chain[-1]['recipient'] != actor or not 0 <= clock_ms - chain[-1]['clock_ms'] <= 5000):
            flush()
        for person in (sender, receiver):
            players[str(person['id'])] = person.get('name') or f"球员 {person['id']}"
        chain.append({'index': event['index'], 'player': actor, 'recipient': recipient,
                      'minute': event['minute'], 'second': event['second'], 'clock_ms': clock_ms,
                      'segment': period_segment(period, event['minute']),
                      'start': point(event.get('location')), 'end': point(detail.get('end_location'))})
        audit[team]['eligible_passes'] += 1
    flush()
    return {'schema': 'Touchline.PassingMotifs.v1', 'max_gap_seconds': 5,
            'players': players, 'chains': chains, 'audit': audit}


def refresh(data_dir, scouting_report, summaries_dir, receipt):
    """Add only this derivative to existing summaries from verified source bytes."""
    from app.data_service.archive import read_asset_json

    data_dir, summaries_dir = Path(data_dir), Path(summaries_dir)
    manifest_path = data_dir / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assets = {row['path']: row for row in manifest['files']}
    matches = json.loads(Path(scouting_report).read_text())['matches']
    entries, outputs = [], []
    for match in matches:
        if match['competition_id'] != 55 or match['season'] != '2024' or match['gender'] != 'male':
            continue
        relative = f"data/events/{match['match_id']}.json"
        events = read_asset_json(data_dir, relative, assets)
        target = summaries_dir / f"statsbomb-{match['match_id']}.json"
        original = json.loads(target.read_text())
        if original['id'] != target.stem or original['teams'] != [match['home_team'], match['away_team']]:
            raise ValueError('Passing motif summary does not match source')
        derivative = summarize(match, events)
        entries.append({'match_id': target.stem, 'source_sha256': assets[relative]['sha256'],
                        'audit': derivative['audit']})
        outputs.append((target, {**original, 'passing_motifs': derivative}))
    # Read and validate every source before changing any summary.
    for target, summary in outputs:
        write_json_atomic(target, summary)
    result = {'schema': 'Touchline.PassingMotifBuild.v1', 'matches': entries,
              'revision': manifest['revision'], 'extractor_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'totals': dict(sum((Counter(team) for entry in entries for team in entry['audit'].values()), Counter()))}
    write_json_atomic(Path(receipt), result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', default='.tmp/open-data/statsbomb')
    parser.add_argument('--scouting-report', default='.tmp/scouting/statsbomb/report.json')
    parser.add_argument('--summaries-dir', default='results/exploration/matches')
    parser.add_argument('--receipt', default='.tmp/passing-motifs-20261015/build-receipt.json')
    args = parser.parse_args()
    result = refresh(args.data_dir, args.scouting_report, args.summaries_dir, args.receipt)
    print(json.dumps({'matches': len(result['matches']), 'totals': result['totals']}))
