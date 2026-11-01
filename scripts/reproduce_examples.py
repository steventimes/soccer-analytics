"""Answer three bounded questions from committed public summaries, without training."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]


def answer_question(question, *, root=ROOT):
    root = Path(root)
    sources = []

    def read(name):
        path = root / 'docs/data/research' / name
        raw = path.read_bytes()
        sources.append({'path': path.relative_to(root).as_posix(), 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    if question == 'prediction':
        group = next(row for row in read('model-lab.json')['groups'] if row['kind'] == 'scores' and row['competition'] == 'PL')
        selected = next(row for row in group['candidates'] if row['id'] == group['selected'])
        baseline = next(row for row in group['candidates'] if row['id'] == 'poisson_constant')
        observations = {'selected_on': group['selected_on'], 'selected': selected['id'],
                        'baseline': baseline['id'], 'holdout_rows': group['holdout_rows'],
                        'holdout_interval': [group['holdout_start'], group['holdout_end']],
                        'selected_log_loss': selected['holdout']['log_loss'],
                        'baseline_log_loss': baseline['holdout']['log_loss'],
                        'log_loss_difference': selected['holdout']['log_loss'] - baseline['holdout']['log_loss']}
        scope = 'PL 2025/26 历史固定留出；按开发期选择，差值为已选模型减固定进球率基线，越低越好。重现公开结果比较，不重训或声称前瞻效果。'
        route = 'models.html?experiment=scores-PL&stage=holdout&metric=log_loss#model-lab'
    elif question == 'events':
        match = read('matches/statsbomb-3943043.json')
        observations = {'match': match['id'], 'date': match['date'], 'teams': [
            {'team': team, **{key: sum(row[key] for row in match['segments'] if row['team'] == team)
                              for key in ('shots', 'completed_passes', 'goals')}} for team in match['teams']]}
        scope = 'StatsBomb 2024 男足欧洲杯决赛；将非重叠时段的公开计数相加，包含未定位射门，不重建原始事件。'
        route = 'matches.html?match=statsbomb-3943043&team=Spain&view=network#match-explorer'
    elif question == 'players':
        data = read('comparisons.json')
        cohort = [row for row in data['players'] if row['league'] == 'England' and row['season'] == '2017/18'
                  and row['position'] == 'Forward' and row['minutes'] >= 900
                  and isinstance(row['metrics'].get('shots_per90'), (int, float))]
        rows = sorted(cohort, key=lambda row: (-row['metrics']['shots_per90'], row['id']))[:2]
        observations = {'source': data['player_source'], 'league': 'England', 'season': '2017/18',
                        'minimum_minutes': 900, 'eligible_players': len(cohort),
                        'players': [{k: row[k] for k in ('id', 'name', 'position', 'minutes', 'matches', 'date_end', 'metrics')} for row in rows]}
        scope = '同联赛、同赛季、前锋且至少 900 分钟；每 90 分钟射门用于观察，不是能力、潜力或跨来源排名。'
        route = 'entities.html?' + urlencode({'kind':'players','league':'England','season':'2017/18',
                  'position':'Forward','minutes':900,'a':rows[0]['id'],'b':rows[1]['id']}) + '#comparisons'
    else:
        raise ValueError('Unknown example question')
    return {'schema': 'SoccerAnalytics.PublicExample.v1', 'question': question,
            'scope': scope, 'route': route, 'sources': sources, 'observations': observations}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--question', choices=('prediction', 'events', 'players', 'all'), default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    questions = ('prediction', 'events', 'players') if args.question == 'all' else (args.question,)
    result = [answer_question(question) for question in questions]
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
