import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class TestResearchExport(unittest.TestCase):
    def test_search_projection_checks_shared_budget_and_strips_private_trace(self):
        import copy
        from app.ml.experiments import search_parameters, shared_initial_parameters
        from app.run_receipts import content_digest
        from app.web.public_experiments import public_search_comparison

        protocol = {'model':'xgboost','model_seed':42,'timing_policy':'observed_result','evaluation_level':'full',
            'parallel_trials':1,'duplicate_policy':'count_and_evaluate','trials':6,'folds':3,'search_seeds':[11],
            'arms':['random','bayesian','genetic','beam2'],'as_of':'2025-04-01T00:00:00Z','evaluation_start':'2025-03-01T00:00:00Z',
            'evaluation_end':'2025-03-09T00:00:00Z','final_holdout_start':'2025-04-01T00:00:00Z'}
        points = shared_initial_parameters('xgboost',seed=11,include_features=True)
        runs=[]
        for arm in protocol['arms']:
            fits=[]
            def objective(params): fits.extend([1,2,3]); return 1. + params['learning_rate']
            search = search_parameters('xgboost','beam' if arm == 'beam2' else 'random' if arm == 'bayesian' else arm,objective,
                trials=6,seed=11,initial_points=points,include_features=True,
                beam_width=2 if arm == 'beam2' else None,fit_counter=lambda:len(fits))
            if arm == 'bayesian':
                search['protocol']['optimizer']='bayesian'
                search['initialization']['gp_tpe_startup']=4
            runs.append({'arm':arm,'search_seed':11,'model_seed':42,'state':'complete','search':search,
                'selection_log_loss':search['best_loss'],'objective_requests':6,'inner_fit_attempts':18,
                'outer_fit_attempts':1,'outer_rows':8,'outer_log_loss':1.2,'elapsed_seconds':search['elapsed_seconds']+1,
                'private_predictions':[{'match_id':123456789,'secret':'private-sentinel'}]})
        result = {'competition':'PL','schema':'SoccerAnalytics.SearchComparison.v1','status':'complete',
            'adoption_decision':'not_determined','protocol':protocol,'protocol_sha256':content_digest(protocol),
            'admission':{'state':'admitted','development_rows':48,'outer_rows':8,'development_sha256':'a'*64,
                         'validation_identity':'b'*64,'outer_identity':'c'*64,'pending_outer_rows':0,
                         'configuration':{k:protocol[k] for k in ('as_of','evaluation_start','evaluation_end','final_holdout_start','folds')}},'runs':runs,
            'budget':{'objective_requests':24,'inner_fit_attempts':72,'outer_fit_attempts':4,'cached_full_evaluations':0},
            'source_snapshots':[{'provider':'fixture','source':'football-data','competition':'PL','season':'2025',
                'fetched_at':'2025-04-01T00:00:00Z','sha256':'d'*64,'private_path':'private-sentinel'}]}
        report = {'errors':[],'results':[result]}
        public = public_search_comparison(report)
        encoded = json.dumps(public)
        for secret in ('private-sentinel','123456789','best_params','parent_trial_id','match_ids'):
            self.assertNotIn(secret, encoded)
        self.assertEqual(public['comparisons'][0]['runs'][0]['curve'][-1]['fit_attempts'],18)
        warm = copy.deepcopy(report); item=warm['results'][0]
        item['protocol']['initial_training_rows']=30
        item['protocol_sha256']=content_digest(item['protocol'])
        item['admission']['configuration']['initial_training_rows']=30
        item['admission']['effective_initial_training_rows']=32
        item['admission']['validation_rows']=16
        self.assertEqual(public_search_comparison(warm)['comparisons'][0]['validation_rows'],16)
        item['admission']['validation_rows']=17
        with self.assertRaises(ValueError):public_search_comparison(warm)
        for change in ('shared','count','protocol','cache','partial','initial-trial','startup','population','admission','pending','admission-config','space','source-type'):
            bad = copy.deepcopy(report); row=bad['results'][0]
            if change == 'shared': row['runs'][1]['search']['initialization']['points'][0]['l2']=2.
            if change == 'count': row['runs'][0]['inner_fit_attempts']=17
            if change == 'protocol': row['protocol']['model_seed']=7
            if change == 'cache': row['runs'][0]['search']['trials'][0]['cache_hit']=True
            if change == 'partial': row['status']='partial'
            if change == 'initial-trial': row['runs'][0]['search']['trials'][0]['params']['l2']=2.
            if change == 'startup': row['runs'][1]['search']['initialization']['gp_tpe_startup']=1
            if change == 'population': row['runs'][2]['search']['initialization']['ga_population']=6
            if change == 'admission': row['admission']['development_rows']=0
            if change == 'pending': row['admission']['pending_outer_rows']=1
            if change == 'admission-config': row['admission']['configuration']['folds']=2
            if change == 'source-type': row['source_snapshots'][0]['provider']={'private_predictions':[123456789]}
            if change == 'space':
                for run in row['runs']:
                    for point in run['search']['initialization']['points']: point.pop('max_depth')
                    for trial in run['search']['trials'][:4]: trial['params'].pop('max_depth')
            with self.subTest(change=change), self.assertRaises(ValueError): public_search_comparison(bad)

    def test_readiness_projection_strips_private_ids_and_accepts_no_fit_reports(self):
        import copy
        from app.ml.feature_experiments import run_search_comparison
        from app.run_receipts import content_digest
        from app.web.public_experiments import public_search_comparison
        from tests.ml.test_experiments import TestFeatureSearch

        matches = TestFeatureSearch._observed_matches()
        result = run_search_comparison(matches, as_of=matches[-1].result_available_at.isoformat(),
            evaluation_start=matches[64].kickoff.isoformat(), evaluation_end=matches[72].kickoff.isoformat(),
            final_holdout_start=matches[75].kickoff.isoformat(), initial_training_rows=40, admit_only=True)
        result.update(competition='PL', source_snapshots=[{'provider':'fixture','source':'football-data',
            'competition':'PL','season':'2025','fetched_at':result['protocol']['as_of'],'sha256':'a'*64,
            'private_path':'private-sentinel'}])
        ready = result['admission']['readiness']
        ready['private_debug'] = {'path':'private-sentinel', 'match_ids':[987654321]}
        ready['sha256'] = content_digest({k:v for k,v in ready.items() if k!='sha256'})
        report = {'results':[result], 'errors':[]}
        public = public_search_comparison(report)
        self.assertEqual(public['comparisons'], [])
        self.assertEqual(public['readiness'][0]['state'], 'not_blocked')
        self.assertEqual(public['readiness'][0]['development_rows'], 62)
        self.assertEqual(public['readiness'][0]['outer_rows'], 8)
        self.assertEqual(public['readiness'][0]['execution_status'], 'admitted')
        self.assertEqual(public['readiness'][0]['provenance']['code'], result['run_receipt']['code'])
        from app.web.research import export_research_data
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); raw=root/'private-report.json'; raw.write_text(json.dumps(report))
            output=root/'public'; output.mkdir(); existing=output/'model-lab.json'
            existing.write_bytes(b'preserve-existing-history')
            exported=export_research_data(root/'unused-results',output,search_comparison_report=raw)
            self.assertEqual(exported['files'],[str(output/'search-comparison.json')])
            self.assertEqual(exported['figure_files'],[])
            self.assertEqual(existing.read_bytes(),b'preserve-existing-history')
            generated=json.loads((output/'search-comparison.json').read_text())
            self.assertEqual(generated['readiness'][0]['development_rows'],62)
            self.assertNotIn('private-sentinel',json.dumps(generated))

        for secret in ('private-sentinel', '987654321', 'match_ids', 'row_history', 'private_debug'):
            self.assertNotIn(secret, json.dumps(public))
        for change in ('reason', 'feature', 'count', 'hash'):
            bad = copy.deepcopy(report); r=bad['results'][0]['admission']['readiness']
            if change=='reason': r['reason_codes']=['private-sentinel']
            if change=='feature': r['folds'][0]['constant_features']=['private-sentinel']
            if change=='count': r['unique_training_rows']='private-sentinel'
            if change!='hash': r['sha256']=content_digest({k:v for k,v in r.items() if k!='sha256'})
            else: r['sha256']='f'*64
            with self.subTest(change=change), self.assertRaises(ValueError): public_search_comparison(bad)

    def test_analysis_provenance_preserves_missing_history_and_filters_recorded_metadata(self):
        import copy
        import hashlib
        from app.run_receipts import build_run_receipt, content_digest
        from app.web.provenance import public_analysis_provenance, result_artifact

        path = Path('results/paper-followup-20261019/scores.json')
        artifact = result_artifact(path, Path('results'))
        legacy = public_analysis_provenance({}, artifact=artifact)
        self.assertEqual(legacy['status'], 'legacy_missing')
        self.assertIsNone(legacy['code'])
        self.assertEqual(legacy['timing_policy'], 'not_recorded')
        self.assertEqual(artifact['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        receipt = build_run_receipt('score-models', configuration={'seed':42,'timing_policy':'kickoff_order',
                                   'private_path':'/private/input'}, inputs={}, stages={'PL':'complete'})
        public = public_analysis_provenance({'run_receipt':receipt}, artifact=artifact)
        self.assertEqual(public['run_id'], receipt['run_id'])
        self.assertEqual(public['configuration'], {'seed':42,'timing_policy':'kickoff_order'})
        self.assertEqual(public['configuration_sha256'], content_digest(receipt['configuration']))
        self.assertNotIn('/private/input', json.dumps(public))
        self.assertEqual(public['code']['application_sha256'], receipt['code']['application_sha256'])
        tampered = copy.deepcopy(receipt)
        tampered['configuration']['seed'] = 7
        with self.assertRaisesRegex(ValueError, 'receipt'):
            public_analysis_provenance({'run_receipt':tampered})
        with self.assertRaisesRegex(ValueError, 'result identity'):
            public_analysis_provenance({}, artifact={'path':'../private','sha256':'a'*64})

    def test_failed_batch_publish_restores_earlier_outputs_and_removes_new_files(self):
        from app import file_io

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "site"
            root.mkdir()
            (root / "coverage.json").write_bytes(b"previous coverage")
            (root / "case.json").write_bytes(b"previous case")
            original = file_io.os.replace

            def fail_last(source, destination):
                if Path(destination) == root / "case.json" and Path(source).parent.name == "new":
                    raise OSError("synthetic mid-publish failure")
                return original(source, destination)

            with patch.object(file_io.os, "replace", side_effect=fail_last):
                with self.assertRaisesRegex(OSError, "mid-publish"):
                    file_io.write_files_recoverable(root, {"coverage.json": b"new coverage",
                        "added.json": b"new file", "case.json": b"new case"})
            self.assertEqual((root / "coverage.json").read_bytes(), b"previous coverage")
            self.assertEqual((root / "case.json").read_bytes(), b"previous case")
            self.assertFalse((root / "added.json").exists())
            self.assertEqual(list(Path(temporary).glob(".research-export-*")), [])

            def fail_publish_and_restore(source, destination):
                destination, source = Path(destination), Path(source)
                if destination == root / "case.json" or (destination == root / "coverage.json" and source.parent.name != "new"):
                    raise OSError("synthetic publish and rollback failure")
                return original(source, destination)

            with patch.object(file_io.os, "replace", side_effect=fail_publish_and_restore):
                with self.assertRaisesRegex(OSError, "backups retained"):
                    file_io.write_files_recoverable(root, {"coverage.json": b"new coverage", "case.json": b"new case"})
            recovery, = Path(temporary).glob(".research-export-*-recovery")
            self.assertEqual((recovery / "previous-0").read_bytes(), b"previous coverage")
            self.assertEqual((root / "case.json").read_bytes(), b"previous case")

    def test_public_json_transport_bounds_body_reads_cancels_and_deduplicates(self):
        import subprocess

        script = r"""
        const assert=require('node:assert/strict');
        const {createDataClient}=require('./docs/assets/js/core/data-client.js');
        (async()=>{
          let calls=0,release;
          const shared=createDataClient({fetchData:()=>{calls++;return new Promise(resolve=>{release=resolve})}});
          const a=shared.read('representation',{remember:true}),b=shared.read('representation',{remember:true});
          assert.equal(a,b);assert.equal(calls,1);
          release({ok:true,text:async()=>'{"value":0.123456789}'});
          assert.equal((await a).raw,'{"value":0.123456789}');await b;
          assert.equal((await shared.read('representation',{remember:true})).payload.value,0.123456789);
          assert.equal(calls,1);shared.invalidate('representation');
          const again=shared.read('representation');assert.equal(calls,2);
          shared.cancel('representation');await assert.rejects(again,/取消/);
          for(const bodyHangs of [false,true]){
            let signal;
            const client=createDataClient({timeoutMs:10,fetchData:async(_,options)=>{
              signal=options.signal;
              return bodyHangs?{ok:true,text:()=>new Promise(()=>{})}:new Promise(()=>{});
            }});
            await assert.rejects(client.read('held'),/超时/);assert.equal(signal.aborted,true);
          }
          const controller=new AbortController();
          const cancellable=createDataClient({fetchData:()=>new Promise(()=>{})});
          const cancelled=cancellable.read('own',{signal:controller.signal});controller.abort();
          await assert.rejects(cancelled,/取消/);
          let attempt=0;
          const retry=createDataClient({fetchData:async()=>({ok:true,text:async()=>++attempt===1?'broken':'{"ready":true}'})});
          await assert.rejects(retry.read('retry',{remember:true}),SyntaxError);
          assert.equal((await retry.read('retry',{remember:true})).payload.ready,true);
        })().catch(error=>{console.error(error);process.exitCode=1});
        """
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_figure_alternatives_transcribe_public_values_without_new_source_rows(self):
        import xml.etree.ElementTree as ET
        from app.web.public_figures import public_figures

        catalog, assets = public_figures(Path("results"))
        for figure in catalog["figures"]:
            with self.subTest(figure=figure["id"]):
                alternative = figure["text_alternative"]
                self.assertEqual(set(alternative), {"description", "reading_order", "panels", "notes"})
                self.assertTrue(alternative["description"])
                root = ET.fromstring(assets[figure["filename"]])
                original = [" ".join("".join(node.itertext()).split())
                            for node in root.iter("{http://www.w3.org/2000/svg}text")
                            if "".join(node.itertext()).strip()]
                transcribed = alternative["notes"] + [entry for panel in alternative["panels"] for entry in panel["entries"]]
                self.assertCountEqual(transcribed, original)
                self.assertTrue(transcribed)
        chance = next(figure for figure in catalog["figures"] if figure["id"] == "chance-ablations")
        self.assertIn("+0.003156", [entry for panel in chance["text_alternative"]["panels"] for entry in panel["entries"]])

    def test_model_lab_keeps_cohort_selection_and_only_aggregate_fields(self):
        import copy
        from app.web.research import public_model_lab

        source = Path("results/paper-followup-20261019")
        inputs = {name: json.loads((source / f"{name}.json").read_text())
                  for name in ("scores", "ratings", "calibration", "trees")}
        public = public_model_lab(inputs)
        self.assertEqual([len(group["candidates"]) for group in public["groups"]], [5, 12, 6, 9])
        self.assertEqual([group["holdout_rows"] for group in public["groups"]], [76, 70, 76, 76])
        for group, raw in zip(public["groups"], inputs.values()):
            for candidate, original in zip(group["candidates"], raw["results"][0]["results"]):
                self.assertEqual(candidate["selection_log_loss"], original["selection_log_loss"])
                expected = original.get("holdout_metrics", {}).get("log_loss", original.get("holdout_log_loss"))
                self.assertEqual(candidate["holdout"]["log_loss"], expected)
            self.assertIn(group["selected"], {row["id"] for row in group["candidates"]})
        contaminated = copy.deepcopy(inputs)
        for value in contaminated.values():
            value["results"][0]["private_predictions"] = [{"match_id": "private", "probabilities": [0, 0, 1]}]
            value["results"][0]["results"][0]["fitted_state"] = {"secret": 123}
        self.assertEqual(public_model_lab(contaminated), public)
        for mutation in ("nonfinite", "cohort", "selected", "source", "failed"):
            broken = copy.deepcopy(inputs)
            experiment = broken["scores"]["results"][0]
            if mutation == "nonfinite":
                experiment["results"][0]["holdout_metrics"]["log_loss"] = float("nan")
            elif mutation == "cohort":
                experiment["results"][0]["holdout_metrics"]["rows"] -= 1
            elif mutation == "selected":
                experiment["selected"] = "absent"
            elif mutation == "source":
                experiment["source_snapshots"][0]["sha256"] = "invalid"
            else:
                broken["scores"]["errors"] = ["incomplete"]
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                public_model_lab(broken)

    def test_research_links_preserve_provider_selection_and_model_stages(self):
        import subprocess

        script = r"""
        const assert = require('node:assert/strict');
        const {researchRoute} = require('./docs/assets/js/core/research-route.js');
        const {modelLabRows,searchComparisonRows} = require('./docs/assets/js/research/model-lab.js');
        const cases = {'fifa-offers':'matches.html','comparisons':'entities.html',
          'style-retrieval':'entities.html','representation-paired':'models.html',
          'vaep-calibration':'models.html','xt-grid':'models.html','literature':'library.html','coverage':'library.html'};
        for(const [hash,page] of Object.entries(cases)) {
          const url = new URL(researchRoute(`https://example.test/soccer/research.html?team=Netherlands&match=statsbomb-3942819#${hash}`));
          assert.equal(url.pathname,`/soccer/${page}`); assert.equal(url.hash,`#${hash}`);
          assert.equal(url.searchParams.get('team'),'Netherlands'); assert.equal(url.searchParams.get('match'),'statsbomb-3942819');
        }
        assert.equal(new URL(researchRoute('https://example.test/soccer/research.html#unknown')).hash,'#match-explorer');
        const group = {candidates:[{id:'development-choice',selection_log_loss:0.8,holdout:{log_loss:1.2}},
          {id:'other',selection_log_loss:1.1,holdout:{log_loss:0.7}}]};
        assert.deepEqual(modelLabRows(group,'development','log_loss').map(row=>row.value),[0.8,1.1]);
        assert.deepEqual(modelLabRows(group,'holdout','log_loss').map(row=>row.value),[1.2,0.7]);
        assert.deepEqual(modelLabRows(group,'development','accuracy'),[]);
        assert.deepEqual(group.candidates.map(row=>row.id),['development-choice','other']);
        const comparison={timing_policy:'observed_result',outer_rows:8,runs:[{arm:'beam2',search_seed:11,
          selection_log_loss:1.1,outer_log_loss:1.2,elapsed_seconds:3,objective_requests:6,inner_fit_attempts:18,
          outer_fit_attempts:1,failed_trials:0,curve:[{fit_attempts:3,elapsed_seconds:.1,best_log_loss:null}],
          parent_diversity:[{parent_count:2,parent_distance_unit_l2:.4}]}]};
        assert.equal(searchComparisonRows(comparison)[0].parentDistance,.4);
        comparison.runs[0].inner_fit_attempts=null;
        assert.throws(()=>searchComparisonRows(comparison));
        """
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_research_sections_retry_independently_without_reloading_ready_sections(self):
        import subprocess

        script = """
        const assert=require('node:assert/strict');
        const {loadResearchSection,validResearchData}=require('./docs/assets/js/research/research.js');
        const fs=require('node:fs');
        (async()=>{
          const calls=[],renders=[],states=[];
          for(const name of ['coverage','case','methods','representation']){
            const good=JSON.parse(fs.readFileSync(`docs/data/research/${name}.json`,'utf8'));
            assert.equal(validResearchData(name,good),true,name);
            const bad=structuredClone(good);
            if(name==='coverage')bad.coverage=null;
            if(name==='case')bad.team_comparisons=[];
            if(name==='methods')bad.methods=[{}];
            if(name==='representation')bad.data=null;
            assert.equal(validResearchData(name,bad),false,name);
          }
          const sections=['coverage','case','methods','representation'].map(name=>({
            name,url:name,validate:data=>Array.isArray(data.rows),
            render:data=>renders.push([name,data.rows]),
            onState:section=>states.push([section.name,section.state])
          }));
          let fail=true;
          const fetchData=async url=>{
            calls.push(url);
            return {ok:!(fail&&url==='representation'),status:503,
              json:async()=>({rows:[url]})};
          };
          await Promise.all(sections.map(section=>loadResearchSection(section,fetchData)));
          assert.deepEqual(sections.map(s=>s.state),['ready','ready','ready','error']);
          assert.deepEqual(renders.map(r=>r[0]),['coverage','case','methods']);
          assert.equal(sections[3].error,'服务器返回 HTTP 503');
          const healthyData=sections.slice(0,3).map(s=>s.data);
          fail=false;
          await Promise.all(sections.map(section=>loadResearchSection(section,fetchData)));
          assert.deepEqual(calls,['coverage','case','methods','representation','representation']);
          assert.deepEqual(sections.map(s=>s.state),['ready','ready','ready','ready']);
          assert.deepEqual(renders.map(r=>r[0]),['coverage','case','methods','representation']);
          sections.slice(0,3).forEach((s,i)=>assert.equal(s.data,healthyData[i]));
          assert.deepEqual(states.filter(s=>s[0]==='representation').map(s=>s[1]),
            ['loading','error','loading','ready']);
          const bad={name:'invalid',url:'invalid',validate:data=>Array.isArray(data.rows),
            render:()=>assert.fail('invalid data must not render'),onState:()=>{}};
          await loadResearchSection(bad,async()=>({ok:true,json:async()=>({rows:null})}));
          assert.equal(bad.state,'error');assert.equal(bad.error,'返回的数据不完整');
          for(const [failure,want] of [
            [async()=>{throw Error('network detail')},'网络连接失败'],
            [async()=>({ok:true,json:async()=>{throw Error('decoder detail')}}),'返回的数据无法读取']
          ]){
            const section={name:'broken',url:'broken',validate:()=>true,
              render:()=>assert.fail('failed request must not render'),onState:()=>{}};
            await loadResearchSection(section,failure);
            assert.equal(section.state,'error');assert.equal(section.error,want);
          }
          let release,requests=0;
          const waiting={name:'waiting',url:'waiting',validate:()=>true,
            render:()=>{},onState:()=>{}};
          const delayed=()=>{requests++;return new Promise(resolve=>{release=resolve})};
          const first=loadResearchSection(waiting,delayed);
          const second=loadResearchSection(waiting,delayed);
          assert.equal(requests,1);assert.equal(waiting.state,'loading');
          release({ok:true,json:async()=>({})});await Promise.all([first,second]);
          assert.equal(waiting.state,'ready');
        })().catch(error=>{console.error(error);process.exitCode=1});
        """
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_paired_comparison_view_preserves_signed_bounds_and_rejects_stale_cohorts(self):
        import subprocess

        script = """
        const assert=require('node:assert/strict');
        const {representationComparisonRows:f}=require('./docs/assets/js/research/representation-comparison.js');
        const report={data:{splits:{test:{sequences_used:4,matches_used:2}}},
          summaries:[{method:'reference',runs:1},{method:'other',runs:3}],
          test_comparisons:{method:'paired_match_case_bootstrap',reference:'reference',
            test_sequences:4,test_matches:2,replicates:200,comparisons:[{method:'other',runs:3,
              completion_log_loss:{difference:-.1,lower:-.2,upper:.05},
              completion_brier:{difference:-.01,lower:-.02,upper:.005},
              forward_displacement_mae_pitch_pct:{difference:-2,lower:-3,upper:1}}]}};
        const rows=f(report);assert.equal(rows.length,1);
        assert.equal(rows[0].completion_log_loss.difference,-.1);
        assert.equal(rows[0].completion_log_loss.lower,-.2);
        assert.equal(rows[0].completion_log_loss.upper,.05);
        for(const kind of ['legacy','cohort','bounds','nan','duplicate','missing']) {
          const bad=structuredClone(report);
          if(kind==='legacy')delete bad.test_comparisons;
          if(kind==='cohort')bad.test_comparisons.test_sequences=5;
          if(kind==='bounds')bad.test_comparisons.comparisons[0].completion_log_loss.lower=1;
          if(kind==='nan')bad.test_comparisons.comparisons[0].completion_brier.upper=NaN;
          if(kind==='duplicate')bad.test_comparisons.comparisons.push(bad.test_comparisons.comparisons[0]);
          if(kind==='missing')bad.test_comparisons.comparisons=[];
          assert.throws(()=>f(bad),Error);
        }
        """
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_paired_representation_export_omits_predictions_and_checks_cohort_counts(self):
        import copy
        import app.web.research as research

        self.assertTrue(callable(getattr(research, 'public_representation_comparisons', None)))
        metrics = ('completion_log_loss', 'completion_brier', 'forward_displacement_mae_pitch_pct')
        report = {'data': {'splits': {'test': {'sequences_used': 4, 'matches_used': 2}}},
                  'summaries': [{'method': 'reference', 'runs': 1,
                                 'test': {k: {'mean': v} for k, v in zip(metrics, [.5, .25, 10])}},
                                {'method': 'other', 'runs': 2,
                                 'test': {k: {'mean': v} for k, v in zip(metrics, [.6, .3, 12])}}]}
        raw = dict(method='paired_match_case_bootstrap', reference='reference',
                   test_sequences=4, test_matches=2, replicates=200, seed=7,
                   rng='numpy.PCG64', scipy_version='1.18.0', quantiles=[.025, .975],
                   interval_method='percentile', weighting='pooled_sampled_passes',
                   seed_aggregation='mean_loss_across_fixed_seeds',
                   cohort_sha256='a'*64, predictions_sha256='b'*64,
                   method_url='https://journal.r-project.org/articles/RJ-2023-015/',
                   engine_url='https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html',
                   scope='Conditional on fixed models.', limits=['Not simultaneous coverage.'],
                   private_predictions=['omit'], comparisons=[{'method': 'other', 'runs': 2,
                       **{k: dict(difference=d, lower=d-.01, upper=d+.01, private_match='omit')
                          for k, d in zip(metrics, [.1, .05, 2])}}])
        public = research.public_representation_comparisons(report, raw)
        self.assertNotIn('omit', json.dumps(public))
        for key, value in [('difference', .1), ('lower', .09), ('upper', .11)]:
            self.assertAlmostEqual(public['comparisons'][0]['completion_log_loss'][key], value)
        for field, value in [('test_sequences', 5), ('test_matches', 3), ('reference', 'missing')]:
            broken = copy.deepcopy(raw);broken[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                research.public_representation_comparisons(report, broken)
        broken = copy.deepcopy(raw);broken['comparisons'][0]['completion_brier']['difference'] = .2
        with self.assertRaises(ValueError):
            research.public_representation_comparisons(report, broken)

    def test_xt_grid_view_preserves_axes_bounds_and_sample_counts(self):
        import subprocess

        script = """
        const assert=require('node:assert/strict'), fs=require('node:fs');
        const {xtGridRows}=require('./docs/assets/js/research/xt-grid.js');
        const report=JSON.parse(fs.readFileSync('docs/data/research/xt-numerics.json','utf8'));
        const rows=xtGridRows(report);
        assert.equal(rows.length,96);assert.equal(rows[16].x,40);assert.equal(rows[16].y,10);
        assert.equal(rows[16].value,report.grid[1][4]);
        assert.equal(rows[16].width,report.resampling.upper_grid[1][4]-report.resampling.lower_grid[1][4]);
        assert.equal(rows.reduce((n,r)=>n+r.shots,0),report.counts.shots);
        for(const change of ['bound','nan','shape','support','failed']) {
          const bad=structuredClone(report);
          if(change==='bound')bad.resampling.upper_grid[0][0]=-1;
          if(change==='nan')bad.grid[0][0]=NaN;
          if(change==='shape')bad.grid.pop();
          if(change==='support')bad.resampling.match_support.shots[0][0]=999;
          if(change==='failed')bad.resampling.numerical_failures=1;
          assert.throws(()=>xtGridRows(bad),Error);
        }
        """
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_xt_numerics_export_preserves_support_and_omits_private_inputs(self):
        import copy
        from app.web.research import public_xt_numerics

        report = json.loads(Path('results/xt-numerics.json').read_text())
        report['private_matches'] = ['omit']
        report['source']['local_path'] = '/private/omit'
        report['solver']['diagnostics']['raw_payload'] = 'omit'
        report['provenance']['private_receipt'] = 'omit'
        report['resampling']['private_matches'] = ['omit']
        public = public_xt_numerics(report)
        self.assertEqual(public['counts'], report['counts'])
        self.assertEqual(public['support'], report['support'])
        self.assertEqual(public['grid'], report['grid'])
        self.assertEqual(public['resampling']['lower_grid'], report['resampling']['lower_grid'])
        self.assertEqual(public['resampling']['replicates'], 1000)
        self.assertNotIn('omit', json.dumps(public))
        legacy = copy.deepcopy(report); legacy.pop('resampling')
        self.assertNotIn('resampling', public_xt_numerics(legacy))
        for invalid in ('count', 'nonfinite', 'support', 'convergence', 'future', 'digest', 'bounds', 'draw_failure', 'match_support', 'missing_draws'):
            changed = copy.deepcopy(report)
            if invalid == 'count': changed['counts']['attempts'] += 1
            elif invalid == 'nonfinite': changed['grid'][0][0] = float('nan')
            elif invalid == 'support': changed['support']['shots'][0][0] = -1
            elif invalid == 'convergence': changed['solver']['converged'] = False
            elif invalid == 'future': changed['window']['end'] = '2018-04-01'
            elif invalid == 'digest': changed['provenance']['solver_sha256'] = 'invalid'
            elif invalid == 'bounds': changed['resampling']['upper_grid'][0][0] = changed['resampling']['lower_grid'][0][0] - .01
            elif invalid == 'draw_failure': changed['resampling']['numerical_failures'] = 1
            elif invalid == 'match_support': changed['resampling']['match_support']['shots'][0][0] = 209
            else: changed['resampling']['unobserved_replicates'][0][0] = 1001
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'xT'):
                public_xt_numerics(changed)

    def test_passing_motif_window_filter_uses_third_pass_segment(self):
        import subprocess

        payload = {'schema': 'Touchline.PassingMotifs.v1', 'chains': [
            {'id': '1-1', 'team': 'A', 'period': 1, 'passes': [
                {'index': 1, 'segment': 1}, {'index': 2, 'segment': 2},
                {'index': 3, 'segment': 2}, {'index': 4, 'segment': 2}],
             'motifs': ['ABAC', 'ABCD']}]}
        script = "const {passingMotifWindows:f}=require('./docs/assets/js/events/motifs.js'); const d=JSON.parse(process.argv[1]); process.stdout.write(JSON.stringify([f(d,'A',[0,30]),f(d,'A',[30,45]),f(d,'A',[30,45],'ABAC'),f(d,'B',[30,45])]));"
        process = subprocess.run(['node', '-e', script, json.dumps(payload)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        early, later, selected, other = json.loads(process.stdout)
        self.assertEqual(early, [])
        self.assertEqual([r['id'] for r in later], ['1-1:0', '1-1:1'])
        self.assertEqual([p['index'] for p in later[0]['passes']], [1, 2, 3])
        self.assertEqual([r['id'] for r in selected], ['1-1:0'])
        self.assertEqual(other, [])

    def test_passing_motif_export_keeps_only_valid_continuous_derivatives(self):
        import copy
        from app.analytics.events.passing_motifs import summarize
        from app.web.research import public_passing_motifs

        events = [{'index': i, 'period': 1, 'possession': 1, 'possession_team': {'name': 'A'},
                   'team': {'name': 'A'}, 'type': {'name': 'Pass'}, 'minute': 0, 'second': i,
                   'timestamp': f'00:00:0{i}.000', 'player': {'id': i, 'name': str(i)},
                   'pass': {'recipient': {'id': i + 1, 'name': str(i + 1)}}} for i in [1, 2, 3]]
        result = summarize({'home_team': 'A', 'away_team': 'B'}, events)
        result['raw_events'] = ['omit'];result['chains'][0]['uuid'] = 'omit'
        result['chains'][0]['passes'][0]['tracking'] = 'omit'
        public = public_passing_motifs(result, ['A', 'B'])
        self.assertEqual(public['chains'][0]['motifs'], ['ABCD'])
        self.assertNotIn('raw_events', public)
        self.assertNotIn('uuid', public['chains'][0])
        self.assertNotIn('tracking', public['chains'][0]['passes'][0])
        for change in [lambda r: r['chains'][0]['passes'][1].update(player='4'),
                       lambda r: r['chains'][0]['passes'][1].update(clock_ms=9000),
                       lambda r: r['chains'][0]['passes'][2].update(minute=44, second=59, segment=2),
                       lambda r: r['chains'][0].update(motifs=['ABAB']),
                       lambda r: r['audit']['A'].update(windows=0)]:
            broken = copy.deepcopy(result);change(broken)
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'Invalid passing motif'):
                public_passing_motifs(broken, ['A', 'B'])

    def test_passing_motifs_keep_overlap_and_break_on_ball_interruption(self):
        from app.analytics.events.passing_motifs import summarize

        def event(i, sender, recipient, seconds, **extra):
            return {'index': i, 'period': 1, 'possession': 8, 'possession_team': {'name': 'A'},
                    'team': {'name': 'A'}, 'type': {'name': 'Pass'}, 'minute': 0, 'second': int(seconds),
                    'timestamp': f'00:00:{seconds:06.3f}', 'player': {'id': sender, 'name': str(sender)},
                    'location': [30, 40], 'pass': {'recipient': {'id': recipient, 'name': str(recipient)}, 'end_location': [50, 40]}, **extra}
        events = [event(1, 1, 2, 0), event(2, 2, 1, 4, type={'name': 'Pressure'}, team={'name': 'B'}),
                  event(4, 2, 1, 4), event(5, 1, 3, 6, type={'name': 'Carry'}),
                  event(6, 1, 3, 8, location=None), event(7, 3, 4, 13),
                  event(8, 4, 3, 14, type={'name': 'Interception'}, team={'name': 'B'}),
                  event(9, 4, 3, 15), event(10, 3, 2, 19),
                  event(11, 2, 1, 20, **{'pass': {'outcome': {'name': 'Incomplete'}}}),
                  event(12, 2, 1, 21)]
        result = summarize({'home_team': 'A', 'away_team': 'B'}, list(reversed(events)))
        self.assertEqual(len(result['chains']), 1)
        chain = result['chains'][0]
        self.assertEqual([p['index'] for p in chain['passes']], [1, 4, 6, 7])
        self.assertEqual(chain['motifs'], ['ABAC', 'ABCD'])
        self.assertIsNone(chain['passes'][2]['start'])
        self.assertEqual(result['audit']['A']['windows'], 2)
        self.assertEqual(result['audit']['A']['chain_passes'], 4)

    def test_passing_motifs_use_period_clock_and_five_second_boundary(self):
        from app.analytics.events.passing_motifs import motif_label, summarize

        self.assertEqual([motif_label(p) for p in [(1, 2, 1, 2), (1, 2, 1, 3),
                         (1, 2, 3, 1), (1, 2, 3, 2), (1, 2, 3, 4)]],
                         ['ABAB', 'ABAC', 'ABCA', 'ABCB', 'ABCD'])
        def passes(period, offset, last):
            return [{'index': offset + i, 'period': period, 'possession': 9,
                     'possession_team': {'name': 'A'}, 'team': {'name': 'A'},
                     'type': {'name': 'Pass'}, 'minute': 47 if period == 1 else 45,
                     'second': int(t), 'timestamp': f'00:{47 if period == 1 else 0:02}:{t:06.3f}',
                     'player': {'id': a, 'name': str(a)}, 'pass': {'recipient': {'id': b, 'name': str(b)}}}
                    for i, (a, b, t) in enumerate([(1, 2, 0), (2, 3, 4), (3, 1, last)])]
        result = summarize({'home_team': 'A', 'away_team': 'B'}, passes(1, 1, 9) + passes(2, 4, 9.001))
        self.assertEqual(len(result['chains']), 1)
        self.assertEqual(result['chains'][0]['motifs'], ['ABCA'])
        self.assertEqual(result['chains'][0]['passes'][-1]['segment'], 2)
        with self.assertRaises(ValueError):
            summarize({'home_team': 'A', 'away_team': 'B'}, passes(1, 1, 9) * 2)

    def test_calibration_export_checks_cohort_decomposition_and_omits_private_rows(self):
        import copy
        from app.web.research import public_vaep_calibration

        source = json.loads(Path('results/vaep-spadl.json').read_text())
        report = json.loads(Path('results/vaep-calibration.json').read_text())
        report['predictions'] = ['private']
        report['groups'][0]['match_ids'] = ['private']
        report['groups'][0]['pav'][0]['row_ids'] = ['private']
        public = public_vaep_calibration(report, source)
        self.assertEqual(len(public['groups']), 12)
        self.assertNotIn('provenance', public)
        self.assertNotIn('predictions', public)
        self.assertNotIn('match_ids', public['groups'][0])
        self.assertNotIn('row_ids', public['groups'][0]['pav'][0])
        for mutate in [lambda r: r.update(source_run_id='0' * 20),
                       lambda r: r['groups'][0].update(mcb=.5),
                       lambda r: r['groups'][0]['pav'][0].update(count=0),
                       lambda r: r['groups'][0]['binning']['10']['bins'][0].update(positives=-1)]:
            broken = copy.deepcopy(report); mutate(broken)
            with self.subTest(mutation=mutate), self.assertRaisesRegex(ValueError, 'Invalid calibration'):
                public_vaep_calibration(broken, source)

    def test_spadl_vaep_exports_aggregates_and_rejects_split_or_metric_contradictions(self):
        from app.web.research import public_spadl_vaep

        report = json.loads(Path('results/vaep-spadl.json').read_text())
        report['private_predictions'] = ['omit']
        report['leagues'][0]['private_actions'] = ['omit']
        report['leagues'][0]['results'][0]['weights'] = ['omit']
        public = public_spadl_vaep(report)
        self.assertNotIn('private_predictions', public)
        self.assertNotIn('private_actions', public['leagues'][0])
        self.assertNotIn('weights', public['leagues'][0]['results'][0])
        self.assertEqual(len(public['leagues']), 2)
        report['leagues'][0]['results'][0]['metrics']['score']['observed_rate'] = 1.1
        with self.assertRaisesRegex(ValueError, 'Invalid SPADL'):
            public_spadl_vaep(report)
        report = json.loads(Path('results/vaep-spadl.json').read_text())
        report['leagues'][0]['splits']['train']['end'] = report['leagues'][0]['splits']['test']['start']
        with self.assertRaisesRegex(ValueError, 'Invalid SPADL'):
            public_spadl_vaep(report)

    def test_spatial_style_exports_only_aggregate_scores_and_rejects_invalid_metrics(self):
        import copy
        from app.web.research import public_spatial_style

        report = json.loads(Path("results/spatial-style.json").read_text())
        report["private_cohort"] = [{"player_id": "omit"}]
        report["summary"][0]["player_scores"] = ["omit"]
        public = public_spatial_style(report)
        self.assertNotIn("private_cohort", public)
        self.assertNotIn("runs", public)
        self.assertNotIn("inputs", public)
        self.assertNotIn("player_scores", public["summary"][0])
        self.assertEqual(len(public["summary"]), len(report["summary"]))
        self.assertEqual(public["cohort"]["players"], report["cohort"]["players"])
        # Publish seed diagnostics at aggregate precision, never bases or arrays.
        report['parameters'].update(nmf_init='random', nmf_solver='cd', nmf_tol=1e-4)
        report['seed_stability'] = {
            'initialization': 'random', 'metric': 'cosine', 'alignment': 'one-to-one active basis matching',
            'pairs': [dict(method=method, kind=kind, seed_a=a, seed_b=b,
                           components=n, matched_cosines=[1.] * n, dead_components=[0, 0],
                           unassessed_pairs=0, mean_cosine=1., min_cosine=1., private_basis=['omit'])
                      for method in ('nmf_start_only', 'nmf_start_end')
                      for kind, n in [('shots', 4), ('crosses', 4), ('take_ons', 5), ('passes', 5)]
                      for a, b in [(7, 17), (7, 42), (17, 42)]]}
        public = public_spatial_style(report)
        self.assertEqual(len(public['seed_stability']['pairs']), 24)
        self.assertNotIn('private_basis', public['seed_stability']['pairs'][0])
        self.assertEqual(public['parameters']['nmf_init'], 'random')
        for mutation in ('metric', 'missing_pair', 'mismatched_mean', 'invalid_cosine', 'duplicate_pair', 'dead_count'):
            broken = copy.deepcopy(report)
            pair = broken['seed_stability']['pairs'][0]
            if mutation == 'metric': broken['seed_stability']['metric'] = 'unknown'
            elif mutation == 'missing_pair': broken['seed_stability']['pairs'].pop()
            elif mutation == 'mismatched_mean': pair['mean_cosine'] = .4
            elif mutation == 'invalid_cosine': pair['matched_cosines'][0] = 1.1
            elif mutation == 'duplicate_pair': broken['seed_stability']['pairs'][-1] = dict(pair)
            else: pair['dead_components'] = [1, 0]
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, 'Invalid spatial style'):
                public_spatial_style(broken)
        report["summary"][0]["top1"]["mean"] = 1.1
        with self.assertRaisesRegex(ValueError, "Invalid spatial style"):
            public_spatial_style(report)

    def test_displayed_csv_roundtrips_unicode_quotes_and_keeps_spreadsheet_text_inert(self):
        import csv
        import io
        import subprocess

        payload = {"headings": ["球员", "次数", "备注"],
                   "rows": [['O\"Brien,\n球队', -3, '=HYPERLINK("https://example.org")'],
                            ["中文", 0, " \t@SUM(1,2)"]],
                   "context": {"比赛": "2024 欧洲杯", "缺失说明": "无记录不等于不能发生"}}
        script = "const {displayedTableCsv}=require('./docs/assets/js/core/table-export.js'); const p=JSON.parse(process.argv[1]); process.stdout.write(JSON.stringify(displayedTableCsv(p.headings,p.rows,p.context)));"
        result = subprocess.run(["node", "-e", script, json.dumps(payload)], capture_output=True, text=True, check=True)
        content = json.loads(result.stdout)
        self.assertTrue(content.startswith("\ufeff"))
        rows = list(csv.reader(io.StringIO(content.removeprefix("\ufeff"))))
        self.assertEqual(rows[0], ["比赛", "缺失说明", "球员", "次数", "备注"])
        self.assertEqual(rows[1][2:4], ['O\"Brien,\n球队', "-3"])
        self.assertEqual(rows[1][-1], "'" + payload["rows"][0][-1])
        self.assertEqual(rows[2][-1], "'" + payload["rows"][1][-1])
        self.assertTrue(all(row[:2] == list(payload["context"].values()) for row in rows[1:]))

    def test_displayed_snapshot_preserves_types_and_cannot_drift_with_input_mutation(self):
        import subprocess

        script = """
        const assert = require('node:assert/strict');
        const {displayedTableSnapshot} = require('./docs/assets/js/core/table-export.js');
        assert.equal(typeof displayedTableSnapshot, 'function', 'displayed snapshot export missing');
        const headings = ['球员', '次数', '占比'];
        const rows = [['=外部文本', 0, '0.0%'], ['中文', null, '—']];
        const context = {筛选: '英超 / 同位置', 来源: 'https://example.org/source'};
        const snapshot = displayedTableSnapshot(headings, rows, context);
        rows[0][1] = 99; headings[0] = 'changed'; context.筛选 = 'changed';
        assert.deepEqual(snapshot, {
          schema: 'SoccerAnalytics.DisplayedTable.v1', precision: 'displayed',
          row_count: 2, column_count: 3, columns: ['球员', '次数', '占比'],
          rows: [['=外部文本', 0, '0.0%'], ['中文', null, '—']],
          context: {筛选: '英超 / 同位置', 来源: 'https://example.org/source'}
        });
        assert.equal(displayedTableSnapshot(['x'], []).row_count, 0);
        assert.throws(() => displayedTableSnapshot(['x'], [[1, 2]]), TypeError);
        for (const value of [NaN, Infinity, {raw: 'private'}]) {
          assert.throws(() => displayedTableSnapshot(['x'], [[value]]), TypeError);
        }
        """
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_literature_export_separates_read_papers_from_metadata_and_omits_scratch(self):
        from app.web.research import public_literature

        catalog = json.loads(Path("results/literature.json").read_text())
        catalog["local_pdf_root"] = "/private/scratch"
        first = catalog["papers"][0]
        first["raw_pdf_text"] = "omit"
        first["source_checks"] = [{"label": "官方数据快照", "url": "https://github.com/statsbomb/open-data", "note": "只核对公开范围", "local_pdf": "/private/omit"}]
        public = public_literature(catalog)
        self.assertNotIn("local_pdf_root", public)
        self.assertNotIn("raw_pdf_text", public["papers"][0])
        self.assertNotIn("code_links", public["papers"][0])
        self.assertEqual(public["papers"][0]["source_checks"], [{"label": "官方数据快照", "url": "https://github.com/statsbomb/open-data", "note": "只核对公开范围"}])
        first["source_checks"][0]["url"] = "file:///private/data.json"
        with self.assertRaisesRegex(ValueError, "Literature source check"):
            public_literature(catalog)
        first["source_checks"][0]["url"] = "https://github.com/statsbomb/open-data"
        first["read_scope"] = "metadata_only"
        with self.assertRaises(ValueError):
            public_literature(catalog)
        first["read_scope"] = "methods and evaluation"
        first["source_url"] = "file:///private/raw.pdf"
        with self.assertRaises(ValueError):
            public_literature(catalog)
        first["source_url"] = "https://example.org/paper.pdf"
        catalog["candidates"][0]["id"] = first["id"]
        with self.assertRaises(ValueError):
            public_literature(catalog)

    def test_market_reference_exports_matched_aggregates_and_rejects_bad_accounting(self):
        from app.web.research import public_market_reference

        report = json.loads(Path("results/market-reference.json").read_text())
        report["private_predictions"] = ["omit"]
        first = report["results"][0]
        first["match_odds"] = ["omit"]
        first["metrics"]["closing"]["private_values"] = ["omit"]
        public = public_market_reference(report)
        self.assertNotIn("private_predictions", public)
        self.assertNotIn("match_odds", public["results"][0])
        self.assertNotIn("private_values", public["results"][0]["metrics"]["closing"])
        first["paired_matches"] -= 1
        with self.assertRaises(ValueError):
            public_market_reference(report)
        first["paired_matches"] += 1
        first["training_end"] = first["test_start"]
        with self.assertRaises(ValueError):
            public_market_reference(report)

    def test_fifa_offers_export_allows_team_aggregates_only_and_validates_sources(self):
        from app.web.research import public_fifa_offers

        report = json.loads(Path("results/fifa-offers.json").read_text())
        report["private_player_rows"] = ["omit"]
        report["matches"][0]["raw_pdf_text"] = "omit"
        report["matches"][0]["teams"][0]["private_roster"] = ["omit"]
        published = public_fifa_offers(report)
        self.assertNotIn("private_player_rows", published)
        self.assertNotIn("raw_pdf_text", published["matches"][0])
        self.assertNotIn("private_roster", published["matches"][0]["teams"][0])
        first = report["matches"][0]
        for field, value in [("source_url", "https://example.invalid/report.pdf"),
                             ("source_sha256", "not-a-sha"), ("id", "../private")]:
            with self.subTest(field=field):
                previous = first[field]; first[field] = value
                with self.assertRaises(ValueError):
                    public_fifa_offers(report)
                first[field] = previous
        first["teams"][0]["in_front"] += 1
        with self.assertRaises(ValueError):
            public_fifa_offers(report)

    def test_passing_graph_keeps_direction_and_receivers_without_locations(self):
        import subprocess

        rows = [
            {"completed_passes": 8, "nodes": {"1": [20, 40, 2], "2": [30, 20, 1]},
             "edges": [["1", "2", 2], ["2", "1", 1], ["1", "3", 4]]},
            {"completed_passes": 4, "nodes": {"1": [40, 20, 1]},
             "edges": [["1", "2", 3], ["3", "2", 1]]},
        ]
        script = "const {passingGraph} = require('./docs/assets/js/events/passing.js'); process.stdout.write(JSON.stringify(passingGraph(JSON.parse(process.argv[1]))));"
        result = subprocess.run(["node", "-e", script, json.dumps(rows)],
                                capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        graph = json.loads(result.stdout)
        self.assertEqual(graph["total"], 11)
        self.assertEqual(graph["unlinked"], 1)
        edges = {(r["source"], r["target"]): r["count"] for r in graph["edges"]}
        self.assertEqual(edges, {("1", "2"): 5, ("2", "1"): 1,
                                 ("1", "3"): 4, ("3", "2"): 1})
        players = {r["id"]: r for r in graph["players"]}
        self.assertEqual(players["3"]["incoming"], 4)
        self.assertEqual(players["3"]["outgoing"], 1)
        self.assertIsNone(players["3"]["position"])
        self.assertEqual(players["1"]["position"], [20, 20])
        self.assertEqual(players["1"]["outgoing"], 9)

    def test_exploration_bins_keep_stoppage_and_extra_time_separate(self):
        from app.analytics.events.exploration import summarize_match

        match = {"match_id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"}
        def shot(period, minute, team="A"):
            return {"period": period, "minute": minute, "team": {"name": team},
                    "type": {"name": "Shot"}, "location": [120, 80],
                    "shot": {"statsbomb_xg": .2, "outcome": {"name": "Goal"}}}
        data = summarize_match(match, [shot(1, 47), shot(2, 47), shot(3, 94), shot(5, 121)])
        self.assertEqual([r["segment"] for r in data["segments"]], [2, 3, 6])
        self.assertEqual(sum(r["shots"] for r in data["segments"]), 3)
        self.assertEqual(data["segments"][0]["cells"][0]["cell"], 95)
        self.assertNotIn("events", data)

    def test_exploration_network_preserves_direction_and_counts_only_completed_passes(self):
        from app.analytics.events.exploration import summarize_match

        def event(a, b, complete=True):
            return {"period": 1, "minute": 10, "team": {"name": "A"},
                    "type": {"name": "Pass"}, "location": [20, 30],
                    "player": {"id": a, "name": str(a)},
                    "pass": {"recipient": {"id": b, "name": str(b)},
                             **({} if complete else {"outcome": {"name": "Incomplete"}})}}
        data = summarize_match({"match_id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"},
                               [event(1, 2), event(1, 2), event(2, 1), event(1, 2, False)])
        row = data["segments"][0]
        self.assertEqual(row["passes"], 4)
        self.assertEqual(row["completed_passes"], 3)
        self.assertEqual(row["edges"], [["1", "2", 2], ["2", "1", 1]])

    def test_shot_sequences_preserve_order_rebounds_and_period_boundaries(self):
        from app.analytics.events.exploration import summarize_match

        def event(index, kind, period=1, minute=44, team="A", possession=7, owner="A", **extra):
            return {"index": index, "type": {"name": kind}, "period": period,
                    "minute": minute, "second": 12, "team": {"name": team},
                    "possession": possession, "possession_team": {"name": owner},
                    "location": [30, 40], "player": {"id": 1, "name": "Player"},
                    "shot": {"statsbomb_xg": .2, "outcome": {"name": "Saved"},
                             "end_location": [120, 40, 1]}, **extra}
        events = [event(1, "Pass", **{"pass": {"end_location": [50, 40]}}),
                  event(2, "Pressure", team="B"),
                  event(3, "Carry", location=[float("nan"), 40], carry={"end_location": [70, 40]}),
                  event(4, "Shot", minute=47),
                  event(5, "Shot", minute=47, shot={"statsbomb_xg": .1, "outcome": {"name": "Goal"}}),
                  event(6, "Pass", minute=48),
                  event(10, "Shot", period=2, minute=45),
                  event(11, "Shot", period=5, minute=121),
                  event(12, "Shot", period=2, minute=49, possession=None),
                  event(13, "Shot", period=2, minute=49, possession=8, owner="B")]
        data = summarize_match({"match_id": 1, "home_team": "A", "away_team": "B", "date": "2024-01-01"},
                               list(reversed(events)))
        sequences = data.get("shot_sequences", {})
        phases = sequences.get("phases", [])
        self.assertEqual(len(phases), 2)
        self.assertEqual([p["id"] for p in phases], ["1-7", "2-7"])
        self.assertEqual([a["index"] for a in phases[0]["actions"]], [1, 3, 4, 5])
        self.assertEqual(phases[0]["actions"][0]["end"], [50, 40])
        self.assertIsNone(phases[0]["actions"][1]["start"])
        self.assertEqual(phases[0]["actions"][2]["end"], [120, 40])
        self.assertEqual((phases[0]["shots"], phases[0]["goals"]), (2, 1))
        self.assertAlmostEqual(phases[0]["xg"], .3)
        self.assertEqual(phases[0]["opponent_events"], 1)
        self.assertEqual(sequences["unassigned_shots"], {"A": 2, "B": 0})

    def test_calibration_uses_summed_counts_and_retains_empty_bins(self):
        from app.analytics.events.exploration import validation_periods

        rows = [{"date": "2025-01-01", "target": 2, "probabilities": [.1, .2, .7]},
                {"date": "2025-01-02", "target": 0, "probabilities": [.1, .2, .7]},
                {"date": "2025-02-01", "target": 2, "probabilities": [.2, .2, .6]}]
        periods = validation_periods(rows, [.3, .3, .4])
        self.assertEqual([r["count"] for r in periods], [2, 1])
        self.assertEqual(len(periods[0]["calibration"]["top"]), 10)
        bucket = periods[0]["calibration"]["top"][7]
        self.assertEqual(bucket, {"count": 2, "probability_sum": 1.4, "hits": 1})
        self.assertAlmostEqual(periods[0]["accuracy"], .5)
        self.assertEqual(periods[0]["calibration"]["top"][0]["count"], 0)

    def test_shot_sequence_time_filter_keeps_stoppage_in_its_half(self):
        import subprocess

        phases = [{"id": "1-7", "team": "A", "period": 1, "actions": [
                    {"kind": "Pass", "minute": 29}, {"kind": "Shot", "minute": 47}]},
                  {"id": "2-7", "team": "A", "period": 2, "actions": [{"kind": "Shot", "minute": 47}]},
                  {"id": "3-8", "team": "A", "period": 3, "actions": [{"kind": "Shot", "minute": 94}]}]
        script = """const {shotSequencePhases}=require('./docs/assets/js/events/sequences.js');
        const p=JSON.parse(process.argv[1]);
        process.stdout.write(JSON.stringify([[30,45],[45,60],[90,105],[0,15]].map(range=>shotSequencePhases(p,'A',range))));"""
        result = subprocess.run(["node", "-e", script, json.dumps(phases)], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        first, second, extra, empty = json.loads(result.stdout)
        self.assertEqual([p["id"] for p in first], ["1-7"])
        self.assertEqual(len(first[0]["actions"]), 2)
        self.assertEqual([p["id"] for p in second], ["2-7"])
        self.assertEqual([p["id"] for p in extra], ["3-8"])
        self.assertEqual(empty, [])

    def test_team_comparisons_never_merge_leagues_seasons_or_duplicate_fixtures(self):
        from app.analytics.events.exploration import team_comparisons

        row = {"id": 1, "competition": "PL", "season": 2025, "date": "2025-09-01",
               "home_id": "1", "away_id": "2", "home": "A", "away": "B", "home_goals": 2, "away_goals": 0}
        data = team_comparisons([row, row, {**row, "id": 2, "season": 2026}])
        self.assertEqual(len(data), 4)
        self.assertTrue(all(r["matches"] == 1 for r in data))
        self.assertEqual(next(r for r in data if r["name"] == "A")["metrics"]["points_per_match"], 3)

    def test_exploration_export_drops_private_fields_and_rejects_path_traversal(self):
        import shutil
        from app.web.research import public_exploration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "exploration"
            shutil.copytree("results/exploration", root)
            path = root / "comparisons.json"
            comparisons = json.loads(path.read_text())
            comparisons["players"][0]["private_birthday"] = "omit"
            comparisons["players"][0]["metrics"]["game_potential"] = 99
            comparisons["private_training_rows"] = ["omit"]
            path.write_text(json.dumps(comparisons))
            validation_path = root / "validation.json"
            validation = json.loads(validation_path.read_text())
            validation["source"]["private_model_path"] = "/private/model"
            validation_path.write_text(json.dumps(validation))
            match_path = root / "matches" / (json.loads((root / "index.json").read_text())["matches"][0]["id"] + ".json")
            match = json.loads(match_path.read_text())
            match["shot_sequences"] = {"schema": "Touchline.ShotSequences.v1", "private_input": "omit",
                "unassigned_shots": dict.fromkeys(match["teams"], 0), "phases": [{
                "id": "1-7", "period": 1, "possession": 7, "team": match["teams"][0],
                "shots": 1, "goals": 0, "xg": .2, "opponent_events": 0, "raw_events": ["omit"],
                "actions": [{"index": 3, "minute": 1, "second": 2, "kind": "Shot", "player": "P",
                    "recipient": "", "start": [100, 40], "end": [120, 40], "outcome": "Saved", "xg": .2,
                    "freeze_frame": "omit"}]}]}
            match_path.write_text(json.dumps(match))
            published = public_exploration(root)
            sequences = published["matches/" + match["id"] + ".json"].get("shot_sequences", {})
            self.assertEqual(len(sequences.get("phases", [])), 1)
            self.assertNotIn("private_input", sequences)
            self.assertNotIn("raw_events", sequences["phases"][0])
            self.assertNotIn("freeze_frame", sequences["phases"][0]["actions"][0])
            self.assertNotIn("private_birthday", published["comparisons.json"]["players"][0])
            self.assertNotIn("game_potential", published["comparisons.json"]["players"][0]["metrics"])
            self.assertNotIn("private_training_rows", published["comparisons.json"])
            self.assertNotIn("private_model_path", published["validation.json"]["source"])
            index = root / "index.json"
            catalog = json.loads(index.read_text())
            self.assertEqual(published["matches.json"]["matches"][0]["date"], catalog["matches"][0]["date"])
            self.assertEqual(published["matches/" + match["id"] + ".json"]["date"], catalog["matches"][0]["date"])
            match["date"] = "2099-01-01"
            match_path.write_text(json.dumps(match))
            with self.assertRaisesRegex(ValueError, "Public match summary does not match catalog"):
                public_exploration(root)
            match["date"] = catalog["matches"][0]["date"]
            match_path.write_text(json.dumps(match))
            catalog["matches"][0]["id"] = "../private"
            index.write_text(json.dumps(catalog))
            with self.assertRaisesRegex(ValueError, "Invalid public match identity"):
                public_exploration(root)
            catalog["matches"] = []
            index.write_text(json.dumps(catalog))
            empty = public_exploration(root)
            self.assertEqual(empty["matches.json"]["matches"], [])
            self.assertEqual(empty["matches.json"]["scope"], catalog["scope"])
            self.assertFalse(any(name.startswith("matches/") for name in empty))

    def test_exploration_export_does_not_follow_summary_symlinks(self):
        import shutil
        from app.web.research import public_exploration

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "exploration"
            shutil.copytree("results/exploration", root)
            private = Path(directory) / "private.json"
            private.write_text('{"private": true}')
            path = root / "comparisons.json"
            path.unlink()
            path.symlink_to(private)
            with self.assertRaisesRegex(ValueError, "escapes exploration root"):
                public_exploration(root)

    def test_full_experiment_report_exports_aggregates_without_match_ids_or_private_fields(
        self,
    ):
        from app.web.research import export_research_data

        full = json.loads(Path("results/representation-learning.json").read_text())
        full["runs"] = [{"private_prediction": "omit"}]
        full["private_players"] = ["omit"]
        full["configuration"]["local_private_path"] = "omit"
        full["configuration"]["tabular_baselines"] = {
            "inputs": "same observed history", "tree": "fixed train-only tree",
            "mlp": "validation-selected MLP", "local_private_path": "omit",
        }
        full["data"]["splits"]["test"]["match_ids"] = [12345]
        full["data"]["splits"]["test"]["sample_match_ids"] = [12345]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "full.json"
            source.write_text(json.dumps(full))
            export_research_data(
                Path("results"), root / "site", experiment_report=source
            )
            published = json.loads((root / "site" / "representation.json").read_text())
            self.assertNotIn("runs", published)
            self.assertNotIn("private_players", published)
            self.assertNotIn("local_private_path", published["configuration"])
            self.assertEqual(published["configuration"]["tabular_baselines"], {
                "inputs": "same observed history", "tree": "fixed train-only tree",
                "mlp": "validation-selected MLP",
            })
            self.assertNotIn("match_ids", published["data"]["splits"]["test"])
            self.assertNotIn("sample_match_ids", published["data"]["splits"]["test"])
            self.assertEqual(published["summaries"], full["summaries"])

    def test_export_limits_case_to_public_aggregates_and_keeps_source_counts_separate(
        self,
    ):
        from app.web.research import export_research_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = export_research_data(Path("results"), root)
            exported = {Path(p).relative_to(root).as_posix() for p in result["files"]}
            required = {"coverage.json", "case.json", "methods.json", "representation.json", "figures.json"}
            if Path("results/fifa-offers.json").is_file():
                required.add("fifa-offers.json")
            if Path("results/market-reference.json").is_file():
                required.add("market-reference.json")
            if Path("results/literature.json").is_file():
                required.add("literature.json")
            if Path("results/spatial-style.json").is_file():
                required.add("spatial-style.json")
            if Path("results/xt-numerics.json").is_file():
                required.add("xt-numerics.json")
            if Path("results/vaep-spadl.json").is_file():
                required.add("vaep-spadl.json")
            if Path("results/vaep-calibration.json").is_file():
                required.add("vaep-calibration.json")
            if Path("results/paper-followup-20261019/scores.json").is_file():
                required.add("model-lab.json")
            self.assertTrue(required <= exported)
            if (root / "matches.json").exists():
                catalog = json.loads((root / "matches.json").read_text())
                expected = required | {"matches.json", "comparisons.json", "validation.json"}
                expected.update(f"matches/{m['id']}.json" for m in catalog["matches"])
                self.assertEqual(exported, expected)
                for match in catalog["matches"]:
                    payload = json.loads((root / "matches" / f"{match['id']}.json").read_text())
                    self.assertEqual(payload["id"], match["id"])
                    self.assertNotIn("events", payload)
                    self.assertNotIn("location", payload)
            else:
                self.assertEqual(exported, required)
            case = json.loads((root / "case.json").read_text())
            self.assertNotIn("metrics_by_team", case["match"])
            self.assertNotIn("physical_player_rows", case["match"])
            self.assertEqual(len(case["team_comparisons"]), 2)
            self.assertEqual(
                case["team_comparisons"][0]["metrics"]["controlled_possession_pct"][
                    "pre_final_available_matches"
                ],
                6,
            )
            coverage = json.loads((root / "coverage.json").read_text())
            self.assertNotIn("total_unique_matches", coverage)
            source_coverage = json.loads(Path("results/core-coverage.json").read_text())
            source_rows = source_coverage["coverage"]
            self.assertEqual(coverage["summary_scope"], source_coverage.get("summary_scope"))
            self.assertEqual(
                [{key: r[key] for key in ("competition", "matched_summary_documents", "missing_summary_documents", "invalid_summary_documents")}
                 for r in coverage["coverage"] if r["provider"] == "espn_recent"],
                [{key: r[key] for key in ("competition", "matched_summary_documents", "missing_summary_documents", "invalid_summary_documents")}
                 for r in source_rows if r["provider"] == "espn_recent"],
            )
            self.assertCountEqual(
                [
                    (r["provider"], r["competition"], r["source_season"], r["grain"], r["source_match_records"])
                    for r in coverage["coverage"]
                ],
                [
                    (r["provider"], r["competition"], r["season"], r["grain"], r["source_match_records"])
                    for r in source_rows
                ],
            )
            seasons = [
                r
                for r in coverage["coverage"]
                if r["provider"] == "understat" and r["source_season"] == "2021"
            ]
            self.assertTrue(seasons)
            self.assertTrue(all(r["season"] == "2020/2021" for r in seasons))

    def test_missing_input_prevents_any_export(self):
        from app.web.research import export_research_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(FileNotFoundError):
                export_research_data(root / "missing", root / "output")
            self.assertFalse((root / "output").exists())

    def test_figure_export_uses_fixed_allowlist_and_prefers_svg(self):
        from app.web.research import public_figures

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "football").mkdir()
            (root / "football" / "shot-map.svg").write_text("<svg>approved vector</svg>")
            (root / "football" / "shot-map.png").write_bytes(b"fallback bitmap")
            (root / "player-assessment.png").write_bytes(b"approved bitmap")
            (root / "private-player-records.svg").write_text("omit private file")
            (root / "football" / "unreviewed.svg").write_text("omit unreviewed file")
            catalog, assets = public_figures(root)
            self.assertEqual(set(assets), {"football-shot-map.svg", "player-assessment.png"})
            self.assertEqual(assets["football-shot-map.svg"], b"<svg>approved vector</svg>")
            self.assertEqual(len(catalog["figures"]), 2)
            for figure in catalog["figures"]:
                self.assertEqual(set(figure), {"id", "title", "topic", "filename", "format", "alt", "scope", "source", "text_alternative", "aggregate_url"})
                self.assertNotIn(str(root), json.dumps(figure))
            self.assertEqual(public_figures(root / "missing"), ({"figures": []}, {}))

    def test_figure_export_does_not_follow_asset_outside_results(self):
        from app.web.research import public_figures

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "results"
            source.mkdir()
            private = root / "private.svg"
            private.write_text("private material")
            (source / "team-assessment.svg").symlink_to(private)
            self.assertEqual(public_figures(source), ({"figures": []}, {}))

    def test_interrupted_figure_replace_preserves_published_image_and_catalog(self):
        from app import file_io
        from app.web.research import export_research_data

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "site"
            export_research_data(Path("results"), output)
            target = output / "figures" / "analysis-upgrades.svg"
            target.write_bytes(b"previous complete figure")
            catalog = (output / "figures.json").read_bytes()
            replace = file_io.os.replace

            def interrupted_replace(source, destination):
                if Path(destination) == target:
                    raise OSError("interrupted image replacement")
                return replace(source, destination)

            with patch.object(file_io.os, "replace", side_effect=interrupted_replace):
                with self.assertRaisesRegex(OSError, "interrupted image replacement"):
                    export_research_data(Path("results"), output)
            self.assertEqual(target.read_bytes(), b"previous complete figure")
            self.assertEqual((output / "figures.json").read_bytes(), catalog)
            self.assertEqual(list(output.rglob(".*.tmp")), [])

    def test_figure_export_replaces_link_without_overwriting_its_target(self):
        from app.web.research import export_research_data

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "site"
            (output / "figures").mkdir(parents=True)
            original = root / "original.svg"
            original.write_bytes(b"keep original")
            target = output / "figures" / "analysis-upgrades.svg"
            target.symlink_to(original)
            export_research_data(Path("results"), output)
            self.assertEqual(original.read_bytes(), b"keep original")
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.read_bytes(), Path("results/analysis-upgrades.svg").read_bytes())
            self.assertEqual(target.stat().st_mode & 0o777, 0o644)
