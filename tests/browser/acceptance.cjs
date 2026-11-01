const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
// Use an explicit or project-local runtime; never pick an arbitrary npm cache.
function loadPlaywright() {
 if(process.env.PLAYWRIGHT_MODULE)return require(path.resolve(process.env.PLAYWRIGHT_MODULE));
 try{return require('playwright');}catch(error){if(error.code!=='MODULE_NOT_FOUND')throw error;}
 throw new Error('Playwright is unavailable. Install it locally or set PLAYWRIGHT_MODULE to an existing runtime.');
}
const {chromium}=loadPlaywright();
const AxeBuilder=require('@axe-core/playwright').default;
const root=path.resolve(__dirname,'../..'),docs=path.join(root,'docs'),out=path.resolve(process.env.TOUCHLINE_BROWSER_OUTPUT||path.join(root,'output/playwright/frontend'));fs.mkdirSync(out,{recursive:true});
const server=http.createServer((req,res)=>{const url=new URL(req.url,'http://localhost'),p=path.resolve(docs,'.'+(url.pathname==='/'?'/index.html':url.pathname));if(!p.startsWith(docs+path.sep)||!fs.existsSync(p)||!fs.statSync(p).isFile()){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png'})[path.extname(p)]||'application/octet-stream');res.end(fs.readFileSync(p));});
const selectedCases=process.env.FRONTEND_CASE===undefined?null:new Set(process.env.FRONTEND_CASE.split(',').map(name=>name.trim()).filter(Boolean));
const knownCases=new Set();
let browser;const results=[],errors=[];const reportName=selectedCases?'acceptance-affected.json':'acceptance.json';
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r)); const base=`http://127.0.0.1:${server.address().port}`;
 const executable=process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE||chromium.executablePath();
 browser=await chromium.launch({executablePath:executable,headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true});const page=await context.newPage();page.on('pageerror',error=>errors.push({url:page.url(),message:error.message}));
 async function go(url){await page.goto(base+'/'+url);await page.waitForLoadState('networkidle');}
 async function check(name,fn){knownCases.add(name);if(selectedCases&&!selectedCases.has(name))return;try{await fn();results.push({name,status:'passed'});}catch(error){results.push({name,status:'failed',error:error.message,stack:error.stack});await page.screenshot({path:path.join(out,name+'-failure.png')}).catch(()=>{});} console.log(JSON.stringify(results.at(-1)));fs.writeFileSync(path.join(out,reportName),JSON.stringify({results,errors},null,2));}
 await check('routes-and-responsive-layout',async()=>{
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:width===1440?1000:844});
   for(const file of ['index.html','studio.html','matches.html','entities.html','models.html','library.html','predictions.html']){
    await go(file);assert.equal(await page.locator('h1').count(),1,file);
    assert.equal(await page.locator('.workspace-nav a[aria-current=page]').count(),1,file);
    const metrics=await page.evaluate(()=>({width:innerWidth,document:document.documentElement.scrollWidth,height:document.documentElement.scrollHeight,chart:document.querySelector('#match-chart')?.getBoundingClientRect().top,invalid:[...document.querySelectorAll('select,input')].filter(el=>!el.closest('label')&&!el.labels?.length&&!el.getAttribute('aria-label')).map(el=>el.id)}));
    assert.ok(metrics.document<=width+1,`${file} ${width} overflow ${metrics.document}`);assert.deepEqual(metrics.invalid,[],file);
    if(width!==320)await page.screenshot({path:path.join(out,file.replace('.html','')+'-'+width+'.png')});
    results.push({name:`layout-${file}-${width}`,status:'passed',...metrics});
   }
  }
 });
 await page.setViewportSize({width:1440,height:1000});
 await check('match-controls-and-export',async()=>{
  await go('matches.html?match=statsbomb-3942819&team=Netherlands&view=network&period=second#match-explorer');
  assert.equal(await page.locator('#match-team').inputValue(),'Netherlands');assert.equal(await page.locator('#match-view').inputValue(),'network');assert.equal(await page.locator('#match-period').inputValue(),'second');
  await page.locator('.match-catalog > summary').click();await page.locator('#match-search').fill('no_such_team');assert.equal(await page.locator('#match-catalog button').count(),0);assert.match(await page.locator('#match-search-status').innerText(),/没有匹配/);await page.locator('#match-search').fill('Spain');assert.ok(await page.locator('#match-catalog button').count()>0);
  await page.locator('.match-catalog > summary').click();await page.locator('.filter-disclosure > summary').click();
  await page.locator('#match-view').selectOption('sequence');await page.locator('#match-period').selectOption('all');assert.ok(await page.locator('#match-phase option').count()>0);
  await page.locator('#match-action').selectOption('0');await page.locator('#match-next-action').click();assert.equal(await page.locator('#match-action').inputValue(),'1');
  const before=await page.locator('#match-chart svg').getAttribute('viewBox');const downloadEvent=page.waitForEvent('download');await page.locator('#match-export').click();const download=await downloadEvent;await download.saveAs(path.join(out,'sequence.svg'));const svg=fs.readFileSync(path.join(out,'sequence.svg'),'utf8');assert.match(svg,/viewBox="0 0 760/);assert.match(svg,/StatsBomb/);assert.notEqual(before,'0 0 760 636');
  await page.locator('#match-view').selectOption('motifs');assert.ok(await page.locator('#match-table table').count());const motif=page.waitForEvent('download');await page.locator('#match-export').click();await (await motif).saveAs(path.join(out,'motifs.svg'));assert.match(fs.readFileSync(path.join(out,'motifs.svg'),'utf8'),/窗口/);
  await page.locator('#match-view').selectOption('shots');const circle=await page.locator('#match-chart svg circle').first().getAttribute('r');assert.equal(circle,'56');
 });
 await check('tabs-keyboard-back-and-legacy-link',async()=>{
  await go('research.html?fifa=fifa-fcwc2025-1&offers_view=shares#fifa-offers');assert.match(page.url(),/matches.html/);assert.equal(await page.locator('[role=tabpanel]:visible').getAttribute('id'),'fifa-offers');assert.equal(await page.locator('#fifa-mode').inputValue(),'shares');
  await page.locator('#tab-fifa-offers').focus();await page.keyboard.press('ArrowRight');assert.equal(await page.locator('[role=tabpanel]:visible').getAttribute('id'),'case');assert.equal(await page.locator('#tab-case').evaluate(el=>el===document.activeElement),true);
  await page.goBack();await page.waitForLoadState('networkidle');assert.equal(await page.locator('[role=tabpanel]:visible').getAttribute('id'),'fifa-offers');
  await page.locator('#tab-fifa-offers').focus();await page.keyboard.press('Home');assert.equal(await page.locator('[role=tabpanel]:visible').getAttribute('id'),'match-explorer');
 });
 await check('native-chart-dialog-focus-and-events',async()=>{
  await go('matches.html?view=network');const opener=page.locator('.chart-open').first();await opener.click();assert.equal(await page.locator('.chart-reader').evaluate(el=>el.open),true,'dialog opens');assert.ok(await page.locator('.chart-reader #match-chart svg').count());
  for(let i=0;i<5;i++)await page.keyboard.press('Tab');assert.equal(await page.locator('.chart-reader').evaluate(el=>el.contains(document.activeElement)),true,'focus stays inside');
  await page.keyboard.press('Escape');await page.waitForFunction(()=>!document.querySelector('.chart-reader').open&&document.activeElement.classList.contains('chart-open'));assert.equal(await opener.evaluate(el=>el===document.activeElement),true,'opener regains focus');assert.equal(await page.locator('#match-explorer #match-chart').count(),1);
 });
 await check('entity-search-and-selection-restoration',async()=>{
  await go('entities.html?kind=players&league=England&season=2017%2F18&minutes=900');assert.equal(await page.locator('#compare-kind').inputValue(),'players');assert.equal(await page.locator('#compare-season').inputValue(),'2017/18');
  const previous=await page.locator('#compare-b').inputValue(),a=await page.locator('#compare-a').inputValue();await page.locator('#compare-filters > summary').click();await page.locator('#compare-search').fill('Pickford');assert.ok(await page.locator('.subject-search-result').count());assert.equal(await page.locator('#compare-b').inputValue(),previous);assert.equal(await page.locator('#compare-a').inputValue(),a);
  await page.getByRole('button',{name:'Jordan Pickford 设为 A',exact:true}).click();assert.equal(await page.locator('#compare-a').inputValue(),'wyscout:10131');
  const url=page.url();await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('#compare-a').inputValue(),'wyscout:10131');assert.match(url,/a=wyscout/);
  assert.equal(await page.locator('#compare-chart .paired-metric').count(),8);
  await page.locator('#comparisons .chart-data > summary').click();const download=page.waitForEvent('download');await page.locator('#compare-table .table-csv-export').click();await (await download).saveAs(path.join(out,'comparison.csv'));assert.match(fs.readFileSync(path.join(out,'comparison.csv'),'utf8'),/2017\/18/);
 });
 await check('model-groups-selection-and-diagnostics',async()=>{
  await go('models.html');assert.equal(await page.locator('#lab-group option').count(),4);assert.equal(await page.locator('.lab-result').count(),5);assert.match(await page.locator('#lab-cohort').innerText(),/76 场/);
  await page.locator('#lab-group').selectOption('ratings-PL');assert.equal(await page.locator('.lab-result').count(),12);assert.match(await page.locator('#lab-cohort').innerText(),/70 场/);
  await page.locator('#lab-stage').selectOption('development');await page.locator('#lab-metric').selectOption('accuracy');assert.match(await page.locator('#lab-chart').innerText(),/仅提供开发期/);assert.equal(new URL(page.url()).searchParams.get('stage'),'development');assert.equal(new URL(page.url()).searchParams.get('metric'),'accuracy');await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('#lab-stage').inputValue(),'development');assert.equal(await page.locator('#lab-metric').inputValue(),'accuracy');await page.locator('#lab-metric').selectOption('log_loss');assert.equal(await page.locator('.lab-result').count(),12);
  await page.locator('#tab-representations').click();await page.waitForFunction(()=>document.querySelectorAll('#experiment-chart .interval-result').length>0);assert.equal(await page.locator('#experiment-chart .interval-result').count(),JSON.parse(fs.readFileSync(path.join(docs,'data/research/representation.json'))).summaries.length);
  await page.locator('#tab-vaep-calibration').click();await page.waitForFunction(()=>!document.querySelector('#vaep-calibration-bins').disabled);await page.locator('#vaep-calibration-bins').selectOption('9');assert.match(await page.locator('#vaep-calibration-summary').innerText(),/9 箱/);
  await page.locator('#tab-xt-grid').click();await page.waitForFunction(()=>!document.querySelector('#xt-grid-view').disabled);await page.locator('#xt-grid-view').selectOption('shots');assert.equal(await page.locator('#xt-grid-table tbody tr').count(),96);
  const controlled=JSON.parse(fs.readFileSync(path.join(docs,'data/research/model-lab.json')));
  controlled.search_comparisons=[{competition:"PL",timing_policy:"observed_result",development_rows:48,outer_rows:8,evaluation_start:"2025-03-01T00:00:00Z",evaluation_end:"2025-03-09T00:00:00Z",final_holdout_start:"2025-04-01T00:00:00Z",protocol_sha256:"a".repeat(64),runs:["beam1","beam2"].map((arm,index)=>({arm,search_seed:11,selection_log_loss:1.12,outer_log_loss:1.2,objective_requests:4,inner_fit_attempts:12,outer_fit_attempts:1,failed_trials:0,elapsed_seconds:2+index,curve:[1,2,3,4].map(i=>({objective_requests:i,fit_attempts:i*3,elapsed_seconds:i/3,best_log_loss:1.2-i*.02}))}))}];
  await page.route('**/data/research/model-lab.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(controlled)}));
  try{await go('models.html?experiment=trees-PL#model-lab');await page.locator('#lab-search-table-0').waitFor();assert.equal(await page.locator('#lab-search-table-0 tbody tr').count(),2);assert.equal(await page.locator('#lab-search-table-0 tbody tr').first().locator('td').count(),await page.locator('#lab-search-table-0 thead th').count());assert.equal(await page.locator('#lab-search-table-0 tbody tr').first().locator('td').last().innerText(),'暂无');assert.match(await page.locator('#lab-search-comparisons').innerText(),/开发期外层检验/);assert.match(await page.locator('#lab-search-table-0').innerText(),/Beam width 2/);const detail=page.locator('#lab-search-curve-0').locator('..');await detail.locator(':scope > summary').click();assert.equal(await page.locator('#lab-search-curve-0 tbody tr').count(),8);assert.match(await page.locator('#lab-search-curve-0').innerText(),/实际拟合累计/);for(const width of [390,320]){await page.setViewportSize({width,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth),'search comparison mobile overflow');if(width===390)await page.locator('#lab-search-comparisons').screenshot({path:path.join(out,'search-comparison-390.png')});}await page.setViewportSize({width:1440,height:1000});await page.locator('#lab-search-comparisons').screenshot({path:path.join(out,'search-comparison-1440.png')});const scan=await new AxeBuilder({page}).include('#lab-search-comparisons').analyze();fs.writeFileSync(path.join(out,'search-accessibility.json'),JSON.stringify(scan,null,2));assert.deepEqual(scan.violations,[]);}finally{await page.unroute('**/data/research/model-lab.json');}
 });
 await check('search-readiness-states-isolation-and-export',async()=>{
  // Controlled match inputs passed through public_search_comparison; never published research results.
  const fixtures={"insufficient":{"schema":"Touchline.SearchComparison.v1","comparisons":[],"readiness":[{"competition":"PL","timing_state":"admitted","state":"insufficient","sha256":"334851716b43581ed4b42fdc675259afc4d72686147f5e89b0da91bb597f58a5","reason_codes":["NO_OBSERVED_HISTORY_IN_TRAINING_ROWS"],"warning_codes":["INDEPENDENT_CONFIRMATION_NOT_REGISTERED","CONSTANT_BASELINE_TRAINING_FEATURES"],"unique_training_rows":50,"unique_validation_rows":10,"folds":[{"fit_as_of":"2024-02-20T00:00:00+00:00","training_rows":50,"validation_rows":4,"labels_purged_at_fit":0,"training_rows_without_history":50,"constant_features":["home_elo","away_elo","elo_diff","home_form_points","away_form_points","form_points_diff","home_goals_for","away_goals_for","attack_diff","home_goals_against","away_goals_against","defense_diff","home_goal_difference","away_goal_difference","goal_difference_diff","home_win_rate","away_win_rate","win_rate_diff","home_clean_sheet_rate","away_clean_sheet_rate","home_rest_days","away_rest_days","rest_diff","home_venue_points","away_venue_points","venue_points_diff","home_matches_seen","away_matches_seen","experience_diff","home_advantage"],"varying_features":[],"validation_constant_features":["home_advantage"]},{"fit_as_of":"2024-02-24T00:00:00+00:00","training_rows":50,"validation_rows":3,"labels_purged_at_fit":4,"training_rows_without_history":50,"constant_features":["home_elo","away_elo","elo_diff","home_form_points","away_form_points","form_points_diff","home_goals_for","away_goals_for","attack_diff","home_goals_against","away_goals_against","defense_diff","home_goal_difference","away_goal_difference","goal_difference_diff","home_win_rate","away_win_rate","win_rate_diff","home_clean_sheet_rate","away_clean_sheet_rate","home_rest_days","away_rest_days","rest_diff","home_venue_points","away_venue_points","venue_points_diff","home_matches_seen","away_matches_seen","experience_diff","home_advantage"],"varying_features":[],"validation_constant_features":["home_clean_sheet_rate","away_clean_sheet_rate","home_advantage"]},{"fit_as_of":"2024-02-27T00:00:00+00:00","training_rows":50,"validation_rows":3,"labels_purged_at_fit":7,"training_rows_without_history":50,"constant_features":["home_elo","away_elo","elo_diff","home_form_points","away_form_points","form_points_diff","home_goals_for","away_goals_for","attack_diff","home_goals_against","away_goals_against","defense_diff","home_goal_difference","away_goal_difference","goal_difference_diff","home_win_rate","away_win_rate","win_rate_diff","home_clean_sheet_rate","away_clean_sheet_rate","home_rest_days","away_rest_days","rest_diff","home_venue_points","away_venue_points","venue_points_diff","home_matches_seen","away_matches_seen","experience_diff","home_advantage"],"varying_features":[],"validation_constant_features":["home_advantage"]}],"model_fit_attempts":0,"followup":{"state":"available_unregistered","registered":false,"observed_rows":3,"pending_rows":0},"sources":[{"provider":"controlled-browser-fixture","source":"fixture","competition":"PL","season":"2025","fetched_at":"2024-03-20T02:00:00+00:00","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"protocol_sha256":"0dc6af7f36a4955d888fc082258258787b02cea90f83b4f04e3758839dad9493"}]},"not_blocked":{"schema":"Touchline.SearchComparison.v1","comparisons":[],"readiness":[{"competition":"PL","timing_state":"admitted","state":"not_blocked","sha256":"0f15b4427d822c83ee8bd1db537bc5556aceeee846866bb3f8d896bba3bd2310","reason_codes":[],"warning_codes":["INDEPENDENT_CONFIRMATION_NOT_REGISTERED"],"unique_training_rows":55,"unique_validation_rows":22,"folds":[{"fit_as_of":"2024-02-12T00:00:00+00:00","training_rows":40,"validation_rows":8,"labels_purged_at_fit":0,"training_rows_without_history":3,"constant_features":["home_advantage"],"varying_features":["home_elo","away_elo","elo_diff","home_form_points","away_form_points","form_points_diff","home_goals_for","away_goals_for","attack_diff","home_goals_against","away_goals_against","defense_diff","home_goal_difference","away_goal_difference","goal_difference_diff","home_win_rate","away_win_rate","win_rate_diff","home_clean_sheet_rate","away_clean_sheet_rate","home_rest_days","away_rest_days","rest_diff","home_venue_points","away_venue_points","venue_points_diff","home_matches_seen","away_matches_seen","experience_diff"],"validation_constant_features":["home_advantage"]},{"fit_as_of":"2024-02-20T00:00:00+00:00","training_rows":48,"validation_rows":7,"labels_purged_at_fit":0,"training_rows_without_history":3,"constant_features":["home_advantage"],"varying_features":["home_elo","away_elo","elo_diff","home_form_points","away_form_points","form_points_diff","home_goals_for","away_goals_for","attack_diff","home_goals_against","away_goals_against","defense_diff","home_goal_difference","away_goal_difference","goal_difference_diff","home_win_rate","away_win_rate","win_rate_diff","home_clean_sheet_rate","away_clean_sheet_rate","home_rest_days","away_rest_days","rest_diff","home_venue_points","away_venue_points","venue_points_diff","home_matches_seen","away_matches_seen","experience_diff"],"validation_constant_features":["home_advantage"]},{"fit_as_of":"2024-02-27T00:00:00+00:00","training_rows":55,"validation_rows":7,"labels_purged_at_fit":0,"training_rows_without_history":3,"constant_features":["home_advantage"],"varying_features":["home_elo","away_elo","elo_diff","home_form_points","away_form_points","form_points_diff","home_goals_for","away_goals_for","attack_diff","home_goals_against","away_goals_against","defense_diff","home_goal_difference","away_goal_difference","goal_difference_diff","home_win_rate","away_win_rate","win_rate_diff","home_clean_sheet_rate","away_clean_sheet_rate","home_rest_days","away_rest_days","rest_diff","home_venue_points","away_venue_points","venue_points_diff","home_matches_seen","away_matches_seen","experience_diff"],"validation_constant_features":["home_advantage"]}],"model_fit_attempts":0,"followup":{"state":"available_unregistered","registered":false,"observed_rows":3,"pending_rows":0},"sources":[{"provider":"controlled-browser-fixture","source":"fixture","competition":"PL","season":"2025","fetched_at":"2024-03-20T02:00:00+00:00","sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}],"protocol_sha256":"59c2f91d677cae5f8193137cad2fede86c84de99d952d5a6c58a440f0a03edfd"}]}};
  fixtures.insufficient.readiness[0].provenance={configuration:{as_of:'2024-03-21T10:15:30.123456+08:00'}};
  fixtures.insufficient.readiness[0].sources[0].season=2025; // Both integer and string seasons are valid public source values.
  fixtures.legacy=structuredClone(fixtures.not_blocked);
  delete fixtures.legacy.readiness[0].sources;
  for(const fold of fixtures.legacy.readiness[0].folds)delete fold.fit_as_of;
  let state='insufficient',searchFailed=false,modelFailed=false,invalid=null,withComparisons=false;
  const model=JSON.parse(fs.readFileSync(path.join(docs,'data/research/model-lab.json')));
  const comparison={competition:'PL',timing_policy:'observed_result',development_rows:48,outer_rows:8,evaluation_start:'2025-03-01T00:00:00Z',evaluation_end:'2025-03-09T00:00:00Z',final_holdout_start:'2025-04-01T00:00:00Z',protocol_sha256:'a'.repeat(64),runs:[{arm:'beam2',search_seed:11,selection_log_loss:1.1,outer_log_loss:1.2,objective_requests:4,inner_fit_attempts:12,outer_fit_attempts:1,failed_trials:0,elapsed_seconds:3,curve:[{objective_requests:4,fit_attempts:12,elapsed_seconds:2,best_log_loss:1.1}],parent_diversity:[{parent_count:2,parent_distance_unit_l2:.4}]}]};
  const requests={model:0,search:0};
  async function observeValidation(){
   await page.evaluate(()=>{
    window.labValidationCalls={comparison:0,readiness:0,utc:0};
    for(const [name,key] of [['searchComparisonRows','comparison'],['searchReadinessRows','readiness'],['readinessUtc','utc']]){
     // Observe the real validators without replacing their validation or normalization.
     window[name]=new Proxy(window[name],{apply(target,thisArg,args){window.labValidationCalls[key]++;return Reflect.apply(target,thisArg,args);}});
    }
   });
  }
  async function inspectTimeScope(expected,file){
   const panel=page.locator('#lab-readiness-0'),scope=panel.locator(':scope > .readiness-time-scope');
   assert.equal(await scope.isVisible(),true,'diagnostic time scope must be visible in body');
   assert.ok((await scope.innerText()).includes(`诊断观察时点（UTC）：${expected.asOf}`),'diagnostic as_of must be shown without substituting a source or historical time');
   for(const line of expected.sources)assert.ok((await scope.innerText()).includes(line),'source season and fetch time must be visible');
   if(!expected.sources.length)assert.match(await scope.innerText(),/来源赛季与获取时刻：未记录/);
   assert.match(await scope.innerText(),/当前无拟合准备盘点/);assert.match(await scope.innerText(),/来源获取时刻不代表比赛完赛或结果最早可用时刻/);assert.match(await scope.innerText(),/每折训练截点不是最终留出/);
   await panel.getByText('查看逐折训练、验证与常数特征',{exact:true}).click();
   const table=page.locator('#lab-readiness-table-0'),rows=table.locator('tbody tr');
   assert.equal(await table.locator('thead th').nth(1).innerText(),'训练截点（UTC）');
   assert.equal(await rows.count(),expected.folds.length);
   for(let i=0;i<expected.folds.length;i++)assert.equal(await rows.nth(i).locator('td').nth(1).innerText(),expected.folds[i],'fold fit cutoff must match its recorded UTC time');
   await table.locator('.table-inspector > summary').click();
   const event=page.waitForEvent('download');await table.getByRole('button',{name:'下载当前视图 JSON'}).click();
   await (await event).saveAs(path.join(out,file));const view=JSON.parse(fs.readFileSync(path.join(out,file),'utf8'));
   assert.equal(view.context['诊断观察时点（UTC）'],expected.asOf);
   assert.equal(view.context['来源赛季与获取时刻（UTC）'],expected.sources.join(' / ')||'未记录');
   assert.equal(view.columns[1],'训练截点（UTC）');assert.deepEqual(view.rows.map(row=>row[1]),expected.folds);
   assert.match(view.context['范围'],/每折训练截点不是最终留出/);
   await table.locator('.table-inspector > summary').click();
   await panel.getByText('查看逐折训练、验证与常数特征',{exact:true}).click();
  }
  // Read the installed real public summary through the production page before any data routes are installed.
  const realBytes=fs.readFileSync(path.join(docs,'data/research/search-comparison.json')),real=JSON.parse(realBytes).readiness[0];
  const fetched=page.waitForResponse(response=>response.url().endsWith('/data/research/search-comparison.json')&&response.status()===200);
  await go('models.html?experiment=trees-PL#model-lab');assert.deepEqual(await (await fetched).body(),realBytes,'real page must receive the installed summary bytes');
  const recordedUtc=value=>{assert.match(value,/(?:Z|\+00:00)$/,'real public timestamps are recorded in UTC');return value.replace('T',' ').replace(/(?:Z|\+00:00)$/,' UTC');};
  await inspectTimeScope({asOf:recordedUtc(real.provenance.configuration.as_of),sources:real.sources.map(source=>`${source.competition} · 来源赛季 ${source.season} · ${source.provider} · 获取时刻（UTC）：${recordedUtc(source.fetched_at)}`),folds:real.folds.map(fold=>recordedUtc(fold.fit_as_of))},'real-time-view.json');
  assert.ok((await page.locator('#lab-cohort').innerText()).includes(model.groups.find(group=>group.id==='trees-PL').holdout_start.slice(0,10)),'historical model retains its own holdout range');
  for(const width of [390,320]){
   await page.setViewportSize({width,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`real time scope overflow ${width}`);
   await page.locator('#lab-readiness-0').getByText('查看逐折训练、验证与常数特征',{exact:true}).click();
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`real cutoff table overflow ${width}`);
   await page.locator('#lab-readiness-0').screenshot({path:path.join(out,`real-time-scope-${width}.png`)});
   await page.locator('#lab-readiness-0').getByText('查看逐折训练、验证与常数特征',{exact:true}).click();
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#lab-readiness-0').getByText('查看逐折训练、验证与常数特征',{exact:true}).click();
  const realScan=await new AxeBuilder({page}).include('#lab-search-comparisons').analyze();
  fs.writeFileSync(path.join(out,'real-time-accessibility.json'),JSON.stringify(realScan,null,2));assert.deepEqual(realScan.violations,[]);
  const secret='private-browser-sentinel';
  await page.route('**/data/research/model-lab.json',route=>{
   requests.model++;
   if(modelFailed)return route.fulfill({status:503,body:'unavailable'});
   const payload=structuredClone(model);
   if(withComparisons)payload.search_comparisons=[structuredClone(comparison)];
   if(invalid==='model')payload.search_comparisons[0].runs[0].inner_fit_attempts=null;
   return route.fulfill({contentType:'application/json',body:JSON.stringify(payload)});
  });
  await page.route('**/data/research/search-comparison.json',route=>{
   requests.search++;
   if(searchFailed)return route.fulfill({status:503,body:'unavailable'});
   const payload=structuredClone(fixtures[state]);
   payload.readiness[0].private_debug={match_ids:[987654321],row_history:[secret],private_path:secret};
   if(withComparisons){payload.comparisons=[structuredClone(comparison)];payload.comparisons[0].runs[0].outer_log_loss=1.2345;}
   if(invalid==='readiness')payload.readiness[0].folds[0].validation_rows=-1;
   if(invalid==='comparison')payload.comparisons[0].runs[0].inner_fit_attempts=null;
   if(invalid==='time')payload.readiness[0].provenance={configuration:{as_of:'2024-03-21T10:15:30'}};
   return route.fulfill({contentType:'application/json',body:JSON.stringify(payload)});
  });
  try{
   await go('models.html?experiment=trees-PL#model-lab');
   assert.equal(await page.locator('#lab-readiness-0').count(),1,'readiness must render through production model page');
   const panel=page.locator('#lab-readiness-0');
   assert.match(await panel.innerText(),/时间准入：通过/);
   assert.match(await panel.innerText(),/效益准备不足 · 未执行搜索/);
   assert.match(await panel.innerText(),/去重训练 50 场 · 去重验证 10 场/);
   assert.match(await panel.innerText(),/后续已观察 3 场/);
   assert.match(await panel.innerText(),/所有折的训练行均无已观察历史/);
   assert.equal(await page.locator('[id^=lab-search-curve-]').count(),0,'readiness is not an optimization curve');
   assert.ok(await page.locator('.lab-result').count()>0,'historical metrics remain separate');
   await inspectTimeScope({asOf:'2024-03-21 02:15:30.123456 UTC',sources:['PL · 来源赛季 2025 · controlled-browser-fixture · 获取时刻（UTC）：2024-03-20 02:00:00 UTC'],folds:['2024-02-20 00:00:00 UTC','2024-02-24 00:00:00 UTC','2024-02-27 00:00:00 UTC']},'controlled-time-view.json');
   await panel.locator(':scope > details > summary').click();
   assert.equal(await page.locator('#lab-readiness-table-0 tbody tr').count(),3);
   assert.equal(await page.locator('#lab-readiness-table-0 tbody tr').first().locator('td').nth(5).innerText(),'30');
   for(const width of [1440,390,320]){
    await page.setViewportSize({width,height:width===1440?1000:844});
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`readiness overflow ${width}`);
    if(width!==320)await panel.screenshot({path:path.join(out,`readiness-${width}.png`)});
   }
   await page.setViewportSize({width:1440,height:1000});
   const scan=await new AxeBuilder({page}).include('#lab-search-comparisons').analyze();
   fs.writeFileSync(path.join(out,'readiness-accessibility.json'),JSON.stringify(scan,null,2));assert.deepEqual(scan.violations,[]);
   await page.locator('#lab-readiness-table-0 .table-inspector > summary').click();
   const downloadEvent=page.waitForEvent('download');
   await page.locator('#lab-readiness-table-0').getByRole('button',{name:'下载当前视图 JSON'}).click();
   const download=await downloadEvent;await download.saveAs(path.join(out,'readiness-view.json'));
   for(const text of [await page.locator('#lab-search-comparisons').innerText(),fs.readFileSync(path.join(out,'readiness-view.json'),'utf8')]){
    for(const forbidden of [secret,'987654321','row_history','match_ids','private_path'])assert.ok(!text.includes(forbidden),`private field leaked: ${forbidden}`);
   }
   await observeValidation();const beforeRedraw={...requests};
   await page.locator('#lab-group').selectOption('ratings-PL');await page.locator('#lab-group').selectOption('trees-PL');
   await page.locator('#lab-stage').selectOption('development');await page.locator('#lab-metric').selectOption('accuracy');
   assert.equal(await page.locator('#lab-readiness-0').isVisible(),true,'readiness survives missing development metric');
   assert.equal(new URL(page.url()).searchParams.get('stage'),'development');
   assert.deepEqual(await page.evaluate(()=>window.labValidationCalls),{comparison:0,readiness:0,utc:0},'redraw must reuse accepted search rows and normalized times');
   assert.deepEqual(requests,beforeRedraw,'redraw must not fetch the same summaries again');
   state='legacy';await go('models.html?experiment=trees-PL#model-lab');
   await inspectTimeScope({asOf:'未记录',sources:[],folds:['未记录','未记录','未记录']},'legacy-time-view.json');
   state='not_blocked';await go('models.html?experiment=trees-PL#model-lab');
   assert.match(await panel.innerText(),/尚未发现结构性阻断 · 有效性未证明/);
   assert.equal(await page.locator('[id^=lab-search-table-]').count(),0);
   assert.match(await panel.innerText(),/后续独立确认尚未注册/);
   await page.locator('#tab-representations').click();await page.locator('#tab-model-lab').click();
   assert.equal(await panel.isVisible(),true,'existing tab navigation retains readiness');
   searchFailed=true;await go('models.html?experiment=trees-PL#model-lab');
   assert.match(await page.locator('#lab-search-comparisons').innerText(),/搜索准备摘要暂不可用/);
   assert.equal(await page.locator('#lab-readiness-0').count(),0,'failed source cannot retain a stale readiness state');
   assert.equal(await page.locator('#lab-group').isEnabled(),true);assert.ok(await page.locator('.lab-result').count()>0);
   searchFailed=false;await page.getByRole('button',{name:'重试搜索准备摘要'}).click();await page.locator('#lab-readiness-0').waitFor();
   modelFailed=true;await go('models.html#model-lab');
   assert.match(await page.locator('#lab-status').innerText(),/实验汇总暂不可用/);
   assert.equal(await page.locator('#lab-group').isDisabled(),true);assert.equal(await panel.isVisible(),true,'search source survives model source failure');
   modelFailed=false;await page.locator('#lab-retry').click();await page.waitForFunction(()=>!document.querySelector('#lab-group').disabled);
   withComparisons=true;
   for(const broken of ['readiness','comparison','model','time']){
    invalid=broken;await go('models.html?experiment=trees-PL#model-lab');
    if(broken==='model'){
     assert.match(await page.locator('#lab-status').innerText(),/实验汇总暂不可用/,'malformed embedded costs must fail at the model boundary');
     assert.equal(await panel.isVisible(),true,'valid independent readiness survives malformed model source');
     assert.equal(await page.locator('#lab-group').isDisabled(),true);
    }else{
     assert.match(await page.locator('#lab-search-comparisons').innerText(),/搜索准备摘要暂不可用/,`malformed ${broken} must fail at the search boundary`);
     assert.equal(await panel.count(),0,'malformed source must not be partially accepted');
     assert.ok(await page.locator('.lab-result').count()>0,'valid historical source survives malformed search source');
    }
    assert.equal(await page.locator('#lab-search-table-0 tbody tr').count(),1,'valid comparison from the other source stays visible');
    invalid=null;await observeValidation();
    const beforeRetry={...requests};
    await (broken==='model'?page.locator('#lab-retry'):page.getByRole('button',{name:'重试搜索准备摘要'})).click();
    await page.waitForFunction(()=>!document.querySelector('#lab-group').disabled&&document.querySelector('#lab-readiness-0'));
    assert.equal(requests.model,beforeRetry.model+1);assert.equal(requests.search,beforeRetry.search+1);
    assert.deepEqual(await page.evaluate(()=>window.labValidationCalls),{comparison:2,readiness:1,utc:5},'retry validates each newly received item and timestamp once');
    assert.equal(await page.locator('[id^=lab-search-table-]').count(),1,'same competition and protocol remain deduplicated');
    assert.match(await page.locator('#lab-search-table-0').innerText(),/1\.2345/,'independent search source retains precedence');
    assert.match(await page.locator('#lab-search-table-0').innerText(),/0\.400/,'normalized parent distance is preserved');
    assert.equal(await page.locator('#lab-search-curve-0 tbody tr').count(),1,'normalized cost curve is preserved');
    const accepted={...requests};
    for(const [id,value] of [['lab-group','scores-PL'],['lab-group','trees-PL'],['lab-stage','development'],['lab-metric','accuracy'],['lab-metric','log_loss'],['lab-stage','holdout']])await page.locator('#'+id).selectOption(value);
    await page.locator('#tab-representations').click();await page.locator('#tab-model-lab').click();
    assert.deepEqual(await page.evaluate(()=>window.labValidationCalls),{comparison:2,readiness:1,utc:5},'filter and tab redraw must not validate accepted comparisons, readiness or times again');
    assert.deepEqual(requests,accepted,'filter and tab redraw do not reload accepted sources');
   }
  }finally{
   await page.unroute('**/data/research/model-lab.json');await page.unroute('**/data/research/search-comparison.json');
   await page.setViewportSize({width:1440,height:1000});
  }
 });
 await check('library-deep-links-gallery-dialog-and-coverage',async()=>{
  await go('library.html?figure=paper-followup-20261019-global-contributions#figures');assert.match(await page.locator('#figure-reader-title').innerText(),/同一输出/);
  await page.locator('#figure-enlarge').click();assert.equal(await page.locator('#figure-dialog').evaluate(el=>el.open),true);await page.keyboard.press('Escape');assert.equal(await page.locator('#figure-enlarge').evaluate(el=>el===document.activeElement),true);
  await page.locator('#tab-coverage').click();await page.waitForFunction(()=>!document.querySelector('#competition-filter').disabled);await page.locator('#competition-filter').selectOption('champions_league');assert.match(await page.locator('#coverage-rows').innerText(),/欧冠/);await page.locator('#coverage-rows button').first().click();const eventDetail=await page.locator('#coverage-detail').innerText();assert.match(eventDetail,/来源/);assert.match(eventDetail,/比赛日期范围/);
  const coverage=JSON.parse(fs.readFileSync(path.join(docs,'data/research/coverage.json'),'utf8'));
  const csvRows=coverage.coverage.filter(row=>row.provider==='football_csv'),latestCsvSeason=csvRows.map(row=>row.season).sort().at(-1);assert.ok(latestCsvSeason);
  for(const row of csvRows.filter(row=>row.season===latestCsvSeason)){
   await page.locator('#reset-filters').click();await page.locator('#provider-filter').selectOption(row.provider);await page.locator('#season-filter').selectOption(row.season);await page.locator('#competition-filter').selectOption(row.competition);
   assert.equal(await page.locator('#coverage-rows tr').count(),1);assert.equal(await page.locator('#coverage-rows tr td').nth(2).innerText(),row.source_match_records.toLocaleString());await page.locator('#coverage-rows button').click();assert.ok((await page.locator('#coverage-detail').innerText()).includes(`${row.date_start} 至 ${row.date_end}`));
  }
  const schedule=coverage.coverage.find(row=>row.provider==='understat'&&row.grain==='schedule_with_xg'&&row.date_end>coverage.observed_at.slice(0,10))||coverage.coverage.find(row=>row.provider==='understat'&&row.grain==='schedule_with_xg');assert.ok(schedule);
  await page.locator('#reset-filters').click();await page.locator('#provider-filter').selectOption(schedule.provider);await page.locator('#season-filter').selectOption(schedule.season);await page.locator('#competition-filter').selectOption(schedule.competition);await page.locator('#coverage-rows button').first().click();
  const detail=await page.locator('#coverage-detail').innerText();assert.match(detail,/赛程日期范围（含未赛）/);assert.ok(detail.includes(`${schedule.date_start} 至 ${schedule.date_end}`));assert.ok((await page.locator('#snapshot-date').innerText()).includes(`覆盖核对日期 ${coverage.observed_at.slice(0,10)}`));
  await page.locator('#tab-literature').click();await page.waitForFunction(()=>!document.querySelector('#paper-search').disabled);await page.locator('#paper-search').fill('XGBoost');assert.ok(await page.locator('#paper-cards .method-card').count()>0);await page.locator('#paper-search').fill('no_such_paper');assert.match(await page.locator('#paper-cards').innerText(),/没有匹配/);
 });
 await check('failure-isolation-and-retry',async()=>{
  let block=true;await page.route('**/data/research/representation.json',route=>block?route.fulfill({status:503,body:'unavailable'}):route.continue());await go('models.html#representations');assert.equal(await page.locator('#representation-retry').isVisible(),true);assert.equal(await page.locator('#lab-group').isDisabled(),true);await page.locator('#tab-model-lab').click();await page.waitForFunction(()=>!document.querySelector('#lab-group').disabled);assert.equal(await page.locator('#lab-group option').count(),4);await page.locator('#tab-representations').click();block=false;await page.locator('#representation-retry').click();await page.waitForFunction(()=>document.querySelector('#representation-retry').hidden);assert.equal(await page.locator('#representation-retry').isVisible(),false);await page.unroute('**/data/research/representation.json');
  let matchBlock=true;await page.route('**/data/research/matches/statsbomb-3943043.json',route=>matchBlock?route.fulfill({status:503,body:'unavailable'}):route.continue());await go('matches.html?match=statsbomb-3943043&team=England&view=network&period=second');assert.equal(await page.locator('#match-retry').isVisible(),true);matchBlock=false;await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);assert.equal(await page.locator('#match-export').isEnabled(),true);assert.equal(await page.locator('#match-team').inputValue(),'England');assert.equal(await page.locator('#match-period').inputValue(),'second');assert.equal(await page.locator('#match-view').inputValue(),'network');await page.unroute('**/data/research/matches/statsbomb-3943043.json');
 });
 await check('match-catalog-empty-and-failure-states',async()=>{
  const real=JSON.parse(fs.readFileSync(path.join(docs,'data/research/matches.json'),'utf8'));
  const empty={...real,matches:[],scope:'controlled-empty-catalog-fixture · '+real.scope};
  const deepLink='matches.html?match=statsbomb-3943043&team=England&view=network&period=second#match-explorer';
  let mode='empty',details=0,catalogRequests=0;
  async function openCatalog(){await page.goto('about:blank');await go(deepLink);}
  await page.route('**/data/research/matches/*.json',route=>{details++;return route.continue();});
  await page.route('**/data/research/matches.json',route=>{
   catalogRequests++;
   if(mode==='http')return route.fulfill({status:503,body:'Unavailable'});
   if(mode==='json')return route.fulfill({status:200,contentType:'application/json',body:'{invalid'});
   if(mode==='shape')return route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({...empty,matches:null})});
   return mode==='real'?route.continue():route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(empty)});
  });
  try{
   for(const width of [1440,390]){
    await page.setViewportSize({width,height:width===1440?1000:844});mode='empty';const before=details;
    await openCatalog();
    assert.match(await page.locator('#match-status').innerText(),/当前公开样本没有比赛记录/,'a legal empty catalog is not a loading failure');
    assert.equal(await page.locator('#match-retry').isHidden(),true);
    assert.equal(await page.locator('#match-explorer .chart-actions').isHidden(),true);
    assert.equal(await page.locator('#match-explorer .chart-data').isHidden(),true);
    assert.equal(await page.locator('#match-catalog-count').innerText(),'0 场公开样本');
    assert.equal(await page.locator('#match-select').isDisabled(),true);
    assert.equal(await page.locator('#match-select').innerText(),'暂无公开比赛');
    for(const id of ['match-search','match-team','match-view','match-period','match-export'])assert.equal(await page.locator('#'+id).isDisabled(),true,id);
    assert.equal(await page.locator('#match-time-control').isHidden(),true);assert.equal(await page.locator('#match-chart svg').count(),0);assert.equal(details,before,'empty catalogs request no match detail');
    assert.equal(new URL(page.url()).searchParams.get('match'),'statsbomb-3943043');
    assert.match(await page.locator('#match-status').innerText(),/controlled-empty-catalog-fixture/);
    assert.equal(await page.locator('#match-explorer a[href="https://github.com/hudl/open-data"]').count(),1);
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
    await page.locator('#match-explorer').screenshot({path:path.join(out,`empty-catalog-${width}.png`)});
    const scan=await new AxeBuilder({page}).include('#match-explorer').analyze();fs.writeFileSync(path.join(out,`empty-catalog-axe-${width}.json`),JSON.stringify(scan,null,2));assert.deepEqual(scan.violations,[]);
    mode='http';await openCatalog();assert.match(await page.locator('#match-status').innerText(),/比赛目录未就绪/);assert.equal(await page.locator('#match-catalog-count').innerText(),'目录未加载');assert.equal(await page.locator('#match-retry').isVisible(),true);
    await page.locator('#match-explorer').screenshot({path:path.join(out,`failed-catalog-${width}.png`)});
    mode='real';await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);
    assert.equal(await page.locator('#match-explorer .chart-actions').isVisible(),true);
    assert.equal(await page.locator('#match-team').inputValue(),'England');assert.equal(await page.locator('#match-view').inputValue(),'network');assert.equal(await page.locator('#match-period').inputValue(),'second');
    assert.ok((await page.locator('#match-status').innerText()).includes(real.matches.find(row=>row.id==='statsbomb-3943043').date));
    await page.locator('.match-catalog > summary').click();await page.locator('#match-search').fill('no_such_team');assert.match(await page.locator('#match-search-status').innerText(),/没有匹配比赛/);assert.equal(await page.locator('#match-export').isEnabled(),true,'filtering the list keeps the current match');
    await page.locator('#match-search').fill('Spain');assert.ok(await page.locator('#match-catalog button').count()>0);await page.locator('#match-search').fill('');assert.equal(await page.locator('#match-catalog button').count(),real.matches.length);
    await page.locator('#match-explorer').screenshot({path:path.join(out,`populated-catalog-${width}.png`)});
   }
   for(const failure of ['json','shape']){
    mode=failure;await openCatalog();assert.equal(await page.locator('#match-retry').isVisible(),true);assert.match(await page.locator('#match-status').innerText(),/比赛目录未就绪/);
    mode='empty';const before=catalogRequests;await page.locator('#match-retry').click();await page.waitForFunction(()=>document.querySelector('#match-status').textContent.includes('当前公开样本没有比赛记录'));assert.equal(catalogRequests,before+1,'retry re-reads an invalid catalog');assert.equal(await page.locator('#match-retry').isHidden(),true);
   }
  }finally{await page.unroute('**/data/research/matches.json');await page.unroute('**/data/research/matches/*.json');await page.setViewportSize({width:1440,height:1000});}
 });
 await check('catalog-failure-malformed-model-and-retry',async()=>{
  let block=true;await page.route('**/data/research/matches.json',route=>block?route.fulfill({status:503,body:'unavailable'}):route.continue());await go('matches.html?match=statsbomb-3943043&team=England');assert.equal(await page.locator('#match-retry').isVisible(),true);block=false;await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);assert.equal(await page.locator('#match-team').inputValue(),'England');await page.unroute('**/data/research/matches.json');
  let invalid=true;await page.route('**/data/research/model-lab.json',route=>invalid?route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({schema:'Touchline.ModelLab.v1',groups:[{id:'bad',candidates:[]}]})}):route.continue());await go('models.html');assert.equal(await page.locator('#lab-retry').isVisible(),true);assert.equal(await page.locator('#lab-metric').isDisabled(),true);invalid=false;await page.locator('#lab-retry').click();await page.waitForFunction(()=>!document.querySelector('#lab-group').disabled);assert.equal(await page.locator('#lab-group option').count(),4);await page.unroute('**/data/research/model-lab.json');
 });
 await check('style-retry-replaces-options-and-handler-state',async()=>{
  const good=JSON.parse(fs.readFileSync(path.join(docs,'data/research/spatial-style.json'))),bad=structuredClone(good);const index=bad.summary.findIndex(row=>row.scope==='all'&&row.pool==='same_league');assert.ok(index>=0);bad.summary.splice(index,1);
  let invalid=true;await page.route('**/data/research/spatial-style.json',route=>invalid?route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(bad)}):route.continue());await go('entities.html#style-retrieval');assert.equal(await page.locator('#style-retry').isVisible(),true);invalid=false;await page.locator('#style-retry').click();await page.waitForFunction(()=>!document.querySelector('#style-league').disabled);assert.equal(await page.locator('#style-league option').count(),6);await page.locator('#style-league').selectOption('England');await page.locator('#style-league').selectOption('all');assert.equal(await page.locator('#style-league').isDisabled(),false);assert.equal(await page.locator('#style-pool').isDisabled(),false);assert.equal(await page.locator('#style-table tbody tr').count(),4);await page.unroute('**/data/research/spatial-style.json');
 });
 await check('malformed-match-is-not-cached-and-can-recover',async()=>{
  let unavailable=true;await page.route('**/data/research/matches/statsbomb-3942819.json',route=>unavailable?route.fulfill({status:503,body:'unavailable'}):route.continue());
  try {await go('matches.html?match=statsbomb-3942819&team=missing-team&view=network&period=second#match-explorer');assert.equal(await page.locator('#match-retry').isVisible(),true);assert.equal(await page.locator('#match-restore-status').innerText(),'');assert.equal(await page.locator('#match-export').isDisabled(),true);unavailable=false;await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);assert.equal(await page.locator('#match-team').inputValue(),'Netherlands');assert.equal(new URL(page.url()).searchParams.get('team'),'Netherlands');assert.match(await page.locator('#match-restore-status').innerText(),/球队.*不可用/);}finally{await page.unroute('**/data/research/matches/statsbomb-3942819.json');}
  const good=JSON.parse(fs.readFileSync(path.join(docs,'data/research/matches/statsbomb-3943043.json'))),bad=structuredClone(good);delete bad.segments.find(row=>row.team==='Spain').cells;
  let invalid=true,requests=0;await page.route('**/data/research/matches/statsbomb-3943043.json',route=>{requests++;return invalid?route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(bad)}):route.continue();});await go('matches.html?match=statsbomb-3943043&team=Spain&view=shots');assert.equal(await page.locator('#match-retry').isVisible(),true);assert.equal(await page.locator('#match-export').isDisabled(),true);assert.equal(await page.locator('#match-time-control').isHidden(),true);assert.equal(await page.locator('#match-summary').innerText(),'');invalid=false;await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);assert.equal(requests,2);assert.equal(await page.locator('#match-retry').isVisible(),false);assert.equal(await page.locator('#match-time-control').isVisible(),true);await page.unroute('**/data/research/matches/statsbomb-3943043.json');
 });
 await check('responsive-comparison-and-fifa-reading',async()=>{
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:width===1440?1000:844});
   for(const target of ['entities.html','matches.html#fifa-offers']){
    await go(target);
    const semantics=await page.evaluate(()=>{
     const chart=document.querySelector('#comparisons:not([hidden]) #compare-chart,#fifa-offers:not([hidden]) #fifa-chart');
     const label=chart.querySelector('.paired-value.is-b > span:first-child');
     const svgLabel=[...chart.querySelectorAll('svg text')].find(el=>el.textContent.startsWith('B '));
     return{pageColor:getComputedStyle(label).color,svgColor:getComputedStyle(svgLabel).fill,symbol:label.textContent,fill:svgLabel.getAttribute('fill'),configured:window.TouchlineVisuals.comparison[1].textColor};
    });
    assert.equal(semantics.pageColor,semantics.svgColor);assert.equal(semantics.fill,semantics.configured);assert.equal(semantics.symbol,'B ◇');
    const info=await page.evaluate(()=>{const plot=document.querySelector('#comparisons:not([hidden]) .paired-metric,#fifa-offers:not([hidden]) .paired-metric');return{width:document.documentElement.scrollWidth,top:plot.getBoundingClientRect().top};});assert.ok(info.width<=width+1);assert.ok(info.top<844,`first metric below viewport ${target} ${width}: ${info.top}`);await page.screenshot({path:path.join(out,(target.startsWith('entities')?'entities-reading-':'fifa-reading-')+width+'.png')});
   }
  }
 });
 await check('lazy-model-data-and-primary-metric-url',async()=>{
  let requests=0;const count=request=>{if(request.url().endsWith('/data/research/representation.json'))requests++;};page.on('request',count);
  try{
   await go('models.html');assert.equal(requests,0);assert.equal(await page.locator('#vaep-calibration-league').isDisabled(),true);
   await page.locator('#tab-representations').click();await page.waitForFunction(()=>document.querySelectorAll('#experiment-chart .interval-result').length>0&&!document.querySelector('#representation-paired-metric').disabled);assert.equal(requests,1);
   await page.locator('#experiment-metric').selectOption('forward_displacement_mae_pitch_pct');assert.equal(new URL(page.url()).searchParams.get('experiment_metric'),'forward_displacement_mae_pitch_pct');
   await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('#experiment-metric').inputValue(),'forward_displacement_mae_pitch_pct');assert.equal(requests,2);
  }finally{page.off('request',count);}
 });
 await check('cancelled-snapshot-and-research-can-retry',async()=>{
  let held=true;await page.route('**/data/scores.json',route=>held?undefined:route.continue());
  try{
   const request=page.waitForRequest('**/data/scores.json');await page.goto(base+'/predictions.html',{waitUntil:'domcontentloaded'});await request;await page.waitForFunction(()=>performance.getEntriesByType('resource').some(row=>row.name.endsWith('/data/predictions.json')));
   await page.evaluate(()=>window.TouchlineData.cancel('./data/scores.json'));await page.waitForFunction(()=>!document.querySelector('#snapshot-retry').hidden&&!document.querySelector('#snapshot-retry').disabled);
   await page.locator('details:has(#prediction-errors) > summary').click();
   assert.match(await page.locator('#prediction-errors').innerText(),/SCORES_ARTIFACT_UNAVAILABLE/);assert.match(await page.locator('#recommended-bets').innerText(),/候选列表未就绪/);assert.equal(await page.locator('#recommended-bets article').count(),0);
   held=false;await page.locator('#snapshot-retry').click();await page.waitForFunction(()=>document.querySelector('#snapshot-retry').hidden);assert.doesNotMatch(await page.locator('#data-health').innerText(),/正在读取|完整性检查未通过/);
  }finally{await page.unroute('**/data/scores.json');}
  let researchHeld=true,requests=0;await page.route('**/data/research/representation.json',route=>{requests++;if(!researchHeld)return route.continue();});
  try{
   const request=page.waitForRequest('**/data/research/representation.json');await page.goto(base+'/models.html#representations',{waitUntil:'domcontentloaded'});await request;await page.waitForFunction(()=>document.querySelector('#representation-retry').disabled);
   await page.evaluate(()=>window.TouchlineData.cancel('./data/research/representation.json'));await page.waitForFunction(()=>!document.querySelector('#representation-retry').hidden&&!document.querySelector('#representation-retry').disabled);
   researchHeld=false;await page.locator('#representation-retry').click();await page.waitForFunction(()=>document.querySelector('#representation-retry').hidden);
   await page.locator('#representation-paired > summary').click();await page.locator('#representation-paired-retry').click();await page.waitForFunction(()=>!document.querySelector('#representation-paired-metric').disabled);assert.equal(requests,2);
  }finally{await page.unroute('**/data/research/representation.json');}
 });
 await check('readable-mobile-chart-and-public-figure-text',async()=>{
  await page.setViewportSize({width:390,height:844});await go('matches.html?view=shots');
  const fonts=await page.locator('#match-chart svg text').evaluateAll(rows=>rows.filter(row=>/^\d+$/.test(row.textContent)).map(row=>parseFloat(getComputedStyle(row).fontSize)*Math.hypot(row.getScreenCTM().a,row.getScreenCTM().b)));assert.ok(fonts.length>0);assert.ok(fonts.every(value=>value>=12));
  await page.locator('#match-explorer .chart-table-open').click();assert.equal(await page.locator('#match-explorer .chart-data').evaluate(el=>el.open),true);assert.equal(await page.locator('#match-explorer .chart-data > summary').evaluate(el=>el===document.activeElement),true);
  await go('library.html?figure=chance-ablations#figures');await page.locator('#figure-reader-alternative > summary').click();assert.match(await page.locator('#figure-reader-text').innerText(),/\+0\.003156/);assert.match(await page.locator('#figure-reader-text').innerText(),/Brier/);
  await page.locator('#figure-enlarge').click();await page.locator('#figure-dialog-alternative > summary').click();assert.match(await page.locator('#figure-dialog-text').innerText(),/\+0\.003156/);await page.keyboard.press('Escape');
 });
 await check('match-time-selection-and-snapshot-consistency',async()=>{
  await page.setViewportSize({width:390,height:844});
  const match=JSON.parse(fs.readFileSync(path.join(docs,'data/research/matches/statsbomb-3943043.json'),'utf8'));
  const segment=match.segments.find(row=>row.team==='Spain'&&row.segment===3);
  await go('matches.html?match=statsbomb-3943043&team=Spain&view=shots');
  assert.equal(await page.locator('#match-timeline button').count(),6);
  const bin=page.locator('#match-timeline [data-segment="3"]');
  assert.equal(await bin.locator('strong').innerText(),String(segment.shots));
  await bin.focus();await page.keyboard.press('Enter');
  assert.equal(await page.locator('#match-period').inputValue(),'custom');
  assert.equal(await page.locator('#match-start').inputValue(),'45');assert.equal(await page.locator('#match-end').inputValue(),'60');
  assert.equal(await bin.evaluate(el=>el===document.activeElement),true,'keyboard focus survives render');
  assert.equal(await page.locator('#match-timeline [aria-pressed=true]').count(),1);
  assert.match(await page.locator('#match-summary').innerText(),new RegExp(`${segment.shots} 次射门`));
  const numbers=await page.locator('#match-table tbody tr').evaluateAll(rows=>rows.map(row=>[...row.cells].map(cell=>cell.textContent)));
  assert.equal(numbers.reduce((sum,row)=>sum+Number(row[1]),0),segment.shots-segment.unlocated_shots,'table reconciles with located shots');
  const selectedUrl=page.url();assert.equal(new URL(selectedUrl).searchParams.get('start'),'45');
  await page.reload();await page.waitForLoadState('networkidle');assert.equal(await bin.getAttribute('aria-pressed'),'true');
  await page.locator('#match-explorer .chart-table-open').click();
  const jsonEvent=page.waitForEvent('download');await page.locator('#match-table .table-inspector > summary').click();await page.locator('#match-table .table-inspector button').click();
  const download=await jsonEvent;await download.saveAs(path.join(out,'selected-match-table.json'));const snapshot=JSON.parse(fs.readFileSync(path.join(out,'selected-match-table.json'),'utf8'));
  assert.deepEqual(snapshot.rows.map(row=>row.map(String)),numbers);assert.equal(typeof snapshot.rows[0][1],'number');assert.match(snapshot.context['来源与范围'],/45–60/);assert.match(snapshot.context['筛选'],/自选时段/);
  const svgEvent=page.waitForEvent('download');await page.locator('#match-export').click();await(await svgEvent).saveAs(path.join(out,'selected-match.svg'));assert.match(fs.readFileSync(path.join(out,'selected-match.svg'),'utf8'),/45–60/);
  await bin.click();assert.equal(await page.locator('#match-period').inputValue(),'all');assert.equal(new URL(page.url()).searchParams.has('start'),false);
  await page.locator('#match-view').selectOption('network');await bin.click();assert.match(await page.locator('#match-time-selection').innerText(),new RegExp(`完成传球 ${segment.completed_passes} 次`));
  await page.locator('#match-time-clear').click();assert.equal(await page.locator('#match-period').inputValue(),'all');
  await page.locator('.filter-disclosure > summary').click();await page.locator('#match-period').selectOption('extra1');assert.equal(await page.locator('#match-table table').count(),0);assert.equal(await page.locator('#match-export').isDisabled(),true);
  await page.locator('#match-time-clear').click();assert.equal(await page.locator('#match-table table').count(),1);
  await page.locator('#match-view').selectOption('sequence');assert.equal(await page.locator('#match-time-control').isHidden(),true);
 });
 await check('dashboard-filter-and-scope',async()=>{
  await page.setViewportSize({width:1440,height:1000});await go('index.html');
  const catalog=JSON.parse(fs.readFileSync(path.join(docs,'data/research/matches.json'),'utf8')).matches;
  const coverage=JSON.parse(fs.readFileSync(path.join(docs,'data/research/coverage.json'),'utf8'));
  assert.equal(Number(await page.locator('#overview-match-count').innerText()),catalog.length);
  assert.equal(Number(await page.locator('#overview-source-count').innerText()),new Set(coverage.coverage.map(row=>row.provider)).size);
  assert.equal(await page.locator('#overview-matches a').count(),6);await page.locator('#overview-more').click();assert.equal(await page.locator('#overview-matches a').count(),12);
  await page.locator('#overview-team-query').fill('Spain');await page.locator('#overview-result-filter').selectOption('first');
  const expected=catalog.filter(row=>row.teams.join(' ').toLowerCase().includes('spain')&&row.score[0]>row.score[1]);
  assert.equal(await page.locator('#overview-outcome-sample').innerText(),`${expected.length} 场`);
  assert.equal(await page.locator('#overview-matches a').count(),Math.min(6,expected.length));
  assert.equal(await page.locator('#overview-outcomes b').first().innerText(),`${expected.length} 场`);
  await page.locator('#overview-team-query').fill('no_such_team');assert.match(await page.locator('#overview-match-filter-status').innerText(),/没有匹配/);assert.equal(await page.locator('#overview-outcome-sample').innerText(),'0 场');
  await page.locator('#overview-team-query').fill('');await page.locator('#overview-result-filter').selectOption('all');assert.equal(await page.locator('#overview-outcome-sample').innerText(),`${catalog.length} 场`);
 });
 await check('dashboard-responsive-and-empty',async()=>{
  const real=JSON.parse(fs.readFileSync(path.join(docs,'data/research/matches.json'),'utf8'));
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:width===1440?1000:844});await go('index.html');
   assert.equal(await page.locator('#overview-matches a').count(),6);
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),`dashboard overflow at ${width}`);
   const welcome=page.locator('.dashboard-welcome');assert.equal(await welcome.locator('a').isVisible(),true);
   if(width<500){const [copy,action]=await Promise.all([welcome.locator('div').nth(1).boundingBox(),welcome.locator('a').boundingBox()]);assert.ok(action.y>=copy.y+copy.height,`welcome action wraps at ${width}`);}
   await page.locator('#overview-team-query').fill('Spain');await page.locator('#overview-result-filter').selectOption('first');
   assert.ok(await page.locator('#overview-matches a').count()>0);await page.locator('#overview-team-query').fill('no_such_team');assert.match(await page.locator('#overview-match-filter-status').innerText(),/没有匹配/);
   await page.locator('#overview-team-query').fill('');await page.locator('#overview-result-filter').selectOption('all');
   await page.screenshot({path:path.join(out,`dashboard-${width}.png`),fullPage:true});
   await page.locator('#overview-matches a').first().click();await page.waitForLoadState('networkidle');assert.match(page.url(),/matches\.html\?match=/);assert.equal(await page.locator('#match-export').isEnabled(),true);
  }
  await page.route('**/data/research/matches.json',route=>route.fulfill({status:200,contentType:'application/json',body:JSON.stringify({...real,matches:[]})}));
  try{for(const width of [1440,390,320]){await page.setViewportSize({width,height:844});await go('index.html');assert.equal(await page.locator('#overview-match-count').innerText(),'0');assert.match(await page.locator('#overview-match-filter-status').innerText(),/目录暂无比赛/);assert.equal(await page.locator('#overview-retry').isHidden(),true);assert.equal(await page.locator('#overview-matches a').count(),0);}}finally{await page.unroute('**/data/research/matches.json');}
  await page.route('**/data/research/matches.json',route=>route.fulfill({status:503,body:'unavailable'}));
  try{await go('index.html');assert.match(await page.locator('#overview-status').innerText(),/暂不可用/);assert.equal(await page.locator('#overview-retry').isVisible(),true);}finally{await page.unroute('**/data/research/matches.json');}
 });
 await check('analysis-provenance-view-and-export',async()=>{
  await page.setViewportSize({width:1440,height:1000});await go('models.html?experiment=scores-PL#model-lab');
  await page.locator('#lab-provenance > summary').click();assert.match(await page.locator('#lab-provenance').innerText(),/未保存运行收据/);
  const event=page.waitForEvent('download');await page.locator('#lab-trace-export').click();const file=await event;await file.saveAs(path.join(out,'analysis-trace.json'));
  const exported=JSON.parse(fs.readFileSync(path.join(out,'analysis-trace.json'),'utf8'));assert.equal(exported.group,'scores-PL');assert.equal(exported.provenance.status,'legacy_missing');assert.equal(exported.provenance.code,null);assert.equal(exported.provenance.timing_policy,'not_recorded');
  const source=fs.readFileSync(path.join(docs,'../results/paper-followup-20261019/scores.json'));assert.equal(exported.provenance.result.sha256,require('node:crypto').createHash('sha256').update(source).digest('hex'));assert.ok(exported.sources.every(row=>/^[a-f0-9]{64}$/.test(row.sha256)));
  for(const width of [390,320]){await page.setViewportSize({width,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
  await page.screenshot({path:path.join(out,'analysis-trace-mobile.png')});
  let malformed=false,badLink=false,missing=false;
  const fixedUrl='https://github.com/steventimes/soccer-analytics/blob/'+'c'.repeat(40)+'/results/paper-followup-20261019/scores.json';
  // Controlled provenance, not an actual new model run or a fetched remote result.
  await page.route('**/data/research/model-lab.json',route=>{const body=JSON.parse(fs.readFileSync(path.join(docs,'data/research/model-lab.json'),'utf8'));body.groups[0].provenance={schema:'Touchline.AnalysisProvenance.v1',status:'recorded',timing_policy:'observed_result',run_id:malformed?'broken':'a'.repeat(64),code:{application_sha256:'b'.repeat(64),git_head:'c'.repeat(40)},configuration_sha256:'d'.repeat(64),configuration:{seed:42,timing_policy:'observed_result'},environment:{python:'3.12',packages:{}},result:{path:'results/paper-followup-20261019/scores.json',sha256:require('node:crypto').createHash('sha256').update(source).digest('hex'),repository_url:badLink?'https://example.invalid/result.json':fixedUrl}};if(missing)delete body.groups[0].provenance;return route.fulfill({contentType:'application/json',body:JSON.stringify(body)});});
  try{await go('models.html?experiment=scores-PL#model-lab');await page.locator('#lab-provenance > summary').click();assert.match(await page.locator('#lab-provenance').innerText(),/结果实际可用时间/);assert.match(await page.locator('#lab-provenance pre').innerText(),/seed/);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
   assert.equal(await page.getByRole('link',{name:'打开固定版本结果文件'}).getAttribute('href'),fixedUrl);
   await page.evaluate(()=>{window.traceValidationCalls=0;window.modelLabTrace=new Proxy(window.modelLabTrace,{apply(target,thisArg,args){window.traceValidationCalls++;return Reflect.apply(target,thisArg,args);}});});
   for(const [id,value] of [['lab-group','ratings-PL'],['lab-group','scores-PL'],['lab-stage','development'],['lab-metric','accuracy'],['lab-metric','log_loss'],['lab-stage','holdout']])await page.locator('#'+id).selectOption(value);
   await page.locator('#tab-representations').click();await page.locator('#tab-model-lab').click();
   assert.equal(await page.evaluate(()=>window.traceValidationCalls),0,'trace validation must not repeat during redraw');
   await page.locator('#lab-provenance > summary').click();
   assert.equal(await page.getByRole('link',{name:'打开固定版本结果文件'}).getAttribute('href'),fixedUrl);
   const recordedEvent=page.waitForEvent('download');await page.locator('#lab-trace-export').click();await (await recordedEvent).saveAs(path.join(out,'analysis-recorded-trace.json'));
   const recorded=JSON.parse(fs.readFileSync(path.join(out,'analysis-recorded-trace.json'),'utf8'));assert.equal(recorded.provenance.run_id,'a'.repeat(64));assert.equal(recorded.provenance.result.repository_url,fixedUrl);
   for(const broken of ['identity','link']){
    await page.evaluate(()=>{window.traceValidationCalls=0;});malformed=broken==='identity';badLink=broken==='link';
    await page.locator('#lab-retry').evaluate(el=>el.click());await page.waitForFunction(()=>!document.querySelector('#lab-retry').hidden);
    assert.match(await page.locator('#lab-status').innerText(),broken==='identity'?/追溯身份不完整/:/结果链接格式不完整/);assert.equal(await page.locator('#lab-trace').innerText(),'');
    assert.equal(await page.locator('#lab-readiness-0').isVisible(),true,'independent search data survives malformed model provenance');
    assert.equal(await page.evaluate(()=>window.traceValidationCalls),1,'bad provenance must fail in the loading boundary');
    malformed=false;badLink=false;await page.locator('#lab-retry').click();await page.waitForFunction(()=>!document.querySelector('#lab-group').disabled);
    assert.equal(await page.evaluate(()=>window.traceValidationCalls),5,'retry validates each of four newly received groups once');
   }
   missing=true;await page.locator('#lab-retry').evaluate(el=>el.click());await page.waitForFunction(()=>!document.querySelector('#lab-group').disabled);
   await page.locator('#lab-provenance > summary').click();assert.match(await page.locator('#lab-provenance').innerText(),/未保存运行收据/);
   const legacyEvent=page.waitForEvent('download');await page.locator('#lab-trace-export').click();await (await legacyEvent).saveAs(path.join(out,'analysis-missing-trace.json'));
   const legacy=JSON.parse(fs.readFileSync(path.join(out,'analysis-missing-trace.json'),'utf8'));assert.equal(legacy.provenance.status,'legacy_missing');assert.equal(legacy.provenance.result,null);assert.equal(legacy.provenance.code,null);assert.equal(legacy.provenance.timing_policy,'not_recorded');
  }finally{await page.unroute('**/data/research/model-lab.json');}
 });
 await check('saved-analysis-view-roundtrip',async()=>{
  await go('matches.html?match=statsbomb-3942819&team=Netherlands&view=network&period=second#match-explorer');
  const original=new URL(page.url());await page.locator('#save-view').click();await page.locator('#saved-view-name').fill('荷兰 · 下半场传球');await page.locator('#save-view-form button[type=submit]').click();
  assert.equal(await page.locator('#save-view-dialog').isVisible(),false);await go('studio.html#saved-views');
  const saved=page.locator('#studio-saved a').filter({hasText:'荷兰 · 下半场传球'});assert.equal(await saved.count(),1);await saved.click();await page.waitForLoadState('networkidle');
  assert.equal(new URL(page.url()).search,original.search);assert.equal(new URL(page.url()).hash,original.hash);assert.equal(await page.locator('#match-period').inputValue(),'second');assert.equal(await page.locator('#match-view').inputValue(),'network');
  await go('studio.html#analysis-tools');await page.locator('#studio-query').fill('校准');assert.ok(await page.locator('#studio-destinations a').count()>0);await page.locator('#studio-query').fill('no_such_tool');assert.match(await page.locator('#studio-count').innerText(),/没有匹配/);
  await page.locator('#studio-query').fill('');await page.locator('#studio-group').selectOption('比赛研究');assert.equal(await page.locator('#studio-destinations a').count(),3);
  const event=page.waitForEvent('download');await page.locator('#export-workspace').click();const download=await event;await download.saveAs(path.join(out,'saved-workspace.json'));const exported=JSON.parse(fs.readFileSync(path.join(out,'saved-workspace.json'),'utf8'));assert.ok(exported.saved.some(row=>row.title==='荷兰 · 下半场传球'&&row.href.includes('period=second')));
  await page.locator('#tab-saved-views').click();await page.locator('#studio-saved button').filter({hasText:'移除'}).first().click();await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('#studio-saved a').filter({hasText:'荷兰 · 下半场传球'}).count(),0);
  await page.locator('#tab-recent-views').click();await page.locator('#clear-recent').click();assert.match(await page.locator('#studio-recent').innerText(),/还没有最近访问/);
 });
 await check('match-saved-view-boundaries',async()=>{
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:width===1440?1000:844});
   await go('matches.html?match=statsbomb-3942819&team=Netherlands&view=network&period=second#match-explorer');
   await page.waitForFunction(()=>!document.querySelector('#match-select').disabled);
   assert.match(await page.locator('#match-restore-status').innerText(),/当前数据/);
   assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
   await page.screenshot({path:path.join(out,`saved-match-${width}.png`)});
   if(width===390){const axe=await new AxeBuilder({page}).include('#match-explorer').withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa']).analyze();fs.writeFileSync(path.join(out,'saved-match-axe.json'),JSON.stringify({violations:axe.violations,incomplete:axe.incomplete},null,2));assert.deepEqual(axe.violations,[]);}
  }
  await page.setViewportSize({width:1440,height:1000});
  await page.locator('#save-view').click();await page.locator('#saved-view-name').fill('   ');await page.locator('#save-view-form button[type=submit]').click();assert.match(await page.locator('#save-view-message').innerText(),/请输入/);
  await page.locator('#saved-view-name').fill('比赛视图');await page.locator('#save-view-form button[type=submit]').click();
  const local=await page.evaluate(()=>JSON.parse(localStorage.getItem('touchline.workspace.v1')));assert.ok(local.saved.some(row=>row.title==='比赛视图'&&row.href.includes('period=second')));assert.deepEqual(Object.keys(local.saved.find(row=>row.title==='比赛视图')).sort(),['href','title']);
  await page.locator('#match-view').selectOption('shots');await page.locator('#save-view').click();await page.locator('#saved-view-name').fill('比赛视图');await page.locator('#save-view-form button[type=submit]').click();assert.match(await page.locator('#save-view-message').innerText(),/同名/);await page.keyboard.press('Escape');
  await go('studio.html#saved-views');const saved=page.locator('#studio-saved a').filter({hasText:'比赛视图'});assert.equal(await saved.count(),1);await saved.focus();await page.keyboard.press('Enter');await page.waitForLoadState('networkidle');assert.equal(await page.locator('#match-view').inputValue(),'network');assert.equal(await page.locator('#match-period').inputValue(),'second');
  await go('matches.html?match=missing-match&team=missing-team&view=network&period=second#match-explorer');assert.match(await page.locator('#match-restore-status').innerText(),/比赛、球队/);assert.notEqual(await page.locator('#match-select').inputValue(),'missing-match');
  await go('matches.html?match=statsbomb-3942819&team=Netherlands&view=network&period=second&player=missing-player#match-explorer');assert.match(await page.locator('#match-restore-status').innerText(),/球员/);
  await page.route('**/data/research/matches.json',route=>route.fulfill({json:{schema:'Touchline.MatchExplorer.v1',matches:[],scope:'受控空目录'}}));
  try {await go('matches.html?match=statsbomb-3942819#match-explorer');assert.match(await page.locator('#match-restore-status').innerText(),/空目录/);assert.match(await page.locator('#match-status').innerText(),/没有比赛记录/);}finally{await page.unroute('**/data/research/matches.json');}
  await go('studio.html#saved-views');await page.getByRole('button',{name:'移除收藏：比赛视图'}).click();assert.equal(await page.locator('#studio-saved a').filter({hasText:'比赛视图'}).count(),0);
  await page.evaluate(()=>localStorage.setItem('touchline.workspace.v1','{broken'));await page.reload();await page.waitForLoadState('networkidle');assert.match(await page.locator('#shell-status').innerText(),/原记录未覆盖/);assert.equal(await page.evaluate(()=>localStorage.getItem('touchline.workspace.v1')),'{broken');await page.evaluate(()=>localStorage.removeItem('touchline.workspace.v1'));
  const restricted=await context.newPage();try{await restricted.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw new DOMException('Storage disabled','SecurityError');}}));await restricted.goto(base+'/matches.html');await restricted.waitForLoadState('networkidle');await restricted.locator('#save-view').click();await restricted.locator('#saved-view-name').fill('临时比赛视图');await restricted.locator('#save-view-form button[type=submit]').click();assert.match(await restricted.locator('#shell-status').innerText(),/不会保留/);}finally{await restricted.close();}
 });
 await check('saved-view-backup-import',async()=>{
  async function importBackup(target, file) {
   await target.locator('#import-workspace').setInputFiles(file);
   await target.waitForFunction(() => document.getElementById('import-workspace').value === '');
  }
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:width===1440?1000:844});await go('matches.html?match=statsbomb-3942819&team=Netherlands&view=network&period=second#match-explorer');
   await page.evaluate(()=>localStorage.removeItem('touchline.workspace.v1'));await page.reload();await page.waitForLoadState('networkidle');
   const original=new URL(page.url()), chart=await page.locator('#match-chart svg').innerHTML();
   await page.locator('#save-view').click();await page.locator('#saved-view-name').fill('荷兰备份');await page.locator('#save-view-form button[type=submit]').click();
   await go('studio.html#saved-views');const event=page.waitForEvent('download');await page.locator('#export-workspace').click();const download=await event;const backupPath=path.join(out,`view-backup-${width}.json`);await download.saveAs(backupPath);
   await page.getByRole('button',{name:'移除收藏：荷兰备份'}).click();assert.equal(await page.getByRole('link',{name:'荷兰备份'}).count(),0);
   await importBackup(page,backupPath);assert.match(await page.locator('#import-workspace-status').innerText(),/已导入 1/);assert.equal(await page.getByRole('link',{name:'荷兰备份'}).count(),1);
   await page.screenshot({path:path.join(out,`view-import-${width}.png`)});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));
   const before=await page.evaluate(()=>localStorage.getItem('touchline.workspace.v1'));
   await importBackup(page,{name:'broken.json',mimeType:'application/json',buffer:Buffer.from('{broken')});assert.match(await page.locator('#import-workspace-status').innerText(),/不是有效 JSON/);assert.equal(await page.evaluate(()=>localStorage.getItem('touchline.workspace.v1')),before);
   const conflict={schema:'Touchline.SavedViews.v1',saved:[{title:'新视图',href:'matches.html?match=new#match-explorer'},{title:'荷兰备份',href:'matches.html?match=other#match-explorer'}]};await importBackup(page,{name:'conflict.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(conflict))});assert.match(await page.locator('#import-workspace-status').innerText(),/冲突/);assert.equal(await page.evaluate(()=>localStorage.getItem('touchline.workspace.v1')),before);assert.equal(await page.getByRole('link',{name:'新视图'}).count(),0);
   await importBackup(page,backupPath);assert.match(await page.locator('#import-workspace-status').innerText(),/已导入 0/);
   await page.getByRole('link',{name:'荷兰备份'}).click();await page.waitForLoadState('networkidle');assert.equal(new URL(page.url()).search,original.search);assert.equal(await page.locator('#match-team').inputValue(),'Netherlands');assert.equal(await page.locator('#match-period').inputValue(),'second');assert.equal(await page.locator('#match-view').inputValue(),'network');assert.equal(await page.locator('#match-chart svg').innerHTML(),chart);
  }
  await page.setViewportSize({width:1440,height:1000});
  await go('studio.html#saved-views');await page.evaluate(()=>localStorage.setItem('touchline.workspace.v1','{broken'));await page.reload();await page.waitForLoadState('networkidle');
  await importBackup(page,path.join(out,'view-backup-1440.json'));assert.match(await page.locator('#import-workspace-status').innerText(),/本地记录损坏/);assert.equal(await page.evaluate(()=>localStorage.getItem('touchline.workspace.v1')),'{broken');await page.evaluate(()=>localStorage.removeItem('touchline.workspace.v1'));
  await page.evaluate(()=>localStorage.setItem('touchline.workspace.v1','{"saved":"broken"}'));await page.reload();await page.waitForLoadState('networkidle');await importBackup(page,path.join(out,'view-backup-1440.json'));assert.match(await page.locator('#import-workspace-status').innerText(),/本地记录损坏/);assert.equal(await page.evaluate(()=>localStorage.getItem('touchline.workspace.v1')),'{"saved":"broken"}');await page.evaluate(()=>localStorage.removeItem('touchline.workspace.v1'));
  const restricted=await context.newPage();try{await restricted.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw new DOMException('Storage disabled','SecurityError');}}));await restricted.goto(base+'/studio.html#saved-views');await restricted.waitForLoadState('networkidle');await importBackup(restricted,path.join(out,'view-backup-1440.json'));assert.match(await restricted.locator('#import-workspace-status').innerText(),/未允许保存/);assert.equal(await restricted.getByRole('link',{name:'荷兰备份'}).count(),0);}finally{await restricted.close();}
 });
 await check('application-navigation-keyboard-and-storage',async()=>{
  await page.setViewportSize({width:1440,height:1000});await go('index.html');await page.locator('#sidebar-toggle').click();assert.equal(await page.locator('body').evaluate(el=>el.classList.contains('sidebar-collapsed')),true);await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('body').evaluate(el=>el.classList.contains('sidebar-collapsed')),true);await page.locator('#sidebar-toggle').click();
  await page.locator('#density-toggle').click();assert.equal(await page.locator('html').getAttribute('data-density'),'compact');await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('html').getAttribute('data-density'),'compact');await page.locator('#density-toggle').click();
  await page.keyboard.press('Control+k');assert.equal(await page.locator('#navigation-search').isVisible(),true);await page.locator('#navigation-query').fill('接应');assert.equal(await page.locator('#navigation-results a').count(),1);await page.keyboard.press('Escape');assert.equal(await page.locator('#navigation-search').isVisible(),false);assert.equal(await page.locator('#density-toggle').evaluate(el=>el===document.activeElement),true);
  await page.locator('#navigation-search-open').click();await page.keyboard.press('Escape');assert.equal(await page.locator('#navigation-search-open').evaluate(el=>el===document.activeElement),true);
  for(const width of [390,320]){await page.setViewportSize({width,height:844});await page.locator('#sidebar-toggle').click();assert.equal(await page.locator('#sidebar-toggle').getAttribute('aria-expanded'),'true');await page.locator('.workspace-nav a[href="./studio.html"]').click();await page.waitForLoadState('networkidle');assert.equal(await page.locator('#sidebar-toggle').getAttribute('aria-expanded'),'false');assert.equal(await page.locator('.workspace-sidebar').evaluate(el=>el.inert),true);assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1));}
  await page.setViewportSize({width:1440,height:1000});
  await page.evaluate(()=>localStorage.setItem('touchline.workspace.v1',JSON.stringify({saved:[{title:'bad',href:'https://example.com/'}],recent:[{title:'bad',href:'../README.md'}]})));await page.reload();await page.waitForLoadState('networkidle');assert.equal(await page.locator('#studio-saved a').count(),0);
  await page.evaluate(()=>localStorage.setItem('touchline.workspace.v1',JSON.stringify({saved:Array.from({length:40},(_,index)=>({title:`视图 ${index}`,href:`models.html?slot=${index}`}))})));await go('index.html');await page.locator('#save-view').click();await page.locator('#saved-view-name').fill('不能静默覆盖旧收藏');await page.locator('#save-view-form button[type=submit]').click();assert.equal(await page.locator('#save-view-dialog').isVisible(),true);assert.match(await page.locator('#save-view-message').innerText(),/40/);assert.equal(await page.evaluate(()=>window.TouchlineShell.views().saved.length),40);await page.keyboard.press('Escape');await page.evaluate(()=>localStorage.removeItem('touchline.workspace.v1'));
  const restricted=await context.newPage();try{await restricted.addInitScript(()=>Object.defineProperty(window,'localStorage',{get(){throw new DOMException('Storage disabled','SecurityError');}}));await restricted.goto(base+'/index.html');await restricted.waitForLoadState('networkidle');await restricted.locator('#save-view').click();await restricted.locator('#saved-view-name').fill('仅本次会话');await restricted.locator('#save-view-form button[type=submit]').click();assert.match(await restricted.locator('#shell-status').innerText(),/不会保留/);assert.equal(await restricted.evaluate(()=>window.TouchlineShell.views().saved.length),1);}finally{await restricted.close();}
 });
 await check('accessibility-pages-and-interactive-states',async()=>{
  const scans=[];
  async function scan(name){
   const result=await new AxeBuilder({page}).withTags(['wcag2a','wcag2aa','wcag21a','wcag21aa','best-practice']).analyze();
   scans.push({name,url:page.url(),width:page.viewportSize().width,engine:result.testEngine,violations:result.violations,incomplete:result.incomplete,passed_rules:result.passes.length});
   fs.writeFileSync(path.join(out,'accessibility.json'),JSON.stringify({browser:browser.version(),scans},null,2)+'\n');
  }
  for(const width of [1440,390]){
   await page.setViewportSize({width,height:width===1440?1000:844});
   for(const file of ['index.html','studio.html','matches.html','entities.html','models.html','library.html','predictions.html']){await go(file);await scan(`${file}-${width}`);}
  }
  await go('matches.html?match=statsbomb-3943043&view=network');await page.locator('#match-timeline button').first().click();await page.locator('#match-explorer .chart-table-open').click();await scan('selected-time-and-table');
  await page.locator('#match-explorer .chart-open').click();await scan('chart-reader-dialog');await page.keyboard.press('Escape');
  await go('matches.html?offers_view=shares#fifa-offers');await page.locator('#fifa-offers .chart-table-open').click();await scan('fifa-shares-and-table');
  await go('models.html#xt-grid');await page.waitForFunction(()=>!document.querySelector('#xt-grid-view').disabled);await page.locator('#xt-grid .chart-table-open').click();await scan('dense-grid-and-table');
  await go('models.html?experiment=scores-PL#model-lab');await page.locator('#lab-provenance > summary').click();await scan('analysis-provenance-expanded');
  await go('library.html#figures');await page.locator('#figure-enlarge').click();await scan('figure-reader-dialog');await page.keyboard.press('Escape');
  await go('library.html#literature');await page.locator('#paper-search').fill('no_such_paper');await scan('empty-literature');
  await page.setViewportSize({width:1440,height:1000});await go('studio.html');await page.locator('#navigation-search-open').click();await scan('navigation-search-dialog');await page.keyboard.press('Escape');await page.locator('#save-view').click();await scan('save-view-dialog');await page.keyboard.press('Escape');
  await page.setViewportSize({width:390,height:844});await page.locator('#sidebar-toggle').click();await scan('mobile-navigation-open');await page.locator('#navigation-shade').click({position:{x:380,y:20}});
  await page.route('**/data/research/matches.json',route=>route.fulfill({status:503,body:'Unavailable'}));
  try{await go('matches.html');assert.equal(await page.locator('#match-time-control').isHidden(),true);await scan('match-catalog-error');}finally{await page.unroute('**/data/research/matches.json');}
  const violations=scans.flatMap(result=>result.violations.map(violation=>({state:result.name,rule:violation.id,impact:violation.impact,targets:violation.nodes.map(node=>node.target)})));
  assert.deepEqual(violations,[],'Axe findings; full evidence is in accessibility.json');
 });
 await context.close();
 if(selectedCases)assert.deepEqual([...selectedCases].filter(name=>!knownCases.has(name)),[],'Unknown FRONTEND_CASE');
 assert.ok(results.length>0,'No browser scenarios selected');assert.deepEqual(errors,[],'Browser runtime errors');assert.ok(results.every(result=>result.status==='passed'),'Acceptance failures');
})().catch(error=>{console.error(error);errors.push({message:error.message});process.exitCode=1;}).finally(async()=>{fs.writeFileSync(path.join(out,reportName),JSON.stringify({results,errors},null,2)+'\n');if(browser)await browser.close();server.close();});
