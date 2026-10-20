const fs=require('node:fs'),path=require('node:path'),http=require('node:http'),assert=require('node:assert/strict');
// Prefer an explicit or project-local runtime; the reviewed CLI cache is a local fallback.
function loadPlaywright() {
 if(process.env.PLAYWRIGHT_MODULE)return require(path.resolve(process.env.PLAYWRIGHT_MODULE));
 try{return require('playwright');}catch(error){if(error.code!=='MODULE_NOT_FOUND')throw error;}
 const cache=path.join(require('node:os').homedir(),'.npm','_npx');
 if(fs.existsSync(cache))for(const name of fs.readdirSync(cache).sort()){
  const candidate=path.join(cache,name,'node_modules','playwright');
  if(fs.existsSync(path.join(candidate,'package.json')))return require(candidate);
 }
 throw new Error('Playwright is unavailable. Install it locally or set PLAYWRIGHT_MODULE to an existing runtime.');
}
const {chromium}=loadPlaywright();
const root=path.resolve(__dirname,'..'),docs=path.join(root,'docs'),out=path.resolve(process.env.TOUCHLINE_BROWSER_OUTPUT||path.join(root,'output/playwright/frontend'));fs.mkdirSync(out,{recursive:true});
const server=http.createServer((req,res)=>{const url=new URL(req.url,'http://localhost'),p=path.resolve(docs,'.'+(url.pathname==='/'?'/index.html':url.pathname));if(!p.startsWith(docs+path.sep)||!fs.existsSync(p)||!fs.statSync(p).isFile()){res.writeHead(404).end();return;}res.setHeader('Content-Type',({'.html':'text/html','.js':'text/javascript','.css':'text/css','.json':'application/json','.svg':'image/svg+xml','.png':'image/png'})[path.extname(p)]||'application/octet-stream');res.end(fs.readFileSync(p));});
const selectedCases=process.env.FRONTEND_CASE===undefined?null:new Set(process.env.FRONTEND_CASE.split(',').map(name=>name.trim()).filter(Boolean));
const knownCases=new Set();
let browser;const results=[],errors=[];const reportName=selectedCases?'acceptance-affected.json':'acceptance.json';
(async()=>{
 await new Promise(r=>server.listen(0,'127.0.0.1',r)); const base=`http://127.0.0.1:${server.address().port}`;
 const executable=process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE||(fs.existsSync('/opt/google/chrome/google-chrome')?'/opt/google/chrome/google-chrome':chromium.executablePath());
 browser=await chromium.launch({executablePath:executable,headless:true});
 const context=await browser.newContext({viewport:{width:1440,height:1000},reducedMotion:'reduce',acceptDownloads:true});const page=await context.newPage();page.on('pageerror',error=>errors.push({url:page.url(),message:error.message}));
 async function go(url){await page.goto(base+'/'+url);await page.waitForLoadState('networkidle');}
 async function check(name,fn){knownCases.add(name);if(selectedCases&&!selectedCases.has(name))return;try{await fn();results.push({name,status:'passed'});}catch(error){results.push({name,status:'failed',error:error.message,stack:error.stack});await page.screenshot({path:path.join(out,name+'-failure.png')}).catch(()=>{});} console.log(JSON.stringify(results.at(-1)));fs.writeFileSync(path.join(out,reportName),JSON.stringify({results,errors},null,2));}
 await check('routes-and-responsive-layout',async()=>{
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:width===1440?1000:844});
   for(const file of ['index.html','matches.html','entities.html','models.html','library.html','predictions.html']){
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
  await page.locator('#tab-representations').click();assert.equal(await page.locator('#experiment-chart .interval-result').count(),JSON.parse(fs.readFileSync(path.join(docs,'data/research/representation.json'))).summaries.length);
  await page.locator('#tab-vaep-calibration').click();await page.locator('#vaep-calibration-bins').selectOption('9');assert.match(await page.locator('#vaep-calibration-summary').innerText(),/9 箱/);
  await page.locator('#tab-xt-grid').click();await page.locator('#xt-grid-view').selectOption('shots');assert.equal(await page.locator('#xt-grid-table tbody tr').count(),96);
 });
 await check('library-deep-links-gallery-dialog-and-coverage',async()=>{
  await go('library.html?figure=paper-followup-20261019-global-contributions#figures');assert.match(await page.locator('#figure-reader-title').innerText(),/同一输出/);
  await page.locator('#figure-enlarge').click();assert.equal(await page.locator('#figure-dialog').evaluate(el=>el.open),true);await page.keyboard.press('Escape');assert.equal(await page.locator('#figure-enlarge').evaluate(el=>el===document.activeElement),true);
  await page.locator('#tab-coverage').click();await page.locator('#competition-filter').selectOption('champions_league');assert.match(await page.locator('#coverage-rows').innerText(),/欧冠/);await page.locator('#coverage-rows button').first().click();assert.match(await page.locator('#coverage-detail').innerText(),/来源/);
  await page.locator('#tab-literature').click();await page.locator('#paper-search').fill('XGBoost');assert.ok(await page.locator('#paper-cards .method-card').count()>0);await page.locator('#paper-search').fill('no_such_paper');assert.match(await page.locator('#paper-cards').innerText(),/没有匹配/);
 });
 await check('failure-isolation-and-retry',async()=>{
  let block=true;await page.route('**/data/research/representation.json',route=>block?route.fulfill({status:503,body:'unavailable'}):route.continue());await go('models.html#representations');assert.equal(await page.locator('#representation-retry').isVisible(),true);assert.ok(await page.locator('#lab-group option').count()===4);block=false;await page.locator('#representation-retry').click();await page.waitForFunction(()=>document.querySelector('#representation-retry').hidden);assert.equal(await page.locator('#representation-retry').isVisible(),false);await page.unroute('**/data/research/representation.json');
  let matchBlock=true;await page.route('**/data/research/matches/statsbomb-3943043.json',route=>matchBlock?route.fulfill({status:503,body:'unavailable'}):route.continue());await go('matches.html?match=statsbomb-3943043&team=England&view=network&period=second');assert.equal(await page.locator('#match-retry').isVisible(),true);matchBlock=false;await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);assert.equal(await page.locator('#match-export').isEnabled(),true);assert.equal(await page.locator('#match-team').inputValue(),'England');assert.equal(await page.locator('#match-period').inputValue(),'second');assert.equal(await page.locator('#match-view').inputValue(),'network');await page.unroute('**/data/research/matches/statsbomb-3943043.json');
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
  const good=JSON.parse(fs.readFileSync(path.join(docs,'data/research/matches/statsbomb-3943043.json'))),bad=structuredClone(good);delete bad.segments.find(row=>row.team==='Spain').cells;
  let invalid=true,requests=0;await page.route('**/data/research/matches/statsbomb-3943043.json',route=>{requests++;return invalid?route.fulfill({status:200,contentType:'application/json',body:JSON.stringify(bad)}):route.continue();});await go('matches.html?match=statsbomb-3943043&team=Spain&view=shots');assert.equal(await page.locator('#match-retry').isVisible(),true);assert.equal(await page.locator('#match-export').isDisabled(),true);invalid=false;await page.locator('#match-retry').click();await page.waitForFunction(()=>!document.querySelector('#match-export').disabled);assert.equal(requests,2);assert.equal(await page.locator('#match-retry').isVisible(),false);await page.unroute('**/data/research/matches/statsbomb-3943043.json');
 });
 await check('responsive-comparison-and-fifa-reading',async()=>{
  for(const width of [1440,390,320]){
   await page.setViewportSize({width,height:width===1440?1000:844});
   for(const target of ['entities.html','matches.html#fifa-offers']){
    await go(target);const info=await page.evaluate(()=>{const plot=document.querySelector('#comparisons:not([hidden]) .paired-metric,#fifa-offers:not([hidden]) .paired-metric');return{width:document.documentElement.scrollWidth,top:plot.getBoundingClientRect().top};});assert.ok(info.width<=width+1);assert.ok(info.top<844,`first metric below viewport ${target} ${width}: ${info.top}`);await page.screenshot({path:path.join(out,(target.startsWith('entities')?'entities-reading-':'fifa-reading-')+width+'.png')});
   }
  }
 });
 await context.close();
 if(selectedCases)assert.deepEqual([...selectedCases].filter(name=>!knownCases.has(name)),[],'Unknown FRONTEND_CASE');
 assert.ok(results.length>0,'No browser scenarios selected');assert.deepEqual(errors,[],'Browser runtime errors');assert.ok(results.every(result=>result.status==='passed'),'Acceptance failures');
})().catch(error=>{console.error(error);errors.push({message:error.message});process.exitCode=1;}).finally(async()=>{fs.writeFileSync(path.join(out,reportName),JSON.stringify({results,errors},null,2)+'\n');if(browser)await browser.close();server.close();});
