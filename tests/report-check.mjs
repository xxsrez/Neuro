// Source/data checks and a JS smoke test with a minimal DOM/canvas double.
// This is not a real browser or a visual/layout test.
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
import vm from 'node:vm';
const directory = process.argv[2];
if (!directory) throw new Error('Usage: node tests/report-check.mjs runs/name');
const html = readFileSync(`${directory}/report.html`, 'utf8');
const report = JSON.parse(readFileSync(`${directory}/report-data.json`, 'utf8'));
const history = JSON.parse(readFileSync(`${directory}/history.json`, 'utf8'));
const summary = JSON.parse(readFileSync(`${directory}/summary.json`, 'utf8'));
const embedded = JSON.parse(html.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/)[1]);
assert.deepEqual(embedded, report);
assert.deepEqual(history, report.history);
assert.deepEqual(summary, report.summary);
assert(!html.includes('@@STATIC@@') && !html.includes('@@DATA@@'));
assert(!/<(?:script|link)[^>]+(?:src|href)=/i.test(html), 'External dependencies');
const staticPart = html.slice(0, html.indexOf('<section id="interactive">'));
assert(staticPart.includes('<table>') && staticPart.includes('<svg '));
assert(staticPart.includes('Тестовая MSE') && staticPart.includes('Ошибка по эпохам'));
if (report.config.blocks) assert(staticPart.includes('Активности первого блока'));
const csv = readFileSync(`${directory}/metrics.csv`, 'utf8').trim().split('\n');
assert.equal(csv.length, history.length + 1);
for (let i = 0; i < history.length; i++) {
  const row = csv[i + 1].split(',').map(Number), r = history[i];
  assert.deepEqual(row.slice(0, 5), [r.epoch, r.train.mse, r.validation.mse, r.train.meanDistance, r.validation.meanDistance]);
}
for (const p of report.grid) {
  const error = Math.hypot(p.predictedX - p.targetX, p.predictedY - p.targetY);
  assert(Math.abs(error - p.error) < 1e-12);
}
assert.equal(report.grid.length, report.config.gridSide ** 2);
assert.equal(report.activations.length, report.config.blocks);
for (const values of report.activations) assert.equal(values.length, report.activationSide ** 2 * report.config.width);
assert.equal(summary.bestEpoch, history.reduce((best, r) => r.validation.mse < best.validation.mse ? r : best).epoch);
assert.equal(summary.bestValidation.mse, history[summary.bestEpoch].validation.mse);

const script = html.match(/<script>\s*([\s\S]*?)<\/script>/)[1];
function makeEnvironment(search='') {
  const elements = new Map();
  const ctx = new Proxy({}, {get: (_, key) => key === 'createImageData' ? (w,h)=>({data:new Uint8ClampedArray(w*h*4)}) : ()=>{}, set:()=>true});
  function element(tag='div') {
    return {tag,children:[],listeners:{},width:440,height:440,hidden:true,value:'0',checked:false,textContent:'',
      append(...items){this.children.push(...items)},replaceChildren(){this.children=[]},
      getContext(){return ctx},setAttribute(){},addEventListener(event,fn){this.listeners[event]=fn},
      getBoundingClientRect(){return {left:0,top:0,width:this.width,height:this.height}}};
  }
  for (const id of ['data','status','controls','loss','log','error','point','motion','block','neurons','activation-scale','gradients']) elements.set(id,element());
  elements.get('data').textContent=JSON.stringify(report);
  elements.get('loss').width=860;elements.get('loss').height=330;
  return {elements,context:{document:{getElementById:id=>elements.get(id),createElement:element},location:{search},URLSearchParams,console}};
}
const normal=makeEnvironment();vm.runInNewContext(script, normal.context);
assert(normal.elements.get('status').textContent.includes('готов'), normal.elements.get('status').textContent);
assert.equal(normal.elements.get('controls').hidden,false);
assert.equal(normal.elements.get('neurons').children.length,report.config.blocks ? report.config.width : 0);
normal.elements.get('log').checked=true;normal.elements.get('log').listeners.change();
normal.elements.get('error').listeners.click({clientX:220,clientY:220});
assert(normal.elements.get('point').textContent.includes('Расстояние'));
if(report.config.blocks>1){normal.elements.get('block').value=String(report.config.blocks-1);normal.elements.get('block').listeners.change();assert.equal(normal.elements.get('neurons').children.length,report.config.width)}
const failure=makeEnvironment('?fail=1');vm.runInNewContext(script,failure.context);
assert.equal(failure.elements.get('controls').hidden,true);
assert(failure.elements.get('status').textContent.includes('недоступен'));
console.log('PASS report: embedded data, CSV, error distances, static fallback, JS controls, forced failure. Visual layout unverified.');
if (process.argv[3]) {
  const other=process.argv[3];
  const otherHistory=JSON.parse(readFileSync(`${other}/history.json`,'utf8'));
  const metrics=rows=>rows.map(r=>({epoch:r.epoch,train:r.train,validation:r.validation}));
  assert.deepEqual(metrics(history),metrics(otherHistory));
  for(const name of ['best-weights.json','last-weights.json']) {
    const a=JSON.parse(readFileSync(`${directory}/${name}`,'utf8')),b=JSON.parse(readFileSync(`${other}/${name}`,'utf8'));
    assert.deepEqual(a.parameters,b.parameters);assert.equal(a.epoch,b.epoch);
    const {output:_,...ca}=a.configuration,{output:__,...cb}=b.configuration;assert.deepEqual(ca,cb);
  }
  const otherSummary=JSON.parse(readFileSync(`${other}/summary.json`,'utf8'));
  for(const key of ['bestEpoch','bestTrain','bestValidation','test','initialValidation','identityValidation','affineValidation','identityTest','affineTest','affineCoefficients','accepted'])
    assert.deepEqual(summary[key],otherSummary[key]);
  console.log('PASS reproducibility: all epoch metrics and best/last weights exactly match. Timing excluded.');
}
