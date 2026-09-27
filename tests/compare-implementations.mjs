// Compare saved results with a tolerance, not bitwise equality across languages.
import { readFileSync } from 'node:fs';
import { strict as assert } from 'node:assert';
const [first, second] = process.argv.slice(2);
if (!first || !second) throw new Error('Usage: node tests/compare-implementations.mjs runs/prototype runs/go-prototype');
const load=(directory,file)=>JSON.parse(readFileSync(`${directory}/${file}`,'utf8'));
const a=load(first,'history.json'),b=load(second,'history.json');
assert.equal(a.length,b.length);
let maxDifference=0;
function near(x,y){assert(Number.isFinite(x)&&Number.isFinite(y));const difference=Math.abs(x-y);maxDifference=Math.max(maxDifference,difference);assert(difference<=1e-8+1e-6*Math.max(Math.abs(x),Math.abs(y)),`Mismatch ${x} vs ${y}`)}
for(let i=0;i<a.length;i++){
  assert.equal(a[i].epoch,b[i].epoch);
  for(const split of ['train','validation'])for(const metric of ['mse','meanDistance'])near(a[i][split][metric],b[i][split][metric]);
}
const sa=load(first,'summary.json'),sb=load(second,'summary.json');
assert.equal(sa.parameterCount,sb.parameterCount);assert.equal(sa.bestEpoch,sb.bestEpoch);assert.equal(sa.accepted,sb.accepted);
for(const split of ['initialValidation','bestTrain','bestValidation','test','identityValidation','affineValidation','identityTest','affineTest'])
  for(const metric of ['mse','meanDistance'])near(sa[split][metric],sb[split][metric]);
console.log(`PASS cross-language training comparison: ${a.length} evaluations; max metric difference=${maxDifference}. Timing excluded.`);
