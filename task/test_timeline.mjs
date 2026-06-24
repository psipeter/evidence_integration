/**
 * test_timeline.mjs
 * Tests build-trial-timeline.js logic directly — no browser needed.
 * Run: node test_timeline.mjs
 */
import { buildTrialTimeline } from './src/shared/build-trial-timeline.js';

let _log = [], _stack = [], _scenario = [];
const log  = (...a) => _log.push(a.join(' '));
class TL   { constructor() { this.aborted=false; } abort() { this.aborted=true; } }
const jsPsych = { data: { addProperties: ()=>{} } };

async function runNode(node, depth=0) {
  const indent = '  '.repeat(depth);
  if (node.timeline) {
    if (node.conditional_function && !node.conditional_function()) { log(indent+'[SKIP]'); return; }
    if (node.on_timeline_start) node.on_timeline_start();
    const tl = new TL(); _stack.push(tl);
    let i=0;
    do {
      if (tl.aborted) break;
      if (++i>20) { log(indent+'[LOOP ERR]'); break; }
      for (const child of node.timeline) { if (tl.aborted) break; await runNode(child,depth+1); }
    } while (!tl.aborted && node.loop_function && node.loop_function());
    _stack.pop(); return;
  }
  const screen = node.data?.screen ?? '?';
  if (node.type?._mock === 'OBS') {
    const a = _scenario.shift();
    if (a==='timeout') { log(indent+'[OBS] '+screen+' TIMEOUT'); node.on_finish({timed_out:true,response:null}); }
    else               { log(indent+'[OBS] '+screen+' SUBMIT('+a+')'); node.on_finish({timed_out:false,response:a}); }
  } else if (node.type?._mock === 'ITI') {
    const to  = typeof node.timed_out==='function' ? node.timed_out() : (node.timed_out??false);
    const rem = typeof node.timeouts_remaining==='function' ? node.timeouts_remaining() : '-';
    log(indent+'[ITI] '+screen+(to?' TOO_SLOW rem='+rem:''));
    if (node.on_finish) node.on_finish({});
  } else if (node.type?._mock === 'SUMMARY') {
    const resp = typeof node.responses==='function' ? node.responses() : [];
    log(indent+'[SUMMARY] '+screen+' responses='+JSON.stringify(resp));
    if (node.on_finish) node.on_finish({});
  } else if (node.type?._mock === 'BTI') {
    log(indent+'[BTI] '+screen);
    if (node.on_finish) node.on_finish({});
  } else {
    log(indent+'[???] type='+(node.type?.name||node.type));
    if (node.on_finish) node.on_finish({});
  }
}

const OBS={_mock:'OBS'}, ITI={_mock:'ITI'}, SUMMARY={_mock:'SUMMARY'}, BTI={_mock:'BTI'};
let passed=0, failed=0;

async function test(name, sequences, scenario) {
  _log=['\n=== '+name+' ===']; _scenario=[...scenario]; _stack=[];
  const earlyExit = () => log('\n*** EARLY EXIT ***');
  const { timeline: tl } = buildTrialTimeline(
    { sequences, sliderDefault:'none', defaultValue:50, btiMs:0, tObsMs:0,
      showSliderValue:true, showTrialPerformance:true, MAX_TIMEOUTS_PER_TRIAL:3 },
    { ItiClockPlugin:ITI, TrialObsPlugin:OBS, TrialSummaryPlugin:SUMMARY,
      InterTrialPlugin:BTI, isBinary:false },
    jsPsych, earlyExit,
  );
  for (const node of tl) await runNode(node, 0);
  await new Promise(r=>setTimeout(r,10));
  const out = _log.join('\n');
  console.log(out);
  return out;
}

function assert(condition, msg) {
  if (!condition) { console.log('  FAIL: '+msg); failed++; }
  else            { console.log('  PASS: '+msg); passed++; }
}

const S1=[{values:[1,2,3],iti_ms:0,true_mean:50,true_std:10,true_p:null,qid:0}];
const S2=[
  {values:[1,2,3],iti_ms:0,true_mean:50,true_std:10,true_p:null,qid:0},
  {values:[4,5],  iti_ms:0,true_mean:60,true_std:10,true_p:null,qid:1},
];

let out;

out = await test('Normal submits',        S1, [70,65,60]);
assert(!out.includes('Too slow'),         'No too-slow on normal submit');
assert(out.includes('SUMMARY'),           'Summary appears');
assert(out.includes('responses=[70,65,60]'),'Responses recorded correctly');

out = await test('1 timeout then ok',     S1, ['timeout',70,65,60]);
assert(out.includes('TOO_SLOW rem=2'),    'Shows 2 remaining after 1st timeout');
assert(!out.includes('EARLY EXIT'),       'No early exit after 1 timeout');
assert(out.includes('responses=[70,65,60]'),'All responses recorded');

out = await test('3 timeouts exit',       S1, ['timeout','timeout','timeout']);
assert(out.includes('TOO_SLOW rem=2'),    'Shows 2 remaining');
assert(out.includes('TOO_SLOW rem=1'),    'Shows 1 remaining');
assert(!out.includes('SUMMARY'),          'No summary after exit');
assert(!out.includes('obs_0_1'),          'No obs_0_1 after exit');

out = await test('2 timeouts then ok',    S1, ['timeout','timeout',70,65,60]);
assert(!out.includes('EARLY EXIT'),       'No exit after 2 timeouts');
assert(out.includes('SUMMARY'),           'Summary appears');
assert(out.includes('responses=[70,65,60]'),'All responses recorded');

out = await test('timeout on obs1',       S1, [70,'timeout',65,60]);
assert(out.includes('TOO_SLOW rem=2'),    'Shows 2 remaining');
assert(out.includes('responses=[70,65,60]'),'All responses recorded');

out = await test('2-trial normal',        S2, [70,65,60,40,50]);
assert(out.includes('BTI'),               'BTI appears between trials');
assert(out.includes('responses=[40,50]'), 'Trial 2 responses correct');

out = await test('exit trial1 skip t2',   S2, ['timeout','timeout','timeout']);
assert(!out.includes('Trial 1 start'),    'Trial 2 skipped');

out = await test('exit trial2',           S2, [70,65,60,'timeout','timeout','timeout']);
assert(out.includes('SUMMARY'), 'Trial 1 summary appears');
assert(out.split('SUMMARY').length-1 === 1, 'Only 1 summary (trial 2 exit skips its summary)');

console.log('\n'+'='.repeat(40));
console.log('Results: '+passed+' passed, '+failed+' failed');
if (failed>0) process.exit(1);
