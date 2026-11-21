import {betaDensity, betaUpdate, coinExample, countHeads, normalDensity, probability} from './bayes-math.mjs?v=704fc235ecc0';
import {
  betaBinomial, discretize, highestDensity, logBayesFactor, mixtureDensity,
  modelCriteria, quantile, regressionPosteriorSampler, shrinkageModes
} from './inference-math.mjs?v=704fc235ecc0';
import {setupMcmcLab} from './mcmc-lab.mjs?v=704fc235ecc0';

const {Plot, d3} = window;
const byId = id => document.getElementById(id);
const value = id => byId(id).valueAsNumber;
const write = (id, text) => { byId(id).textContent = text; };
const percent = p => `${(100 * p).toFixed(1)}%`;
const format = d3.format('.3~f');
const style = getComputedStyle(document.querySelector('.bayes-page'));
const ink = style.getPropertyValue('--ink').trim();
const teal = style.getPropertyValue('--posterior').trim();
const purple = style.getPropertyValue('--evidence').trim();
const rust = style.getPropertyValue('--prior').trim();
const priors = {uniform: [1, 1], jeffreys: [0.5, 0.5], strong: [20, 20]};
const shared = new URLSearchParams(location.search);
const sharedTosses = shared.get('tosses');
const sharedPrior = [Number(shared.get('a')), Number(shared.get('b'))];
const validShare = sharedTosses !== null && /^[HT]*$/.test(sharedTosses)
  && sharedPrior.every(x => x >= 0.5 && x <= 30 && Number.isInteger(x * 2));
let tosses = validShare ? sharedTosses : coinExample;
if (validShare) {
  const known = Object.keys(priors).find(key => priors[key].every((x, i) => x === sharedPrior[i]));
  if (known) byId('prediction-prior').value = known;
  else {
    priors.shared = sharedPrior;
    byId('prediction-prior').add(new Option(`From Chapter 1 · Beta(${sharedPrior.join(', ')})`, 'shared', true, true));
  }
}

const prior = () => priors[byId('prediction-prior').value];
const counts = sequence => {
  const heads = countHeads(sequence);
  return [heads, sequence.length - heads];
};
function resizeTosses(sequence, heads, tails) {
  const remaining = {H: heads, T: tails};
  let kept = '';
  for (const toss of sequence) {
    if (remaining[toss] === 0) continue;
    kept += toss;
    remaining[toss] -= 1;
  }
  return kept + 'H'.repeat(remaining.H) + 'T'.repeat(remaining.T);
}
const [importedHeads, importedTails] = counts(tosses);
tosses = resizeTosses(tosses, Math.min(importedHeads, 20), Math.min(importedTails, 20));
byId('import-limit').hidden = importedHeads + importedTails === tosses.length;

const posterior = sequence => betaUpdate(...prior(), ...counts(sequence));
const densityLine = (data, stroke = teal, strokeDasharray) => Plot.lineY(data, {x: 'x', y: 'y', stroke, strokeDasharray, strokeWidth: 2.2});
const range = (a, b, n = 320) => Array.from({length: n + 1}, (_, i) => a + (b - a) * i / n);
const curve = (density, domain, n) => range(...domain, n).map(x => ({x, y: density(x)}));
const bins = probabilities => probabilities.map((p, k) => ({k, p}));
const betaWindow = (a, b) => [a < 1 ? 0.01 : 0, b < 1 ? 0.99 : 1];

function draw(id, label, options, height = 236) {
  const host = byId(id);
  const width = host.clientWidth;
  const chart = Plot.plot({
    width, height, marginTop: 24, marginBottom: 42, marginLeft: 48, marginRight: 12,
    ariaLabel: label, style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
    ...options, x: {ticks: width < 370 ? 4 : 7, ...options.x},
    y: {grid: true, ticks: 4, zero: true, ...options.y}
  });
  host.replaceChildren(chart);
  return chart;
}

let predictiveDraws = [];
let predictionRandom = d3.randomLcg(6207);

function renderInterval() {
  const {a, b, mean} = posterior(tosses);
  const [priorA, priorB] = prior();
  const mixture = byId('interval-example').value === 'mixture';
  const density = mixture ? mixtureDensity : betaDensity(a, b);
  const domain = mixture ? [-6, 6] : [0, 1];
  const visible = mixture ? domain : betaWindow(a, b);
  const priorData = mixture ? curve(() => 1 / 12, domain) : curve(betaDensity(priorA, priorB), betaWindow(priorA, priorB));
  const coordinate = mixture ? t => 12 * t - 6 : t => Math.sin(Math.PI * t / 2) ** 2;
  const cells = discretize(density, coordinate);
  const flat = !mixture && a === 1 && b === 1;
  const regions = flat ? [[quantile(cells, 0.025), quantile(cells, 0.975)]] : highestDensity(cells, 0.95);
  const median = quantile(cells, 0.5);
  const center = mixture ? 0 : mean;
  const densityData = cells.filter(({x}) => x >= visible[0] && x <= visible[1]).map(({x, height}) => ({x, y: height}));
  const shade = regions.flatMap(([lo, hi], region) => {
    const left = Math.max(lo, visible[0]);
    const right = Math.min(hi, visible[1]);
    return left < right ? curve(density, [left, right], 100).map(point => ({...point, region})) : [];
  });
  const bounds = pair => `[${pair.map(format).join(', ')}]`;
  const interval = ` 95% HPD ${regions.map(bounds).join(' ∪ ')}.`;
  const clipped = !mixture && (priorA < 1 || priorB < 1) ? ' The plot trims 0.01 near each diverging endpoint.' : '';
  write('interval-summary', `${mixture ? 'Two-peak example' : `Beta(${a}, ${b})`} · Mean ${format(center)}.${interval}${clipped}`);
  draw('interval-plot', 'Prior and posterior densities with mean, median, and the 95% highest-density region', {
    x: {domain, label: mixture ? 'Parameter value' : 'Chance of heads, θ'},
    y: {label: 'Density'},
    marks: [
      Plot.areaY(shade, {x: 'x', y: 'y', z: 'region', fill: teal, fillOpacity: 0.27}),
      densityLine(densityData),
      densityLine(priorData, rust, '6 4'),
      Plot.dot([{x: center, y: density(center)}], {x: 'x', y: 'y', fill: teal, r: 4, title: () => 'Mean'}),
      Plot.dot([{x: median, y: density(median)}], {x: 'x', y: 'y', stroke: rust, fill: 'white', r: 4, symbol: 'diamond', title: () => 'Median'})
    ]
  });
}

function setRange(from, to) {
  byId('range-from').value = from;
  byId('range-to').value = to;
}

function renderPredictive() {
  const future = value('future');
  const {a, b, mean} = posterior(tosses);
  for (const id of ['range-from', 'range-to']) byId(id).max = future;
  const from = value('range-from');
  const to = value('range-to');
  const exact = betaBinomial(future, a, b);
  const selected = k => k >= from && k <= to;
  const mass = exact.slice(from, to + 1).reduce((sum, p) => sum + p, 0);
  const frequencies = Array(future + 1).fill(0);
  for (const k of predictiveDraws) frequencies[k] += 1 / predictiveDraws.length;
  write('future-value', future);
  write('range-from-value', from);
  write('range-to-value', to);
  write('predictive-summary', `P(${from} ≤ heads ≤ ${to}) = ${percent(mass)}.`);
  byId('predictive-simulated').hidden = !predictiveDraws.length;
  if (predictiveDraws.length) {
    write('predictive-simulated', `Simulated ${percent(predictiveDraws.filter(selected).length / predictiveDraws.length)} of ${predictiveDraws.length} draws.`);
  }
  write('next-toss', percent(mean));
  const [heads, tails] = counts(tosses);
  byId('observed-heads').value = heads;
  byId('observed-tails').value = tails;
  write('observed-heads-value', heads);
  write('observed-tails-value', tails);
  write('observed-total', `n = ${tosses.length}`);
  const chart = draw('predictive-plot', 'Exact predictive probabilities with highlighted range and simulated frequencies', {
    x: {domain: [-0.5, future + 0.5], label: 'Heads in the new batch', tickFormat: k => Number.isInteger(k) ? k : ''},
    y: {label: 'Probability'},
    marks: [
      Plot.rectY(bins(exact), {x1: d => d.k - 0.39, x2: d => d.k + 0.39, y: 'p', fill: d => selected(d.k) ? teal : '#aabeb3'}),
      Plot.dot(predictiveDraws.length ? bins(frequencies) : [], {x: 'k', y: 'p', fill: 'white', stroke: purple, r: 3})
    ]
  });
  const scale = chart.scale('x');
  const width = Number(chart.getAttribute('width'));
  const height = Number(chart.getAttribute('height'));
  const brush = d3.brushX().extent([[48, 24], [width - 12, height - 42]]).on('end', ({selection}) => {
    if (!selection) return;
    const [lo, hi] = selection.map(px => Math.max(0, Math.min(future, Math.round(scale.invert(px)))));
    setRange(lo, hi);
    renderPredictive();
  });
  d3.select(chart).append('g').attr('aria-hidden', 'true').call(brush);
}

function samplePrediction() {
  const {a, b} = posterior(tosses);
  const cumulative = d3.cumsum(betaBinomial(value('future'), a, b));
  predictiveDraws = Array.from({length: 200}, () => d3.bisectLeft(cumulative, predictionRandom()));
  renderPredictive();
}

const checkNormal = d3.randomNormal.source(d3.randomLcg(6208))();
const makeCheckData = size => Array.from({length: size}, () => ({x: [checkNormal(), checkNormal()], error: checkNormal()}));
const checkTrain = makeCheckData(60);
const checkTest = makeCheckData(120).sort((a, b) => a.x[1] - b.x[1]);
const checkGroups = d3.range(0, checkTest.length, 20).map(start => ({
  start, x: d3.mean(checkTest.slice(start, start + 20), row => row.x[1])
}));
const checkBandwidth = 0.45;
const checkKernel = normalDensity(0, checkBandwidth);
let checkSeed = 6211;
let checkResult;

function sampleCheck() {
  const strength = value('check-strength');
  const outcome = ({x, error}) => x[0] + strength * x[1] + error;
  const response = checkTrain.map(outcome);
  const normal = d3.randomNormal.source(d3.randomLcg(checkSeed))();
  const replicate = predictors => {
    const design = checkTrain.map(({x}) => x.slice(0, predictors));
    const sample = regressionPosteriorSampler(design, response, 2);
    return Array.from({length: 30}, () => {
      const beta = sample(normal);
      return checkTest.map(({x}) => beta.reduce((mean, b, j) => mean + x[j] * b, 0) + normal());
    });
  };
  checkResult = {observed: checkTest.map(outcome), full: replicate(2), reduced: replicate(1)};
  write('check-strength-value', format(strength));
}

function renderCheck() {
  const {observed, full, reduced} = checkResult;
  const trend = byId('check-view').value === 'trend';
  const xDomain = trend ? d3.extent(checkGroups, d => d.x)
    : d3.extent([observed, ...full, ...reduced].flat()).map((bound, i) => bound + (i ? 3 : -3) * checkBandwidth);
  const xs = range(...xDomain, 100);
  const series = trend
    ? outcomes => checkGroups.map(({x, start}) => ({x, y: d3.mean(outcomes.slice(start, start + 20))}))
    : outcomes => xs.map(x => ({x, y: d3.mean(outcomes, y => checkKernel(x - y))}));
  const observedSeries = series(observed);
  const models = [
    {id: 'check-full-plot', name: 'True model', replicas: full.map(series)},
    {id: 'check-reduced-plot', name: 'Alternative model', replicas: reduced.map(series)}
  ];
  const allPoints = [observedSeries, ...models.flatMap(model => model.replicas)].flat();
  const yDomain = trend ? d3.extent(allPoints, d => d.y) : [0, d3.max(allPoints, d => d.y)];
  byId('check-guide').hidden = !trend;
  for (const {id, name, replicas} of models) {
    const lines = replicas.flatMap((points, replica) => points.map(point => ({...point, replica})));
    draw(id, `${name}. Observed test ${trend ? 'means by x₂' : 'outcome density'} and 30 posterior predictive replicas.`, {
      x: {domain: xDomain, label: trend ? 'Mean x₂ in group' : 'Outcome y'},
      y: {domain: yDomain, label: trend ? 'Mean outcome' : 'Density', nice: true, zero: false},
      marks: [
        Plot.lineY(lines, {x: 'x', y: 'y', z: 'replica', stroke: teal, strokeOpacity: 0.27, strokeWidth: 1.1}),
        Plot.lineY(observedSeries, {x: 'x', y: 'y', stroke: ink, strokeWidth: 2.8}),
        trend ? Plot.dot(observedSeries, {x: 'x', y: 'y', fill: ink, r: 3}) : null
      ]
    }, 246);
  }
}

function renderFactor() {
  if (!byId('factor-lab').open) return;
  const heads = value('factor-heads');
  const tails = value('factor-tails');
  const p = value('model-prob') / 100;
  const shape = value('factor-shape');
  const logPrior = Math.log(p / (1 - p));
  const logFactor = logBayesFactor(heads, tails, shape);
  const logPost = logPrior + logFactor;
  const after = probability(logPost);
  const nullAfter = probability(-logPost);
  const odds = log => Math.abs(log) < 12 ? d3.format('.3~g')(Math.exp(log)) : `exp(${format(log)})`;
  write('factor-heads-value', heads);
  write('factor-tails-value', tails);
  write('factor-total', `n = ${heads + tails}`);
  write('model-prob-value', `${value('model-prob')}%`);
  write('factor-shape-value', shape);
  byId('factor-bar-null').value = nullAfter;
  byId('factor-bar-alternative').value = after;
  write('factor-null', percent(nullAfter));
  write('factor-alternative', percent(after));
  write('factor-summary', `Prior P(M₁) ${percent(p)} · Prior odds ${odds(logPrior)} × BF₁₀ ${odds(logFactor)} = posterior odds ${odds(logPost)} → Posterior P(M₁) ${percent(after)}`);
}

const criterionNames = ['AIC', 'BIC', 'DIC'];
const criteriaRandom = d3.randomNormal.source(d3.randomLcg(6210))();
const criteriaData = Array.from({length: 100}, () => ({
  x: Array.from({length: 10}, criteriaRandom),
  error: criteriaRandom()
}));

function renderCriteria() {
  if (!byId('criteria-lab').open) return;
  const n = value('criteria-n');
  const p = value('criteria-p');
  const strength = value('criteria-strength');
  const sd = value('criteria-sd');
  const sample = criteriaData.slice(0, n);
  const full = sample.map(({x}) => x.slice(0, p));
  const reduced = full.map(row => row.slice(0, -1));
  const response = sample.map(({x, error}, i) =>
    error + reduced[i].reduce((sum, predictor) => sum + 0.5 * predictor, 0) + strength * x[p - 1]
  );
  const names = ['M₀', 'M₁'];
  const rows = [modelCriteria(reduced, response, sd), modelCriteria(full, response, sd)];
  for (const [id, number] of [['criteria-n', n], ['criteria-p', p], ['criteria-strength', strength], ['criteria-sd', sd]]) {
    write(`${id}-value`, format(number));
  }
  byId('criteria-table').innerHTML = rows.map(({k, pD, scores}, i) => {
    const cells = criterionNames.map(name => {
      const score = scores[name];
      const other = rows[1 - i].scores[name];
      if (score === null) return '<td>—</td>';
      return other !== null && score < other ? `<td><strong>${format(score)}</strong></td>` : `<td>${format(score)}</td>`;
    });
    return `<tr><th scope="row">${names[i]}</th><td>${k}</td>${cells.join('')}<td>${format(pD)}</td></tr>`;
  }).join('');
  const scores = rows.flatMap(({scores}, i) => criterionNames.map(name => ({
    model: names[i], name, score: scores[name]
  }))).filter(d => d.score !== null);
  draw('criteria-plot', 'AIC, BIC, and DIC scores grouped by regression model', {
    x: {domain: criterionNames, label: null},
    y: {label: 'Score'},
    fx: {domain: names, label: null},
    marginTop: 32,
    marks: [Plot.barY(scores, {fx: 'model', x: 'name', y: 'score', fill: teal, inset: 5})]
  }, 225);
  const penalties = criterionNames.map(name => ({name, strength: rows[1].penalties[name]}));
  draw('criteria-penalties', 'AIC, BIC, and DIC penalty strength for M1', {
    x: {domain: criterionNames, label: 'Criterion'},
    y: {label: 'Penalty strength'},
    marks: [Plot.barY(penalties, {x: 'name', y: 'strength', fill: teal, inset: 10})]
  }, 225);
  const largest = penalties.reduce((highest, item) => item.strength > highest.strength ? item : highest);
  const nominal = n < p ? ' nominal' : '';
  write('criteria-insight', `With n = ${n} and ${p} ${p === 1 ? 'predictor' : 'predictors'} in M₁, ${largest.name} applies the largest${nominal} penalty to M₁ (${format(largest.strength)}).`);
  byId('criteria-rank-note').hidden = n >= p;
}

function renderLasso() {
  if (!byId('lasso-lab').open) return;
  const lambda = value('lasso-rate');
  const estimate = [1.6, 2.4];
  const rho = 0.6;
  const residualSd = Math.sqrt(1 - rho ** 2);
  const {lasso, ridge} = shrinkageModes(estimate, rho, lambda);
  const angles = range(0, 2 * Math.PI, 160);
  const likelihoodContour = radius => angles.map(t => ({
    x: estimate[0] + radius * Math.cos(t),
    y: estimate[1] + radius * (rho * Math.cos(t) + residualSd * Math.sin(t))
  }));
  const domain = [-4, 5];
  const axes = [{domain, label: 'β₁'}, {domain, label: 'β₂', grid: false}];
  write('lasso-rate-value', lambda);

  for (const {name, mode, circle} of [
    {name: 'lasso', mode: lasso, circle: false}, {name: 'ridge', mode: ridge, circle: true}
  ]) {
    const boundary = radius => {
      const xs = circle ? range(-radius, radius, 100) : [-radius, 0, radius];
      return xs.map(x => {
        const y = circle ? Math.sqrt(Math.max(0, radius ** 2 - x ** 2)) : radius - Math.abs(x);
        return {x, lo: -y, hi: y};
      });
    };
    const densityBands = [0.15, 0.3, 0.5, 0.7, 0.85].map(level => {
      const radius = circle ? Math.sqrt(-2 * Math.log(level) / lambda) : -Math.log(level) / lambda;
      return Plot.areaY(boundary(radius), {x: 'x', y1: 'lo', y2: 'hi', fill: teal, fillOpacity: 0.1, clip: true});
    });
    const radius = circle ? Math.hypot(...mode) : Math.abs(mode[0]) + Math.abs(mode[1]);
    const shell = boundary(radius);
    const outline = [...shell.map(d => ({x: d.x, y: d.hi})), ...shell.reverse().map(d => ({x: d.x, y: d.lo}))];
    const dx = mode[0] - estimate[0];
    const dy = mode[1] - estimate[1];
    const touching = Math.hypot(dx, (dy - rho * dx) / residualSd);
    const width = Math.min(byId(`${name}-plot`).clientWidth, 455);
    draw(`${name}-plot`, `${name} likelihood contours, prior density, and posterior mode`, {
      width, x: axes[0], y: axes[1],
      marks: [
        ...densityBands,
        Plot.ruleX([0], {stroke: '#89948a'}),
        Plot.ruleY([0], {stroke: '#89948a'}),
        ...[0.55, 1.05, 1.65].map(t => Plot.lineY(likelihoodContour(t), {x: 'x', y: 'y', stroke: rust, strokeOpacity: 0.4, strokeDasharray: '4 4', clip: true})),
        Plot.lineY(likelihoodContour(touching), {x: 'x', y: 'y', stroke: rust, strokeWidth: 2, strokeDasharray: '5 3', clip: true}),
        Plot.lineY(outline, {x: 'x', y: 'y', stroke: teal, strokeWidth: 2.3, clip: true}),
        Plot.dot([estimate], {x: d => d[0], y: d => d[1], fill: 'white', stroke: rust, r: 5, title: () => 'Unregularized estimate'}),
        Plot.dot([mode], {x: d => d[0], y: d => d[1], fill: teal, stroke: 'white', r: 5.5, title: () => 'Posterior mode'})
      ]
    }, width + 6);
    write(`${name}-summary`, `Posterior mode (β₁, β₂) = (${mode.map(format).join(', ')}) · ${circle ? 'L₂' : 'L₁'} radius ${format(radius)}`);
  }
}

const labs = [
  ['factor-lab', renderFactor],
  ['criteria-lab', renderCriteria], ['lasso-lab', renderLasso]
];
for (const [id, render] of labs) byId(id).addEventListener('toggle', render);
for (const [id, render] of [
  ['interval-example', renderInterval], ['check-view', renderCheck],
  ['factor-heads', renderFactor], ['factor-tails', renderFactor], ['model-prob', renderFactor], ['factor-shape', renderFactor],
  ['criteria-n', renderCriteria], ['criteria-p', renderCriteria],
  ['criteria-strength', renderCriteria], ['criteria-sd', renderCriteria], ['lasso-rate', renderLasso]
]) byId(id).addEventListener('input', render);

byId('draw-prediction').addEventListener('click', samplePrediction);
byId('future').addEventListener('input', () => {
  predictiveDraws = [];
  renderPredictive();
});
for (const id of ['range-from', 'range-to']) byId(id).addEventListener('input', () => {
  const from = value('range-from');
  const to = value('range-to');
  if (from > to) setRange(id === 'range-from' ? from : to, id === 'range-from' ? from : to);
  renderPredictive();
});
function updatePrediction() {
  predictiveDraws = [];
  renderInterval();
  renderPredictive();
}
byId('prediction-prior').addEventListener('change', updatePrediction);
for (const id of ['observed-heads', 'observed-tails']) byId(id).addEventListener('input', () => {
  tosses = resizeTosses(tosses, value('observed-heads'), value('observed-tails'));
  byId('import-limit').hidden = true;
  updatePrediction();
});
byId('reset-inference').addEventListener('click', () => {
  tosses = coinExample;
  byId('import-limit').hidden = true;
  byId('prediction-prior').value = 'uniform';
  byId('interval-example').value = 'coin';
  byId('future').value = 10;
  for (const id of ['range-from', 'range-to']) byId(id).max = 10;
  setRange(7, 10);
  predictiveDraws = [];
  predictionRandom = d3.randomLcg(6207);
  renderAll();
});
byId('check-strength').addEventListener('input', () => {
  sampleCheck();
  renderCheck();
});
byId('resample-check').addEventListener('click', () => {
  checkSeed += 1;
  sampleCheck();
  renderCheck();
});

function renderAll() {
  renderInterval();
  renderPredictive();
  renderCheck();
  for (const [, render] of labs) render();
}

setupMcmcLab(draw);
window.addEventListener('resize', renderAll);
sampleCheck();
renderAll();
