import {
  betaDensity, betaUpdate, coinExample, countHeads, logOdds, logOddsDensity, normalDensity, normalUpdate, relativeLikelihood
} from './bayes-math.mjs?v=d6303284914f';

const {Plot, d3} = window;
const byId = id => document.getElementById(id);
const value = id => byId(id).valueAsNumber;
const write = (id, text) => { byId(id).textContent = text; };
const percent = x => `${(100 * x).toFixed(1)}%`;
const chartColors = getComputedStyle(document.querySelector('.bayes-page'));
const priorColor = chartColors.getPropertyValue('--prior').trim();
const posteriorColor = chartColors.getPropertyValue('--posterior').trim();
const evidenceColor = chartColors.getPropertyValue('--evidence').trim();
const examples = {default: coinExample, heads: 'H'.repeat(17), tails: 'T'.repeat(17), empty: ''};
const priors = {uniform: ['Uniform', 1, 1], jeffreys: ['Jeffreys', 0.5, 0.5], haldane: ['Haldane', 0, 0]};
const growthSizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000];
const stepManually = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
let tosses = [...examples.default];
let random = d3.randomLcg(6206);
let growthTimer = null;

function grid(start, end, steps = 240) {
  return Array.from({length: steps + 1}, (_, i) => i === steps ? end : start + (end - start) * i / steps);
}

function points(xs, fn) {
  return xs.map(x => ({x, y: fn(x)}));
}

function betaPoints(a, b) {
  const low = a < 1 ? 0.01 : 0;
  const high = b < 1 ? 0.99 : 1;
  const density = betaDensity(a, b);
  if (a < 1 || b < 1) return points(grid(low, high), density);
  const mean = a / (a + b);
  const sd = Math.sqrt(mean * (1 - mean) / (a + b + 1));
  const xs = [...grid(low, high), ...grid(Math.max(low, mean - 6 * sd), Math.min(high, mean + 6 * sd))];
  return points(xs.sort((x, y) => x - y), density);
}

function draw(id, label, x, y, marks, height = 215) {
  const container = byId(id);
  const width = container.clientWidth;
  const {label: leftLabel, rightLabel, ...scale} = y;
  const axisLabel = document.createElement('div');
  axisLabel.className = 'bayes-axis-label';
  axisLabel.textContent = leftLabel;
  if (rightLabel) {
    const secondaryLabel = document.createElement('span');
    secondaryLabel.textContent = rightLabel;
    axisLabel.append(secondaryLabel);
  }
  const svg = Plot.plot({
    width, height: height - 14, marginLeft: 45, marginRight: rightLabel ? 36 : 12, marginTop: 9, marginBottom: 41,
    ariaLabel: label,
    style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
    x: {ticks: width < 215 ? 2 : width < 380 ? 4 : 7, ...x},
    y: {grid: true, ticks: 4, ...scale, label: null},
    marks
  });
  container.replaceChildren(axisLabel, svg);
}

const probAxis = {domain: [0, 1], label: 'Chance of heads, θ'};
const densityAxis = {label: 'Density', zero: true};
const line = (data, stroke, pattern) => Plot.lineY(data, {x: 'x', y: 'y', stroke, strokeWidth: 2.3, strokeDasharray: pattern});
const reference = (x, stroke = '#7c887e') => Plot.ruleX([x], {stroke, strokeDasharray: '2 4'});
const counts = () => {
  const heads = countHeads(tosses);
  return [heads, tosses.length - heads];
};
const countText = (heads, tails) => `${heads} ${heads === 1 ? 'head' : 'heads'} · ${tails} ${tails === 1 ? 'tail' : 'tails'} · ${heads + tails} tosses`;
const betaText = ({a, b}) => `Beta(${a}, ${b})`;

function drawUpdate(id, label, x, {prior, updated, likelihood, marker, densityLimit = 1.07 * d3.max([...prior, ...updated], p => p.y), height}) {
  const axis = {...densityAxis, domain: [0, densityLimit], rightLabel: 'Relative likelihood'};
  const evidence = likelihood.map(({x, y}) => ({x, y: y * densityLimit}));
  draw(id, `${label}. Prior and posterior density on the left axis, relative likelihood on the right axis`, x, axis, [
    marker, line(prior, priorColor, '6 4'), line(updated, posteriorColor), line(evidence, evidenceColor, '2 4'),
    Plot.axisY({anchor: 'left', ticks: 4}),
    Plot.axisY({
      anchor: 'right', color: evidenceColor, ticks: [0, densityLimit / 2, densityLimit],
      tickFormat: y => (y / densityLimit).toFixed(1)
    })
  ], height);
}

function coinLikelihood(heads, tails) {
  const total = heads + tails;
  const xs = total ? [...grid(0, 1), heads / total].sort((a, b) => a - b) : grid(0, 1);
  return points(xs, relativeLikelihood(heads, tails));
}

function renderCustom(heads, tails, likelihood = coinLikelihood(heads, tails)) {
  const a = value('custom-alpha');
  const b = value('custom-beta');
  const posterior = betaUpdate(a, b, heads, tails);
  write('custom-alpha-value', a);
  write('custom-beta-value', b);
  drawUpdate('custom-plot', 'Your prior', probAxis, {
    prior: betaPoints(a, b), updated: betaPoints(posterior.a, posterior.b), likelihood, marker: reference(0.5)
  });
  const boundaryNote = a < 1 || b < 1 ? ' A density that diverges at a boundary is drawn up to 0.01 from that boundary.' : '';
  write('custom-summary', `Prior ${betaText({a, b})} · Posterior ${betaText(posterior)} · Posterior mean ${percent(posterior.mean)}.${boundaryNote}`);
}

function renderCoin() {
  const [heads, tails] = counts();
  write('coin-bias-value', percent(value('coin-bias')));
  write('toss-count', countText(heads, tails) + (tosses.length > 40 ? ' · Last 40 shown' : ''));
  byId('toss-log').replaceChildren(...tosses.slice(-40).map((toss, index, recent) => {
    const item = document.createElement('li');
    const label = toss === 'H' ? 'Heads' : 'Tails';
    item.textContent = toss;
    item.title = `Toss ${tosses.length - recent.length + index + 1}: ${label}`;
    item.setAttribute('aria-label', item.title);
    item.toggleAttribute('data-tails', toss === 'T');
    return item;
  }));

  const comparisons = [['weak', 1], ['strong', 20]].map(([name, shape]) => {
    const posterior = betaUpdate(shape, shape, heads, tails);
    return {name, posterior, prior: betaPoints(shape, shape), updated: betaPoints(posterior.a, posterior.b)};
  });
  write('mobile-means', `Posterior means · Weak ${percent(comparisons[0].posterior.mean)} · Strong ${percent(comparisons[1].posterior.mean)}`);
  const maximum = d3.max(comparisons, ({prior, updated}) => d3.max([...prior, ...updated], p => p.y));
  const likelihood = coinLikelihood(heads, tails);
  const total = heads + tails;
  for (const {name, posterior, prior, updated} of comparisons) {
    drawUpdate(`${name}-plot`, `${name} prior`, probAxis, {
      prior, updated, likelihood, marker: reference(0.5), densityLimit: maximum * 1.07
    });
    write(`${name}-summary`, `${betaText(posterior)} · Posterior mean ${percent(posterior.mean)}`);
  }

  write('likelihood-summary', total
    ? `The same likelihood curve peaks at the observed proportion, ${percent(heads / total)}.`
    : 'Add a toss to see which values are better supported. The likelihood is flat before any data.');

  renderCustom(heads, tails, likelihood);
  renderObjective();
}

function renderObjective() {
  if (!byId('objective').open) return;
  const [heads, tails] = counts();
  const selected = byId('objective-prior').value;
  write('objective-count', countText(heads, tails));
  byId('prior-summary').innerHTML = Object.values(priors).map(([name, a, b]) => {
    const posterior = betaUpdate(a, b, heads, tails);
    return `<tr><th scope="row">${name}</th><td>${posterior ? betaText(posterior) : 'Improper'}</td><td>${posterior ? percent(posterior.mean) : 'Undefined'}</td></tr>`;
  }).join('');

  const [name, a, b] = priors[selected];
  const posterior = betaUpdate(a, b, heads, tails);
  const haldane = selected === 'haldane';
  byId('kernel-figure').hidden = !haldane;
  byId('objective-prior-key').hidden = haldane;
  byId('objective-legend').hidden = !posterior;
  write('objective-title', haldane ? 'Haldane posterior density' : `${name} prior and posterior density`);
  byId('objective-plot').hidden = !posterior;
  if (posterior) {
    const marks = haldane ? [] : [line(betaPoints(a, b), priorColor, '6 4')];
    marks.push(line(betaPoints(posterior.a, posterior.b), posteriorColor));
    draw('objective-plot', `${name} prior comparison, posterior ${betaText(posterior)}`, probAxis, densityAxis, marks, 245);
  }
  const description = posterior
    ? `${betaText(posterior)} has mean ${percent(posterior.mean)}.`
    : 'The posterior kernel has infinite integral. Haldane needs at least one head and one tail before a posterior density and mean exist.';
  write('objective-note', description + (selected === 'jeffreys'
    ? ' A curve that diverges at a boundary stops 0.01 from that boundary.'
    : ''));
  if (haldane) {
    draw('kernel-plot', 'Unnormalized Haldane prior kernel', probAxis, {label: 'Kernel height', zero: true}, [
      line(points(grid(0.01, 0.99), x => 1 / (x * (1 - x))), priorColor, '6 4')
    ], 245);
  }
}

function renderCoordinates() {
  if (!byId('coordinates').open) return;
  const cutoff = value('cutoff');
  const transformed = logOdds(cutoff);
  write('cutoff-value', cutoff.toFixed(2));
  write('coordinate-summary', `P(θ ≤ ${cutoff.toFixed(2)}) = P(γ ≤ ${transformed.toFixed(2)}) = ${percent(cutoff)}.`);
  for (const {id, axis, fn, threshold} of [
    {id: 'probability-plot', axis: probAxis, fn: () => 1, threshold: cutoff},
    {id: 'log-odds-plot', axis: {domain: [-8, 8], label: 'Log-odds, γ'}, fn: logOddsDensity, threshold: transformed}
  ]) {
    const [low, high] = axis.domain;
    draw(id, `Density on the ${axis.label} scale, ${percent(cutoff)} below the cutoff`, axis, densityAxis, [
      Plot.areaY(points(grid(low, threshold), fn), {x: 'x', y: 'y', fill: posteriorColor, fillOpacity: 0.2}),
      line(points(grid(low, high), fn), posteriorColor),
      reference(threshold, evidenceColor)
    ], 245);
  }
}

function normalPoints(domain, mean, sd) {
  const xs = [
    ...grid(...domain),
    ...grid(Math.max(domain[0], mean - 4 * sd), Math.min(domain[1], mean + 4 * sd)),
    mean
  ];
  return points(xs.sort((a, b) => a - b), normalDensity(mean, sd));
}

function renderNormal() {
  if (!byId('normal').open) return;
  const ids = ['prior-mean', 'prior-sd', 'sample-mean', 'sample-size'];
  const [priorMean, priorSd, sampleMean, sampleSize] = ids.map(value);
  for (const id of ids) write(`${id}-value`, value(id));
  const observationSd = 2;
  const posterior = normalUpdate(priorMean, priorSd, sampleMean, sampleSize, observationSd);
  const domain = [
    Math.min(priorMean - 4 * priorSd, sampleMean - 4 * observationSd),
    Math.max(priorMean + 4 * priorSd, sampleMean + 4 * observationSd)
  ];
  const likelihood = normalPoints(domain, sampleMean, observationSd / Math.sqrt(sampleSize));
  const peak = d3.max(likelihood, p => p.y);
  drawUpdate('normal-plot', 'Normal mean', {domain, label: 'Unknown mean, μ'}, {
    prior: normalPoints(domain, priorMean, priorSd),
    updated: normalPoints(domain, posterior.mean, posterior.sd),
    likelihood: likelihood.map(({x, y}) => ({x, y: y / peak})),
    marker: reference(sampleMean),
    height: 260
  });
  const priorWeight = 1 - posterior.sampleWeight;
  write('normal-summary', `Posterior mean ${posterior.mean.toFixed(2)} · Posterior SD ${posterior.sd.toFixed(2)}`);
  write('prior-weight', `Prior mean weight ${percent(priorWeight)}`);
  write('sample-weight', `Sample mean weight ${percent(posterior.sampleWeight)}`);
  byId('sample-weight-bar').value = posterior.sampleWeight;
}

function showSampleSize(size) {
  byId('sample-size').value = size;
  renderNormal();
}

function stopGrowth() {
  clearInterval(growthTimer);
  growthTimer = null;
  write('play-growth', stepManually ? 'Step through sample growth' : 'Play sample growth');
}

document.querySelectorAll('[data-toss]').forEach(button => button.addEventListener('click', () => {
  tosses.push(button.dataset.toss);
  renderCoin();
}));
byId('coin-bias').addEventListener('input', () => write('coin-bias-value', percent(value('coin-bias'))));
byId('simulate').addEventListener('click', () => {
  const bias = value('coin-bias');
  tosses.push(...Array.from({length: 10}, () => random() < bias ? 'H' : 'T'));
  renderCoin();
});
byId('clear-tosses').addEventListener('click', () => {
  tosses = [];
  renderCoin();
});
byId('reset-tosses').addEventListener('click', () => {
  tosses = [...examples.default];
  random = d3.randomLcg(6206);
  byId('coin-bias').value = 0.75;
  renderCoin();
});
for (const id of ['custom-alpha', 'custom-beta']) {
  byId(id).addEventListener('input', () => renderCustom(...counts()));
}
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  tosses = [...examples[button.dataset.example]];
  renderCoin();
}));
byId('objective').addEventListener('toggle', renderObjective);
byId('objective-prior').addEventListener('change', renderObjective);
byId('coordinates').addEventListener('toggle', renderCoordinates);
byId('cutoff').addEventListener('input', renderCoordinates);
byId('normal').addEventListener('toggle', () => {
  if (!byId('normal').open) stopGrowth();
  renderNormal();
});
byId('normal').querySelectorAll('input').forEach(input => input.addEventListener('input', () => {
  if (growthTimer) stopGrowth();
  renderNormal();
}));
byId('play-growth').addEventListener('click', () => {
  if (stepManually) {
    showSampleSize(growthSizes.find(size => size > value('sample-size')) ?? growthSizes[0]);
    return;
  }
  if (growthTimer) {
    stopGrowth();
    return;
  }
  let frame = 0;
  showSampleSize(growthSizes[frame++]);
  write('play-growth', 'Pause sample growth');
  growthTimer = setInterval(() => {
    showSampleSize(growthSizes[frame++]);
    if (frame === growthSizes.length) stopGrowth();
  }, 320);
});
window.addEventListener('resize', () => {
  renderCoin();
  renderCoordinates();
  renderNormal();
});
stopGrowth();
renderCoin();
