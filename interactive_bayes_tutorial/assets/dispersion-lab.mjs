import {countRegressionFits, countLogLikelihood, countPredictionInterval} from './count-math.mjs?v=608a0d229a30';

const {d3, Plot} = window;
const byId = id => document.getElementById(id);
const lab = byId('overdispersion');
const styles = getComputedStyle(lab);
const color = styles.getPropertyValue('--posterior').trim();
const evidence = styles.getPropertyValue('--evidence').trim();
const ink = styles.getPropertyValue('--ink').trim();
const phi = byId('dispersion-phi');
const slope = byId('dispersion-slope');
const trainSize = 100;
const testSize = 100;
const defaultSeed = 6254;
const style = {background: 'transparent', fontFamily: 'inherit', fontSize: 12};
const format = d3.format('.2~f');

let seed = defaultSeed;
let example = generate(seed);
let charts;
let frame = 0;

function generate(seed) {
  const random = d3.randomLcg(seed);
  const normal = d3.randomNormal.source(random)(0, Math.sqrt(2));
  return Array.from({length: trainSize + testSize}, () => ({x: normal(), seed: random()}));
}

function draw() {
  const {training, xDomain, yDomain, fits, bins, variance, dispersion} = charts;
  for (const {id, label, curve} of fits) {
    const host = byId(id);
    host.replaceChildren(Plot.plot({
      ariaLabel: `${label} regression with 100 training counts and a 95% prediction interval`,
      width: host.clientWidth, height: 270, marginLeft: 46, marginRight: 14, marginBottom: 40,
      style, x: {domain: xDomain, label: 'Predictor x'}, y: {domain: yDomain, label: 'Count', grid: true, nice: true},
      marks: [
        Plot.areaY(curve, {x: 'x', y1: 'lower', y2: 'upper', fill: '#d6e4dc', curve: 'step'}),
        Plot.dot(training, {x: 'x', y: 'y', fill: ink, fillOpacity: 0.45, r: 3,
          title: ({x, y}) => `x = ${format(x)}, count = ${y}`}),
        Plot.lineY(curve, {x: 'x', y: 'mean', stroke: color, strokeWidth: 2.5})
      ]
    }));
  }
  const host = byId('dispersion-variance');
  host.replaceChildren(Plot.plot({
    ariaLabel: 'Five observed bin variances compared with the Poisson and negative binomial mean–variance curves',
    width: host.clientWidth, height: 250, marginLeft: 50, marginRight: 20, marginBottom: 40,
    style, x: {label: 'Mean count', nice: true}, y: {label: 'Variance', grid: true},
    marks: [
      Plot.lineY(variance, {x: 'mean', y: 'mean', stroke: color, strokeWidth: 2}),
      Plot.lineY(variance, {x: 'mean', y: ({mean}) => mean + mean ** 2 / dispersion,
        stroke: evidence, strokeWidth: 2, strokeDasharray: '5 4'}),
      Plot.dot(bins, {x: 'mean', y: 'variance', fill: ink, r: 4.5,
        title: ({mean, variance, range}) => `20 counts at x = ${range} · Mean ${format(mean)} · Variance ${format(variance)}`})
    ]
  }));
}

function fit() {
  frame = 0;
  const dispersion = phi.valueAsNumber;
  const beta1 = slope.valueAsNumber;
  const data = example.map(({x, seed}) => {
    const random = d3.randomLcg(seed);
    const mean = Math.exp(1 + beta1 * x);
    const rate = d3.randomGamma.source(random)(dispersion, mean / dispersion)();
    return {x, y: d3.randomPoisson.source(random)(rate)()};
  });
  const training = data.slice(0, trainSize);
  const test = data.slice(trainSize);
  const [left, right] = d3.extent(data, ({x}) => x);
  const xGrid = d3.range(101).map(i => left + (right - left) * i / 100);
  const coverage = [];
  const fits = countRegressionFits(training).map((model, i) => {
    const id = i === 0 ? 'dispersion-poisson' : 'dispersion-nb';
    const label = i === 0 ? 'Poisson' : 'Negative binomial';
    const curve = xGrid.map(x => {
      const mean = model.mean(x);
      const [lower, upper] = countPredictionInterval(mean, model.phi);
      return {x, mean, lower, upper};
    });
    const covered = test.filter(({x, y}) => {
      const [lower, upper] = countPredictionInterval(model.mean(x), model.phi);
      return lower <= y && y <= upper;
    }).length;
    coverage.push(covered);
    const logLik = countLogLikelihood(test, model);
    byId(`${id}-score`).innerHTML = `<strong>Test coverage: ${d3.format('.0%')(covered / testSize)}</strong> · Test log likelihood: ${format(logLik)}`
      + (i === 1 ? ` · Fitted dispersion: ${format(model.phi)}` : '');
    return {id, label, curve};
  });
  const [poissonCovered, nbCovered] = coverage;
  byId('dispersion-comparison').textContent = poissonCovered === nbCovered
    ? 'Both intervals cover the same number of held-out counts in this simulation.'
    : `${nbCovered > poissonCovered ? 'The negative binomial interval' : 'The Poisson interval'} covers more held-out counts in this simulation.`;

  const sorted = d3.sort(training, ({x}) => x);
  const bins = d3.range(0, trainSize, 20).map(i => {
    const bin = sorted.slice(i, i + 20);
    return {mean: d3.mean(bin, row => row.y), variance: d3.variance(bin, row => row.y),
      range: `${format(bin[0].x)} to ${format(bin.at(-1).x)}`};
  });
  const maximumMean = Math.max(1, d3.max(bins, row => row.mean));
  const variance = d3.range(61).map(i => ({mean: i / 60 * maximumMean}));
  const top = Math.max(d3.max(training, row => row.y), ...fits.map(({curve}) => d3.max(curve, row => row.upper)));
  charts = {training, xDomain: [left, right], yDomain: [0, top], fits, bins, variance, dispersion};
  if (lab.open) draw();
}

function scheduleFit() {
  byId('dispersion-phi-value').value = phi.valueAsNumber.toFixed(2);
  byId('dispersion-slope-value').value = slope.valueAsNumber.toFixed(2);
  frame ||= requestAnimationFrame(fit);
}

phi.addEventListener('input', scheduleFit);
slope.addEventListener('input', scheduleFit);
byId('dispersion-resimulate').addEventListener('click', () => {
  example = generate(++seed);
  scheduleFit();
});
byId('dispersion-reset').addEventListener('click', () => {
  phi.value = phi.defaultValue;
  slope.value = slope.defaultValue;
  seed = defaultSeed;
  example = generate(seed);
  scheduleFit();
});
lab.addEventListener('toggle', () => {
  if (!lab.open) return;
  if (charts) draw();
  else scheduleFit();
});
window.addEventListener('resize', () => {
  if (lab.open && charts) draw();
});
if (lab.open) scheduleFit();
