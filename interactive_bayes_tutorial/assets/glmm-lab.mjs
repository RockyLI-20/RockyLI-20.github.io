import {poissonGroupMeans} from './glmm-math.mjs?v=608a0d229a30';

const {d3, Plot} = window;
const byId = id => document.getElementById(id);
const styles = getComputedStyle(byId('glmm'));
const trueColor = styles.getPropertyValue('--evidence').trim();
const fittedColor = styles.getPropertyValue('--posterior').trim();
const tau = byId('glmm-tau');
const slope = byId('glmm-slope');
const groupCount = 5;
const testSize = 40;
const defaultSeed = 6206;
const format = d3.format('.3~g');
const labels = {glm: 'Poisson GLM', glmm: 'Poisson GLMM'};

let seed = defaultSeed;
let example = generate(seed);
let charts;
let frame = 0;

function generate(seed) {
  const random = d3.randomLcg(seed);
  const normal = d3.randomNormal.source(random)();
  const sizes = d3.shuffler(random)(d3.range(5, 31)).slice(0, groupCount);
  return {
    groups: sizes.map(n => ({
      n, z: normal(),
      training: Array.from({length: n}, () => ({x: normal(), seed: random()}))
    })),
    test: Array.from({length: testSize}, normal)
  };
}

function draw() {
  const {points, domain} = charts;
  for (const [model, name] of Object.entries(labels)) {
    const host = byId(`glmm-${model}-plot`);
    host.replaceChildren(Plot.plot({
      ariaLabel: `${name}: true and predicted mean counts and training sample sizes for five groups`,
      width: host.clientWidth, height: 225, marginLeft: 115, marginRight: 22, marginBottom: 46,
      style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
      x: {domain, nice: true, grid: true, ticks: 5, label: 'Mean count'},
      y: {domain: points.map(row => row.label), label: null},
      marks: [
        Plot.ruleY(points, {y: 'label', x1: 'truth', x2: model, stroke: '#8a9990', strokeWidth: 1.5}),
        Plot.dot(points, {y: 'label', x: 'truth', stroke: trueColor, r: 6,
          title: row => `${row.label} · True mean ${format(row.truth)}`}),
        Plot.dot(points, {y: 'label', x: model, fill: fittedColor, r: 3.5,
          title: row => `${row.label} · Predicted mean ${format(row[model])}`})
      ]
    }));
  }
}

function fit() {
  frame = 0;
  const between = tau.valueAsNumber;
  const beta1 = slope.valueAsNumber;
  const {groups, test} = example;
  const meanCount = (x, z) => Math.exp(-0.5 + beta1 * x + between * z);
  const training = groups.flatMap(({z, training}, group) => training.map(({x, seed}) => ({
    group, x, y: d3.randomPoisson.source(d3.randomLcg(seed))(meanCount(x, z))()
  })));
  const {glm, glmm} = poissonGroupMeans(training, test, groupCount);
  const points = groups.map(({n, z}, j) => ({
    label: `Group ${j + 1} (n = ${n})`,
    truth: d3.mean(test, x => meanCount(x, z)),
    glm: glm[j], glmm: glmm[j]
  }));
  charts = {points, domain: [0, d3.max(points, row => Math.max(row.truth, row.glm, row.glmm)) * 1.05]};
  draw();

  const report = model => {
    const rmse = format(Math.sqrt(d3.mean(points, row => (row[model] - row.truth) ** 2)));
    byId(`glmm-${model}-score`).innerHTML = `<strong>Group mean RMSE: ${rmse}</strong>`;
    return Number(rmse);
  };
  const pooled = report('glm');
  const mixed = report('glmm');
  const result = pooled === mixed
    ? 'The two models have the same RMSE at the displayed precision.'
    : `${mixed < pooled ? 'Partial pooling' : 'Complete pooling'} gives a lower RMSE for this simulated sample.`;
  byId('glmm-comparison').textContent = (between === 0 ? 'At τ = 0, all groups have the same true mean. ' : '') + result;
}

function scheduleFit() {
  byId('glmm-tau-value').value = tau.valueAsNumber.toFixed(2);
  byId('glmm-slope-value').value = slope.valueAsNumber.toFixed(2);
  frame ||= requestAnimationFrame(fit);
}

tau.addEventListener('input', scheduleFit);
slope.addEventListener('input', scheduleFit);
byId('glmm-resimulate').addEventListener('click', () => {
  example = generate(++seed);
  scheduleFit();
});
byId('glmm-reset').addEventListener('click', () => {
  tau.value = tau.defaultValue;
  slope.value = slope.defaultValue;
  seed = defaultSeed;
  example = generate(seed);
  scheduleFit();
});
window.addEventListener('resize', draw);
fit();
