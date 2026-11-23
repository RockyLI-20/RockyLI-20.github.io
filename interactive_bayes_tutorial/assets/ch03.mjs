import {axisMath, math} from './math.mjs?v=b22720b201be';

const {d3, Plot} = window;
const byId = id => document.getElementById(id);
const styles = getComputedStyle(document.querySelector('.bayes-page'));
const ink = styles.getPropertyValue('--ink').trim();
const fittedColor = styles.getPropertyValue('--posterior').trim();
const trueColor = styles.getPropertyValue('--evidence').trim();
const slope = byId('glm-slope');
const trainingSize = 120;
const testSize = 60;
const drawPredictor = d3.randomPoisson.source(d3.randomLcg(6209))(3);
const outcomeSeed = d3.randomLcg(6212);
const observations = Array.from({length: trainingSize + testSize}, () => ({
  x: drawPredictor(), seed: outcomeSeed()
}));

function fitOLS(data) {
  const meanX = d3.mean(data, row => row.x);
  const meanY = d3.mean(data, row => row.y);
  const slope = d3.sum(data, ({x, y}) => (x - meanX) * (y - meanY))
    / d3.sum(data, ({x}) => (x - meanX) ** 2);
  return x => meanY + slope * (x - meanX);
}

function fitPoisson(data) {
  const total = d3.sum(data, row => row.y);
  const observedMean = d3.sum(data, ({x, y}) => x * y) / total;
  const weight = (b, x) => Math.exp(b * x);
  const sumWeights = b => d3.sum(data, ({x}) => weight(b, x));
  // After profiling out the intercept, the score matches the mean predictor per event.
  let low = -8;
  let high = 8;
  for (let i = 0; i < 45; i += 1) {
    const mid = (low + high) / 2;
    const fittedMean = d3.sum(data, ({x}) => x * weight(mid, x)) / sumWeights(mid);
    if (fittedMean < observedMean) low = mid;
    else high = mid;
  }
  const b = (low + high) / 2;
  const scale = total / sumWeights(b);
  return x => scale * weight(b, x);
}

function draw(id, name, dots, means, fit, xDomain, yDomain) {
  const host = byId(id);
  const width = host.clientWidth;
  const height = 275;
  const chart = Plot.plot({
    ariaLabel: `${name} fitted means and true means versus the same observed test counts`,
    width, height, marginLeft: 40, marginRight: 15, marginTop: 24, marginBottom: 45,
    style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
    x: {domain: xDomain, tickFormat: x => Number.isInteger(x) ? x : '', label: null},
    y: {domain: yDomain, nice: true, grid: true, label: null},
    marks: [
      Plot.ruleY([0], {stroke: '#87928a'}),
      Plot.dot(dots, {
        x: 'x', y: 'y', r: 'frequency', fill: ink, fillOpacity: 0.42,
        title: ({x, y, frequency}) => `Predictor ${x}, count ${y} (${frequency} test observations)`
      }),
      Plot.lineY(means, {x: 'x', y: 'y', stroke: trueColor, strokeDasharray: '5 4'}),
      Plot.lineY(fit, {x: 'x', y: 'y', stroke: fittedColor, strokeWidth: 2.6})
    ]
  });
  axisMath(chart, `Predictor ${math('x')} →`, {x: 4, y: height - 22, width: width - 19, align: 'right'});
  axisMath(chart, `↑ Count ${math('y')}`, {x: 4, width: width - 8});
  host.replaceChildren(chart);
}

function render() {
  const b = slope.valueAsNumber;
  const trueMean = x => Math.exp(-0.9 + b * x);
  const data = observations.map(({x, seed}) => ({
    x, y: d3.randomPoisson.source(d3.randomLcg(seed))(trueMean(x))()
  }));
  const training = data.slice(0, trainingSize);
  const test = data.slice(trainingSize);
  const dots = d3.rollups(test, group => group.length, row => row.x, row => row.y)
    .flatMap(([x, yy]) => yy.map(([y, frequency]) => ({x, y, frequency})));
  const maxX = d3.max(test, row => row.x);
  const denseGrid = d3.range(101).map(i => i * maxX / 100);
  const points = predict => denseGrid.map(x => ({x, y: predict(x)}));
  const means = points(trueMean);
  const models = [
    {id: 'ols', name: 'OLS', predict: fitOLS(training)},
    {id: 'poisson', name: 'Poisson GLM', predict: fitPoisson(training)}
  ].map(model => ({...model, curve: points(model.predict)}));
  const values = [...dots, ...means, ...models.flatMap(model => model.curve)];
  const min = Math.min(0, d3.min(values, row => row.y));
  const max = d3.max(values, row => row.y);
  const padding = (max - min) * 0.05;
  const xDomain = [-0.25, maxX + 0.25];
  const yDomain = [min - padding, max + padding];

  byId('glm-slope-value').value = b.toFixed(2);
  const [ols, poisson] = models.map(({id, name, predict, curve}) => {
    const rmse = Math.sqrt(d3.mean(test, ({x, y}) => (y - predict(x)) ** 2)).toFixed(2);
    const negative = test.filter(({x}) => predict(x) < 0).length;
    byId(`${id}-score`).innerHTML = `<strong>Test RMSE: ${rmse}</strong> · Negative predictions ${negative}/${testSize}`;
    draw(`${id}-plot`, name, dots, means, curve, xDomain, yDomain);
    return Number(rmse);
  });
  const winner = ols < poisson ? 'OLS' : 'Poisson GLM';
  byId('glm-comparison').textContent = ols === poisson
    ? 'Both models have the same test RMSE to two decimal places.'
    : `${winner} has the lower test RMSE for these counts.`;
}

slope.addEventListener('input', render);
window.addEventListener('resize', render);
render();
