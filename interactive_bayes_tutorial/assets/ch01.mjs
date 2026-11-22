import {
  betaDensity, betaUpdate, countHeads, logOdds, logOddsDensity, relativeLikelihood
} from './bayes-math.mjs?v=71a58d6efbfb';
import {clopperPearson, discretize, highestDensity} from './inference-math.mjs?v=71a58d6efbfb';
import {axisMath, betaMath, math, tex} from './math.mjs?v=71a58d6efbfb';

const {Plot, d3} = window;
const byId = id => document.getElementById(id);
const value = id => byId(id).valueAsNumber;
const write = (id, text) => { byId(id).textContent = text; };
const writeMath = (id, markup) => { byId(id).innerHTML = markup; };
const percent = x => `${(100 * x).toFixed(1)}%`;
const chartColors = getComputedStyle(document.querySelector('.bayes-page'));
const priorColor = chartColors.getPropertyValue('--prior').trim();
const posteriorColor = chartColors.getPropertyValue('--posterior').trim();
const evidenceColor = chartColors.getPropertyValue('--evidence').trim();
const priors = {uniform: ['Uniform', 1, 1], jeffreys: ['Jeffreys', 0.5, 0.5], haldane: ['Haldane', 0, 0]};
const growthLimit = 500;
const growthSizes = [1, 2, 4, 8, 12, 18, 26, 38, 55, 80, 115, 160, 220, 300, 400, growthLimit];
const growthFrameMs = 96;
const stepManually = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const coinPlots = new Map();
const coinGrid = grid(0, 1, 160).map(t => Math.sin(Math.PI * t / 2) ** 2);
let tosses = [];
let random = d3.randomLcg(6206);
let growthFrame = 0;
let lastGrowth = 0;

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
    const secondary = document.createElement('span');
    secondary.textContent = rightLabel;
    axisLabel.append(secondary);
  }
  const svg = Plot.plot({
    width, height: height - 14, marginLeft: 45, marginRight: rightLabel ? 38 : 12, marginTop: 9, marginBottom: 41,
    ariaLabel: label,
    style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
    x: {ticks: width < 215 ? 2 : width < 380 ? 4 : 7, domain: x.domain, label: null},
    y: {grid: true, ticks: 4, ...scale, label: null},
    marks
  });
  axisMath(svg, `${x.label}, ${math(x.symbol)} →`, {x: 4, y: height - 35, width: width - 16, align: 'right'});
  container.replaceChildren(axisLabel, svg);
}

const probAxis = {domain: [0, 1], label: 'Chance of heads', symbol: tex`\theta`};
const densityAxis = {label: 'Density', zero: true};
const line = (data, stroke, pattern) => Plot.lineY(data, {x: 'x', y: 'y', stroke, strokeWidth: 2.3, strokeDasharray: pattern});
const reference = (x, stroke = '#7c887e') => Plot.ruleX([x], {stroke, strokeDasharray: '2 4'});
const counts = () => {
  const heads = countHeads(tosses);
  return [heads, tosses.length - heads];
};
const countText = (heads, tails) => {
  const total = heads + tails;
  return `${heads} ${heads === 1 ? 'head' : 'heads'} · ${tails} ${tails === 1 ? 'tail' : 'tails'} · ${total} ${total === 1 ? 'toss' : 'tosses'}`;
};
const betaText = ({a, b}) => `Beta(${a}, ${b})`;

function likelihoodMarks(likelihood, limit) {
  return [
    line(likelihood.map(({x, y}) => ({x, y: limit * y})), evidenceColor, '2 4'),
    Plot.axisY({
      anchor: 'right', color: evidenceColor, ticks: [0, limit / 2, limit],
      tickFormat: y => (y / limit).toFixed(1), label: null
    })
  ];
}

function renderSmallSample() {
  const n = value('small-sample-n');
  const x = value('small-sample-x');
  const estimate = x / n;
  const posterior = betaUpdate(1, 1, x, n - x);
  const [lo, hi] = clopperPearson(x, n);
  const density = betaDensity(posterior.a, posterior.b);
  const regions = highestDensity(discretize(density, t => t), 0.95);
  const shade = regions.flatMap(([lo, hi], region) => points(grid(lo, hi, 100), density).map(point => ({...point, region})));
  const intervalText = bounds => `[${bounds.map(bound => bound.toFixed(3)).join(', ')}]`;
  const ci = intervalText([lo, hi]);
  const hpd = regions.map(intervalText).join(' ∪ ');
  const posteriorData = coinDensity(posterior.a, posterior.b);
  const limit = 1.18 * d3.max(posteriorData, point => point.y);
  const comparison = 2 * x === n ? 'same as' : 'pulled back from';

  write('small-sample-x-value', x);
  write('small-sample-n-value', n);
  writeMath('small-sample-mle', math(tex`\hat\theta=x/n=${estimate.toFixed(3)}`));
  write('small-sample-mean', posterior.mean.toFixed(3));
  write('small-sample-comparison', `(${comparison} ${estimate.toFixed(3)})`);
  writeMath('small-sample-ci', `95% Clopper–Pearson CI ${math(ci)}`);
  writeMath('small-sample-hpd', `${betaMath(posterior.a, posterior.b)} posterior · 95% HPD ${math(regions.map(intervalText).join(tex` \cup `))}`);

  draw('small-sample-ci-plot', `Frequentist estimate ${estimate.toFixed(3)} and 95% confidence interval ${ci}`, probAxis,
    {label: '', domain: [-1, 1], axis: null, grid: false}, [
      Plot.ruleY([0], {x1: 0, x2: 1, stroke: '#d8d8c8'}),
      Plot.ruleY([0], {x1: lo, x2: hi, stroke: evidenceColor, strokeWidth: 4}),
      Plot.ruleX([lo, hi], {y1: -0.07, y2: 0.07, stroke: evidenceColor, strokeWidth: 2}),
      Plot.dot([{x: estimate, y: 0}], {x: 'x', y: 'y', fill: evidenceColor, stroke: 'white', r: 6})
    ]);
  draw('small-sample-density', `Uniform prior, ${betaText(posterior)} posterior, 95% highest posterior density interval ${hpd}, and relative likelihood on the right axis`,
    probAxis, {...densityAxis, rightLabel: 'Relative likelihood', domain: [0, limit]}, [
      Plot.areaY(shade, {x: 'x', y: 'y', z: 'region', fill: posteriorColor, fillOpacity: 0.27}),
      line(coinDensity(1, 1), priorColor, '6 4'),
      line(posteriorData, posteriorColor),
      Plot.axisY({anchor: 'left', ticks: 4}),
      ...likelihoodMarks(coinLikelihood(x, n - x), limit)
    ]);
}

function makeCoinPlot(id) {
  const host = byId(id);
  const width = host.clientWidth;
  const height = 201;
  const left = 45;
  const right = width - 36;
  const top = 10;
  const bottom = height - 38;
  const x = d3.scaleLinear([0, 1], [left, right]);
  const y = d3.scaleLinear([0, 1], [bottom, top]);
  const relative = y.copy();
  const legend = d3.create('div').attr('class', 'bayes-axis-label').text('Density');
  const rightLabel = legend.append('span').text('Relative likelihood');
  const svg = d3.create('svg').attr('width', width).attr('height', height).attr('role', 'img');
  const grid = svg.append('g').attr('transform', `translate(${left},0)`).attr('class', 'bayes-coin-grid');
  const yAxis = svg.append('g').attr('transform', `translate(${left},0)`);
  const rightAxis = svg.append('g').attr('transform', `translate(${right},0)`).attr('color', evidenceColor)
    .call(d3.axisRight(relative).tickValues([0, 0.5, 1]).tickFormat(d3.format('.1f')).tickSizeOuter(0));
  svg.append('g').attr('transform', `translate(0,${bottom})`)
    .call(d3.axisBottom(x).ticks(width < 215 ? 2 : width < 380 ? 4 : 7).tickSizeOuter(0));
  axisMath(svg.node(), `${probAxis.label}, ${math(probAxis.symbol)} →`, {
    x: left, y: height - 20, width: right - left, align: 'right'
  });
  svg.append('line').attr('x1', x(0.5)).attr('x2', x(0.5)).attr('y1', top).attr('y2', bottom)
    .attr('stroke', '#7c887e').attr('stroke-dasharray', '2 4');
  const path = (stroke, pattern) => svg.append('path').attr('fill', 'none')
    .attr('stroke', stroke).attr('stroke-width', 2.3).attr('stroke-dasharray', pattern);
  const prior = path(priorColor, '6 4');
  const posterior = path(posteriorColor, null);
  const likelihood = path(evidenceColor, '2 4');
  host.replaceChildren(legend.node(), svg.node());
  return {svg, x, y, relative, grid, yAxis, rightAxis, rightLabel, prior, posterior, likelihood};
}

function drawUpdate(id, label, {prior, updated, likelihood, densityLimit = 1.07 * d3.max([...prior, ...updated], p => p.y), animate}) {
  if (!coinPlots.has(id)) coinPlots.set(id, makeCoinPlot(id));
  const plot = coinPlots.get(id);
  const hasData = likelihood.length > 0;
  const transition = animate ? plot.svg.transition('growth').duration(growthFrameMs - 8).ease(d3.easeLinear) : null;
  const paint = selection => transition ? selection.transition(transition) : selection.interrupt('growth');
  plot.svg.attr('aria-label', hasData
    ? `${label}. Prior and posterior density on the left axis, relative likelihood on the right axis`
    : `${label}. Prior density before any tosses`);
  plot.rightLabel.style('visibility', hasData ? null : 'hidden');
  plot.rightAxis.style('display', hasData ? null : 'none');
  const limit = plot.y.domain([0, densityLimit]).nice(4).domain()[1];
  if (plot.limit !== limit || !animate) {
    plot.limit = limit;
    if (!animate) {
      plot.yAxis.selectAll('*').interrupt('growth');
      plot.grid.selectAll('*').interrupt('growth');
    }
    paint(plot.yAxis).call(d3.axisLeft(plot.y).ticks(4).tickSizeOuter(0));
    paint(plot.grid).call(d3.axisLeft(plot.y).ticks(4).tickSize(plot.x.range()[0] - plot.x.range()[1]).tickFormat(() => ''));
  }
  const path = y => d3.line().x(d => plot.x(d.x)).y(d => y(d.y));
  paint(plot.prior).attr('d', path(plot.y)(prior));
  paint(plot.posterior).attr('d', hasData ? path(plot.y)(updated) : null);
  paint(plot.likelihood).attr('d', hasData ? path(plot.relative)(likelihood) : null);
}

function coinDensity(a, b) {
  const low = a < 1 ? 0.01 : 0;
  const high = b < 1 ? 0.99 : 1;
  return points(coinGrid.map(x => low + (high - low) * x), betaDensity(a, b));
}

function coinLikelihood(heads, tails) {
  const total = heads + tails;
  if (!total) return [];
  const peak = heads / total;
  const xs = [...coinGrid, peak].sort((a, b) => a - b);
  return points(xs, relativeLikelihood(heads, tails));
}

function renderCustom(heads, tails, likelihood = coinLikelihood(heads, tails), animate = false) {
  const a = value('custom-alpha');
  const b = value('custom-beta');
  const posterior = betaUpdate(a, b, heads, tails);
  write('custom-alpha-value', a);
  write('custom-beta-value', b);
  drawUpdate('custom-plot', 'Your prior', {
    prior: coinDensity(a, b), updated: likelihood.length ? coinDensity(posterior.a, posterior.b) : [], likelihood, animate
  });
  const boundaryNote = a < 1 || b < 1 ? ' A density that diverges at a boundary is drawn up to 0.01 from that boundary.' : '';
  const update = likelihood.length ? ` · Posterior ${betaMath(posterior.a, posterior.b)} · Posterior mean` : ' · Prior mean';
  writeMath('custom-summary', `Prior ${betaMath(a, b)}${update} ${percent(posterior.mean)}.${boundaryNote}`);
}

function renderCoin(animate = false) {
  const [heads, tails] = counts();
  const total = heads + tails;
  const distribution = total ? 'Posterior' : 'Prior';
  write('coin-bias-value', percent(value('coin-bias')));
  byId('play-growth').disabled = total >= growthLimit;
  for (const item of document.querySelectorAll('[data-observed-curve]')) item.hidden = !total;
  write('toss-count', countText(heads, tails) + (tosses.length > 40 ? ' · Last 40 shown' : ''));
  const recent = tosses.slice(-40).map((toss, i, history) => ({toss, step: total - history.length + i + 1}));
  d3.select(byId('toss-log')).selectAll('li').data(recent, d => `${d.step}-${d.toss}`).join(enter =>
    enter.append('li').text(d => d.toss).attr('data-tails', d => d.toss === 'T' ? '' : null)
      .attr('title', d => `Toss ${d.step}: ${d.toss === 'H' ? 'Heads' : 'Tails'}`)
      .attr('aria-label', function () { return this.title; })
  );

  const comparisons = [['weak', 1], ['strong', 20]].map(([name, shape]) => {
    const posterior = betaUpdate(shape, shape, heads, tails);
    return {name, posterior, prior: coinDensity(shape, shape), updated: total ? coinDensity(posterior.a, posterior.b) : []};
  });
  write('mobile-means', `${distribution} means · Weak ${percent(comparisons[0].posterior.mean)} · Strong ${percent(comparisons[1].posterior.mean)}`);
  const maximum = d3.max(comparisons, ({prior, updated}) => d3.max([...prior, ...updated], p => p.y));
  const likelihood = coinLikelihood(heads, tails);
  for (const {name, posterior, prior, updated} of comparisons) {
    drawUpdate(`${name}-plot`, `${name} prior`, {
      prior, updated, likelihood, densityLimit: maximum * 1.07, animate
    });
    writeMath(`${name}-summary`, `${betaMath(posterior.a, posterior.b)} · ${distribution} mean ${percent(posterior.mean)}`);
  }

  write('likelihood-summary', total
    ? `The same likelihood curve peaks at the observed proportion, ${percent(heads / total)}.`
    : 'Add or simulate tosses to reveal the likelihood and the updated posteriors.');

  renderCustom(heads, tails, likelihood, animate);
}

function renderObjective() {
  if (!byId('objective').open) return;
  const heads = value('objective-heads');
  const tails = value('objective-tails');
  const selected = byId('objective-prior').value;
  write('objective-heads-value', heads);
  write('objective-tails-value', tails);
  writeMath('objective-count', math(`n=${heads + tails}`));
  writeMath('prior-summary', Object.values(priors).map(([name, a, b]) => {
    const posterior = betaUpdate(a, b, heads, tails);
    const distribution = a ? betaMath(a, b) : 'improper';
    return `<tr><th scope="row">${name} · ${distribution}</th><td>${posterior ? betaMath(posterior.a, posterior.b) : 'Improper'}</td><td>${posterior ? percent(posterior.mean) : 'Undefined'}</td></tr>`;
  }).join(''));

  const [name, a, b] = priors[selected];
  const posterior = betaUpdate(a, b, heads, tails);
  const haldane = selected === 'haldane';
  byId('kernel-figure').hidden = !haldane;
  byId('objective-prior-key').hidden = haldane;
  byId('objective-posterior-key').hidden = !posterior;
  const title = haldane ? (posterior ? 'Haldane posterior density and likelihood' : 'Relative likelihood') : `${name} prior, likelihood and posterior`;
  write('objective-title', title);
  const priorData = haldane ? [] : betaPoints(a, b);
  const posteriorData = posterior ? betaPoints(posterior.a, posterior.b) : [];
  const hasDensity = priorData.length > 0 || posteriorData.length > 0;
  const limit = hasDensity ? 1.07 * d3.max([...priorData, ...posteriorData], d => d.y) : 1;
  const likelihood = heads + tails ? coinLikelihood(heads, tails) : points(coinGrid, () => 1);
  draw('objective-plot', `${title}. Relative likelihood is read on the right axis.`, probAxis, {
    label: hasDensity ? 'Density' : '', rightLabel: 'Relative likelihood', domain: [0, limit], grid: hasDensity
  }, [
    priorData.length ? line(priorData, priorColor, '6 4') : null,
    posterior ? line(posteriorData, posteriorColor) : null,
    hasDensity ? Plot.axisY({anchor: 'left', ticks: 4}) : null,
    ...likelihoodMarks(likelihood, limit)
  ], 245);
  const description = posterior
    ? `${betaMath(posterior.a, posterior.b)} has mean ${percent(posterior.mean)}.`
    : 'The posterior kernel has infinite integral. Haldane needs at least one head and one tail before a posterior density and mean exist.';
  writeMath('objective-note', description + (selected === 'jeffreys'
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
  writeMath('coordinate-summary', math(tex`P(\theta\le ${cutoff.toFixed(2)}) = P(\gamma\le ${transformed.toFixed(2)}) = ${percent(cutoff).slice(0, -1)}\%`) + '.');
  for (const {id, axis, fn, threshold} of [
    {id: 'probability-plot', axis: probAxis, fn: () => 1, threshold: cutoff},
    {id: 'log-odds-plot', axis: {domain: [-8, 8], label: 'Log-odds', symbol: tex`\gamma`}, fn: logOddsDensity, threshold: transformed}
  ]) {
    const [low, high] = axis.domain;
    draw(id, `Density on the ${axis.label} scale, ${percent(cutoff)} below the cutoff`, axis, densityAxis, [
      Plot.areaY(points(grid(low, threshold), fn), {x: 'x', y: 'y', fill: posteriorColor, fillOpacity: 0.2}),
      line(points(grid(low, high), fn), posteriorColor),
      reference(threshold, evidenceColor)
    ], 245);
  }
}

function simulateTosses(count, animate = false) {
  const bias = value('coin-bias');
  tosses.push(...Array.from({length: count}, () => random() < bias ? 'H' : 'T'));
  renderCoin(animate);
}

function updateSimulateButton() {
  const count = value('simulate-count');
  write('simulate-count-value', count);
  write('simulate', `Simulate ${count} ${count === 1 ? 'toss' : 'tosses'}`);
}

function animateGrowth(now) {
  growthFrame = requestAnimationFrame(animateGrowth);
  if (now - lastGrowth < growthFrameMs) return;
  lastGrowth = now;
  const increment = Math.max(1, Math.ceil(tosses.length / 13));
  simulateTosses(Math.min(growthLimit - tosses.length, increment), true);
  if (tosses.length === growthLimit) stopGrowth();
}

function stopGrowth() {
  if (growthFrame) {
    cancelAnimationFrame(growthFrame);
    growthFrame = 0;
  }
  write('play-growth', stepManually ? 'Step through sample growth' : 'Play sample growth to 500');
}

function clearTosses() {
  stopGrowth();
  tosses = [];
  renderCoin();
}

document.querySelectorAll('[data-toss]').forEach(button => button.addEventListener('click', () => {
  stopGrowth();
  tosses.push(button.dataset.toss);
  renderCoin();
}));
byId('coin-bias').addEventListener('input', () => {
  stopGrowth();
  write('coin-bias-value', percent(value('coin-bias')));
});
byId('simulate-count').addEventListener('input', updateSimulateButton);
byId('simulate').addEventListener('click', () => {
  stopGrowth();
  simulateTosses(value('simulate-count'));
});
byId('clear-tosses').addEventListener('click', clearTosses);
byId('reset-tosses').addEventListener('click', () => {
  random = d3.randomLcg(6206);
  byId('coin-bias').value = 0.5;
  byId('simulate-count').value = 10;
  updateSimulateButton();
  clearTosses();
});
for (const id of ['custom-alpha', 'custom-beta']) {
  byId(id).addEventListener('input', () => renderCustom(...counts()));
}
const smallSampleX = byId('small-sample-x');
const smallSampleN = byId('small-sample-n');
smallSampleX.addEventListener('input', () => {
  smallSampleN.value = Math.max(smallSampleN.valueAsNumber, smallSampleX.valueAsNumber);
  renderSmallSample();
});
smallSampleN.addEventListener('input', () => {
  smallSampleX.value = Math.min(smallSampleX.valueAsNumber, smallSampleN.valueAsNumber);
  renderSmallSample();
});
byId('small-sample-extreme').addEventListener('click', () => {
  smallSampleX.value = smallSampleX.max;
  smallSampleN.value = smallSampleN.max;
  renderSmallSample();
});
for (const id of ['objective-heads', 'objective-tails']) byId(id).addEventListener('input', renderObjective);
byId('objective').addEventListener('toggle', renderObjective);
byId('objective-prior').addEventListener('change', renderObjective);
byId('coordinates').addEventListener('toggle', renderCoordinates);
byId('cutoff').addEventListener('input', renderCoordinates);
byId('play-growth').addEventListener('click', () => {
  if (growthFrame) {
    stopGrowth();
    return;
  }
  if (stepManually) {
    simulateTosses(growthSizes.find(next => next > tosses.length) - tosses.length);
    return;
  }
  simulateTosses(1);
  if (tosses.length < growthLimit) {
    write('play-growth', 'Pause sample growth');
    lastGrowth = performance.now();
    growthFrame = requestAnimationFrame(animateGrowth);
  }
});
window.addEventListener('resize', () => {
  coinPlots.clear();
  renderSmallSample();
  renderCoin();
  renderObjective();
  renderCoordinates();
});
stopGrowth();
renderSmallSample();
renderCoin();
