import {
  betaDensity, betaUpdate, coinExample, countHeads, logOdds, logOddsDensity, relativeLikelihood
} from './bayes-math.mjs?v=6325cd8a0930';

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
  const {label: leftLabel, ...scale} = y;
  const axisLabel = document.createElement('div');
  axisLabel.className = 'bayes-axis-label';
  axisLabel.textContent = leftLabel;
  const svg = Plot.plot({
    width, height: height - 14, marginLeft: 45, marginRight: 12, marginTop: 9, marginBottom: 41,
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
const countText = (heads, tails) => {
  const total = heads + tails;
  return `${heads} ${heads === 1 ? 'head' : 'heads'} · ${tails} ${tails === 1 ? 'tail' : 'tails'} · ${total} ${total === 1 ? 'toss' : 'tosses'}`;
};
const betaText = ({a, b}) => `Beta(${a}, ${b})`;

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
  svg.append('text').attr('x', right).attr('y', height - 4).attr('text-anchor', 'end')
    .attr('fill', 'currentColor').attr('font-size', 12).text('Chance of heads, θ →');
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
  const update = likelihood.length ? ` · Posterior ${betaText(posterior)} · Posterior mean` : ' · Prior mean';
  write('custom-summary', `Prior ${betaText({a, b})}${update} ${percent(posterior.mean)}.${boundaryNote}`);
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
    write(`${name}-summary`, `${betaText(posterior)} · ${distribution} mean ${percent(posterior.mean)}`);
  }

  write('likelihood-summary', total
    ? `The same likelihood curve peaks at the observed proportion, ${percent(heads / total)}.`
    : 'Add or simulate tosses to reveal the likelihood and the updated posteriors.');

  renderCustom(heads, tails, likelihood, animate);
  if (!animate) renderObjective();
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
    renderObjective();
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
document.querySelectorAll('[data-example]').forEach(button => button.addEventListener('click', () => {
  stopGrowth();
  tosses = [...examples[button.dataset.example]];
  renderCoin();
}));
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
  renderCoin();
  renderCoordinates();
});
stopGrowth();
renderCoin();
