import {bananaTarget} from './gibbs-math.mjs?v=608a0d229a30';
import {axisMath, math} from './math.mjs?v=608a0d229a30';
import {autocorrelation, effectiveSamples, mhMove, tracePoints} from './mcmc-math.mjs?v=608a0d229a30';

const {d3, Plot} = window;
const byId = id => document.getElementById(id);
const budget = 2000;
const batchSize = 200;
const stepMs = 8;
const traceRefreshMs = 30;
const targetRefreshMs = 120;
const diagnosticRefreshMs = 280;
const recentStates = 1000;
const colors = ['#245d5b', '#a34739'];
const dashes = [null, '5 3'];
const contourColor = '#77877d';
const formatRound = d3.format(',d');
const formatMean = d3.format('.2f');
const controls = [...document.querySelectorAll('input[name="sampler-target"]')];
const swatches = [...document.querySelectorAll('[data-sampler-chain]')];
const proposal = byId('sampler-proposal');
const sigma = byId('sampler-sigma');
const bend = byId('sampler-bend');
const live = byId('sampler-live');
const pause = byId('sampler-pause');
const batch = byId('sampler-batch');
const xLabel = math('x_1');
const yLabel = math('x_2');
const pairLabel = math('(x_1,x_2)');

let target;
let targetKey;
let contours;
let complexBend = bend.value;
let sd;
let samplers;
let round = 0;
let frame = 0;
let lastStep;
let lastTrace;
let lastTarget;
let lastDiagnostic;

function targetContours({domains: [[left, right], [bottom, top]], logDensity}) {
  const columns = 100;
  const rows = 64;
  const x = i => left + (right - left) * i / columns;
  const y = i => bottom + (top - bottom) * i / rows;
  const values = d3.range(rows).flatMap(row => d3.range(columns).map(col =>
    Math.exp(logDensity([x(col + 0.5), y(row + 0.5)]))
  ));
  const peak = d3.max(values);
  const rings = d3.contours().size([columns, rows]).thresholds([0.12, 0.32, 0.65].map(p => p * peak))(values)
    .flatMap(level => level.coordinates.flat());
  return rings.flatMap((ring, group) => ring.map(([col, row]) => ({x: x(col), y: y(row), group})));
}

function makeChains(seed) {
  return target.starts.slice(0, colors.length).map((state, index) => {
    const random = d3.randomLcg(seed + index);
    return {draws: [{state, step: 0}], random, normal: d3.randomNormal.source(random)(), accepted: 0, sums: [0, 0]};
  });
}

function stopLive() {
  cancelAnimationFrame(frame);
  frame = 0;
}

function reset() {
  stopLive();
  round = 0;
  const complex = controls.find(input => input.checked).value === 'complex';
  const sigmaValue = sigma.valueAsNumber;
  bend.disabled = !complex;
  bend.value = complex ? complexBend : 0;
  const bendValue = bend.valueAsNumber;
  const key = `${sigmaValue},${bendValue}`;
  if (key !== targetKey) {
    targetKey = key;
    target = bananaTarget(sigmaValue, bendValue);
    contours = targetContours(target);
  }
  sd = complex ? proposal.valueAsNumber : sigmaValue;
  byId('sampler-proposal-value').value = proposal.value;
  byId('sampler-sigma-value').value = sigmaValue.toFixed(1);
  byId('sampler-bend-value').value = bendValue.toFixed(2);
  byId('sampler-proposal-control').hidden = !complex;
  byId('sampler-target-note').hidden = !bendValue;
  samplers = [
    {id: 'gibbs', name: 'Gibbs', chains: makeChains(6312), move: (state, {normal, random}) =>
      target.gibbs(state, normal, random)},
    {id: 'mh', name: 'MH', chains: makeChains(6322), move: (state, chain) => {
      const result = mhMove(state, sd, target, chain.normal, chain.random);
      chain.accepted += Number(result.accepted);
      return result.next;
    }}
  ];
  render();
}

function advance(steps) {
  const remaining = Math.min(steps, budget - round);
  for (let i = 0; i < remaining; i += 1) {
    round += 1;
    for (const {chains, move} of samplers) {
      for (const chain of chains) {
        const state = move(chain.draws.at(-1).state, chain);
        chain.draws.push({state, step: round});
        chain.sums[0] += state[0];
        chain.sums[1] += state[1];
      }
    }
  }
}

function plotLayer(svg, id, left, right, top, bottom) {
  const chart = d3.select(svg);
  chart.append('defs').append('clipPath').attr('id', id).append('rect')
    .attr('x', left).attr('y', top).attr('width', right - left).attr('height', bottom - top);
  return chart.append('g').attr('clip-path', `url(#${id})`);
}

function makeDensity(sampler, host, width) {
  const [xDomain, yDomain] = target.domains;
  const plotWidth = width - 51;
  const height = Math.min(350, 63 + Math.round(plotWidth * (yDomain[1] - yDomain[0]) / (xDomain[1] - xDomain[0])));
  const svg = Plot.plot({
    width, height, marginLeft: 39, marginRight: 12, marginTop: 22, marginBottom: 41,
    ariaLabel: `${sampler.name} chain draws over the true density contours for x1 and x2`,
    style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
    x: {domain: xDomain, label: null, ticks: 5},
    y: {domain: yDomain, label: null, ticks: 5, grid: true},
    marks: [Plot.line(contours, {x: 'x', y: 'y', z: 'group', stroke: contourColor, strokeWidth: 1.15})]
  });
  const canvas = document.createElement('canvas');
  const pixelRatio = Math.min(window.devicePixelRatio, 2);
  canvas.width = Math.round(width * pixelRatio);
  canvas.height = Math.round(height * pixelRatio);
  canvas.setAttribute('aria-hidden', 'true');
  const context = canvas.getContext('2d');
  context.setTransform(canvas.width / width, 0, 0, canvas.height / height, 0, 0);
  context.beginPath();
  context.rect(39, 22, plotWidth, height - 63);
  context.clip();
  context.globalAlpha = 0.32;

  const layer = plotLayer(svg, `sampler-${sampler.id}-cloud-clip`, 39, width - 12, 22, height - 41);
  const current = sampler.chains.map((_, i) =>
    layer.append('circle').attr('fill', colors[i]).attr('r', 5).attr('stroke', '#fffefa').attr('stroke-width', 1.3)
  );
  axisMath(svg, '↑ ' + yLabel, {x: 4, width: width - 8});
  axisMath(svg, xLabel + ' →', {x: 4, y: height - 22, width: width - 20, align: 'right'});
  const stage = document.createElement('div');
  stage.className = 'sampler-density-stage';
  stage.style.width = `${width}px`;
  stage.append(canvas, svg);
  host.replaceChildren(stage);
  return {width, x: svg.scale('x').apply, y: svg.scale('y').apply, context, current, painted: -1};
}

function densityPlot(sampler) {
  const host = byId(`sampler-${sampler.id}-density`);
  const width = Math.min(host.clientWidth, 530);
  if (!sampler.density || sampler.density.width !== width) {
    sampler.density = makeDensity(sampler, host, width);
  }
  const density = sampler.density;
  if (density.painted === round) return;
  const {x, y, context, current} = density;
  for (let step = density.painted + 1; step <= round; step += 1) {
    sampler.chains.forEach(({draws}, i) => {
      const [x1, x2] = draws[step].state;
      context.fillStyle = colors[i];
      context.beginPath();
      context.arc(x(x1), y(x2), 2.4, 0, 2 * Math.PI);
      context.fill();
    });
  }
  density.painted = round;
  sampler.chains.forEach(({draws}, i) => {
    const [x1, x2] = draws.at(-1).state;
    current[i].attr('cx', x(x1)).attr('cy', y(x2));
  });
}

function makeTrace(sampler, host, width, coordinate) {
  const bottom = 121;
  const right = width - 12;
  const svg = Plot.plot({
    width, height: 160, marginLeft: 39, marginRight: 12, marginTop: 8, marginBottom: 39,
    ariaLabel: `${sampler.name} traces for x${coordinate + 1} from the start of each chain`,
    style: {background: 'transparent', fontFamily: 'inherit', fontSize: 12},
    x: {axis: null},
    y: {domain: target.domains[coordinate], label: null, grid: true, ticks: 5},
    marks: []
  });
  const chart = d3.select(svg);
  const axis = chart.append('g').attr('transform', `translate(0,${bottom})`);
  chart.append('text').attr('x', right).attr('y', 153).attr('text-anchor', 'end')
    .attr('fill', 'currentColor').text('Round →');
  const layer = plotLayer(svg, `${host.id}-clip`, 39, right, 8, bottom);
  const paths = sampler.chains.map((_, i) => layer.append('path').attr('fill', 'none')
    .attr('stroke', colors[i]).attr('stroke-dasharray', dashes[i]).attr('stroke-opacity', 0.75));
  const current = sampler.chains.map((_, i) => layer.append('circle').attr('fill', colors[i]).attr('r', 2.5));
  host.replaceChildren(svg);
  return {width, axis, paths, current, x: d3.scaleLinear([0, 1], [39, right]), y: svg.scale('y').apply};
}

function tracePlot(sampler, coordinate) {
  const host = byId(`sampler-${sampler.id}-trace-${coordinate + 1}`);
  const width = host.clientWidth;
  const traces = sampler.traces ??= [];
  if (traces[coordinate]?.width !== width) traces[coordinate] = makeTrace(sampler, host, width, coordinate);
  const {x, y, axis, paths, current} = traces[coordinate];
  x.domain([0, Math.max(1, round)]);
  axis.call(d3.axisBottom(x).ticks(5).tickFormat(n => Number.isInteger(n) ? formatRound(n) : '').tickSizeOuter(0));
  const buckets = Math.max(1, Math.floor((width - 51) / 2));
  const line = d3.line().x(d => x(d.step)).y(d => y(d.state[coordinate]));
  sampler.chains.forEach(({draws}, i) => {
    paths[i].attr('d', line(tracePoints(draws, coordinate, buckets)));
    current[i].attr('cx', x(round)).attr('cy', y(draws.at(-1).state[coordinate]));
  });
}

function render(update = 'all') {
  for (const sampler of samplers) {
    tracePlot(sampler, 0);
    tracePlot(sampler, 1);
    if (update === 'all') densityPlot(sampler);
  }
  if (update === 'trace') return;
  const complete = round >= budget;
  live.disabled = Boolean(frame) || complete;
  pause.disabled = !frame;
  batch.disabled = Boolean(frame) || complete;
  byId('sampler-rounds').value = `${formatRound(round)} / ${formatRound(budget)} rounds per chain${complete ? ' · Budget reached' : ''}`;
  if (update !== 'all') return;
  const [gibbs, mh] = samplers.map(sampler => {
    const {id, chains} = sampler;
    const axes = [0, 1];
    const mean = round ? axes.map(i => formatMean(d3.sum(chains, c => c.sums[i]) / (round * chains.length))) : null;
    const ess = round < 20 ? null : axes.map(i => Math.round(d3.sum(chains, ({draws}) => {
      const values = draws.slice(Math.max(1, draws.length - recentStates)).map(d => d.state[i]);
      return effectiveSamples(values.length, autocorrelation(values));
    })));
    byId(`sampler-${id}-metrics`).innerHTML =
      `Mean ${pairLabel} ${mean ? '(' + mean.join(', ') + ')' : '—'} · Effective Sample Size ${ess ? ess.join(' / ') : '—'}`;
    return ess;
  });
  const chains = samplers[1].chains;
  const acceptance = round ? d3.sum(chains, c => c.accepted) / (round * chains.length) : 0;
  const rate = round ? `${d3.format('.0%')(acceptance)} accepted. ` : '';
  const hint = byId('sampler-mh-hint');
  hint.dataset.state = round >= 20 && acceptance < 0.2 ? 'rejection' : sd / sigma.valueAsNumber <= 1 / 6 ? 'slow' : 'ready';
  hint.textContent = rate + (hint.dataset.state === 'rejection'
    ? 'High rejection rate. The chain often repeats its current state.'
    : hint.dataset.state === 'slow'
      ? 'Small moves. Watch the chain crawl even when proposals are accepted.'
      : `Normal random walk with SD ${d3.format('.2~f')(sd)}.`);
  const summary = byId('sampler-conclusion');
  const compare = (i, label) => gibbs[i] === mh[i] ? `${label} ESS is tied`
    : `${gibbs[i] > mh[i] ? 'Gibbs' : 'MH'} has higher ${label} ESS`;
  summary.innerHTML = gibbs
    ? `After ${formatRound(round)} rounds per chain, ${compare(0, xLabel)}. ${compare(1, yLabel)}.`
    : 'Run at least 20 rounds to compare the effective sample sizes.';
}

function runLive(now) {
  frame = requestAnimationFrame(runLive);
  const elapsed = now - lastStep;
  const steps = Math.min(8, Math.floor(elapsed / stepMs));
  if (steps > 0) {
    advance(steps);
    samplers.forEach(densityPlot);
    lastStep = now - elapsed % stepMs;
  }
  if (round === budget) {
    stopLive();
    render();
    return;
  }
  if (now - lastDiagnostic >= diagnosticRefreshMs) {
    lastTrace = lastTarget = lastDiagnostic = now;
    render();
  } else if (now - lastTarget >= targetRefreshMs) {
    lastTrace = lastTarget = now;
    render('target');
  } else if (now - lastTrace >= traceRefreshMs) {
    lastTrace = now;
    render('trace');
  }
}

live.addEventListener('click', () => {
  lastStep = lastTrace = lastTarget = lastDiagnostic = performance.now();
  advance(1);
  frame = requestAnimationFrame(runLive);
  render();
});
pause.addEventListener('click', () => {
  stopLive();
  render();
});
batch.addEventListener('click', () => {
  advance(batchSize);
  render();
});
for (const control of controls) control.addEventListener('change', reset);
proposal.addEventListener('input', reset);
sigma.addEventListener('input', reset);
bend.addEventListener('input', () => {
  complexBend = bend.value;
  reset();
});
byId('sampler-reset').addEventListener('click', reset);
window.addEventListener('resize', () => render());
swatches.forEach((item, i) => item.style.setProperty('--chain-color', colors[i]));
reset();
