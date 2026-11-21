import {normalDensity} from './bayes-math.mjs?v=704fc235ecc0';
import {acceptanceTrend, autocorrelation, effectiveSamples, mhMove, targets} from './mcmc-math.mjs?v=704fc235ecc0';

const {Plot, d3} = window;
const byId = id => document.getElementById(id);
const colors = ['#245d5b', '#a34739', '#65517b', '#30678a'];
const dashes = [null, '5 3', '2 3', '7 3 2 3'];
const purple = '#65517b';
const recentStates = 1000;
const kernel = normalDensity(0, 0.3);
const percent = x => `${(100 * x).toFixed(1)}%`;
const grid = ([lo, hi], n = 110) => d3.range(n + 1).map(i => lo + (hi - lo) * i / n);
const integerTick = x => Number.isInteger(x) ? x : '';
const stateText = state => state.map(x => x.toFixed(2)).join(', ');

export function setupMcmcLab(draw) {
  const lab = byId('mh-lab');
  const comparison = byId('mh-compare');
  const distribution = byId('mh-distribution');
  const coordinate = byId('mh-coordinate');
  const root = byId('mh-settings');
  const template = byId('mh-setting-template');
  const groups = [
    {key: 'single', label: 'Single setting', sd: 0.4},
    {key: 'a', label: 'Setting A', sd: 0.2},
    {key: 'b', label: 'Setting B', sd: 2}
  ];
  const activeGroups = () => comparison.checked ? groups.slice(1) : groups.slice(0, 1);
  let timer = null;
  let target;
  let heatmap;
  let marginals;

  function pause() {
    clearInterval(timer);
    timer = null;
    byId('mh-live').textContent = 'Run live';
  }

  function renderActive() {
    if (!lab.open) return;
    for (const group of activeGroups()) render(group);
  }

  function refreshPanels() {
    for (const group of groups) group.card.hidden = !activeGroups().includes(group);
    root.classList.toggle('mh-comparing', comparison.checked);
    renderActive();
  }

  function reset(group) {
    group.step = 0;
    group.chains = group.starts.map((state, i) => {
      const random = d3.randomLcg(6209 + i);
      return {
        random, normal: d3.randomNormal.source(random)(), accepted: 0, sides: 0,
        draws: [{step: 0, state}], last: null
      };
    });
    updateStartFields(group);
  }

  function setTarget() {
    pause();
    target = targets[distribution.value];
    byId('mh-target-note').textContent = target.description;
    const joint = target.domains.length === 2;
    if (!joint) coordinate.value = '0';
    byId('mh-coordinate-label').hidden = !joint;
    marginals = target.marginals.map((density, i) => grid(target.domains[i]).map(x => ({x, y: density(x)})));
    heatmap = [];
    if (joint) {
      const [xs, ys] = target.domains.map(domain => grid(domain, 32));
      heatmap = xs.slice(1).flatMap((x2, i) => ys.slice(1).map((y2, j) => {
        const x1 = xs[i];
        const y1 = ys[j];
        const p = Math.exp(target.logDensity([(x1 + x2) / 2, (y1 + y2) / 2]));
        return {x1, x2, y1, y2, p};
      }));
      const peak = d3.max(heatmap, d => d.p);
      const shade = d3.interpolateRgb('#faf9f5', '#668879');
      for (const tile of heatmap) tile.color = shade(Math.sqrt(tile.p / peak));
    }
    for (const group of groups) {
      group.starts = target.starts.slice(0, Number(byId('mh-chain-count').value)).map(state => [...state]);
      const select = group.parts['start-chain'];
      select.replaceChildren(...group.starts.map((_, i) => new Option(`Chain ${i + 1}`, i)));
      group.parts['start-y-label'].hidden = !joint;
      target.domains.forEach(([min, max], i) => Object.assign(group.parts[i ? 'start-y' : 'start-x'], {min, max, step: 'any'}));
      reset(group);
    }
    refreshPanels();
  }

  function advance(steps) {
    for (const group of activeGroups()) {
      for (let i = 0; i < steps; i += 1) {
        group.step += 1;
        for (const chain of group.chains) {
          const move = mhMove(chain.draws.at(-1).state, group.sd, target, chain.normal, chain.random);
          chain.accepted += Number(move.accepted);
          chain.sides |= move.next[0] < 0 ? 1 : 2;
          chain.last = move;
          chain.draws.push({step: group.step, state: move.next, accepted: move.accepted});
        }
      }
    }
    renderActive();
  }

  function updateStartFields(group) {
    const [x, y] = group.starts[Number(group.parts['start-chain'].value)];
    group.parts['start-x'].value = x;
    group.parts['start-y'].value = y ?? 0;
  }

  function setStart(group, state) {
    pause();
    group.starts[Number(group.parts['start-chain'].value)] = state;
    reset(group);
    render(group);
  }

  function targetMarks(group, states, focus) {
    const joint = target.domains.length === 2;
    const move = group.chains[focus].last;
    const current = states[focus].state;
    const [xDomain, yDomain] = target.domains;
    const shade = {fill: purple, fillOpacity: 0.19, clip: true};
    if (joint) {
      const contour = radius => grid([current[0] - radius, current[0] + radius], 70).map(x => {
        const rise = Math.sqrt(Math.max(0, radius ** 2 - (x - current[0]) ** 2));
        return {x, lo: current[1] - rise, hi: current[1] + rise};
      });
      return {
        x: {domain: xDomain, label: 'X'}, y: {domain: yDomain, label: 'Y', zero: false, grid: false},
        marks: [
          Plot.rect(heatmap, {x1: 'x1', x2: 'x2', y1: 'y1', y2: 'y2', fill: 'color'}),
          Plot.areaY(contour(2 * group.sd), {x: 'x', y1: 'lo', y2: 'hi', ...shade}),
          Plot.areaY(contour(group.sd), {x: 'x', y1: 'lo', y2: 'hi', ...shade}),
          Plot.link(move ? [move] : [], {
            x1: d => d.current[0], y1: d => d.current[1], x2: d => d.proposal[0], y2: d => d.proposal[1],
            stroke: purple, strokeDasharray: '3 3', clip: true
          }),
          Plot.dot(states, {x: d => d.state[0], y: d => d.state[1], fill: 'color', r: 4.5, clip: true}),
          Plot.dot(move ? [move.proposal] : [], {x: d => d[0], y: d => d[1], stroke: purple, r: 7, clip: true})
        ]
      };
    }
    const density = target.marginals[0];
    const proposal = normalDensity(current[0], group.sd);
    const xs = [...grid(xDomain), current[0]].sort((a, b) => a - b);
    const peak = d3.max([
      d3.max(marginals[0], d => d.y),
      ...activeGroups().map(setting => normalDensity(0, setting.sd)(0))
    ]);
    return {
      x: {domain: xDomain, label: 'State'}, y: {domain: [0, 1.06 * peak], label: 'Density'},
      marks: [
        Plot.areaY(xs.map(x => ({x, y: proposal(x)})), {x: 'x', y: 'y', ...shade}),
        Plot.lineY(marginals[0], {x: 'x', y: 'y', stroke: '#434e46'}),
        Plot.ruleX(move ? [move.proposal[0]] : [], {stroke: purple, strokeDasharray: '3 3', clip: true}),
        Plot.dot(states, {x: d => d.state[0], y: d => density(d.state[0]), fill: 'color', r: 4.5, clip: true}),
        Plot.dot(move ? [move.proposal[0]] : [], {x: x => x, y: density, stroke: purple, r: 7, clip: true})
      ]
    };
  }

  function render(group) {
    const {parts, chains, step} = group;
    const axis = Number(coordinate.value);
    const joint = target.domains.length === 2;
    const focus = Number(parts['start-chain'].value);
    const samples = chains.map(chain => chain.draws.slice(Math.max(1, chain.draws.length - recentStates)));
    const acfs = samples.map(draws => autocorrelation(draws.map(draw => draw.state[axis])));
    const rates = chains.map(chain => step ? chain.accepted / step : 0);
    const recentRates = samples.map(draws => {
      const last = draws.slice(-100);
      return last.length ? last.reduce((sum, d) => sum + Number(d.accepted), 0) / last.length : 0;
    });
    const tagged = callback => chains.flatMap((chain, i) => callback(chain, i).map(point => ({...point, chain: i})));
    const line = (data, x, y, i) => Plot.lineY(data, {
      x, y, stroke: colors[i], strokeDasharray: dashes[i], strokeWidth: 1.8, clip: true
    });
    const lines = (data, x, y) => chains.map((_, i) => line(data.filter(d => d.chain === i), x, y, i));
    const chart = (name, label, options, height = 205) => draw(parts[name].id, label, options, height);

    parts['sd-value'].textContent = group.sd.toFixed(1);
    parts['ess-title'].textContent = joint ? `ESS (${axis ? 'Y' : 'X'})` : 'ESS';
    parts.metrics.innerHTML = chains.map((chain, i) => {
      const ess = step >= 20 ? Math.round(effectiveSamples(samples[i].length, acfs[i])) : '—';
      const warn = step >= 20 && recentRates[i] < 0.2 ? ' class="mh-low"' : '';
      return `<tr><th scope="row"><span class="mh-swatch" style="--chain-color:${colors[i]};--chain-style:${i === 0 ? 'solid' : i === 2 ? 'dotted' : 'dashed'}"></span>${i + 1}</th><td>${step ? percent(rates[i]) : '—'}</td><td${warn}>${step ? percent(recentRates[i]) : '—'}</td><td>${ess}</td></tr>`;
    }).join('');

    const rejection = step >= 20 && recentRates.some(rate => rate < 0.2);
    const slow = step >= 20 && acfs.some(acf => acf[1] >= 0.8);
    const stuck = distribution.value === 'mixture' && step >= 50 && chains.some(chain => chain.sides !== 3);
    parts.hint.dataset.state = rejection ? 'rejection' : slow || stuck ? 'slow' : 'ready';
    parts.hint.textContent = `${step.toLocaleString('en')} steps per chain. ${[
      step < 20 ? 'Run at least 20 steps for the first ESS and mixing hints.' : '',
      rejection ? 'Frequent recent rejections (acceptance below 20%). Try a smaller proposal SD.' : '',
      slow ? 'High lag-1 autocorrelation (0.8 or above). If most moves are accepted, try a larger proposal SD.' : '',
      stuck ? 'At least one chain has visited only one side of the two-peak target.' : '',
      step >= 20 && !rejection && !slow && !stuck ? 'Compare the ACF decay and how much the traces overlap.' : ''
    ].filter(Boolean).join(' ')}`;

    const states = chains.map((chain, i) => ({state: chain.draws.at(-1).state, color: colors[i]}));
    const move = chains[focus].last;
    parts['target-key'].textContent = joint
      ? 'Joint density in green. Purple shading marks the 1 and 2 SD proposal contours for the selected chain. Click to set its start.'
      : 'Gray line is the target. Purple shading is the next proposal density for the selected chain. Click to set its start.';
    parts.decision.textContent = move
      ? `Chain ${focus + 1} · ${stateText(move.current)} → ${stateText(move.proposal)} · ${move.accepted ? 'Accepted' : 'Rejected'} (${percent(move.chance)})`
      : 'Step once to see the first proposal.';
    const targetChart = targetMarks(group, states, focus);
    let targetHeight = 224;
    if (joint) {
      const width = Math.min(parts.target.clientWidth, 420);
      const [xSpan, ySpan] = target.domains.map(([lo, hi]) => hi - lo);
      targetChart.width = width;
      targetHeight = 66 + (width - 60) * ySpan / xSpan;
    }
    group.targetPlot = chart('target', 'Target and proposal, with current states and last proposed point', targetChart, targetHeight);

    const stateLabel = joint ? (axis ? 'Y' : 'X') : 'State';
    chart('trace', 'Chain traces from the start of the run', {
      x: {domain: [0, Math.max(1, step)], tickFormat: integerTick, label: 'Iteration'},
      y: {domain: target.domains[axis], label: stateLabel, zero: false},
      marks: [...chains.map((chain, i) => line(chain.draws, 'step', d => d.state[axis], i)), Plot.dot(states, {
        x: () => step, y: d => d.state[axis], fill: 'color', r: 3, clip: true
      })]
    });

    const acfPoints = tagged((_, i) => acfs[i].map((y, lag) => ({lag, y})));
    chart('acf', `Autocorrelation by chain for ${stateLabel}`, {
      x: {domain: [0, 40], tickFormat: integerTick, label: 'Lag'},
      y: {domain: [-1, 1], label: 'Correlation'},
      marks: [Plot.ruleY([0], {stroke: '#aaa'}), ...lines(acfPoints, 'lag', 'y')]
    });

    const marginal = marginals[axis];
    const estimated = tagged((_, i) => {
      const recent = samples[i].slice(-200);
      return recent.length ? marginal.map(({x}) => ({
        x, y: recent.reduce((sum, draw) => sum + kernel(x - draw.state[axis]), 0) / recent.length
      })) : [];
    });
    chart('density', `Estimated density and reference target for ${stateLabel}`, {
      x: {domain: target.domains[axis], label: stateLabel},
      y: {domain: [0, 1.4], label: 'Density'},
      marks: [Plot.lineY(marginal, {x: 'x', y: 'y', stroke: '#646e68'}), ...lines(estimated, 'x', 'y')]
    });

    const trend = tagged((_, i) => acceptanceTrend(samples[i]));
    chart('acceptance', 'Rolling acceptance rates for the latest 100 steps', {
      x: {domain: [Math.max(0, step - 99), Math.max(1, step)], tickFormat: integerTick, label: 'Iteration'},
      y: {domain: [0, 1], label: 'Acceptance', tickFormat: d3.format('.0%')},
      marks: [Plot.ruleY([0.2], {stroke: '#a34739', strokeOpacity: 0.5, strokeDasharray: '3 3'}), ...lines(trend, 'step', 'rate')]
    });
  }

  for (const group of groups) {
    const card = template.content.firstElementChild.cloneNode(true);
    const parts = Object.fromEntries(Array.from(card.querySelectorAll('[data-mh]'), el => [el.dataset.mh, el]));
    Object.assign(group, {card, parts});
    for (const [name, element] of Object.entries(parts)) element.id = `mh-${group.key}-${name}`;
    parts.title.textContent = group.label;
    parts.sd.value = group.sd;
    parts.sd.addEventListener('input', () => {
      group.sd = parts.sd.valueAsNumber;
      renderActive();
    });
    parts['start-chain'].addEventListener('change', () => {
      updateStartFields(group);
      render(group);
    });
    parts['set-start'].addEventListener('click', () => {
      const inputs = [parts['start-x'], parts['start-y']].slice(0, target.domains.length);
      if (inputs.every(input => input.reportValidity())) setStart(group, inputs.map(input => input.valueAsNumber));
    });
    parts.target.addEventListener('click', event => {
      const [x, y] = d3.pointer(event, group.targetPlot);
      const pixels = [x, y].slice(0, target.domains.length);
      const axes = ['x', 'y'].slice(0, target.domains.length).map(axis => group.targetPlot.scale(axis));
      const within = axes.every(({range}, i) => pixels[i] >= Math.min(...range) && pixels[i] <= Math.max(...range));
      if (within) setStart(group, axes.map((scale, i) => Number(scale.invert(pixels[i]).toFixed(2))));
    });
    root.append(card);
  }

  distribution.addEventListener('change', setTarget);
  byId('mh-chain-count').addEventListener('change', setTarget);
  coordinate.addEventListener('change', renderActive);
  comparison.addEventListener('change', () => {
    pause();
    refreshPanels();
  });
  byId('mh-step').addEventListener('click', () => {
    pause();
    advance(1);
  });
  byId('mh-run').addEventListener('click', () => {
    pause();
    advance(200);
  });
  byId('mh-live').addEventListener('click', () => {
    if (timer) {
      pause();
      return;
    }
    byId('mh-live').textContent = 'Pause';
    advance(1);
    timer = setInterval(() => advance(5), 180);
  });
  byId('mh-reset').addEventListener('click', () => {
    pause();
    for (const group of activeGroups()) reset(group);
    renderActive();
  });
  lab.addEventListener('toggle', () => {
    if (!lab.open) pause();
    renderActive();
  });
  window.addEventListener('resize', renderActive);
  setTarget();
}
