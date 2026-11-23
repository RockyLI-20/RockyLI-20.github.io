import {normalDensity} from './bayes-math.mjs?v=608a0d229a30';
import {acceptanceTrend, autocorrelation, effectiveSamples, mhMove, targets, tracePoints} from './mcmc-math.mjs?v=608a0d229a30';
import {math, tex} from './math.mjs?v=608a0d229a30';

const {Plot, d3} = window;
const byId = id => document.getElementById(id);
const colors = ['#245d5b', '#a34739', '#65517b', '#30678a'];
const dashes = [null, '5 3', '2 3', '7 3 2 3'];
const purple = '#65517b';
const recentStates = 1000;
const liveStepMs = 12;
const liveLimit = 2000;
const targetRefreshMs = 135;
const diagnosticRefreshMs = 280;
const traceFrame = {marginLeft: 48, marginRight: 12, marginTop: 24, marginBottom: 42};
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
  const liveButton = byId('mh-live');
  const liveStatus = byId('mh-live-status');
  liveStatus.textContent = `Live limit reached at ${liveLimit.toLocaleString('en-US')} steps per chain. Reset chains to run live again.`;
  const groups = [
    {key: 'single', label: 'Single setting', sd: 0.4},
    {key: 'a', label: 'Setting A', sd: 0.2},
    {key: 'b', label: 'Setting B', sd: 2}
  ];
  const activeGroups = () => comparison.checked ? groups.slice(1) : groups.slice(0, 1);
  let liveRequest = 0;
  let lastStep;
  let lastTarget;
  let lastDiagnostic;
  let target;
  let heatmap;
  let marginals;

  function pause() {
    cancelAnimationFrame(liveRequest);
    liveRequest = 0;
    liveButton.textContent = 'Run live';
  }

  function renderActive(update = 'all') {
    if (!lab.open) return;
    const groups = activeGroups();
    liveButton.disabled = groups.every(group => group.step >= liveLimit);
    liveStatus.hidden = !liveButton.disabled;
    for (const group of groups) render(group, update);
  }

  function refreshPanels() {
    for (const group of groups) group.card.hidden = !activeGroups().includes(group);
    root.classList.toggle('mh-comparing', comparison.checked);
    renderActive();
  }

  function reset(group) {
    group.step = 0;
    group.trace = null;
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
    if (joint) {
      const size = 32;
      const [xs, ys] = target.domains.map(domain => grid(domain, size));
      const tiles = xs.slice(1).flatMap((x2, i) => ys.slice(1).map((y2, j) => {
        const p = Math.exp(target.logDensity([(xs[i] + x2) / 2, (ys[j] + y2) / 2]));
        return {i, j, p};
      }));
      const peak = d3.max(tiles, d => d.p);
      const shade = d3.interpolateRgb('#faf9f5', '#668879');
      const canvas = document.createElement('canvas');
      canvas.width = canvas.height = size;
      const context = canvas.getContext('2d');
      for (const {i, j, p} of tiles) {
        context.fillStyle = shade(Math.sqrt(p / peak));
        context.fillRect(i, size - 1 - j, 1, 1);
      }
      heatmap = canvas.toDataURL();
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

  function advance(steps, update, limit = Infinity) {
    for (const group of activeGroups()) {
      const count = Math.min(steps, limit - group.step);
      for (let i = 0; i < count; i += 1) {
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
    renderActive(update);
  }

  function advanceLive(steps, update = 'all') {
    advance(steps, update, liveLimit);
    if (liveButton.disabled) {
      pause();
      if (update !== 'all') renderActive();
    }
  }

  function runLive(now) {
    liveRequest = requestAnimationFrame(runLive);
    const elapsed = now - lastStep;
    const steps = Math.min(5, Math.floor(elapsed / liveStepMs));
    if (steps < 1) return;
    lastStep = now - elapsed % liveStepMs;
    let update = 'trace';
    if (now - lastDiagnostic >= diagnosticRefreshMs) {
      lastDiagnostic = lastTarget = now;
      update = 'all';
    } else if (now - lastTarget >= targetRefreshMs) {
      lastTarget = now;
      update = 'target';
    }
    advanceLive(steps, update);
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
    renderActive();
  }

  function targetMarks(group, states, focus, width, height) {
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
          Plot.image([[d3.mean(xDomain), d3.mean(yDomain)]], {
            x: d => d[0], y: d => d[1], src: () => heatmap,
            width: width - 60, height: height - 66, preserveAspectRatio: 'none', imageRendering: 'pixelated'
          }),
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

  function renderTrace(group, axis, stateLabel, states) {
    const {parts, chains, step} = group;
    const width = parts.trace.clientWidth;
    const domain = target.domains[axis];
    if (!group.trace || group.trace.width !== width || group.trace.domain !== domain) {
      const height = 205;
      const {marginLeft: left, marginRight, marginTop: top, marginBottom} = traceFrame;
      const right = width - marginRight;
      const bottom = height - marginBottom;
      const svg = draw(parts.trace.id, 'Chain traces from the start of the run', {
        width, ...traceFrame, x: {axis: null}, y: {domain, label: stateLabel, zero: false}, marks: []
      }, height);
      const chart = d3.select(svg);
      const clip = `mh-trace-clip-${group.key}`;
      chart.append('defs').append('clipPath').attr('id', clip)
        .append('rect').attr('x', left).attr('y', top).attr('width', right - left).attr('height', bottom - top);
      const xAxis = chart.append('g').attr('transform', `translate(0,${bottom})`);
      chart.append('text').attr('x', right).attr('y', height - 9).attr('text-anchor', 'end')
        .attr('fill', 'currentColor').text('Iteration →');
      const marks = chart.append('g').attr('clip-path', `url(#${clip})`);
      const paths = chains.map((_, i) => marks.append('path').attr('fill', 'none')
        .attr('stroke', colors[i]).attr('stroke-width', 1.8).attr('stroke-dasharray', dashes[i]));
      const dots = states.map(({color}) => marks.append('circle').attr('fill', color).attr('r', 3));
      const x = d3.scaleLinear([0, 1], [left, right]);
      const y = svg.scale('y').apply;
      group.trace = {width, domain, x, y, xAxis, paths, dots, buckets: Math.max(1, Math.floor((right - left) / 2))};
    }
    const trace = group.trace;
    trace.x.domain([0, Math.max(1, step)]);
    trace.xAxis.call(d3.axisBottom(trace.x).ticks(width < 370 ? 4 : 7).tickFormat(integerTick).tickSizeOuter(0));
    const path = d3.line().x(d => trace.x(d.step)).y(d => trace.y(d.state[axis]));
    chains.forEach((chain, i) => {
      trace.paths[i].attr('d', path(tracePoints(chain.draws, axis, trace.buckets)));
      trace.dots[i].attr('cx', trace.x(step)).attr('cy', trace.y(states[i].state[axis]));
    });
  }

  function render(group, update = 'all') {
    const {parts, chains, step} = group;
    const axis = Number(coordinate.value);
    const joint = target.domains.length === 2;
    const stateLabel = joint ? (axis ? 'Y' : 'X') : 'State';
    const states = chains.map((chain, i) => ({state: chain.draws.at(-1).state, color: colors[i]}));
    renderTrace(group, axis, stateLabel, states);
    if (update === 'trace') return;

    const focus = Number(parts['start-chain'].value);
    const chart = (name, label, options, height = 205) => draw(parts[name].id, label, options, height);
    const move = chains[focus].last;
    parts['target-key'].textContent = joint
      ? 'Joint density in green. Purple shading marks the 1 and 2 SD proposal contours for the selected chain. Click to set its start.'
      : 'Gray line is the target. Purple shading is the next proposal density for the selected chain. Click to set its start.';
    parts.decision.innerHTML = move
      ? `Chain ${focus + 1} · ${math(tex`(${stateText(move.current)})\to(${stateText(move.proposal)})`)} · ${move.accepted ? 'Accepted' : 'Rejected'} (${percent(move.chance)})`
      : 'Step once to see the first proposal.';
    const targetWidth = joint ? Math.min(parts.target.clientWidth, 420) : parts.target.clientWidth;
    let targetHeight = 224;
    if (joint) {
      const [xSpan, ySpan] = target.domains.map(([lo, hi]) => hi - lo);
      targetHeight = 66 + (targetWidth - 60) * ySpan / xSpan;
    }
    group.targetPlot = chart('target', 'Target and proposal, with current states and last proposed point', {
      ...targetMarks(group, states, focus, targetWidth, targetHeight), width: targetWidth
    }, targetHeight);
    if (update === 'target') return;

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

    parts['sd-value'].textContent = group.sd.toFixed(1);
    parts['ess-title'].textContent = joint ? `ESS (${axis ? 'Y' : 'X'})` : 'ESS';
    parts.metrics.innerHTML = chains.map((chain, i) => {
      const ess = step >= 20 ? Math.round(effectiveSamples(samples[i].length, acfs[i])) : '—';
      const warn = step >= 20 && recentRates[i] < 0.2 ? ' class="mh-low"' : '';
      return `<tr><th scope="row"><span class="mh-swatch" style="--chain-color:${colors[i]};--chain-style:${i === 0 ? 'solid' : i === 2 ? 'dotted' : 'dashed'}"></span>${i + 1}</th><td>${step ? percent(rates[i]) : '—'}</td><td${warn}>${step ? percent(recentRates[i]) : '—'}</td><td>${ess}</td></tr>`;
    }).join('');

    const rejection = step >= 20 && recentRates.some(rate => rate < 0.2);
    const slow = step >= 20 && acfs.some(acf => acf[1] >= 0.8);
    const widerProposal = !rejection && slow && acfs.some((acf, i) => acf[1] >= 0.8 && recentRates[i] > 0.5);
    const stuck = distribution.value === 'mixture' && step >= 50 && chains.some(chain => chain.sides !== 3);
    parts.hint.dataset.state = rejection ? 'rejection' : slow || stuck ? 'slow' : 'ready';
    parts.hint.textContent = `${step.toLocaleString('en')} ${step === 1 ? 'step' : 'steps'} per chain. ${[
      step < 20 ? 'Run at least 20 steps for the first ESS and mixing hints.' : '',
      rejection ? 'Frequent recent rejections (acceptance below 20%). Try a smaller proposal SD.' : '',
      slow ? 'High lag-1 autocorrelation (0.8 or above).' : '',
      widerProposal ? 'A highly correlated chain accepts most recent proposals. Try a larger proposal SD.' : '',
      stuck ? 'At least one chain has visited only one side of the two-peak target.' : '',
      step >= 20 && !rejection && !slow && !stuck ? 'Compare the ACF decay and how much the traces overlap.' : ''
    ].filter(Boolean).join(' ')}`;

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
  coordinate.addEventListener('change', () => renderActive());
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
  liveButton.addEventListener('click', () => {
    if (liveRequest) {
      pause();
      renderActive();
      return;
    }
    liveButton.textContent = 'Pause';
    advanceLive(1);
    if (liveButton.disabled) return;
    lastStep = lastTarget = lastDiagnostic = performance.now();
    liveRequest = requestAnimationFrame(runLive);
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
  window.addEventListener('resize', () => renderActive());
  setTarget();
}
