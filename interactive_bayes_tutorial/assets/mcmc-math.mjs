import {normalDensity} from './bayes-math.mjs?v=608a0d229a30';
import {mixtureDensity} from './inference-math.mjs?v=608a0d229a30';

const standard = normalDensity(0, 1);
const logNormal = (x, mean, sd) => -0.5 * ((x - mean) / sd) ** 2 - Math.log(sd * Math.sqrt(2 * Math.PI));
const bananaMean = x => 0.42 * (x ** 2 - 1);

function bananaMarginal(y) {
  let density = 0;
  for (let x = -5.975; x < 6; x += 0.05) {
    density += standard(x) * Math.exp(logNormal(y, bananaMean(x), 0.55)) * 0.05;
  }
  return density;
}

export const targets = {
  normal: {
    description: 'One peak. Compare how proposal width changes movement across the center.',
    domains: [[-4.5, 4.5]],
    starts: [[-2], [2], [-1], [1]],
    logDensity: ([x]) => logNormal(x, 0, 1),
    marginals: [standard]
  },
  mixture: {
    description: 'Two peaks. Check whether every chain can cross the low-density valley.',
    domains: [[-6, 6]],
    starts: [[-2.5], [2.5], [-3.2], [3.2]],
    logDensity: ([x]) => Math.log(mixtureDensity(x)),
    marginals: [mixtureDensity]
  },
  banana: {
    description: 'The high-density region bends. The same round proposal has to follow a curved ridge.',
    domains: [[-4, 4], [-3, 6]],
    starts: [[-1.5, 0.5], [1.5, 0.5], [-2, 1.3], [2, 1.3]],
    logDensity: ([x, y]) => logNormal(x, 0, 1) + logNormal(y, bananaMean(x), 0.55),
    marginals: [standard, bananaMarginal]
  },
  correlated: {
    description: 'The high-density region forms a narrow diagonal. Moves along the ridge fare differently from moves across it.',
    domains: [[-4, 4], [-4, 4]],
    starts: [[-2, -2], [2, 2], [-1, -1], [1, 1]],
    logDensity: ([x, y]) => logNormal(x, 0, 1) + logNormal(y, 0.95 * x, Math.sqrt(1 - 0.95 ** 2)),
    marginals: [standard, standard]
  }
};

export function mhMove(current, stepSize, target, normal, uniform) {
  const proposal = current.map(x => x + stepSize * normal());
  const chance = Math.exp(Math.min(0, target.logDensity(proposal) - target.logDensity(current)));
  const accepted = uniform() < chance;
  return {current, proposal, chance, accepted, next: accepted ? proposal : current};
}

export function autocorrelation(values, maxLag = 40) {
  if (values.length < 2) return [];
  const mean = values.reduce((sum, x) => sum + x, 0) / values.length;
  const centered = values.map(x => x - mean);
  const variance = centered.reduce((sum, x) => sum + x * x, 0);
  if (variance <= 1e-14) return [];
  const result = [1];
  for (let lag = 1; lag <= Math.min(maxLag, values.length - 1); lag += 1) {
    let covariance = 0;
    for (let i = lag; i < values.length; i += 1) covariance += centered[i] * centered[i - lag];
    result.push(covariance / variance);
  }
  return result;
}

export function effectiveSamples(size, acf) {
  if (!acf.length) return 0;
  let integrated = 0;
  for (let lag = 1; lag + 1 < acf.length; lag += 2) {
    const pair = acf[lag] + acf[lag + 1];
    if (pair <= 0) break;
    integrated += pair;
  }
  return size / (1 + 2 * integrated);
}

export function acceptanceTrend(draws, horizon = 100, window = 20) {
  return draws.slice(-horizon).map(({step}, i) => {
    const end = Math.max(0, draws.length - horizon) + i + 1;
    const part = draws.slice(Math.max(0, end - window), end);
    return {step, rate: part.reduce((sum, draw) => sum + Number(draw.accepted), 0) / part.length};
  });
}

export function tracePoints(draws, axis, buckets) {
  const last = draws.length - 1;
  const stride = Math.ceil((last - 1) / buckets);
  if (stride <= 2) return draws;
  const points = [draws[0]];
  // Keep each bucket's extremes in their original time order.
  for (let start = 1; start < last; start += stride) {
    let min = start;
    let max = start;
    for (let i = start + 1; i < Math.min(last, start + stride); i += 1) {
      const state = draws[i].state[axis];
      if (state < draws[min].state[axis]) min = i;
      if (state > draws[max].state[axis]) max = i;
    }
    points.push(draws[Math.min(min, max)]);
    if (min !== max) points.push(draws[Math.max(min, max)]);
  }
  points.push(draws[last]);
  return points;
}
