import {choleskySolver} from './inference-math.mjs?v=608a0d229a30';

const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const softplus = x => Math.max(x, 0) + Math.log1p(Math.exp(-Math.abs(x)));

function poissonLoss(y, eta, logY) {
  const difference = eta - logY;
  return y === 0 ? Math.exp(eta) : y * (Math.expm1(difference) - difference);
}

export function countLaplace(rows, y, start, phi = Infinity) {
  const size = start.length;
  const poisson = phi === Infinity;
  const logPhi = Math.log(phi);
  const logCounts = y.map(Math.log);
  const loss = q => dot(q, q) / 2 + rows.reduce((sum, row, i) => {
    const eta = dot(row, q);
    const shift = eta - logPhi;
    return sum + (poisson ? poissonLoss(y[i], eta, logCounts[i]) : y[i] * softplus(-shift) + phi * softplus(shift));
  }, 0);
  let mode = start;
  for (let iteration = 0; iteration < 40; iteration += 1) {
    const gradient = mode.slice();
    const gram = Array.from({length: size}, () => Array(size).fill(0));
    for (let i = 0; i < rows.length; i += 1) {
      const row = rows[i];
      const eta = dot(row, mode);
      const p = poisson ? 0 : 1 / (1 + Math.exp(logPhi - eta));
      const q = poisson ? 1 : 1 / (1 + Math.exp(eta - logPhi));
      const residual = poisson ? Math.exp(eta) - y[i] : phi * p - y[i] * q;
      const curvature = poisson ? Math.exp(eta) : (y[i] + phi) * p * q;
      for (let j = 0; j < size; j += 1) {
        gradient[j] += row[j] * residual;
        for (let k = 0; k <= j; k += 1) gram[j][k] += curvature * row[j] * row[k];
      }
    }
    const {solve, halfLogDet} = choleskySolver(gram, 1);
    const step = solve(gradient);
    const decrement = dot(gradient, step);
    const currentLoss = loss(mode);
    const tolerance = Math.max(1e-8, 16 * Number.EPSILON * Math.abs(currentLoss));
    if (decrement <= tolerance) return {mode, solve, logWeight: -currentLoss - halfLogDet};
    let fraction = 1;
    let next = mode.map((value, i) => value - step[i]);
    while (loss(next) > currentLoss - 0.0001 * fraction * decrement) {
      fraction /= 2;
      next = mode.map((value, i) => value - fraction * step[i]);
    }
    mode = next;
  }
  throw new Error('Count posterior mode did not converge');
}

function stirlingCorrection(x) {
  const inv = 1 / x;
  const square = inv * inv;
  return inv * (1 / 12 - square * (1 / 360 - square * (1 / 1260 - square * (1 / 1680 - square / 1188))));
}

function logGamma(x) {
  let shift = 0;
  for (; x < 10; x += 1) shift -= Math.log(x);
  return shift + (x - 0.5) * Math.log(x) - x + Math.log(2 * Math.PI) / 2 + stirlingCorrection(x);
}

function logGammaRatio(x, delta) {
  // Evaluate log Γ(x + delta) − log Γ(x) without subtracting two large values.
  const shifted = x + delta;
  if (x < 10 || shifted < 10) return logGamma(shifted) - logGamma(x);
  return delta * Math.log(x) + (shifted - 0.5) * Math.log1p(delta / x) - delta
    + stirlingCorrection(shifted) - stirlingCorrection(x);
}

function logPoissonProbability(y, mu) {
  const eta = Math.log(mu);
  if (y < 10) return y * eta - mu - logGamma(y + 1);
  return -poissonLoss(y, eta, Math.log(y)) - Math.log(2 * Math.PI * y) / 2 - stirlingCorrection(y);
}

function betaFraction(a, b, x) {
  // NIST DLMF 8.17.22–23, evaluated using modified Lentz iteration.
  const nonzero = value => Math.abs(value) < 1e-30 ? 1e-30 : value;
  let c = 1;
  let d = 0;
  let value = 1;
  for (let m = 0; m < 120; m += 1) {
    const offset = a + 2 * m;
    const odd = -(a + m) / offset * ((a + b + m) * x / (offset + 1));
    d = 1 / nonzero(1 + odd * d);
    c = nonzero(1 + odd / c);
    value *= c * d;

    const even = (m + 1) / (offset + 1) * ((b - m - 1) * x / (offset + 2));
    d = 1 / nonzero(1 + even * d);
    c = nonzero(1 + even / c);
    const multiplier = c * d;
    value *= multiplier;
    if (m > 0 && Math.abs(multiplier - 1) < 1e-13) break;
  }
  return 1 / value;
}

function negativeBinomialInterval(mu, phi) {
  const p = phi / (mu + phi);
  const q = mu / (mu + phi);
  const logP = -Math.log1p(mu / phi);
  const logQ = -Math.log1p(phi / mu);
  const logGammaPhi = logGamma(phi);
  const cdf = k => {
    const b = k + 1;
    const front = Math.exp(logGammaRatio(b, phi) - logGammaPhi + phi * logP + b * logQ);
    return p < (phi + 1) / (phi + b + 2)
      ? front * betaFraction(phi, b, p) / phi
      : 1 - front * betaFraction(b, phi, q) / b;
  };
  return [0.025, 0.975].map(probability => {
    let low = -1;
    let high = Math.ceil(mu);
    while (cdf(high) < probability) high *= 2;
    while (high - low > 1) {
      const mid = Math.floor((high + low) / 2);
      if (cdf(mid) < probability) low = mid;
      else high = mid;
    }
    return high;
  });
}

export function countRegressionFits(data) {
  const rows = data.map(({x}) => [2, 2 * x]);
  const y = data.map(row => row.y);
  const count = y.reduce((sum, v) => sum + v, 0);
  const poisson = countLaplace(rows, y, [Math.log((count + 0.5) / y.length) / 2, 0]);
  let start = poisson.mode;
  // shortcut: integrate on a log-spaced grid over [0.03, 20], revisit the grid for wider priors or different data ranges.
  const components = Array.from({length: 64}, (_, i) => {
    const logPhi = Math.log(0.03) + (i + 0.5) / 64 * Math.log(20 / 0.03);
    const phi = Math.exp(logPhi);
    const {mode, logWeight} = countLaplace(rows, y, start, phi);
    start = mode;
    const normalizer = y.reduce((sum, value) => sum + logGammaRatio(value + 1, phi - 1), 0) - y.length * logGamma(phi);
    return {mode, phi, logWeight: logWeight + normalizer + 2 * logPhi - phi};
  });
  const peak = Math.max(...components.map(fit => fit.logWeight));
  const weights = components.map(fit => Math.exp(fit.logWeight - peak));
  const totalWeight = weights.reduce((sum, w) => sum + w, 0);
  const average = value => components.reduce((sum, fit, i) => sum + weights[i] * value(fit), 0) / totalWeight;
  const prediction = ([a, b], phi) => ({phi, mean: x => Math.exp(2 * (a + b * x))});
  return [prediction(poisson.mode, Infinity), prediction([average(fit => fit.mode[0]), average(fit => fit.mode[1])], average(fit => fit.phi))];
}

export function countLogLikelihood(data, {mean, phi}) {
  const logGammaPhi = phi === Infinity ? 0 : logGamma(phi);
  return data.reduce((sum, {x, y}) => {
    const mu = mean(x);
    if (phi === Infinity) return sum + logPoissonProbability(y, mu);
    const coefficient = logGammaRatio(y + 1, phi - 1) - logGammaPhi;
    return sum + coefficient - y * Math.log1p(phi / mu) - phi * Math.log1p(mu / phi);
  }, 0);
}

export function countPredictionInterval(mu, phi) {
  const poisson = phi === Infinity;
  if (!poisson && mu > 100) return negativeBinomialInterval(mu, phi);
  const ratio = mu / (mu + phi);
  // For a large Poisson mean, start far below its center to avoid underflow at zero.
  let k = poisson && mu > 500 ? Math.floor(mu - 12 * Math.sqrt(mu)) : 0;
  let mass = poisson
    ? Math.exp(logPoissonProbability(k, mu))
    : Math.exp(-phi * Math.log1p(mu / phi));
  let cdf = mass;
  const quantile = probability => {
    while (cdf < probability) {
      k += 1;
      mass *= poisson ? mu / k : (k - 1 + phi) / k * ratio;
      cdf += mass;
    }
    return k;
  };
  return [quantile(0.025), quantile(0.975)];
}
