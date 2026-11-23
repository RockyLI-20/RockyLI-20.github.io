import {countLaplace} from './count-math.mjs?v=608a0d229a30';

const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);

function design(x, group, tau, groupCount) {
  // Standardize the coefficients so their prior precision is the identity.
  return [2, 2 * x, ...Array.from({length: groupCount}, (_, j) => j === group ? tau : 0)];
}

function groupMean({mode, solve}, tau, groupCount, group, testX) {
  const intercept = design(0, group, tau, groupCount);
  const slope = mode.map((_, i) => i === 1 ? 2 : 0);
  const interceptCovariance = solve(intercept);
  const varIntercept = dot(intercept, interceptCovariance);
  const cov = dot(slope, interceptCovariance);
  const varSlope = dot(slope, solve(slope));
  const a = dot(intercept, mode) + varIntercept / 2;
  const b = dot(slope, mode) + cov;
  // If eta is Normal, E[exp(eta)] = exp(E[eta] + Var(eta)/2).
  return testX.reduce((sum, x) => sum + Math.exp(a + b * x + varSlope * x ** 2 / 2), 0) / testX.length;
}

export function poissonGroupMeans(training, testX, groupCount) {
  const y = training.map(row => row.y);
  const total = y.reduce((sum, count) => sum + count, 0);
  const pooledRows = training.map(({x}) => design(x, 0, 0, 0));
  const pooled = countLaplace(pooledRows, y, [Math.log((total + 0.5) / y.length) / 2, 0]);
  const commonMean = groupMean(pooled, 0, 0, 0, testX);

  // shortcut: midpoint quadrature on [0, 6] for the Half-Normal(0, 1) prior, revisit the grid if the prior scale changes.
  const grid = Array.from({length: 60}, (_, i) => (i + 0.5) / 10);
  const groups = Array.from({length: groupCount}, (_, j) => j);
  let start = [...pooled.mode, ...groups.map(() => 0)];
  const components = grid.map(tau => {
    const rows = training.map(({x, group}) => design(x, group, tau, groupCount));
    const fit = countLaplace(rows, y, start);
    start = fit.mode;
    return {
      logWeight: fit.logWeight - tau ** 2 / 2,
      means: groups.map(j => groupMean(fit, tau, groupCount, j, testX))
    };
  });
  const peak = Math.max(...components.map(fit => fit.logWeight));
  const weights = components.map(fit => Math.exp(fit.logWeight - peak));
  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);
  return {
    glm: groups.map(() => commonMean),
    glmm: groups.map(j => components.reduce((sum, fit, i) => sum + weights[i] * fit.means[j], 0) / totalWeight)
  };
}
