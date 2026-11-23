import {betaDensity, normalDensity} from './bayes-math.mjs?v=608a0d229a30';

export function betaBinomial(n, a, b) {
  // Neighboring probabilities avoid direct evaluation of binomial coefficients.
  const logMass = [0];
  for (let k = 0; k < n; k += 1) {
    logMass.push(logMass[k] + Math.log((n - k) / (k + 1)) + Math.log((a + k) / (b + n - k - 1)));
  }
  const max = Math.max(...logMass);
  const mass = logMass.map(log => Math.exp(log - max));
  const total = mass.reduce((sum, p) => sum + p, 0);
  return mass.map(p => p / total);
}

export function discretize(density, coordinate, bins = 2048) {
  const cells = Array.from({length: bins}, (_, i) => {
    const lo = coordinate(i / bins);
    const hi = coordinate((i + 1) / bins);
    const x = coordinate((i + 0.5) / bins);
    const height = density(x);
    return {i, x, lo, hi, height, mass: height * (hi - lo)};
  });
  const total = cells.reduce((sum, cell) => sum + cell.mass, 0);
  for (const cell of cells) cell.mass /= total;
  return cells;
}

export function quantile(cells, p) {
  let cumulative = 0;
  for (const {mass, lo, hi} of cells) {
    if (mass > 0 && cumulative + mass >= p) return lo + (hi - lo) * (p - cumulative) / mass;
    cumulative += mass;
  }
  return cells.at(-1).hi;
}

export function clopperPearson(x, n, level = 0.95) {
  const tail = (1 - level) / 2;
  const betaQuantile = (a, b, p) => quantile(discretize(betaDensity(a, b), t => t), p);
  return [
    x === 0 ? 0 : betaQuantile(x, n - x + 1, tail),
    x === n ? 1 : betaQuantile(x + 1, n - x, 1 - tail)
  ];
}

export function highestDensity(cells, level) {
  const ranked = cells.toSorted((a, b) => b.height - a.height);
  const cutoff = new Set();
  let mass = 0;
  for (const cell of ranked) {
    cutoff.add(cell.i);
    mass += cell.mass;
    if (mass >= level) break;
  }
  const regions = [];
  for (const {i, lo, hi} of cells) {
    if (!cutoff.has(i)) continue;
    const last = regions.at(-1);
    if (last?.[1] === lo) last[1] = hi;
    else regions.push([lo, hi]);
  }
  return regions;
}

const leftMode = normalDensity(-2.5, 0.7);
const rightMode = normalDensity(2.5, 0.7);
export const mixtureDensity = x => (leftMode(x) + rightMode(x)) / 2;

export function logBayesFactor(heads, tails, shape) {
  let logFactor = 0;
  for (let i = 0; i < heads; i += 1) logFactor += Math.log(2 * (shape + i) / (2 * shape + i));
  for (let i = 0; i < tails; i += 1) logFactor += Math.log(2 * (shape + i) / (2 * shape + heads + i));
  return logFactor;
}

export function choleskySolver(matrix, diagonal) {
  const size = matrix.length;
  const lower = Array.from({length: size}, () => Array(size).fill(0));
  for (let i = 0; i < size; i += 1) {
    for (let j = 0; j <= i; j += 1) {
      let entry = matrix[i][j];
      for (let k = 0; k < j; k += 1) entry -= lower[i][k] * lower[j][k];
      lower[i][j] = i === j ? Math.sqrt(entry + diagonal) : entry / lower[j][j];
    }
  }
  const solveUpper = rhs => {
    const solution = [...rhs];
    for (let i = size - 1; i >= 0; i -= 1) {
      for (let j = i + 1; j < size; j += 1) solution[i] -= lower[j][i] * solution[j];
      solution[i] /= lower[i][i];
    }
    return solution;
  };
  const solve = rhs => {
    const solution = [...rhs];
    for (let i = 0; i < size; i += 1) {
      for (let j = 0; j < i; j += 1) solution[i] -= lower[i][j] * solution[j];
      solution[i] /= lower[i][i];
    }
    return solveUpper(solution);
  };
  const halfLogDet = lower.reduce((sum, row, i) => sum + Math.log(row[i]), 0);
  return {solve, solveUpper, halfLogDet};
}

function regressionProducts(design, response) {
  const columns = design[0].map((_, j) => j);
  const gram = columns.map(i => columns.map(j => design.reduce((sum, row) => sum + row[i] * row[j], 0)));
  const cross = columns.map(j => design.reduce((sum, row, i) => sum + row[j] * response[i], 0));
  return {columns, gram, cross};
}

export function regressionPosteriorSampler(design, response, priorSd) {
  const {columns, gram, cross} = regressionProducts(design, response);
  const {solve, solveUpper} = choleskySolver(gram, 1 / priorSd ** 2);
  const mean = solve(cross);
  // Solve Lᵀδ = z for z ~ N(0, I) to sample with the inverse precision as covariance.
  return normal => solveUpper(columns.map(() => normal())).map((delta, j) => mean[j] + delta);
}

export function modelCriteria(design, response, priorSd) {
  const n = response.length;
  const {columns, gram, cross} = regressionProducts(design, response);
  const k = columns.length;
  const deviance = beta => design.reduce((sum, row, i) => {
    const fitted = row.reduce((total, x, j) => total + x * beta[j], 0);
    return sum + (response[i] - fitted) ** 2;
  }, 0);
  const priorPrecision = 1 / priorSd ** 2;
  const {solve: solvePosterior} = choleskySolver(gram, priorPrecision);
  const inverseTrace = columns.reduce((sum, j) => sum + solvePosterior(columns.map(i => Number(i === j)))[j], 0);
  // With known noise variance 1, pD = trace(X'X V) = k − priorPrecision * trace(V).
  const pD = Math.max(0, k - priorPrecision * inverseTrace);
  const mleDeviance = n >= k ? deviance(choleskySolver(gram, 0).solve(cross)) : null;
  const penalties = {AIC: 2 * k, BIC: k * Math.log(n), DIC: 2 * pD};
  return {
    k, pD, penalties,
    scores: {
      AIC: mleDeviance === null ? null : mleDeviance + penalties.AIC,
      BIC: mleDeviance === null ? null : mleDeviance + penalties.BIC,
      DIC: deviance(solvePosterior(cross)) + penalties.DIC
    }
  };
}

export function shrinkageModes([z1, z2], rho, lambda) {
  const cutoff = lambda * (1 - rho ** 2);
  const softThreshold = z => Math.sign(z) * Math.max(0, Math.abs(z) - cutoff);
  let b1 = 0;
  let b2 = 0;
  // With the teaching model's rho = 0.6, coordinate descent contracts by rho².
  for (let i = 0; i < 50; i += 1) {
    b1 = softThreshold(z1 + rho * (b2 - z2));
    b2 = softThreshold(z2 + rho * (b1 - z1));
  }
  const diagonal = 1 + lambda;
  const cross = lambda * rho;
  const determinant = diagonal ** 2 - cross ** 2;
  return {
    lasso: [b1, b2],
    ridge: [(diagonal * z1 - cross * z2) / determinant, (diagonal * z2 - cross * z1) / determinant]
  };
}
