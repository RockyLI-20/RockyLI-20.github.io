export const coinExample = 'HHHTHHHHHTHHHHTHH';

export function countHeads(tosses) {
  let heads = 0;
  for (const toss of tosses) heads += Number(toss === 'H');
  return heads;
}

export function betaUpdate(alpha, beta, heads, tails) {
  const a = alpha + heads;
  const b = beta + tails;
  return a > 0 && b > 0 ? {a, b, mean: a / (a + b)} : null;
}

function logPower(power, value) {
  return power === 0 ? 0 : power * Math.log(value);
}

function logGamma(shape) {
  // The presets and integer toss counts only produce integer and half-integer shapes.
  let result = Number.isInteger(shape) ? 0 : Math.log(Math.PI) / 2;
  for (let x = shape - 1; x > 0; x -= 1) result += Math.log(x);
  return result;
}

export function betaDensity(a, b) {
  const logBeta = logGamma(a) + logGamma(b) - logGamma(a + b);
  return x => Math.exp(logPower(a - 1, x) + logPower(b - 1, 1 - x) - logBeta);
}

export function relativeLikelihood(heads, tails) {
  const mode = heads / (heads + tails);
  const logLikelihood = x => logPower(heads, x) + logPower(tails, 1 - x);
  const maximum = logLikelihood(mode);
  return x => Math.exp(logLikelihood(x) - maximum);
}

export function logOdds(theta) {
  return Math.log(theta / (1 - theta));
}

export function probability(logOddsValue) {
  return 1 / (1 + Math.exp(-logOddsValue));
}

export function logOddsDensity(logOddsValue) {
  const theta = probability(logOddsValue);
  return theta * (1 - theta);
}

export function normalUpdate(priorMean, priorSd, sampleMean, sampleSize, observationSd) {
  const priorPrecision = 1 / priorSd ** 2;
  const samplePrecision = sampleSize / observationSd ** 2;
  const precision = priorPrecision + samplePrecision;
  return {
    mean: (priorMean * priorPrecision + sampleMean * samplePrecision) / precision,
    sd: Math.sqrt(1 / precision),
    sampleWeight: samplePrecision / precision
  };
}

export function normalDensity(mean, sd) {
  return x => Math.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * Math.sqrt(2 * Math.PI));
}
