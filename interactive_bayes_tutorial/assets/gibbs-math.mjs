function sampleFirst(v, c, normal, uniform) {
  if (c === 0) return normal();
  const offset = v + c;
  const modeSquared = (c * offset - 0.5) / c ** 2;
  if (modeSquared > 4) {
    // For u >= 0, (u + mode)^2 >= mode^2. This gives a Normal envelope around the positive mode.
    const mode = Math.sqrt(modeSquared);
    let u;
    do {
      do u = mode + normal() / (c * mode);
      while (u < 0);
    } while (uniform() >= Math.exp(-0.5 * (c * (u - mode)) ** 2 * u * (u + 2 * mode)));
    return uniform() < 0.5 ? -u : u;
  }

  // Use the prior Normal as the envelope and divide the likelihood by its maximum.
  const minimumResidual = Math.min(0, offset);
  let u;
  let residual;
  do {
    u = normal();
    residual = offset - c * u ** 2;
  } while (uniform() >= Math.exp(-0.5 * (residual ** 2 - minimumResidual ** 2)));
  return u;
}

export function bananaTarget(sigma, bend) {
  const c = bend * sigma;
  const variance = sigma ** 2;
  const meanSecond = x1 => bend * (x1 ** 2 - variance);
  const logNormalizer = Math.log(2 * Math.PI * variance);
  return {
    domains: [[-4 * sigma, 4 * sigma], [-(4 + c) * sigma, (4 + 4 * c) * sigma]],
    starts: [[-1.6, 0.2], [1.6, -0.2], [0, 0]].map(([u, v]) => [
      sigma * u, sigma * v + meanSecond(sigma * u)
    ]),
    logDensity: ([x1, x2]) =>
      -logNormalizer - 0.5 * ((x1 / sigma) ** 2 + ((x2 - meanSecond(x1)) / sigma) ** 2),
    gibbs: ([, x2], normal, uniform) => {
      const x1 = sigma * sampleFirst(x2 / sigma, c, normal, uniform);
      return [x1, meanSecond(x1) + sigma * normal()];
    }
  };
}
