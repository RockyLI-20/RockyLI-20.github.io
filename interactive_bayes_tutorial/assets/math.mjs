const {katex} = window;
const options = {throwOnError: false};

export function math(tex) {
  return '<span class="math-inline">' + katex.renderToString(tex, options) + '</span>';
}

export const tex = String.raw;
export const betaMath = (a, b) => math(tex`\operatorname{Beta}(${a},${b})`);

for (const formula of document.querySelectorAll('[data-tex]')) {
  const displayMode = formula.hasAttribute('data-display');
  formula.classList.toggle('math-inline', !displayMode);
  katex.render(formula.dataset.tex, formula, {...options, displayMode});
}

export function axisMath(svg, content, {x = 0, y = 0, width, align = 'left'}) {
  const box = document.createElementNS(svg.namespaceURI, 'foreignObject');
  box.setAttribute('x', x);
  box.setAttribute('y', y);
  box.setAttribute('width', width);
  box.setAttribute('height', 22);
  box.setAttribute('aria-hidden', 'true');
  const label = document.createElement('div');
  label.className = 'bayes-axis-math';
  label.style.textAlign = align;
  label.innerHTML = content;
  box.append(label);
  svg.append(box);
}
