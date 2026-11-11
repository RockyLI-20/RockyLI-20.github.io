const menu = document.querySelector('.menu-toggle');
const nav = document.querySelector('#nav');
menu.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  nav.classList.toggle('open', open);
});
document.addEventListener('keydown', event => {
  if (event.key === 'Escape' && menu.getAttribute('aria-expanded') === 'true') {
    menu.setAttribute('aria-expanded', 'false');
    nav.classList.remove('open');
    menu.focus();
  }
});

const buttons = document.querySelectorAll('.topic-button');
function setPanel(button, panel, open) {
  button.setAttribute('aria-expanded', String(open));
  panel.getAnimations().forEach(animation => animation.cancel());
  panel.hidden = false;
  const height = panel.scrollHeight;
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
    panel.hidden = !open;
    return;
  }
  panel.style.overflow = 'hidden';
  const animation = panel.animate(
    open ? [{height:'0px', opacity:0}, {height:`${height}px`, opacity:1}] : [{height:`${height}px`, opacity:1}, {height:'0px', opacity:0}],
    {duration:400, easing:'ease-in-out'}
  );
  animation.onfinish = () => {
    panel.style.overflow = '';
    panel.hidden = !open;
  };
}
document.querySelectorAll('.disclosure-button').forEach(button => button.addEventListener('click', () => {
  const panel = document.getElementById(button.getAttribute('aria-controls'));
  const open = button.getAttribute('aria-expanded') !== 'true';
  setPanel(button, panel, open);
  const icon = button.querySelector('[aria-hidden]');
  if (icon) icon.textContent = open ? '−' : '+';
}));
let topicsPromise;
let current = -1;
buttons.forEach(button => button.addEventListener('click', async () => {
  const index = Number(button.dataset.topic);
  const panel = document.getElementById(button.getAttribute('aria-controls'));
  if (current === index) {
    current = -1;
    setPanel(button, panel, false);
    return;
  }
  current = index;
  buttons.forEach(item => {
    if (item !== button && item.getAttribute('aria-expanded') === 'true') setPanel(item, document.getElementById(item.getAttribute('aria-controls')), false);
  });
  button.setAttribute('aria-expanded', 'true');
  panel.textContent = 'Loading topic…';
  try {
    topicsPromise ||= fetch('topics.json').then(response => { if (!response.ok) throw Error('Unable to load topics'); return response.json(); });
    const topics = await topicsPromise;
    if (current !== index) return;
    const topic = topics[index];
    panel.replaceChildren();
    const heading = document.createElement('h2');
    heading.textContent = topic.title;
    const extract = document.createElement('p');
    extract.textContent = topic.introduction;
    const source = document.createElement('p');
    source.innerHTML = `<small>Abridged from <a href="https://en.wikipedia.org/wiki/${topic.wiki}">Wikipedia</a> · <a href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA</a></small>`;
    const label = document.createElement('h3');
    label.textContent = 'Selected foundational papers';
    const list = document.createElement('ol');
    topic.papers.forEach(paper => {
      const item = document.createElement('li');
      const citation = document.createElement('div');
      citation.textContent = paper.citation;
      const meta = document.createElement('small');
      meta.textContent = `${paper.journal} · `;
      const link = document.createElement('a');
      link.href = `https://doi.org/${paper.doi}`;
      link.textContent = `DOI: ${paper.doi}`;
      meta.append(link);
      const note = document.createElement('div');
      note.className = 'reading-note';
      note.textContent = paper.note;
      item.append(citation, meta, note);
      list.append(item);
    });
    panel.append(heading, extract, source, label, list);
    if (topic.wiki === 'Deep_learning') {
      const scope = document.createElement('p');
      scope.textContent = 'My focus: statistical theory, nonparametric estimation, convergence rates, and inference.';
      panel.insertBefore(scope, label);
    }
    setPanel(button, panel, true);

  } catch {
    if (current === index) panel.textContent = 'Unable to load the reading list. Please try again.';
    topicsPromise = null;
  }
}));