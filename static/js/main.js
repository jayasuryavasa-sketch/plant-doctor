document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('.menu-button'); const nav = document.querySelector('nav');
  if (button && nav) button.addEventListener('click', () => { const open = nav.classList.toggle('open'); button.setAttribute('aria-expanded', open); });
  const input = document.getElementById('image'), preview = document.getElementById('preview'), wrap = document.getElementById('preview-wrap'), analyze = document.getElementById('analyze'), clear = document.getElementById('clear-image');
  function show(file) { if (!file || !file.type.startsWith('image/')) return; preview.src = URL.createObjectURL(file); wrap.hidden = false; analyze.disabled = false; }
  if (input) { input.addEventListener('change', () => show(input.files[0])); clear?.addEventListener('click', () => { input.value=''; wrap.hidden=true; analyze.disabled=true; }); }
  document.getElementById('scan-form')?.addEventListener('submit', () => { analyze.textContent='Analyzing your plant…'; analyze.disabled=true; });
});
