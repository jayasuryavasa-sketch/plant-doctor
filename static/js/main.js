document.addEventListener('DOMContentLoaded', () => {
  const menuButton = document.querySelector('.menu-button');
  const nav = document.querySelector('nav');
  if (menuButton && nav) {
    menuButton.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      menuButton.setAttribute('aria-expanded', String(open));
    });
  }

  const image = document.getElementById('image');
  const preview = document.getElementById('preview');
  const previewWrap = document.getElementById('preview-wrap');
  const clearImage = document.getElementById('clear-image');
  const analyze = document.getElementById('analyze');
  const form = document.getElementById('scan-form');

  image?.addEventListener('change', () => {
    const file = image.files[0];
    if (!file || !file.type.startsWith('image/')) return;
    preview.src = URL.createObjectURL(file);
    previewWrap.hidden = false;
    analyze.disabled = false;
  });
  clearImage?.addEventListener('click', () => {
    image.value = '';
    preview.src = '';
    previewWrap.hidden = true;
    analyze.disabled = true;
  });
  form?.addEventListener('submit', () => {
    analyze.textContent = 'Identifying plant…';
    analyze.disabled = true;
  });
});
