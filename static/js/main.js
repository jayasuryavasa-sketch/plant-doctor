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
  const uploadError = document.getElementById('upload-error');
  let isSubmitting = false;

  const showUploadError = (message) => {
    if (!uploadError) return;
    uploadError.textContent = message;
    uploadError.hidden = false;
  };

  image?.addEventListener('change', () => {
    const file = image.files[0];
    if (!file || !file.type.startsWith('image/')) {
      analyze.disabled = true;
      showUploadError('Please choose a JPG, PNG, or WEBP leaf photo.');
      return;
    }
    preview.src = URL.createObjectURL(file);
    previewWrap.hidden = false;
    analyze.disabled = false;
    if (uploadError) uploadError.hidden = true;
  });
  clearImage?.addEventListener('click', () => {
    image.value = '';
    preview.src = '';
    previewWrap.hidden = true;
    analyze.disabled = true;
    if (uploadError) uploadError.hidden = true;
  });
  form?.addEventListener('submit', (event) => {
    if (!image.files.length) {
      event.preventDefault();
      showUploadError('Choose a leaf photo before asking Plant Doctor to identify it.');
      return;
    }
    if (isSubmitting) {
      event.preventDefault();
      return;
    }
    isSubmitting = true;
    analyze.textContent = 'Identifying plant…';
    analyze.disabled = true;
  });
});
