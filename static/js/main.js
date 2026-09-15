document.addEventListener('DOMContentLoaded', () => {
  const button = document.querySelector('.menu-button');
  const nav = document.querySelector('nav');
  if (button && nav) {
    button.addEventListener('click', () => {
      const open = nav.classList.toggle('open');
      button.setAttribute('aria-expanded', String(open));
    });
  }

  const image = document.getElementById('image');
  const preview = document.getElementById('preview');
  const previewWrap = document.getElementById('preview-wrap');
  const clearImage = document.getElementById('clear-image');
  const crop = document.getElementById('crop-select');
  const condition = document.getElementById('condition-select');
  const openGuidance = document.getElementById('open-guidance');
  const entriesNode = document.getElementById('guide-entries');
  const entries = entriesNode ? JSON.parse(entriesNode.textContent) : [];

  function updateButton() {
    if (openGuidance) openGuidance.disabled = !condition?.value;
  }

  image?.addEventListener('change', () => {
    const file = image.files[0];
    if (!file || !file.type.startsWith('image/')) return;
    preview.src = URL.createObjectURL(file);
    previewWrap.hidden = false;
  });
  clearImage?.addEventListener('click', () => {
    image.value = '';
    preview.src = '';
    previewWrap.hidden = true;
  });

  crop?.addEventListener('change', () => {
    const matches = entries.filter((entry) => entry.plant === crop.value);
    condition.innerHTML = '<option value="">Select a visible condition</option>';
    matches.forEach((entry) => {
      const option = document.createElement('option');
      option.value = entry.slug;
      option.textContent = entry.name;
      condition.appendChild(option);
    });
    condition.disabled = matches.length === 0;
    updateButton();
  });
  condition?.addEventListener('change', updateButton);
  openGuidance?.addEventListener('click', () => {
    if (condition.value) window.location.href = `/guide/${encodeURIComponent(condition.value)}`;
  });
});
