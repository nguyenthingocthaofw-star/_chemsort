(() => {
  const menuButton = document.querySelector('.menu-toggle');
  const navigation = document.querySelector('.site-nav');

  if (menuButton && navigation) {
    menuButton.addEventListener('click', () => {
      const isOpen = navigation.classList.toggle('open');
      menuButton.setAttribute('aria-expanded', String(isOpen));
    });
  }

  const backToTop = document.querySelector('[data-back-to-top]');
  if (backToTop) {
    const updateVisibility = () => {
      backToTop.classList.toggle('visible', window.scrollY > 500);
    };
    window.addEventListener('scroll', updateVisibility, { passive: true });
    updateVisibility();
    backToTop.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }));
  }

  document.querySelectorAll('[data-current-year]').forEach((element) => {
    element.textContent = String(new Date().getFullYear());
  });
})();
