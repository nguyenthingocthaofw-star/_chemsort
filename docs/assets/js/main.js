// ═══════════════════════════════════════════════════════════
// ChemSort Main JavaScript
// ═══════════════════════════════════════════════════════════

(function() {
  'use strict';

  // ───────────────────────────────────────────────────────────
  // Mobile Menu Toggle
  // ───────────────────────────────────────────────────────────
  const menuToggle = document.querySelector('.menu-toggle');
  const siteNav = document.querySelector('.site-nav');

  if (menuToggle && siteNav) {
    menuToggle.addEventListener('click', function() {
      const isExpanded = menuToggle.getAttribute('aria-expanded') === 'true';
      menuToggle.setAttribute('aria-expanded', !isExpanded);
      siteNav.setAttribute('aria-expanded', !isExpanded);
      
      // Update button text
      const srText = menuToggle.querySelector('.sr-only');
      if (srText) {
        srText.textContent = isExpanded ? 'Mở menu' : 'Đóng menu';
      }
    });

    // Close menu when clicking outside
    document.addEventListener('click', function(e) {
      if (!menuToggle.contains(e.target) && !siteNav.contains(e.target)) {
        menuToggle.setAttribute('aria-expanded', 'false');
        siteNav.setAttribute('aria-expanded', 'false');
      }
    });

    // Close menu on escape key
    document.addEventListener('keydown', function(e) {
      if (e.key === 'Escape') {
        menuToggle.setAttribute('aria-expanded', 'false');
        siteNav.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // ───────────────────────────────────────────────────────────
  // Smooth Scroll for Anchor Links
  // ───────────────────────────────────────────────────────────
  document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
      const href = this.getAttribute('href');
      if (href === '#') return;
      
      const target = document.querySelector(href);
      if (target) {
        e.preventDefault();
        target.scrollIntoView({
          behavior: 'smooth',
          block: 'start'
        });
        
        // Update focus for accessibility
        target.setAttribute('tabindex', '-1');
        target.focus();
      }
    });
  });

  // ───────────────────────────────────────────────────────────
  // Header Scroll Effect
  // ───────────────────────────────────────────────────────────
  const header = document.querySelector('.site-header');
  let lastScroll = 0;

  if (header) {
    window.addEventListener('scroll', function() {
      const currentScroll = window.pageYOffset;

      if (currentScroll > 100) {
        header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.08)';
        header.style.background = 'rgba(255, 255, 255, 0.95)';
      } else {
        header.style.boxShadow = '';
        header.style.background = 'rgba(255, 255, 255, 0.8)';
      }

      lastScroll = currentScroll;
    });
  }

  // ───────────────────────────────────────────────────────────
  // Intersection Observer for Animations
  // ───────────────────────────────────────────────────────────
  const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
  };

  const observer = new IntersectionObserver(function(entries) {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, observerOptions);

  // Observe elements that should animate in
  document.querySelectorAll('.document-card, .steps-list li').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(30px)';
    el.style.transition = 'opacity 0.6s ease-out, transform 0.6s ease-out';
    observer.observe(el);
  });

  // ───────────────────────────────────────────────────────────
  // Dynamic Year in Footer
  // ───────────────────────────────────────────────────────────
  const yearElement = document.querySelector('[data-current-year]');
  if (yearElement) {
    yearElement.textContent = new Date().getFullYear();
  }

  // ───────────────────────────────────────────────────────────
  // Card Tilt Effect (3D)
  // ───────────────────────────────────────────────────────────
  const cards = document.querySelectorAll('.document-card');
  
  cards.forEach(card => {
    card.addEventListener('mousemove', function(e) {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      
      const centerX = rect.width / 2;
      const centerY = rect.height / 2;
      
      const rotateX = (y - centerY) / 20;
      const rotateY = (centerX - x) / 20;
      
      card.style.transform = `
        perspective(1000px)
        rotateX(${rotateX}deg)
        rotateY(${rotateY}deg)
        translateY(-8px)
        scale(1.02)
      `;
    });
    
    card.addEventListener('mouseleave', function() {
      card.style.transform = '';
    });
  });

  // ───────────────────────────────────────────────────────────
  // Parallax Effect for Hero Art
  // ───────────────────────────────────────────────────────────
  const heroArt = document.querySelector('.hero-art');
  
  if (heroArt) {
    window.addEventListener('scroll', function() {
      const scrolled = window.pageYOffset;
      const parallax = scrolled * 0.3;
      heroArt.style.transform = `translateY(${parallax}px)`;
    });
  }

  // ───────────────────────────────────────────────────────────
  // Preload Critical Images
  // ───────────────────────────────────────────────────────────
  const preloadImages = [
    'assets/images/chemsort-mark.svg'
  ];

  preloadImages.forEach(src => {
    const link = document.createElement('link');
    link.rel = 'preload';
    link.as = 'image';
    link.href = src;
    document.head.appendChild(link);
  });

  // ───────────────────────────────────────────────────────────
  // Ripple Effect on Buttons
  // ───────────────────────────────────────────────────────────
  const buttons = document.querySelectorAll('.button');
  
  buttons.forEach(button => {
    button.addEventListener('click', function(e) {
      const ripple = document.createElement('span');
      const rect = button.getBoundingClientRect();
      const size = Math.max(rect.width, rect.height);
      const x = e.clientX - rect.left - size / 2;
      const y = e.clientY - rect.top - size / 2;
      
      ripple.style.width = ripple.style.height = size + 'px';
      ripple.style.left = x + 'px';
      ripple.style.top = y + 'px';
      ripple.className = 'ripple-effect';
      
      button.appendChild(ripple);
      
      setTimeout(() => ripple.remove(), 600);
    });
  });

  // ───────────────────────────────────────────────────────────
  // Performance: Lazy Load Images
  // ───────────────────────────────────────────────────────────
  if ('loading' in HTMLImageElement.prototype) {
    const images = document.querySelectorAll('img[loading="lazy"]');
    images.forEach(img => {
      img.src = img.dataset.src || img.src;
    });
  } else {
    // Fallback for browsers that don't support lazy loading
    const script = document.createElement('script');
    script.src = 'https://cdnjs.cloudflare.com/ajax/libs/lazysizes/5.3.2/lazysizes.min.js';
    document.body.appendChild(script);
  }

  // ───────────────────────────────────────────────────────────
  // Console Easter Egg
  // ───────────────────────────────────────────────────────────
  console.log(
    '%c✨ ChemSort %c— Quét vật liệu, mở kiến thức',
    'color: #00d9a3; font-size: 24px; font-weight: bold;',
    'color: #64748b; font-size: 14px;'
  );
  console.log(
    '%cBạn đang xem source code à? 👀\nNếu quan tâm đến dự án, hãy check GitHub repo nhé!',
    'color: #6366f1; font-size: 12px; line-height: 1.8;'
  );

  // ───────────────────────────────────────────────────────────
  // Keyboard Navigation Enhancement
  // ───────────────────────────────────────────────────────────
  document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + K to focus search (if exists)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      const searchInput = document.querySelector('input[type="search"]');
      if (searchInput) searchInput.focus();
    }
  });

  // ───────────────────────────────────────────────────────────
  // Page Load Complete
  // ───────────────────────────────────────────────────────────
  window.addEventListener('load', function() {
    document.body.classList.add('loaded');
    
    // Optional: Send analytics
    if (typeof gtag !== 'undefined') {
      gtag('event', 'page_view', {
        page_title: document.title,
        page_location: window.location.href
      });
    }
  });

})();
