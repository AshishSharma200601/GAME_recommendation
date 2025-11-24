/* Login Page JavaScript */
(function(){
  // Form validation
  const loginForm = document.getElementById('loginForm');
  const registerForm = document.getElementById('registerForm');

  if(loginForm){
    loginForm.addEventListener('submit', function(e){
      const username = document.getElementById('username').value.trim();
      const password = document.getElementById('password').value;
      
      if(!username || !password){
        e.preventDefault();
        showError('Please fill in all fields');
        return false;
      }
    });
  }

  if(registerForm){
    registerForm.addEventListener('submit', function(e){
      const username = document.getElementById('username').value.trim();
      const email = document.getElementById('email').value.trim();
      const password = document.getElementById('password').value;
      const confirmPassword = document.getElementById('confirm_password').value;

      // Username validation
      if(username.length < 3 || username.length > 20){
        e.preventDefault();
        showError('Username must be between 3 and 20 characters');
        return false;
      }

      if(!/^[a-zA-Z0-9_]+$/.test(username)){
        e.preventDefault();
        showError('Username can only contain letters, numbers, and underscores');
        return false;
      }

      // Email validation
      if(!email || !email.includes('@')){
        e.preventDefault();
        showError('Please enter a valid email address');
        return false;
      }

      // Password validation
      if(password.length < 6){
        e.preventDefault();
        showError('Password must be at least 6 characters long');
        return false;
      }

      // Password match validation
      if(password !== confirmPassword){
        e.preventDefault();
        showError('Passwords do not match');
        return false;
      }
    });
  }

  function showError(message){
    // Create error alert if not exists
    let alertContainer = document.querySelector('.alert-container');
    if(!alertContainer){
      alertContainer = document.createElement('div');
      alertContainer.className = 'alert-container';
      const form = loginForm || registerForm;
      form.parentNode.insertBefore(alertContainer, form);
    }

    const alert = document.createElement('div');
    alert.className = 'alert alert-error';
    alert.innerHTML = `<span class="alert-icon">⚠️</span><span>${message}</span>`;
    alertContainer.insertBefore(alert, alertContainer.firstChild);

    // Remove after 5 seconds
    setTimeout(() => {
      alert.style.animation = 'slideOutUp 0.3s ease-out';
      setTimeout(() => alert.remove(), 300);
    }, 5000);
  }

  // Auto-hide alerts after 5 seconds
  document.querySelectorAll('.alert').forEach(alert => {
    setTimeout(() => {
      if(alert.parentNode){
        alert.style.animation = 'slideOutUp 0.3s ease-out';
        setTimeout(() => alert.remove(), 300);
      }
    }, 5000);
  });

  // Add slideOutUp animation
  const style = document.createElement('style');
  style.textContent = `
    @keyframes slideOutUp{
      from{opacity:1;transform:translateY(0)}
      to{opacity:0;transform:translateY(-10px)}
    }
  `;
  document.head.appendChild(style);
})();

