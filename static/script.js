/**
 * Universal JavaScript Utilities for Jain Learning Ecosystem
 * Includes button loading handlers, form utilities, and common functions
 */

// Global utilities
window.JainLearning = {
    // Configuration
    config: {
        loadingTimeout: 30000, // 30 seconds
        animationDuration: 300
    },

    // Initialize application
    init: function() {
        this.setupGlobalEventListeners();
        this.initializeTooltips();
        this.initializeModals();
        this.setupFormValidation();
        console.log('Jain Learning Ecosystem initialized');
    },

    // Setup global event listeners
    setupGlobalEventListeners: function() {
        // Handle all forms with loading states
        document.addEventListener('submit', this.handleFormSubmit.bind(this));

        // Handle button clicks with loading states
        document.addEventListener('click', this.handleButtonClick.bind(this));

        // Handle file uploads
        document.addEventListener('change', this.handleFileInput.bind(this));

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
    },

    // Handle form submissions
    handleFormSubmit: function(e) {
        const form = e.target;
        if (form.tagName !== 'FORM') return;

        const submitBtn = form.querySelector('button[type="submit"]');
        if (submitBtn && !submitBtn.disabled) {
            this.setButtonLoading(submitBtn, true);

            // Set timeout to reset button if needed
            setTimeout(() => {
                if (submitBtn.disabled) {
                    this.setButtonLoading(submitBtn, false);
                }
            }, this.config.loadingTimeout);
        }
    },

    // Handle button clicks
    handleButtonClick: function(e) {
        const button = e.target.closest('button');
        if (!button) return;

        // Skip if already loading or disabled
        if (button.disabled || button.classList.contains('loading')) return;

        // Auto-loading for certain classes
        if (button.classList.contains('auto-loading')) {
            this.setButtonLoading(button, true);
        }
    },

    // Handle file input changes
    handleFileInput: function(e) {
        const input = e.target;
        if (input.type !== 'file') return;

        const files = input.files;
        if (files.length > 0) {
            this.validateFiles(files, input);
        }
    },

    // Handle keyboard shortcuts
    handleKeyboardShortcuts: function(e) {
        // Ctrl/Cmd + Enter to submit forms
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const activeForm = document.activeElement.closest('form');
            if (activeForm) {
                const submitBtn = activeForm.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    submitBtn.click();
                }
            }
        }

        // ESC to close modals
        if (e.key === 'Escape') {
            const openModal = document.querySelector('.modal.show');
            if (openModal) {
                const modal = bootstrap.Modal.getInstance(openModal);
                if (modal) modal.hide();
            }
        }
    },

    // Universal button loading handler
    setButtonLoading: function(button, isLoading, customText = '') {
        if (!button) return;

        if (isLoading) {
            // Save original state
            if (!button.dataset.originalText) {
                button.dataset.originalText = button.innerHTML;
                button.dataset.originalDisabled = button.disabled;
            }

            // Set loading state
            button.disabled = true;
            button.classList.add('loading');

            // Update button content
            const btnText = button.querySelector('.btn-text');
            const btnLoading = button.querySelector('.btn-loading');

            if (btnText && btnLoading) {
                btnText.style.display = 'none';
                btnLoading.style.display = 'inline-block';
                if (customText) {
                    btnLoading.textContent = customText;
                }
            } else {
                // Fallback for buttons without structured content
                const loadingText = customText || 'Loading...';
                button.innerHTML = `
                    <span class="spinner-border spinner-border-sm me-2" role="status">
                        <span class="visually-hidden">Loading...</span>
                    </span>
                    ${loadingText}
                `;
            }
        } else {
            // Restore original state
            button.disabled = button.dataset.originalDisabled === 'true';
            button.classList.remove('loading');

            const btnText = button.querySelector('.btn-text');
            const btnLoading = button.querySelector('.btn-loading');

            if (btnText && btnLoading) {
                btnText.style.display = 'inline-block';
                btnLoading.style.display = 'none';
            } else if (button.dataset.originalText) {
                button.innerHTML = button.dataset.originalText;
            }

            // Clean up datasets
            delete button.dataset.originalText;
            delete button.dataset.originalDisabled;
        }
    },

    // Show loading overlay
    showLoadingOverlay: function(text = 'Processing...') {
        const overlay = document.getElementById('loadingOverlay');
        const loadingText = document.getElementById('loadingText');

        if (overlay) {
            if (loadingText) loadingText.textContent = text;
            overlay.style.display = 'flex';
            document.body.style.overflow = 'hidden';
        }
    },

    // Hide loading overlay
    hideLoadingOverlay: function() {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            overlay.style.display = 'none';
            document.body.style.overflow = '';
        }
    },

    // Initialize Bootstrap tooltips
    initializeTooltips: function() {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    },

    // Initialize Bootstrap modals
    initializeModals: function() {
        // Auto-focus first input in modals
        document.addEventListener('shown.bs.modal', function(e) {
            const firstInput = e.target.querySelector('input, textarea, select');
            if (firstInput) {
                firstInput.focus();
            }
        });
    },

    // Setup form validation
    setupFormValidation: function() {
        // Add Bootstrap validation classes
        const forms = document.querySelectorAll('form');
        forms.forEach(form => {
            form.addEventListener('submit', function(e) {
                if (!form.checkValidity()) {
                    e.preventDefault();
                    e.stopPropagation();
                }
                form.classList.add('was-validated');
            });
        });
    },

    // Validate uploaded files
    validateFiles: function(files, input) {
        const maxSize = 100 * 1024 * 1024; // 100MB
        const allowedTypes = ['application/pdf'];
        let valid = true;

        Array.from(files).forEach(file => {
            // Check file size
            if (file.size > maxSize) {
                this.showAlert(`File ${file.name} is too large. Maximum size is 100MB.`, 'danger');
                valid = false;
            }

            // Check file type for PDF inputs
            if (input.accept === '.pdf' && !allowedTypes.includes(file.type)) {
                this.showAlert(`File ${file.name} is not a valid PDF.`, 'danger');
                valid = false;
            }
        });

        if (!valid) {
            input.value = '';
        }

        return valid;
    },

    // Show alert notification
    showAlert: function(message, type = 'info', duration = 5000) {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
        alertDiv.style.cssText = `
            top: 20px;
            right: 20px;
            z-index: 1060;
            max-width: 400px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;

        alertDiv.innerHTML = `
            <i class="fas ${this.getAlertIcon(type)} me-2"></i>
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
        `;

        document.body.appendChild(alertDiv);

        // Auto-remove after duration
        if (duration > 0) {
            setTimeout(() => {
                if (alertDiv.parentNode) {
                    alertDiv.remove();
                }
            }, duration);
        }

        return alertDiv;
    },

    // Get icon for alert type
    getAlertIcon: function(type) {
        const icons = {
            success: 'fa-check-circle',
            danger: 'fa-exclamation-triangle',
            warning: 'fa-exclamation-circle',
            info: 'fa-info-circle'
        };
        return icons[type] || 'fa-info-circle';
    },

    // Smooth scroll to element
    scrollToElement: function(element, offset = 0) {
        if (typeof element === 'string') {
            element = document.querySelector(element);
        }

        if (element) {
            const elementPosition = element.getBoundingClientRect().top;
            const offsetPosition = elementPosition + window.pageYOffset - offset;

            window.scrollTo({
                top: offsetPosition,
                behavior: 'smooth'
            });
        }
    },

    // Format numbers with commas
    formatNumber: function(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
    },

    // Format time duration
    formatDuration: function(minutes) {
        const hours = Math.floor(minutes / 60);
        const mins = minutes % 60;

        if (hours > 0) {
            return `${hours}h ${mins}m`;
        }
        return `${mins}m`;
    },

    // Debounce function
    debounce: function(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    },

    // API helper functions
    api: {
        // Generic fetch wrapper
        request: function(url, options = {}) {
            const defaultOptions = {
                headers: {
                    'Content-Type': 'application/json',
                },
                credentials: 'same-origin'
            };

            const finalOptions = { ...defaultOptions, ...options };

            return fetch(url, finalOptions)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .catch(error => {
                    console.error('API request failed:', error);
                    JainLearning.showAlert('Network error. Please try again.', 'danger');
                    throw error;
                });
        },

        // GET request
        get: function(url) {
            return this.request(url);
        },

        // POST request
        post: function(url, data) {
            return this.request(url, {
                method: 'POST',
                body: JSON.stringify(data)
            });
        },

        // File upload
        uploadFile: function(url, formData, onProgress = null) {
            return new Promise((resolve, reject) => {
                const xhr = new XMLHttpRequest();

                xhr.upload.addEventListener('progress', (e) => {
                    if (e.lengthComputable && onProgress) {
                        const percentComplete = (e.loaded / e.total) * 100;
                        onProgress(percentComplete);
                    }
                });

                xhr.addEventListener('load', () => {
                    if (xhr.status === 200) {
                        try {
                            resolve(JSON.parse(xhr.responseText));
                        } catch (e) {
                            reject(new Error('Invalid JSON response'));
                        }
                    } else {
                        reject(new Error(`Upload failed: ${xhr.status}`));
                    }
                });

                xhr.addEventListener('error', () => {
                    reject(new Error('Upload failed'));
                });

                xhr.open('POST', url);
                xhr.send(formData);
            });
        }
    },

    // Storage utilities
    storage: {
        // Set item in localStorage
        set: function(key, value) {
            try {
                localStorage.setItem(key, JSON.stringify(value));
                return true;
            } catch (e) {
                console.error('localStorage set failed:', e);
                return false;
            }
        },

        // Get item from localStorage
        get: function(key, defaultValue = null) {
            try {
                const item = localStorage.getItem(key);
                return item ? JSON.parse(item) : defaultValue;
            } catch (e) {
                console.error('localStorage get failed:', e);
                return defaultValue;
            }
        },

        // Remove item from localStorage
        remove: function(key) {
            try {
                localStorage.removeItem(key);
                return true;
            } catch (e) {
                console.error('localStorage remove failed:', e);
                return false;
            }
        },

        // Clear all localStorage
        clear: function() {
            try {
                localStorage.clear();
                return true;
            } catch (e) {
                console.error('localStorage clear failed:', e);
                return false;
            }
        }
    }
};

// Global function aliases for backward compatibility
window.setButtonLoading = JainLearning.setButtonLoading.bind(JainLearning);
window.showLoadingOverlay = JainLearning.showLoadingOverlay.bind(JainLearning);
window.hideLoadingOverlay = JainLearning.hideLoadingOverlay.bind(JainLearning);
window.showAlert = JainLearning.showAlert.bind(JainLearning);

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    JainLearning.init();
});

// Service Worker registration for offline functionality
if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
        navigator.serviceWorker.register('/static/sw.js')
            .then(function(registration) {
                console.log('ServiceWorker registration successful');
            })
            .catch(function(err) {
                console.log('ServiceWorker registration failed: ', err);
            });
    });
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = JainLearning;
}