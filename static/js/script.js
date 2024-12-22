document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('prediction-form');
    
    // Input validation
    const inputs = form.querySelectorAll('input[type="number"]');
    for (const input of inputs) {
        input.addEventListener('input', function() {
            if (this.value < 0) {
                this.value = 0;
            }
        });
    }

    // Form submission handling
    form.addEventListener('submit', function() {
        const loadingIndicator = document.createElement('div');
        loadingIndicator.className = 'loading-indicator';
        loadingIndicator.textContent = 'Calculating prediction...';
        
        const submitButton = form.querySelector('button[type="submit"]');
        submitButton.disabled = true;
        submitButton.parentNode.insertBefore(loadingIndicator, submitButton);
    });

    // Add thousand separators to number inputs
    function formatNumber(input) {
        let value = input.value.replace(/,/g, '');
        if (value !== '') {
            value = Number.parseFloat(value).toLocaleString('en-US', {
                maximumFractionDigits: 2
            });
            input.value = value;
        }
    }

    for (const input of inputs) {
        input.addEventListener('blur', function() {
            formatNumber(this);
        });
    }
});

// Function to update the prediction result dynamically
function updatePrediction(value) {
    const predictionElement = document.querySelector('.prediction-value');
    if (predictionElement) {
        predictionElement.textContent = value;
        predictionElement.classList.add('updated');
        setTimeout(() => {
            predictionElement.classList.remove('updated');
        }, 1000);
    }
}