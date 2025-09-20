// Load statistics on page load
function loadStats() {
    fetch('/stats')
        .then(response => response.json())
        .then(data => {
            const totalDocsElement = document.getElementById('total-docs');
            const vocabSizeElement = document.getElementById('vocab-size');
            
            if (totalDocsElement) {
                totalDocsElement.textContent = data.total_documents;
            }
            if (vocabSizeElement) {
                vocabSizeElement.textContent = data.vocabulary_size;
            }
        })
        .catch(error => {
            console.error('Error loading stats:', error);
            const totalDocsElement = document.getElementById('total-docs');
            const vocabSizeElement = document.getElementById('vocab-size');
            
            if (totalDocsElement) {
                totalDocsElement.textContent = 'Error';
            }
            if (vocabSizeElement) {
                vocabSizeElement.textContent = 'Error';
            }
        });
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    loadStats();
});

// Search form enhancement
function enhanceSearchForm() {
    const searchInput = document.querySelector('.search-input');
    if (searchInput) {
        searchInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                this.closest('form').submit();
            }
        });
    }
}

// Initialize search form enhancement
document.addEventListener('DOMContentLoaded', function() {
    enhanceSearchForm();
});
