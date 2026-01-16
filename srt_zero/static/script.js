document.addEventListener('DOMContentLoaded', () => {
    initWaterfall();
    initTicker();
});

function initWaterfall() {
    const steps = document.querySelectorAll('.waterfall-step');
    steps.forEach((step, index) => {
        // Stagger animations
        step.style.animationDelay = `${index * 0.3}s`;
    });
}

function initTicker() {
    const tickerEl = document.getElementById('prob-ticker');
    if (!tickerEl) return;

    // Get current probability from storage or default to 1.0
    let currentProb = parseFloat(localStorage.getItem('srt_cumulative_prob')) || 1.0;

    // Get the deviation of the current particle to "multiply" the coincidence
    // We'll extract it from the DOM or use a default "hit" value if it's a good match
    const deviationEl = document.querySelector('.metric-val.good');
    
    if (deviationEl) {
        // If we have a good match (green), we multiply by a small probability factor
        // simulating the "unlikelihood" of this match happening by chance.
        // For a 10^-6 match, we might say p=0.01 for visual effect per visit
        // Realistically it's much lower, but we want a nice visual decay.
        const matchQuality = 0.05; 
        
        // Only update if we haven't visited this specific particle in this session (optional complexity)
        // For now, just decay on every load to show the effect
        currentProb = currentProb * matchQuality;
        
        // Cap at extremely small number
        if (currentProb < 1e-50) currentProb = 1e-50;
        
        localStorage.setItem('srt_cumulative_prob', currentProb);
    }

    // Format in scientific notation
    tickerEl.textContent = currentProb.toExponential(2);
}

function toggleAltruxian() {
    const layer = document.getElementById('altruxian-layer');
    layer.classList.toggle('visible');
}
