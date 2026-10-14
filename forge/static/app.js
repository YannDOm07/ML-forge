// ML Forge Dashboard - Client-side JS

document.addEventListener('DOMContentLoaded', () => {
    // Tab switching
    const tabs = document.querySelectorAll('#main-tabs .tab');
    const panels = document.querySelectorAll('.panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetTab = tab.dataset.tab;

            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            panels.forEach(p => {
                p.classList.toggle('active', p.id === `${targetTab}-panel`);
            });
        });
    });

    // Handle HTMX events for project detail expansion
    document.body.addEventListener('htmx:afterSwap', (e) => {
        // Auto-expand experiment detail rows when clicked
        if (e.detail.target.id === 'experiments-table') {
            setupExperimentRows();
        }
    });

    // Setup clickable experiment rows
    function setupExperimentRows() {
        document.querySelectorAll('.clickable-row').forEach(row => {
            row.addEventListener('click', (e) => {
                // Don't trigger if clicking on a button/link inside
                if (e.target.closest('button, a')) return;

                const expId = row.querySelector('code').textContent;
                const detailRow = document.getElementById(`exp-detail-${expId}`);
                const contentDiv = document.getElementById(`exp-detail-content-${expId}`);

                if (detailRow && detailRow.style.display !== 'table-row') {
                    // Load detail
                    detailRow.style.display = 'table-row';
                    contentDiv.innerHTML = '<em>Chargement...</em>';

                    fetch(`/api/projects/{{ project.id }}/experiments/${expId}`)
                        .then(r => r.json())
                        .then(data => renderExperimentDetail(contentDiv, data))
                        .catch(err => {
                            contentDiv.innerHTML = `<span class="badge badge-error">Erreur: ${err}</span>`;
                        });
                } else if (detailRow) {
                    detailRow.style.display = 'none';
                }
            });
        });
    }

    function renderExperimentDetail(container, exp) {
        const decisions = exp.decision ? `<span class="badge badge-${exp.decision}">${exp.decision}</span>` : '—';
        const score = exp.metric_mean !== null ? `${exp.metric_mean.toFixed(5)} ± ${exp.metric_std?.toFixed(5) || '?'}` : '—';
        const delta = exp.delta !== null ? exp.delta.toFixed(5) : '—';

        container.innerHTML = `
            <h4>Détails de ${exp.id}</h4>
            <div class="detail-row">
                <span class="detail-label">Méthode</span>
                <span class="detail-value">${exp.method}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Hypothèse</span>
                <span class="detail-value">${exp.hypothesis}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Statut</span>
                <span class="detail-value"><span class="badge badge-${exp.status}">${exp.status}</span></span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Score CV</span>
                <span class="detail-value">${score}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Delta vs ref</span>
                <span class="detail-value">${delta}</span>
            </div>
            <div class="detail-row">
                <span class="detail-label">Décision</span>
                <span class="detail-value">${decisions}</span>
            </div>
        `;
    }

    // Initial setup for any pre-loaded experiments
    setupExperimentRows();

    // Handle project detail loading
    document.body.addEventListener('htmx:beforeSwap', (e) => {
        if (e.detail.target.id === 'project-detail') {
            // Add a loading indicator
            e.detail.target.innerHTML = '<div class="detail-placeholder">Chargement...</div>';
        }
    });

    // Smooth scroll to project detail when a project is clicked
    document.body.addEventListener('htmx:afterSwap', (e) => {
        if (e.detail.target.id === 'project-detail') {
            e.detail.target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + R to refresh projects list
        if ((e.ctrlKey || e.metaKey) && e.key === 'r' && e.target === document.body) {
            e.preventDefault();
            document.querySelector('#projects-list')?.dispatchEvent(new Event('click'));
        }
    });
});

// Global error handler for HTMX
document.body.addEventListener('htmx:responseError', (e) => {
    console.error('HTMX Error:', e.detail);
    const target = e.detail.target;
    if (target) {
        target.innerHTML = `
            <div class="card" style="border-color: var(--danger);">
                <h3 style="color: var(--danger);">⚠️ Erreur</h3>
                <p>${e.detail.error || 'Une erreur est survenue'}</p>
                <button class="btn btn-primary" onclick="location.reload()">Recharger la page</button>
            </div>
        `;
    }
});

// Confirm before leaving if there are unsaved changes
let hasUnsavedChanges = false;

window.addEventListener('beforeunload', (e) => {
    if (hasUnsavedChanges) {
        e.preventDefault();
        e.returnValue = '';
    }
});

// Utility: format bytes
function formatBytes(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'Ko', 'Mo', 'Go', 'To'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// Utility: format date
function formatDate(isoString) {
    const date = new Date(isoString);
    return date.toLocaleDateString('fr-FR', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}