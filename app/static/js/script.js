function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
}

function showAccountDetails() {
    document.getElementById('account-details').style.display = 'block';
}

window.onload = function() {
    const modal = new bootstrap.Modal(document.getElementById('welcomeModal'));
    modal.show();
}

function toggleSidebar() {
    const sidebar = document.querySelector('.sidebar');
    const mainContent = document.querySelector('.main-content');
    sidebar.classList.toggle('collapsed');
    mainContent.classList.toggle('expanded');
}

function showAccountDetails() {
    document.getElementById('account-details').style.display = 'block';
}

window.onload = function() {
    const modal = new bootstrap.Modal(document.getElementById('welcomeModal'));
    modal.show();

    // Apply hover background colors
    document.querySelectorAll('.glass-card').forEach(card => {
        const hoverBg = card.getAttribute('data-hover-bg');
        card.style.setProperty('--hover-bg', hoverBg);
    });
}