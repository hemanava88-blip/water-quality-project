document.addEventListener('DOMContentLoaded', () => {

    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.add('active');
            
            // Plotly container native resize triggers
            if (targetId === 'tab-viz') {
                setTimeout(() => {
                    Plotly.Plots.resize(document.getElementById('plot-bar'));
                    Plotly.Plots.resize(document.getElementById('plot-pie'));
                }, 50);
            }
            if (targetId === 'tab-comparison') {
                setTimeout(() => {
                    Plotly.Plots.resize(document.getElementById('plot-model-comp'));
                }, 50);
            }
        });
    });

    const initPlotly = () => {
        // Tab 7 Visualization
        const barDiv = document.getElementById('plot-bar');
        const barData = [{
            x: ['pH', 'DO', 'BOD', 'COD', 'Ammonia'],
            y: [6.8, 4.2, 12.0, 45.0, 1.8],
            type: 'bar',
            marker: { color: '#3b82f6' },
            text: ['6.8', '4.2', '12.0', '45.0', '1.8'],
            textposition: 'auto'
        }];
        const barLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, l: 40, r: 20, b: 40 },
            font: { color: '#94a3b8' },
            title: 'Water Parameter Analysis'
        };
        Plotly.newPlot(barDiv, barData, barLayout, { responsive: true });

        const pieDiv = document.getElementById('plot-pie');
        const pieData = [{
            values: [1200, 800],
            labels: ['Good Water', 'Bad Water'],
            type: 'pie',
            textinfo: 'label+percent',
            marker: { colors: ['#10b981', '#ef4444'] }
        }];
        const pieLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, l: 20, r: 20, b: 20 },
            font: { color: '#94a3b8' }
        };
        Plotly.newPlot(pieDiv, pieData, pieLayout, { responsive: true });

        // Tab 8 & 9 Model Metrics
        const models = [
            { id: 'rf', name: 'Random Forest', acc: 0.92 },
            { id: 'svm', name: 'SVM', acc: 0.88 },
            { id: 'knn', name: 'KNN', acc: 0.85 }
        ];

        document.getElementById('metrics-target').innerHTML = models.map(m => 
            `<div class="metric"><span>${m.id}_acc</span><strong>${(m.acc*100).toFixed(1)}%</strong></div>`
        ).join('');

        models.forEach(m => {
            document.getElementById(`acc-${m.id}`).innerText = `${(m.acc*100).toFixed(1)}%`;
        });

        const bestModel = models.reduce((a, b) => a.acc > b.acc ? a : b);
        document.getElementById('best-model-banner').innerHTML = `🏆 Best Model: ${bestModel.name} with accuracy ${(bestModel.acc*100).toFixed(1)}%`;
        document.getElementById(`row-${bestModel.id}`).classList.add('winner-row');

        // Comparison Bar Chart
        const compDiv = document.getElementById('plot-model-comp');
        const compData = [{
            x: models.map(m => m.name),
            y: models.map(m => m.acc * 100),
            type: 'bar',
            marker: { color: ['#10b981', '#3b82f6', '#f59e0b'] },
            text: models.map(m => `${(m.acc*100).toFixed(1)}%`),
            textposition: 'auto'
        }];
        const compLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            margin: { t: 20, l: 40, r: 20, b: 40 },
            font: { color: '#94a3b8' },
            yaxis: { title: 'Accuracy %', range: [60, 100] }
        };
        Plotly.newPlot(compDiv, compData, compLayout, { responsive: true });
    };

    setTimeout(initPlotly, 200);
});
