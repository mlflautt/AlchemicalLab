// AlchemicalLab SynthLab Web Visualization Client

class CAVisualizer {
    constructor() {
        this.isPlaying = false;
        this.stepData = [];
        this.maxHistory = 100;

        // Canvas contexts
        this.aliveCtx = document.getElementById('aliveCanvas').getContext('2d');
        this.energyCtx = document.getElementById('energyCanvas').getContext('2d');
        this.speciesCtx = document.getElementById('speciesCanvas').getContext('2d');
        this.resourcesCtx = document.getElementById('resourcesCanvas').getContext('2d');

        // Chart SVGs
        this.populationSvg = d3.select('#populationChart');
        this.energySvg = d3.select('#energyChart');

        // Set up event listeners
        this.setupEventListeners();

        // Start data fetching
        this.startDataLoop();
    }

    setupEventListeners() {
        document.getElementById('playPause').addEventListener('click', () => {
            this.isPlaying = !this.isPlaying;
            document.getElementById('playPause').textContent = this.isPlaying ? 'Pause' : 'Play';
        });

        document.getElementById('reset').addEventListener('click', () => {
            this.resetSimulation();
        });
    }

    async startDataLoop() {
        while (true) {
            if (this.isPlaying) {
                await this.fetchData();
            }
            await new Promise(resolve => setTimeout(resolve, 100)); // 10 FPS
        }
    }

    async fetchData() {
        try {
            const response = await fetch('/data');
            const data = await response.json();
            this.updateVisualization(data);
        } catch (error) {
            console.error('Error fetching data:', error);
        }
    }

    updateVisualization(data) {
        // Update step counter
        document.getElementById('stepCounter').textContent = `Step: ${data.stats.step || 0}`;

        // Update statistics
        this.updateStats(data.stats);

        // Update grids
        this.updateGrids(data.grids, data.grid_size);

        // Update charts
        this.updateCharts(data.stats);
    }

    updateStats(stats) {
        document.getElementById('aliveCount').textContent = stats.alive_count || 0;
        document.getElementById('avgEnergy').textContent = (stats.avg_energy || 0).toFixed(3);
        document.getElementById('speciesDiversity').textContent = stats.species_diversity || 0;
        document.getElementById('avgAge').textContent = (stats.avg_age || 0).toFixed(2);
        document.getElementById('avgHealth').textContent = (stats.avg_health || 0).toFixed(3);
        document.getElementById('avgWealth').textContent = (stats.avg_wealth || 0).toFixed(3);
        document.getElementById('avgTech').textContent = (stats.avg_tech || 0).toFixed(3);
        document.getElementById('totalResources').textContent = (stats.total_resources || 0).toFixed(1);
    }

    updateGrids(grids, gridSize) {
        const [height, width] = gridSize;

        // Update alive grid
        this.drawGrid(this.aliveCtx, grids.alive, width, height, (val) => val ? [0, 255, 0, 255] : [50, 50, 50, 255]);

        // Update energy grid
        this.drawGrid(this.energyCtx, grids.energy, width, height, (val) => {
            const intensity = Math.floor(val * 255);
            return [intensity, intensity, 255, 255];
        });

        // Update species grid
        this.drawGrid(this.speciesCtx, grids.species, width, height, (val) => {
            const hue = (val * 137.5) % 360; // Golden angle for distinct colors
            return this.hslToRgb(hue / 360, 0.7, 0.5);
        });

        // Update resources grid
        this.drawGrid(this.resourcesCtx, grids.resources, width, height, (val) => {
            const intensity = Math.floor(val * 255);
            return [255, 255 - intensity, 0, 255];
        });
    }

    drawGrid(ctx, data, width, height, colorFunc) {
        const canvas = ctx.canvas;
        const cellWidth = canvas.width / width;
        const cellHeight = canvas.height / height;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const val = data[y][x];
                const color = colorFunc(val);

                ctx.fillStyle = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3] / 255})`;
                ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
            }
        }
    }

    hslToRgb(h, s, l) {
        let r, g, b;
        if (s === 0) {
            r = g = b = l; // achromatic
        } else {
            const hue2rgb = (p, q, t) => {
                if (t < 0) t += 1;
                if (t > 1) t -= 1;
                if (t < 1/6) return p + (q - p) * 6 * t;
                if (t < 1/2) return q;
                if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                return p;
            };
            const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
            const p = 2 * l - q;
            r = hue2rgb(p, q, h + 1/3);
            g = hue2rgb(p, q, h);
            b = hue2rgb(p, q, h - 1/3);
        }
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255), 255];
    }

    updateCharts(stats) {
        // Add current stats to history
        this.stepData.push(stats);
        if (this.stepData.length > this.maxHistory) {
            this.stepData.shift();
        }

        // Update population chart
        this.updateLineChart(this.populationSvg, this.stepData, 'alive_count', 'Population');

        // Update energy chart
        this.updateLineChart(this.energySvg, this.stepData, 'avg_energy', 'Energy');
    }

    updateLineChart(svg, data, key, title) {
        const margin = {top: 20, right: 20, bottom: 30, left: 50};
        const width = +svg.attr('width') - margin.left - margin.right;
        const height = +svg.attr('height') - margin.top - margin.bottom;

        svg.selectAll('*').remove();

        const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`);

        const x = d3.scaleLinear()
            .domain([0, data.length - 1])
            .range([0, width]);

        const y = d3.scaleLinear()
            .domain([0, d3.max(data, d => d[key] || 0) * 1.1])
            .range([height, 0]);

        // Add X axis
        g.append('g')
            .attr('transform', `translate(0,${height})`)
            .call(d3.axisBottom(x).ticks(5));

        // Add Y axis
        g.append('g')
            .call(d3.axisLeft(y).ticks(5));

        // Add line
        const line = d3.line()
            .x((d, i) => x(i))
            .y(d => y(d[key] || 0));

        g.append('path')
            .datum(data)
            .attr('fill', 'none')
            .attr('stroke', '#00d4ff')
            .attr('stroke-width', 2)
            .attr('d', line);
    }

    async resetSimulation() {
        // For now, just reload the page to restart
        window.location.reload();
    }
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    new CAVisualizer();
});