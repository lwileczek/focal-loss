// mycharts.js
// Data for Chart JS to make simple charts in presentation


// {{{ Constants
const CHART_COLORS = {
  red: 'rgb(255, 99, 132)',
  orange: 'rgb(255, 159, 64)',
  yellow: 'rgb(255, 205, 86)',
  green: 'rgb(75, 192, 192)',
  blue: 'rgb(54, 162, 235)',
  purple: 'rgb(153, 102, 255)',
  grey: 'rgb(201, 203, 207)'
};

const BACKGROUND_COLORS = {
  red: 'rgba(255, 99, 132, 0.5)',
  orange: 'rgba(255, 159, 64, 0.5)',
  yellow: 'rgba(255, 205, 86, 0.5)',
  green: 'rgba(75, 192, 192, 0.5)',
  blue: 'rgba(54, 162, 235, 0.5)',
  purple: 'rgba(153, 102, 255, 0.5)',
  grey: 'rgba(201, 203, 207, 0.5)'
};

let _seed = Date.now();

// }}}

// {{{ Functions
let valueOrDefault = (x, defaultValue) => {
  x = (x === undefined) ? defaultValue: x;
  return x;
}

function srand(seed) {
  _seed = seed;
}

function rand(min, max) {
  min = valueOrDefault(min, 0);
  max = valueOrDefault(max, 0);
  _seed = (_seed * 9301 + 49297) % 233280;
  return min + (_seed / 233280) * (max - min);
}

function numbers(config) {
  var cfg = config || {};
  var min = valueOrDefault(cfg.min, 0);
  var max = valueOrDefault(cfg.max, 100);
  var from = valueOrDefault(cfg.from, []);
  var count = valueOrDefault(cfg.count, 8);
  var decimals = valueOrDefault(cfg.decimals, 8);
  var continuity = valueOrDefault(cfg.continuity, 1);
  var dfactor = Math.pow(10, decimals) || 0;
  var data = [];
  var i, value;

  for (i = 0; i < count; ++i) {
    value = (from[i] || 0) + this.rand(min, max);
    if (this.rand() <= continuity) {
      data.push(Math.round(dfactor * value) / dfactor);
    } else {
      data.push(null);
    }
  }

  return data;
}

function points(config) {
  const xs = this.numbers(config);
  const ys = this.numbers(config);
  return xs.map((x, i) => ({x, y: ys[i]}));
}

function bubbles(config) {
  return this.points(config).map(pt => {
    pt.r = this.rand(config.rmin, config.rmax);
    return pt;
  });
}
// }}}

// {{{  Data
const majorityCFG= {count: 20, rmin: 1.5, rmax: 1.5, min: 20, max: 70};
const minorityCFG= {count: 5, rmin: 1.5, rmax: 1.5, min: 0, max: 40};

const data = {
  datasets: [
    {
      label: 'majority',
      data: bubbles(majorityCFG),
      borderColor: CHART_COLORS.blue,
      backgroundColor: BACKGROUND_COLORS.blue,
    },
    {
      label: 'Minority',
      data: bubbles(minorityCFG),
      borderColor: CHART_COLORS.orange,
      backgroundColor: BACKGROUND_COLORS.orange,
    }
  ]
};

// }}}

// {{{  Configurations
const scatterConfig = {
  type: 'scatter',
  data: data,
  options: {
    responsive: false,
    width: 700,
    height: 400,
    plugins: {
      legend: {
        position: 'top',
      },
      title: {
        display: true,
        text: 'Imbalanced Dataset'
      }
    }
  },
};

const trendConfig = {
    data: {
        datasets: [{
            type: 'bar',
            label: 'Bar Dataset',
            data: [10, 20, 30, 40]
        }, {
            type: 'line',
            label: 'Line Dataset',
            data: [50, 50, 50, 50],
        }],
        labels: ['January', 'February', 'March', 'April']
    },
    options: {
    responsive: false,
    width: 700,
    height: 400,
    scales: {
      y: {
        beginAtZero: true
      }
    }
  }
};
// }}}

// {{{ Contexts
const scatter = document.getElementById('scatterChart').getContext('2d');
const trend = document.getElementById('trendChart').getContext('2d');

// }}}

const c0 = new Chart(scatter, scatterConfig)
const c1 = new Chart(trend, trendConfig)

// vim: foldmethod=marker
