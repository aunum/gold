package agent

import (
	"bytes"
	"html/template"
	"net/http"

	g "gorgonia.org/gorgonia"

	"github.com/pbarker/go-rl/pkg/v1/track"
)

// Base is an agents base functionality.
type Base struct {
	// port agent is serving on.
	port string

	// Any graphs being used by the agent.
	graphs []*g.ExprGraph

	// tracker for the agent.
	tracker *track.Tracker
}

// Opt is an option for the base agent.
type Opt func(*Base)

// WithPort sets the port that the agent serves on.
func WithPort(port string) func(*Base) {
	return func(b *Base) {
		b.port = port
	}
}

// WithGraph sets any graphs being used by the agent
func WithGraph(graphs ...*g.ExprGraph) func(*Base) {
	return func(b *Base) {
		b.graphs = graphs
	}
}

// WithTracker sets the tracker being used by the agent
func WithTracker(tracker *track.Tracker) func(*Base) {
	return func(b *Base) {
		b.tracker = tracker
	}
}

// NewBase returns a new base agent.
func NewBase(opts ...Opt) *Base {
	b := &Base{}
	for _, opt := range opts {
		opt(b)
	}
	return b
}

// VisualizeHandler vizualizes the agent.
func (b *Base) VisualizeHandler(w http.ResponseWriter, req *http.Request) {
	t := template.New("data")
	p, err := t.Parse(visualizeTemplate)
	if err != nil {
		w.Write([]byte(err.Error()))
		w.WriteHeader(500)
	}
	var buf bytes.Buffer
	p.Execute(&buf, b)
	w.Write([]byte(buf.Bytes()))
	w.WriteHeader(200)
}

var visualizeTemplate = `
<!DOCTYPE html>
<html>
	<head>
		<script src="https://code.jquery.com/jquery-2.2.4.min.js"></script>
		<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.6.0/Chart.bundle.js"></script>
		<script>
		 null
		</script>
	</head>
	<body>



	<canvas id="canvas0" style="height:400px;width:400px"></canvas>
		<hr>


	</body>
	<script>
	$.ajax({
		url: "http://localhost:8081/chart.php",
		dataType: "json",
		success: function(response)
		{
			var ctx = document.getElementById("canvas0");
			var myBarChart = new Chart(ctx, {
				type: 'line',
				data: {
					labels: response.data.labels,
					datasets: [{
						label: 'Number of Active Customers',
						data: response.data.chartData,
						backgroundColor: [
							'rgba(75, 192, 192, 0.2)',
							'rgba(255, 99, 132, 0.2)'
						],
						borderColor: [
							'rgba(75, 192, 192, 1)',
							'rgba(255,99,132,1)'
						],
						borderWidth: 1
					}]
				}
			});
		}
	});
	Chart.defaults.line.cubicInterpolationMode = 'monotone';
	Chart.defaults.global.animation.duration = 0;
	var charts = []

		var ctx = document.getElementById("canvas0").getContext("2d");
		var chart = new Chart(ctx, {"type":"line","label":"test-chart","data":{"datasets":[{"backgroundColor":"rgba(102, 194, 165, 0.863)","borderColor":"rgba(250, 141, 98, 0.863)","borderWidth":0,"label":"sin(x)","fill":false,"lineTension":0,"pointBorderWidth":4,"pointRadius":10,"pointHitRadius":0,"pointHoverRadius":0,"pointHoverBorderWidth":0,"yAxisID":"yaxis0","data":[{"x":0.00,"y":0.00},{"x":0.10,"y":0.10},{"x":0.20,"y":0.20},{"x":0.30,"y":0.30},{"x":0.40,"y":0.39},{"x":0.50,"y":0.48},{"x":0.60,"y":0.56},{"x":0.70,"y":0.64},{"x":0.80,"y":0.72},{"x":0.90,"y":0.78},{"x":1.00,"y":0.84},{"x":1.10,"y":0.89},{"x":1.20,"y":0.93},{"x":1.30,"y":0.96},{"x":1.40,"y":0.99},{"x":1.50,"y":1.00},{"x":1.60,"y":1.00},{"x":1.70,"y":0.99},{"x":1.80,"y":0.97},{"x":1.90,"y":0.95},{"x":2.00,"y":0.91},{"x":2.10,"y":0.86},{"x":2.20,"y":0.81},{"x":2.30,"y":0.75},{"x":2.40,"y":0.68},{"x":2.50,"y":0.60},{"x":2.60,"y":0.52},{"x":2.70,"y":0.43},{"x":2.80,"y":0.33},{"x":2.90,"y":0.24},{"x":3.00,"y":0.14},{"x":3.10,"y":0.04},{"x":3.20,"y":-0.06},{"x":3.30,"y":-0.16},{"x":3.40,"y":-0.26},{"x":3.50,"y":-0.35},{"x":3.60,"y":-0.44},{"x":3.70,"y":-0.53},{"x":3.80,"y":-0.61},{"x":3.90,"y":-0.69},{"x":4.00,"y":-0.76},{"x":4.10,"y":-0.82},{"x":4.20,"y":-0.87},{"x":4.30,"y":-0.92},{"x":4.40,"y":-0.95},{"x":4.50,"y":-0.98},{"x":4.60,"y":-0.99},{"x":4.70,"y":-1.00},{"x":4.80,"y":-1.00},{"x":4.90,"y":-0.98},{"x":5.00,"y":-0.96},{"x":5.10,"y":-0.93},{"x":5.20,"y":-0.88},{"x":5.30,"y":-0.83},{"x":5.40,"y":-0.77},{"x":5.50,"y":-0.71},{"x":5.60,"y":-0.63},{"x":5.70,"y":-0.55},{"x":5.80,"y":-0.46},{"x":5.90,"y":-0.37},{"x":6.00,"y":-0.28},{"x":6.10,"y":-0.18},{"x":6.20,"y":-0.08},{"x":6.30,"y":0.02},{"x":6.40,"y":0.12},{"x":6.50,"y":0.22},{"x":6.60,"y":0.31},{"x":6.70,"y":0.40},{"x":6.80,"y":0.49},{"x":6.90,"y":0.58},{"x":7.00,"y":0.66},{"x":7.10,"y":0.73},{"x":7.20,"y":0.79},{"x":7.30,"y":0.85},{"x":7.40,"y":0.90},{"x":7.50,"y":0.94},{"x":7.60,"y":0.97},{"x":7.70,"y":0.99},{"x":7.80,"y":1.00},{"x":7.90,"y":1.00},{"x":8.00,"y":0.99},{"x":8.10,"y":0.97},{"x":8.20,"y":0.94},{"x":8.30,"y":0.90},{"x":8.40,"y":0.85},{"x":8.50,"y":0.80},{"x":8.60,"y":0.73},{"x":8.70,"y":0.66},{"x":8.80,"y":0.58},{"x":8.90,"y":0.50},{"x":9.00,"y":0.41}]},{"borderColor":"rgba(230, 138, 195, 0.863)","borderWidth":8,"label":"3*cos(2*x)","fill":false,"lineTension":0,"pointBorderWidth":0,"pointRadius":0,"pointHitRadius":0,"pointHoverRadius":0,"pointHoverBorderWidth":0,"pointStyle":"star","yAxisID":"yaxis1","data":[{"x":0.00,"y":3.00},{"x":0.10,"y":2.94},{"x":0.20,"y":2.76},{"x":0.30,"y":2.48},{"x":0.40,"y":2.09},{"x":0.50,"y":1.62},{"x":0.60,"y":1.09},{"x":0.70,"y":0.51},{"x":0.80,"y":-0.09},{"x":0.90,"y":-0.68},{"x":1.00,"y":-1.25},{"x":1.10,"y":-1.77},{"x":1.20,"y":-2.21},{"x":1.30,"y":-2.57},{"x":1.40,"y":-2.83},{"x":1.50,"y":-2.97},{"x":1.60,"y":-2.99},{"x":1.70,"y":-2.90},{"x":1.80,"y":-2.69},{"x":1.90,"y":-2.37},{"x":2.00,"y":-1.96},{"x":2.10,"y":-1.47},{"x":2.20,"y":-0.92},{"x":2.30,"y":-0.34},{"x":2.40,"y":0.26},{"x":2.50,"y":0.85},{"x":2.60,"y":1.41},{"x":2.70,"y":1.90},{"x":2.80,"y":2.33},{"x":2.90,"y":2.66},{"x":3.00,"y":2.88},{"x":3.10,"y":2.99},{"x":3.20,"y":2.98},{"x":3.30,"y":2.85},{"x":3.40,"y":2.61},{"x":3.50,"y":2.26},{"x":3.60,"y":1.83},{"x":3.70,"y":1.32},{"x":3.80,"y":0.75},{"x":3.90,"y":0.16},{"x":4.00,"y":-0.44},{"x":4.10,"y":-1.02},{"x":4.20,"y":-1.56},{"x":4.30,"y":-2.04},{"x":4.40,"y":-2.43},{"x":4.50,"y":-2.73},{"x":4.60,"y":-2.92},{"x":4.70,"y":-3.00},{"x":4.80,"y":-2.95},{"x":4.90,"y":-2.79},{"x":5.00,"y":-2.52},{"x":5.10,"y":-2.14},{"x":5.20,"y":-1.68},{"x":5.30,"y":-1.16},{"x":5.40,"y":-0.58},{"x":5.50,"y":0.01},{"x":5.60,"y":0.61},{"x":5.70,"y":1.18},{"x":5.80,"y":1.70},{"x":5.90,"y":2.16},{"x":6.00,"y":2.53},{"x":6.10,"y":2.80},{"x":6.20,"y":2.96},{"x":6.30,"y":3.00},{"x":6.40,"y":2.92},{"x":6.50,"y":2.72},{"x":6.60,"y":2.42},{"x":6.70,"y":2.02},{"x":6.80,"y":1.54},{"x":6.90,"y":0.99},{"x":7.00,"y":0.41},{"x":7.10,"y":-0.19},{"x":7.20,"y":-0.78},{"x":7.30,"y":-1.34},{"x":7.40,"y":-1.85},{"x":7.50,"y":-2.28},{"x":7.60,"y":-2.62},{"x":7.70,"y":-2.86},{"x":7.80,"y":-2.98},{"x":7.90,"y":-2.99},{"x":8.00,"y":-2.87},{"x":8.10,"y":-2.64},{"x":8.20,"y":-2.31},{"x":8.30,"y":-1.88},{"x":8.40,"y":-1.38},{"x":8.50,"y":-0.83},{"x":8.60,"y":-0.24},{"x":8.70,"y":0.36},{"x":8.80,"y":0.95},{"x":8.90,"y":1.49},{"x":9.00,"y":1.98}]}],"labels":null},"options":{"responsive":false,"scales":{"xAxes":[{"type":"linear","position":"bottom","id":"xaxis0","scaleLabel":{"display":true,"labelString":"X","fontSize":22}}],"yAxes":[{"type":"linear","position":"left","id":"yaxis0","scaleLabel":{"display":true,"labelString":"sin(x)"}},{"type":"linear","position":"right","id":"yaxis1","scaleLabel":{"display":true,"labelString":"3*cos(2*x)"}}]}}});
		charts.push(chart)

	""
	</script>
</html>
`