package agent

import (
	"bufio"
	"bytes"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"text/template"
	"time"

	"github.com/phayes/freeport"

	"github.com/pbarker/go-rl/pkg/v1/common"
	"github.com/pbarker/go-rl/pkg/v1/common/sse"
	envv1 "github.com/pbarker/go-rl/pkg/v1/env"
	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/log"
	"github.com/skratchdot/open-golang/open"
)

// Base is an agents base functionality.
type Base struct {
	// Name of the agent.
	Name string

	// Port agent is serving on.
	Port string

	// Tracker for the agent.
	Tracker *track.Tracker

	// Logger for the agent.
	Logger *log.Logger

	address   string
	noTracker bool
	noServer  bool
	broker    *sse.Broker
}

// Opt is an option for the base agent.
type Opt func(*Base)

// WithPort sets the port that the agent serves on.
func WithPort(port string) func(*Base) {
	return func(b *Base) {
		b.Port = port
	}
}

// WithTracker sets the tracker being used by the agent
func WithTracker(tracker *track.Tracker) func(*Base) {
	return func(b *Base) {
		b.Tracker = tracker
	}
}

// WithoutTracker prevents tracker from being created.
func WithoutTracker() func(*Base) {
	return func(b *Base) {
		b.noTracker = true
	}
}

// WithLogger adds a logger to the base.
func WithLogger(logger *log.Logger) func(*Base) {
	return func(b *Base) {
		b.Logger = logger
	}
}

// WithoutServer will prevent the provisioning of a server for the agent.
func WithoutServer() func(*Base) {
	return func(b *Base) {
		b.noServer = true
	}
}

// NewBase returns a new base agent. Any errors will be fatal.
func NewBase(name string, opts ...Opt) *Base {
	b := &Base{Name: name}
	for _, opt := range opts {
		opt(b)
	}
	if b.Logger == nil {
		b.Logger = log.DefaultLogger
	}
	if b.Tracker == nil && !b.noTracker {
		tracker, err := track.NewTracker(track.WithLogger(b.Logger))
		if err != nil {
			log.Fatal(err)
		}
		b.Tracker = tracker
	}
	if !b.noServer {
		if b.Port == "" {
			var port int

			// Note: this can panic https://github.com/phayes/freeport/issues/5
			err := common.Retry(10, time.Millisecond*1, func() (err error) {
				defer func() {
					if r := recover(); r != nil {
						err = fmt.Errorf("caught freeport panic: %v", err)
					}
				}()
				port, err = freeport.GetFreePort()
				return err
			})
			if err != nil {
				log.Fatal(err)
			}
			b.Port = strconv.Itoa(port)
		}
		b.broker = sse.NewBroker()
	}
	return b
}

// MakeEpisodes creates a set of episodes for training and stores the number for configuration.
func (b *Base) MakeEpisodes(num int) track.Episodes {
	b.Logger.Infof("running for %d episodes", num)
	eps := b.Tracker.MakeEpisodes(num)
	return eps
}

// Serve the agent api/ui.
func (b *Base) Serve() {
	if b.noServer {
		b.Logger.Fatal("trying to serve an agent that was created with WithoutServer option")
	}
	mux := http.NewServeMux()
	b.Tracker.ApplyHandlers(mux)
	b.ApplyHandlers(mux)
	b.address = fmt.Sprintf("http://localhost:%s", b.Port)
	b.Logger.Infof("serving agent api/ui on %s", b.address)
	go http.ListenAndServe(fmt.Sprintf(":%s", b.Port), mux)
}

// View starts the local agent server and opens a browser to it.
func (b *Base) View() {
	b.Serve()
	err := open.Run(b.address)
	if err != nil {
		b.Logger.Fatal(err)
	}
}

// Wait before exiting with a prompt.
func (b *Base) Wait() {
	fmt.Print("\npress enter to exit\n")
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
}

// Render the given data to the ui.
func (b *Base) Render(env *envv1.Env) error {
	frame, err := env.Render()
	if err != nil {
		return err
	}
	b.broker.Notifier <- frame.Data
	return nil
}

// ApplyHandlers adds the base handlers.
func (b *Base) ApplyHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/", b.VisualizeHandler)
	mux.HandleFunc("/live", b.broker.Handler)
}

// VisualizeHandler vizualizes the agent.
func (b *Base) VisualizeHandler(w http.ResponseWriter, req *http.Request) {
	h, err := b.execTmpl()
	if err != nil {
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
	}
	w.WriteHeader(200)
	w.Write(h)
}

func (b *Base) execTmpl() ([]byte, error) {
	t := template.New("data")
	p, err := t.Parse(visualizeTemplate)
	if err != nil {
		return nil, err
	}
	valueNames := b.Tracker.ValueNames()
	b.Logger.Debugv("value names", valueNames)
	templHelper := struct {
		Name       string
		ValueNames []string
		Port       string
	}{
		Name:       b.Name,
		ValueNames: valueNames,
		Port:       b.Port,
	}
	var buf bytes.Buffer
	p.Execute(&buf, templHelper)
	return buf.Bytes(), nil
}

var visualizeTemplate = `
<!doctype html>
<html lang="en">
	<head>
		<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
		<title>{{.Name}} agent</title>
		<link rel="icon" href="https://avatars1.githubusercontent.com/u/17137938?s=400&v=4">
		<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
	</head>
	<body>
	<ul class="nav">
		<li class="nav-item">
			<a class="nav-link" href="#">Dashboard</a>
		</li>
	  </ul>
	<div class="text-center">
		<img id="live" style="height:400px;width:600px" class="img-fluid"/>
	</div>

	<div class="d-flex justify-content-around flex-wrap" id="metrics">
	</div>

	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.6.0/Chart.bundle.js"></script>
	<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
	</body>
	<script>
		source = "http://localhost:{{.Port}}/live"
		var client = new EventSource(source)
		client.onmessage = function (msg) {
			var metaData = "data:image/jpeg;base64,";
			document.getElementById("live").src = metaData + msg.data;
		}
	</script>
	<script>
		function buildChart(value) {
			var ctx_live = document.getElementById(value);
			var chart = new Chart(ctx_live, {
				type: 'line',
				data: {
				labels: [],
				datasets: [{
					data: [],
					borderWidth: 1,
					borderColor:'#00c0ef',
					fill: false,
				}]
				},
				options: {
				responsive: true,
				title: {
					display: true,
					text: value,
				},
				legend: {
					display: false
				},
				scales: {
					yAxes: [{
						type: 'linear',
						ticks: {
							beginAtZero: true,
					}
					}],
					xAxes: [{
						type: 'linear',
						ticks: {
						beginAtZero: true,
						},
						scaleLabel: {
							display: true,
							labelString: 'Episode'
						}
					}]
				}
				}
			});
			var getData = function() {
				$.ajax({
					url: "http://localhost:{{$.Port}}/api/values/"+value,
					dataType: "json",
					success: function(response) {
						chart.data.datasets[0].data = response.xys;
						chart.options.scales.xAxes[0].scaleLabel.labelString = response.xLabel;

						// re-render the chart
						chart.update();
					}
				});
			};
			getData();
			setInterval(getData, 1000);
		};


		var getValues = function() {
			$.ajax({
				url: "http://localhost:{{$.Port}}/api/values",
				dataType: "json",
				success: function(response) {
					values = window.localStorage.getItem('values');
					console.log('response');
					console.log(response);
					console.log('values');
					console.log(values);

					window.localStorage.setItem('values', response);
					if(!response) {
						console.log('returning')
						return
					};
					if(!values) {
						values = []
					};
					
					newValues = response.filter(e => !values.includes(e))
					console.log('new values')
					console.log(newValues)
					
					if (newValues) {
						var body = document.getElementById("metrics");
						for (value of newValues) {
							console.log('adding value: ' + value)
							var div = document.createElement("div");
							div.id = value + 'Holder';
							div.class = "p-2"

							var canvas = document.createElement('canvas');
							canvas.id = value;
							canvas.width = 400;
							canvas.height = 400;

							div.appendChild(canvas);

							body.appendChild(div);

							buildChart(value)
						};
					};
				}
			});
		}
		getValues()
		setInterval(getValues, 2000);
	</script>
</html>
`

// var visualizeTemplate = `
// <!doctype html>
// <html lang="en">
// 	<head>
// 		<meta name="viewport" content="width=device-width, initial-scale=1, shrink-to-fit=no">
// 		<title>{{.Name}} agent</title>
// 		<link rel="icon" href="https://avatars1.githubusercontent.com/u/17137938?s=400&v=4">
// 		<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
// 	</head>
// 	<body>
// 	<ul class="nav">
// 		<li class="nav-item">
// 			<a class="nav-link" href="#">Dashboard</a>
// 		</li>
// 	  </ul>
// 	<div class="text-center">
// 		<img id="live" style="height:400px;width:600px" class="img-fluid"/>
// 	</div>
// 	{{ range $name := .ValueNames }}
// 	<div class="container">
// 	<canvas id="{{ $name}}" style="height:400px;width:400px"></canvas>
// 		<hr>
// 	</div>
// 	{{ end }}

// 	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
// 	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.6.0/Chart.bundle.js"></script>
// 	<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
// 	</body>
// 	<script>
// 	source = "http://localhost:{{.Port}}/live"
// 	var client = new EventSource(source)
// 	client.onmessage = function (msg) {
// 		var metaData = "data:image/jpeg;base64,";
//         document.getElementById("live").src = metaData + msg.data;
// 	}
// 	</script>
// 	<script>

// 	function arraysEqual(_arr1, _arr2) {
// 		if (!Array.isArray(_arr1) || ! Array.isArray(_arr2) || _arr1.length !== _arr2.length)
// 		return false;

// 		var arr1 = _arr1.concat().sort();
// 		var arr2 = _arr2.concat().sort();

// 		for (var i = 0; i < arr1.length; i++) {
// 			if (arr1[i] !== arr2[i])
// 				return false;
// 		}

// 		return true;
// 	};

// 	var getValues = function() {
// 		$.ajax({
// 			url: "http://localhost:{{$.Port}}/api/values",
// 			dataType: "json",
// 			success: function(response) {
// 				values = window.localStorage.getItem('values');
// 				console.log('response');
// 				console.log(response);
// 				console.log('values');
// 				console.log(values);
// 				if(!values) {
// 					console.log('setting values')
// 					window.localStorage.setItem('values', response);
// 					return
// 				};
// 				if(!arraysEqual(response, values)) {
// 					console.log('reloading window')
// 					location.reload();
// 				};
// 				window.localStorage.setItem('values', response);
// 			}
// 		});
// 	}
// 	getValues()
// 	setInterval(getValues, 2000);
// 	</script>
// 	{{ range $name := .ValueNames }}
// 	<script>
// 	var ctx_{{$name}}_live = document.getElementById("{{$name}}");
// 	var {{$name}}Chart = new Chart(ctx_{{$name}}_live, {
// 		type: 'line',
// 		data: {
// 		  labels: [],
// 		  datasets: [{
// 			data: [],
// 			borderWidth: 1,
// 			borderColor:'#00c0ef',
// 			fill: false,
// 		  }]
// 		},
// 		options: {
// 		  responsive: true,
// 		  title: {
// 			display: true,
// 			text: "{{$name}}",
// 		  },
// 		  legend: {
// 			display: false
// 		  },
// 		  scales: {
// 			yAxes: [{
// 				type: 'linear',
// 				ticks: {
// 					beginAtZero: true,
// 			  }
// 			}],
// 			xAxes: [{
// 				type: 'linear',
// 				ticks: {
// 				  beginAtZero: true,
// 				},
// 				scaleLabel: {
// 					display: true,
// 					labelString: 'Episode'
// 				}
// 			  }]
// 		  }
// 		}
// 	});
// 	var get{{$name}}Data = function() {
// 		$.ajax({
// 			url: "http://localhost:{{$.Port}}/api/values/{{$name}}",
// 			dataType: "json",
// 			success: function(response) {
// 				{{$name}}Chart.data.datasets[0].data = response.xys;
// 				// console.log({{$name}}Chart.data.datasets[0].data)
// 				{{$name}}Chart.options.scales.xAxes[0].scaleLabel.labelString = response.xLabel;

// 				// re-render the chart
// 				{{$name}}Chart.update();
// 			}
// 		});
// 	}
// 	get{{$name}}Data()
// 	setInterval(get{{$name}}Data, 1000);
// 	</script>
// 	{{end}}
// </html>
// `
