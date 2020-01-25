package agent

import (
	"bufio"
	"bytes"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"text/template"

	"github.com/phayes/freeport"

	"github.com/pbarker/go-rl/pkg/v1/track"
	"github.com/pbarker/logger"
	"github.com/skratchdot/open-golang/open"
)

// Base is an agents base functionality.
type Base struct {
	// Port agent is serving on.
	Port string

	// Tracker for the agent.
	Tracker *track.Tracker

	// address to the agent.
	address string
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

// NewBase returns a new base agent. Any errors will be fatal.
func NewBase(opts ...Opt) *Base {
	b := &Base{}
	for _, opt := range opts {
		opt(b)
	}
	if b.Tracker == nil {
		tracker, err := track.NewTracker()
		if err != nil {
			logger.Fatal(err)
		}
		b.Tracker = tracker
	}
	if b.Port == "" {
		port, err := freeport.GetFreePort()
		if err != nil {
			logger.Fatal(err)
		}
		b.Port = strconv.Itoa(port)
	}
	return b
}

// MakeEpisodes creates a set of episodes for training and stores the number for configuration.
func (b *Base) MakeEpisodes(num int) track.Episodes {
	logger.Infof("running for %d episodes", num)
	eps := b.Tracker.MakeEpisodes(num)
	return eps
}

// Serve the agent api/ui.
func (b *Base) Serve() {
	mux := http.NewServeMux()
	b.Tracker.ApplyHandlers(mux)
	b.ApplyHandlers(mux)
	b.address = fmt.Sprintf("http://localhost:%s", b.Port)
	logger.Infof("serving on %s", b.address)
	go http.ListenAndServe(fmt.Sprintf(":%s", b.Port), mux)
}

// View starts the local agent server and opens a browser to it.
func (b *Base) View() {
	b.Serve()
	err := open.Run(b.address)
	if err != nil {
		logger.Fatal(err)
	}
}

// Wait before exiting with a prompt.
func (b *Base) Wait() {
	fmt.Print("\npress enter to exit\n")
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
}

// ApplyHandlers adds the base handlers.
func (b *Base) ApplyHandlers(mux *http.ServeMux) {
	mux.HandleFunc("/", b.VisualizeHandler)
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
	logger.Info("value names: ", valueNames)
	templHelper := struct {
		ValueNames []string
		Port       string
	}{
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
		<link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.4.1/css/bootstrap.min.css" integrity="sha384-Vkoo8x4CGsO3+Hhxv8T/Q5PaXtkKtu6ug5TOeNV6gBiFeWPGFN9MuhOf23Q9Ifjh" crossorigin="anonymous">
	</head>
	<body>
	<ul class="nav">
		<li class="nav-item">
			<a class="nav-link" href="#">Overview</a>
		</li>
		<li class="nav-item">
			<a class="nav-link" href="/graphs">Graphs</a>
		</li>
  	</ul>

	{{ range $name := .ValueNames }}
	<div class="container">
	<canvas id="{{ $name}}" style="height:400px;width:400px"></canvas>
		<hr>
	</div>
	{{ end }}

	<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.4.1/jquery.min.js"></script>
	<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/2.6.0/Chart.bundle.js"></script>
	<script src="https://cdn.jsdelivr.net/npm/popper.js@1.16.0/dist/umd/popper.min.js" integrity="sha384-Q6E9RHvbIyZFJoft+2mJbHaEWldlvI9IOYy5n3zV9zzTtmI3UksdQRVvoxMfooAo" crossorigin="anonymous"></script>
	</body>
	{{ range $name := .ValueNames }}
	<script>
	var ctx_{{$name}}_live = document.getElementById("{{$name}}");
	var {{$name}}Chart = new Chart(ctx_{{$name}}_live, {
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
			text: "{{$name}}",
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
	var get{{$name}}Data = function() {
		$.ajax({
			url: "http://localhost:{{$.Port}}/api/values/{{$name}}",
			dataType: "json",
			success: function(response) {
				// {{$name}}Chart.data.labels.push("Post " + postId++);
				console.log("response")
				console.log(response)
			    // {{$name}}Chart.data.datasets[0].data.push(response);
				{{$name}}Chart.data.datasets[0].data = response;
				console.log({{$name}}Chart.data.datasets[0].data)
				
				// re-render the chart
				{{$name}}Chart.update();
			}
		});
	}
	get{{$name}}Data()
	setInterval(get{{$name}}Data, 1000);
	</script>
	{{end}}
</html>
`
