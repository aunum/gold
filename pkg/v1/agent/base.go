// Package agent provides the agent implementations and base tooling.
package agent

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/aunum/gold/pkg/v1/ui"
	"github.com/aunum/gold/pkg/v1/ui/sse"

	"github.com/phayes/freeport"

	"github.com/aunum/gold/pkg/v1/common"
	envv1 "github.com/aunum/gold/pkg/v1/env"
	"github.com/aunum/gold/pkg/v1/track"
	"github.com/aunum/log"
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

	address string
	broker  *sse.Broker

	noTracker bool
	noServer  bool
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
		b.Logger.Fatalf(`could not open browser; err: %v
			see github.com/skratchdot/open-golang 
			if running in remote terminal consider removing the View() command from the agent`, err)
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
func (b *Base) ApplyHandlers(mux *http.ServeMux) error {
	b.Tracker.ApplyHandlers(mux)
	err := ui.ApplyHandlers(mux)
	if err != nil {
		return err
	}
	mux.HandleFunc("/live", b.broker.Handler)
	mux.HandleFunc("/info", b.InfoHandler)
	return nil
}

// InfoHandler returns info about the agent.
func (b *Base) InfoHandler(w http.ResponseWriter, req *http.Request) {
	info := struct {
		Name string `json:"name"`
	}{
		Name: b.Name,
	}
	bts, err := json.Marshal(info)
	if err != nil {
		w.WriteHeader(500)
		w.Write([]byte(err.Error()))
	}
	w.WriteHeader(200)
	w.Write(bts)
}
