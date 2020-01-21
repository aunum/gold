package track

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/pbarker/logger"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	Values []Value

	Timestep int
	Episode  int

	encoder  *json.Encoder
	f        *os.File
	scanner  *bufio.Scanner
	filePath string
}

// TrackerOpt is a tracker option.
type TrackerOpt func(*Tracker)

// NewTracker returns a new tracker for a graph.
func NewTracker(opts ...TrackerOpt) (*Tracker, error) {
	f, err := ioutil.TempFile("", "stats.*.json")
	if err != nil {
		return nil, err
	}
	logger.Infof("tracking data in %s", f.Name())
	encoder := json.NewEncoder(f)
	scanner := bufio.NewScanner(f)
	t := &Tracker{
		Values:   []Value{},
		encoder:  encoder,
		f:        f,
		scanner:  scanner,
		filePath: f.Name(),
	}
	return t, nil
}

// WithDir is a tracker option to set the directory in which logs are stored.
func WithDir(dir string) func(*Tracker) {
	return func(t *Tracker) {
		f, err := ioutil.TempFile(dir, "stats")
		if err != nil {
			logger.Fatal(err)
		}
		encoder := json.NewEncoder(f)
		t.encoder = encoder
		t.filePath = f.Name()
	}
}

// History is the historical representation of a set of tracked values.
type History struct {
	// Values in the history.
	Values []*HistoricalValue `json:"values"`

	// Timestep of this history.
	Timestep int `json:"timestep"`

	// Episode of this history.
	Episode int `json:"episode"`
}

// Histories are a slice of history.
type Histories []*History

// Get the value history with the given name.
func (h *History) Get(name string) HistoricalValues {
	history := HistoricalValues{}
	for _, value := range h.Values {
		if value.Name == name {
			history = append(history, value)
		}
	}
	return history
}

// HistoricalValue is a historical value.
type HistoricalValue struct {
	// Name of the value.
	Name string `json:"name"`

	// Value of the value.
	Value float64 `json:"value"`

	// Timestep at which the value occured.
	Timestep int `json:"timestep"`

	// Episode at which the value occured.
	Episode int `json:"episode"`
}

// HistoricalValues is a slice of historical values.
type HistoricalValues []*HistoricalValue

// Data yeilds the current tracked values into a historical structure.
func (t *Tracker) Data() *History {
	vals := []*HistoricalValue{}
	for _, v := range t.Values {
		vals = append(vals, v.Data(t.Timestep, t.Episode))
	}
	return &History{
		Values:   vals,
		Timestep: t.Timestep,
		Episode:  t.Episode,
	}
}

// IncTS increments the timestep of the tracker.
func (t *Tracker) IncTS() {
	t.Timestep++
}

// IncEpisode increments the epeisode of the tracker.
func (t *Tracker) IncEpisode() {
	t.Episode++
}

// TrackValue tracks a value. A value can be a graph node or any other value.
func (t *Tracker) TrackValue(name string, value interface{}, index ...int) {
	i := 0
	if len(index) != 0 {
		i = index[0]
	}
	t.checkName(name)
	if n, ok := value.(*g.Node); ok {
		tv := NewTrackedNodeValue(name, i)
		t.Values = append(t.Values, tv)
		g.Read(n, &tv.value)
	} else {
		tv := NewTrackedValue(name, value, i)
		t.Values = append(t.Values, tv)
	}
}

func (t *Tracker) checkName(name string) {
	for _, val := range t.Values {
		if val.Name() == name {
			logger.Fatal("cannot track duplicate name: ", name)
		}
	}
}

// GetValue a tracked value by name.
func (t *Tracker) GetValue(name string) (Value, error) {
	for _, value := range t.Values {
		if value.Name() == name {
			return value, nil
		}
	}
	return nil, fmt.Errorf("%q value does not exist", name)
}

// GetHistory gets all the history for a value.
func (t *Tracker) GetHistory(name string) (HistoricalValues, error) {
	history := HistoricalValues{}
	f, err := os.Open(t.filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		b := scanner.Bytes()

		var h History
		err := json.Unmarshal(b, &h)
		if err != nil {
			return nil, err
		}
		vh := h.Get(name)
		history = append(history, vh...)
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return history, nil
}

// GetHistoryAll gets the history of all values.
func (t *Tracker) GetHistoryAll() ([]HistoricalValues, error) {
	all := []HistoricalValues{}
	for _, v := range t.Values {
		vh, err := t.GetHistory(v.Name())
		if err != nil {
			return nil, err
		}
		all = append(all, vh)
	}
	return all, nil
}

// PrintValue prints a value.
func (t *Tracker) PrintValue(name string) {
	v, err := t.GetValue(name)
	if err != nil {
		logger.Error(err)
	}
	v.Print()
}

// PrintAll values.
func (t *Tracker) PrintAll() {
	for _, value := range t.Values {
		value.Print()
	}
}

// Log tracked values to store.
func (t *Tracker) Log(episode, timestep int) error {
	t.Episode = episode
	t.Timestep = timestep
	return t.encoder.Encode(t.Data())
}

// PrintHistoryAll prints the history of all values.
func (t *Tracker) PrintHistoryAll() error {
	vals, err := t.GetHistoryAll()
	if err != nil {
		return err
	}
	for _, val := range vals {
		val.Print()
	}
	return nil
}

// Print the tracked values.
func (v HistoricalValues) Print() {
	logger.Infoy("history", v)
}

// GetEpisodeHistories returns the episode history.
func (t *Tracker) GetEpisodeHistories() (EpisodeHistories, error) {
	epHistories := EpisodeHistories{}
	f, err := os.Open(t.filePath)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		b := scanner.Bytes()

		var h History
		err := json.Unmarshal(b, &h)
		if err != nil {
			return nil, err
		}

		epHistory, ok := epHistories[h.Episode]
		if !ok {
			epHistory = Histories{}
		}
		epHistory = append(epHistory, &h)
		epHistories[h.Episode] = epHistory
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return epHistories, nil
}
