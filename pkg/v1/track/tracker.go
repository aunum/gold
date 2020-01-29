package track

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/pbarker/log"
	g "gorgonia.org/gorgonia"
)

// Tracker is a means of tracking values on a graph.
type Tracker struct {
	values   map[string]TrackedValue
	episodes Episodes

	timestep int
	episode  int

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
	log.Infof("tracking data in %s", f.Name())
	encoder := json.NewEncoder(f)
	scanner := bufio.NewScanner(f)
	t := &Tracker{
		values:   map[string]TrackedValue{},
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
			log.Fatal(err)
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

	// TrackedValue of the value.
	TrackedValue float64 `json:"value"`

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
	for _, v := range t.values {
		vals = append(vals, v.Data(t.timestep, t.episode))
	}
	return &History{
		Values:   vals,
		Timestep: t.timestep,
		Episode:  t.episode,
	}
}

// TrackValue tracks a graph node or any other scalar value.
func (t *Tracker) TrackValue(name string, value interface{}, opts ...TrackedValueOpt) {
	if n, ok := value.(*g.Node); ok {
		tv := NewTrackedNodeValue(name, opts...)
		t.values[tv.name] = tv
		g.Read(n, &tv.value)
	} else {
		tv := NewTrackedScalarValue(name, value, opts...)
		t.values[tv.name] = tv
	}
}

// ValueNames is the name for all values.
func (t *Tracker) ValueNames() []string {
	names := map[string]string{}
	for name := range t.values {
		names[name] = name
	}
	for _, ep := range t.episodes {
		for name := range ep.Values {
			names[name] = name
		}
	}
	f := []string{}
	for name := range names {
		f = append(f, name)
	}
	return f
}

// IncValue increments a value by a scalar.
func (t *Tracker) IncValue(name string, scalar interface{}) error {
	v, err := t.GetValue(name)
	if err != nil {
		return err
	}
	val, ok := v.(*TrackedScalarValue)
	if !ok {
		return fmt.Errorf("can only increment tracked values")
	}
	val.value = v.Scalar() + toF64(scalar, 0)
	return nil
}

// ZeroValue zeros a scalar value
func (t *Tracker) ZeroValue(name string) error {
	v, err := t.GetValue(name)
	if err != nil {
		return err
	}
	val, ok := v.(*TrackedScalarValue)
	if !ok {
		return fmt.Errorf("can only increment tracked values")
	}
	val.value = 0.0
	return nil
}

func (t *Tracker) checkName(name string) {
	for _, val := range t.values {
		if val.Name() == name {
			log.Fatal("cannot track duplicate name: ", name)
		}
	}
}

// GetValue a tracked value by name.
func (t *Tracker) GetValue(name string) (TrackedValue, error) {
	for _, value := range t.values {
		if value.Name() == name {
			return value, nil
		}
	}
	for _, ep := range t.episodes {
		for _, value := range ep.Values {
			if value.Name() == name {
				return value, nil
			}
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
	for _, v := range t.values {
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
		log.Error(err)
	}
	v.Print()
}

// PrintAll values.
func (t *Tracker) PrintAll() {
	for _, value := range t.values {
		value.Print()
	}
}

// LogStep logs tracked values to store for the given timestep.
// TODO: make time more pluggable so this can be used in other environments.
func (t *Tracker) LogStep(episode, timestep int) error {
	t.episode = episode
	t.timestep = timestep
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
	log.Infoy("history", v)
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
