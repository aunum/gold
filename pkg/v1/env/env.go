// Package env provides a wrapper for interacting with Sphere as well as normalization tools.
package env

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/aunum/log"

	"github.com/aunum/gold/pkg/v1/common/num"

	spherev1alpha "github.com/aunum/sphere/api/gen/go/v1alpha"
	"github.com/skratchdot/open-golang/open"
	"gorgonia.org/tensor"
)

// Env is a convienience environment wrapper.
type Env struct {
	*spherev1alpha.Environment

	// Client to connect to the Sphere server.
	Client spherev1alpha.EnvironmentAPIClient

	// VideoPaths of result videos downloadloaded from the server.
	VideoPaths []string

	// Normalizer normalizes observation data.
	Normalizer Normalizer

	// GoalNormalizer normalizes goal data.
	GoalNormalizer Normalizer

	logger    *log.Logger
	recording bool
	wrappers  []*spherev1alpha.EnvWrapper
	reshape   []int
}

// Opt is an environment option.
type Opt func(*Env)

// Make an environment.
func (s *Server) Make(model string, opts ...Opt) (*Env, error) {
	e := &Env{
		Client: s.Client,
		logger: s.logger,
	}
	for _, opt := range opts {
		opt(e)
	}
	ctx := context.Background()
	resp, err := s.Client.CreateEnv(ctx, &spherev1alpha.CreateEnvRequest{ModelName: model, Wrappers: e.wrappers})
	if err != nil {
		return nil, err
	}
	env := resp.Environment
	s.logger.Successf("created env: %s", env.Id)
	e.Environment = env

	if e.Normalizer != nil {
		err = e.Normalizer.Init(e)
		if err != nil {
			return nil, err
		}
	}
	if e.GoalNormalizer != nil {
		err = e.GoalNormalizer.Init(e)
		if err != nil {
			return nil, err
		}
	}

	if e.recording {
		resp, err := e.Client.StartRecordEnv(ctx, &spherev1alpha.StartRecordEnvRequest{Id: e.Environment.Id})
		if err != nil {
			return nil, err
		}
		e.logger.Success(resp.Message)
	}
	return e, nil
}

// WithRecorder adds a recorder to the environment
func WithRecorder() func(*Env) {
	return func(e *Env) {
		e.recording = true
	}
}

// WithNormalizer adds a normalizer for observation data.
func WithNormalizer(normalizer Normalizer) func(*Env) {
	return func(e *Env) {
		e.Normalizer = normalizer
	}
}

// WithGoalNormalizer adds a normalizer for goal data.
func WithGoalNormalizer(normalizer Normalizer) func(*Env) {
	return func(e *Env) {
		e.GoalNormalizer = normalizer
	}
}

// WithWrapper adds an environment wrapper.
func WithWrapper(wrapper spherev1alpha.IsEnvWrapper) func(*Env) {
	return func(e *Env) {
		e.wrappers = append(e.wrappers, &spherev1alpha.EnvWrapper{Wrapper: wrapper})
	}
}

// WithLogger adds a logger to the env.
func WithLogger(logger *log.Logger) func(*Env) {
	return func(e *Env) {
		e.logger = logger
	}
}

// DefaultAtariWrapper is the default deepmind atari wrapper.
var DefaultAtariWrapper = &spherev1alpha.EnvWrapper_DeepmindAtariWrapper{
	DeepmindAtariWrapper: &spherev1alpha.DeepmindAtariWrapper{EpisodeLife: true, ClipRewards: true},
}

// Outcome of taking an action.
type Outcome struct {
	// Observation of the current state.
	Observation *tensor.Dense

	// Action that was taken
	Action int

	// Reward from action.
	Reward float32

	// Whether the environment is done.
	Done bool
}

// Step through the environment.
func (e *Env) Step(value int) (*Outcome, error) {
	ctx := context.Background()
	resp, err := e.Client.StepEnv(ctx, &spherev1alpha.StepEnvRequest{Id: e.Id, Action: int32(value)})
	if err != nil {
		return nil, err
	}
	observation := resp.Observation.Dense()
	if e.Normalizer != nil {
		observation, err = e.Normalizer.Norm(observation)
		if err != nil {
			return nil, err
		}
	}
	return &Outcome{observation, value, resp.Reward, resp.Done}, nil
}

// SampleAction returns a sample action for the environment.
func (e *Env) SampleAction() (int, error) {
	ctx := context.Background()
	resp, err := e.Client.SampleAction(ctx, &spherev1alpha.SampleActionRequest{Id: e.Id})
	if err != nil {
		return 0, err
	}
	return int(resp.Value), nil
}

// Render the environment.
// TODO: should maybe be a stream.
func (e *Env) Render() (*spherev1alpha.Image, error) {
	ctx := context.Background()
	resp, err := e.Client.RenderEnv(ctx, &spherev1alpha.RenderEnvRequest{Id: e.Id})
	if err != nil {
		return nil, err
	}
	return resp.Frame, nil
}

// InitialState of the environment.
type InitialState struct {
	// Observation of the environment.
	Observation *tensor.Dense

	// Goal if present.
	Goal *tensor.Dense
}

// Reset the environment.
func (e *Env) Reset() (init *InitialState, err error) {
	ctx := context.Background()
	resp, err := e.Client.ResetEnv(ctx, &spherev1alpha.ResetEnvRequest{Id: e.Id})
	if err != nil {
		return nil, err
	}
	observation := resp.Observation.Dense()
	var goal *tensor.Dense
	if resp.GetGoal().Data != nil {
		goal = resp.GetGoal().Dense()
	}
	if e.Normalizer != nil {
		observation, err = e.Normalizer.Norm(observation)
		if err != nil {
			return nil, err
		}
	}
	if e.GoalNormalizer != nil {
		if goal != nil {
			goal, err = e.GoalNormalizer.Norm(goal)
			if err != nil {
				return nil, err
			}
		}
	}
	return &InitialState{Observation: observation, Goal: goal}, nil
}

// Close the environment.
func (e *Env) Close() error {
	ctx := context.Background()
	resp, err := e.Client.DeleteEnv(ctx, &spherev1alpha.DeleteEnvRequest{Id: e.Id})
	if err != nil {
		return err
	}
	e.logger.Success(resp.Message)
	return nil
}

// Results from an environment run.
type Results struct {
	// Episodes is a map of episode id to result.
	Episodes map[int32]*spherev1alpha.EpisodeResult

	// Videos is a map of episode id to result.
	Videos map[int32]*spherev1alpha.Video

	// AverageReward is the average reward of the episodes.
	AverageReward float32
}

// Results results for the environment.
func (e *Env) Results() (*Results, error) {
	ctx := context.Background()
	resp, err := e.Client.Results(ctx, &spherev1alpha.ResultsRequest{Id: e.Id})
	if err != nil {
		return nil, err
	}
	var cumulative float32
	for _, res := range resp.EpisodeResults {
		cumulative += res.Reward
	}
	avg := cumulative / float32(len(resp.EpisodeResults))
	res := &Results{
		Episodes:      resp.EpisodeResults,
		Videos:        resp.Videos,
		AverageReward: avg,
	}
	return res, nil
}

// PrintResults results for the environment.
func (e *Env) PrintResults() error {
	results, err := e.Results()
	if err != nil {
		return err
	}
	e.logger.Infoy("results", results)
	e.logger.Infov("avg reward", results.AverageReward)
	return nil
}

// Videos saves all the videos for the environment episodes to the given path.
// Defaults to current directory. Returns an array of video paths.
func (e *Env) Videos(path string) ([]string, error) {
	if path == "" {
		path = fmt.Sprintf("./results/%s", e.Id)
	}
	ctx := context.Background()
	results, err := e.Results()
	if err != nil {
		return nil, err
	}
	videoPaths := []string{}
	for _, video := range results.Videos {
		stream, err := e.Client.GetVideo(ctx, &spherev1alpha.GetVideoRequest{Id: e.Id, EpisodeId: video.EpisodeId})
		if err != nil {
			return nil, err
		}
		fp := filepath.Join(path, fmt.Sprintf("%s-episode%d.mp4", e.Id, video.EpisodeId))
		f, err := os.Create(fp)
		if err != nil {
			return nil, err
		}
		defer f.Close()

		for {
			resp, err := stream.Recv()
			if err == io.EOF {
				err := stream.CloseSend()
				if err != nil {
					return nil, err
				}
				break
			}
			if err != nil {
				return nil, err
			}
			_, err = f.Write(resp.Chunk)
			if err != nil {
				return nil, err
			}
		}
		videoPaths = append(videoPaths, fp)
	}
	e.VideoPaths = videoPaths
	return videoPaths, nil
}

// End is a helper function that will close an environment and return the
// results and play any videos.
func (e *Env) End() {
	if e.recording {
		err := e.PrintResults()
		if err != nil {
			e.logger.Fatal(err)
		}
		dir, err := ioutil.TempDir("", "sphere")
		if err != nil {
			e.logger.Fatal(err)
		}
		videoPaths, err := e.Videos(dir)
		if err != nil {
			e.logger.Fatal(err)
		}
		e.logger.Successy("saved videos", videoPaths)
	}
	err := e.Close()
	if err != nil {
		e.logger.Fatal(err)
	}
}

// PlayAll videos stored locally.
func (e *Env) PlayAll() {
	if !e.recording {
		log.Fatal("no recordings present, use the WithRecorder() option to record.")
	}
	for _, video := range e.VideoPaths {
		e.logger.Debugf("playing video: %s", video)
		err := open.Run(video)
		if err != nil {
			e.logger.Fatal(err)
		}
	}
	fmt.Print("\npress any key to remove videos or ctrl+c to exit and keep\n")
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
	e.Clean()
}

// Clean any results/videos saved locally.
func (e *Env) Clean() {
	for _, videoPath := range e.VideoPaths {
		err := os.Remove(videoPath)
		if err != nil {
			e.logger.Fatal(err)
		}
		e.logger.Debugf("removed video: %s", videoPath)
	}
	e.logger.Success("removed all local videos")
}

// MaxSteps that can be taken per episode.
func (e *Env) MaxSteps() int {
	return int(e.MaxEpisodeSteps)
}

// ActionSpaceShape is the shape of the action space.
// TODO: should this be in the API of off the generated code?
func (e *Env) ActionSpaceShape() []int {
	return SpaceShape(e.ActionSpace)
}

// ObservationSpaceShape is the shape of the observation space.
func (e *Env) ObservationSpaceShape() []int {
	if len(e.reshape) != 0 {
		return e.reshape
	}
	return SpaceShape(e.ObservationSpace)
}

// SpaceShape return the shape of the given space.
func SpaceShape(space *spherev1alpha.Space) []int {
	shape := []int{}
	switch s := space.GetInfo().(type) {
	case *spherev1alpha.Space_Box:
		shape = num.I32SliceToI(s.Box.GetShape())
	case *spherev1alpha.Space_Discrete:
		shape = []int{1}
	case *spherev1alpha.Space_MultiDiscrete:
		shape = []int{len(s.MultiDiscrete.DiscreteSpaces)}
	case *spherev1alpha.Space_MultiBinary:
		shape = []int{int(s.MultiBinary.GetN())}
	case *spherev1alpha.Space_StructSpace:
		log.Fatalf("struct space not supported")
	default:
		log.Fatalf("unknown action space type: %v", space)
	}
	if len(shape) == 0 {
		log.Fatalf("space had no shape: %v", space)
	}
	return shape
}

// PotentialsShape is an overloaded method that will return a dense tensor of potentials for a given space.
func PotentialsShape(space *spherev1alpha.Space) []int {
	shape := []int{}
	switch s := space.GetInfo().(type) {
	case *spherev1alpha.Space_Box:
		shape = num.I32SliceToI(s.Box.GetShape())
	case *spherev1alpha.Space_Discrete:
		shape = []int{int(s.Discrete.N)}
	case *spherev1alpha.Space_MultiDiscrete:
		shape = num.I32SliceToI(s.MultiDiscrete.DiscreteSpaces)
	case *spherev1alpha.Space_MultiBinary:
		shape = []int{int(s.MultiBinary.N)}
	case *spherev1alpha.Space_StructSpace:
		log.Fatalf("struct space not supported")
	default:
		log.Fatalf("unknown action space type: %v", space)
	}
	if len(shape) == 0 {
		log.Fatalf("space had no shape: %v", space)
	}
	return shape
}

// BoxSpace is a helper for box spaces that converts the values to dense tensors.
// TODO: make proto plugin to do this automagically (protoc-gen-tensor)
type BoxSpace struct {
	// High values for this space.
	High *tensor.Dense

	// Low values for this space.
	Low *tensor.Dense

	// Shape of the space.
	Shape []int
}

// BoxSpace returns the box space as dense tensors.
func (e *Env) BoxSpace() (*BoxSpace, error) {
	space := e.GetObservationSpace()

	if sp := space.GetBox(); sp != nil {
		shape := []int{}
		for _, i := range sp.GetShape() {
			shape = append(shape, int(i))
		}
		return &BoxSpace{
			High:  tensor.New(tensor.WithShape(shape...), tensor.WithBacking(sp.GetHigh())),
			Low:   tensor.New(tensor.WithShape(shape...), tensor.WithBacking(sp.GetLow())),
			Shape: shape,
		}, nil
	}
	return nil, fmt.Errorf("env is not a box space: %+v", space)
}

// Print a YAML representation of the environment.
func (e *Env) Print() {
	e.logger.Infoy("environment", e.Environment)
}
