package env

import (
	"bufio"
	"bytes"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"

	"github.com/ghodss/yaml"
	"github.com/golang/protobuf/jsonpb"
	"github.com/ory/dockertest"
	sphere "github.com/pbarker/sphere/api/gen/go/v1alpha"
	"github.com/skratchdot/open-golang/open"
	"google.golang.org/grpc"
)

// Server of environments.
type Server struct {
	// Client to connect to the Sphere server.
	Client sphere.EnvironmentAPIClient
}

// ServerConfig is the environment server config.
type ServerConfig struct {
	// Docker image of environment.
	Image string
	// Version of the docker image.
	Version string
	// Port the environment is exposed on.
	Port string
}

// GymServerConfig is a configuration for a OpenAI Gym server environment.
var GymServerConfig = &ServerConfig{Image: "sphereproject/gym", Version: "latest", Port: "50051/tcp"}

// NewLocalServer creates a new environment server by launching a docker container and connecting to it.
func NewLocalServer(config *ServerConfig) (*Server, error) {
	pool, err := dockertest.NewPool("")
	if err != nil {
		return nil, fmt.Errorf("Could not connect to docker: %s", err)
	}

	resource, err := pool.Run(config.Image, config.Version, []string{})
	if err != nil {
		return nil, fmt.Errorf("Could not start resource: %s", err)
	}

	var sphereClient sphere.EnvironmentAPIClient

	// exponential backoff-retry, because the application in the container might
	// not be ready to accept connections yet
	if err := pool.Retry(func() error {
		var err error
		address := fmt.Sprintf("localhost:%s", resource.GetPort(config.Port))
		conn, err := grpc.Dial(address, grpc.WithInsecure(), grpc.WithBlock())
		if err != nil {
			return err
		}
		fmt.Println("connected!")
		sphereClient = sphere.NewEnvironmentAPIClient(conn)
		resp, err := sphereClient.Info(context.Background(), &sphere.Empty{})
		fmt.Println(resp)
		return err
	}); err != nil {
		log.Fatalf("Could not connect to docker: %s", err)
	}

	return &Server{
		Client: sphereClient,
	}, nil
}

// Env is a convienience environment wrapper.
type Env struct {
	*sphere.Environment
	// Client to connect to the Sphere server.
	Client sphere.EnvironmentAPIClient
	// VideoPaths of result videos downloadloaded from the server.
	VideoPaths []string
}

// Make an environment.
func (s *Server) Make(model string) (*Env, error) {
	ctx := context.Background()
	resp, err := s.Client.CreateEnv(ctx, &sphere.CreateEnvRequest{ModelName: model})
	if err != nil {
		return nil, err
	}
	env := resp.Environment
	fmt.Printf("created env: %s \n", env.Id)
	rresp, err := s.Client.StartRecordEnv(ctx, &sphere.StartRecordEnvRequest{Id: env.Id})
	if err != nil {
		return nil, err
	}
	fmt.Println(rresp.Message)
	return &Env{
		Environment: env,
		Client:      s.Client,
	}, nil
}

// Step through the environment.
func (e *Env) Step(value int) (*sphere.StepEnvResponse, error) {
	ctx := context.Background()
	resp, err := e.Client.StepEnv(ctx, &sphere.StepEnvRequest{Id: e.Id, Value: int32(value)})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// SampleAction returns a sample action for the environment.
func (e *Env) SampleAction() (int, error) {
	ctx := context.Background()
	resp, err := e.Client.SampleAction(ctx, &sphere.SampleActionRequest{Id: e.Id})
	if err != nil {
		return 0, err
	}
	return int(resp.Value), nil
}

// Reset the environment.
func (e *Env) Reset() (*sphere.Observation, error) {
	ctx := context.Background()
	resp, err := e.Client.ResetEnv(ctx, &sphere.ResetEnvRequest{Id: e.Id})
	if err != nil {
		return nil, err
	}
	return resp.Observation, nil
}

// Close the environment.
func (e *Env) Close() error {
	ctx := context.Background()
	resp, err := e.Client.DeleteEnv(ctx, &sphere.DeleteEnvRequest{Id: e.Id})
	if err != nil {
		return err
	}
	fmt.Println(resp.Message)
	return nil
}

// Results results for the environment.
func (e *Env) Results() (*sphere.ResultsResponse, error) {
	ctx := context.Background()
	resp, err := e.Client.Results(ctx, &sphere.ResultsRequest{Id: e.Id})
	if err != nil {
		return nil, err
	}
	return resp, nil
}

// PrintResults results for the environment.
func (e *Env) PrintResults() error {
	results, err := e.Results()
	if err != nil {
		return err
	}
	marshaller := &jsonpb.Marshaler{}
	var b bytes.Buffer
	err = marshaller.Marshal(&b, results)
	if err != nil {
		return err
	}
	yam, err := yaml.JSONToYAML(b.Bytes())
	if err != nil {
		return err
	}
	fmt.Println(string(yam))
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
		stream, err := e.Client.GetVideo(ctx, &sphere.GetVideoRequest{Id: e.Id, EpisodeId: video.EpisodeId})
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
	return videoPaths, nil
}

// End is a helper function that will close an environment and return the
// results and play any videos.
func (e *Env) End() {
	err := e.PrintResults()
	if err != nil {
		log.Fatal(err)
	}
	dir, err := ioutil.TempDir("", "sphere")
	if err != nil {
		log.Fatal(err)
	}
	videoPaths, err := e.Videos(dir)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("saved videos: ", videoPaths)
	err = e.Close()
	if err != nil {
		log.Fatal(err)
	}
}

// PlayAll videos stored locally.
func (e *Env) PlayAll() {
	for _, video := range e.VideoPaths {
		err := open.Run(video)
		if err != nil {
			log.Fatal(err)
		}
	}
	fmt.Print("press any key to remove videos or ctrl+c to exit and keep")
	input := bufio.NewScanner(os.Stdin)
	input.Scan()
	e.Clean()
}

// Clean any results/videos saved locally.
func (e *Env) Clean() {
	for _, videoPath := range e.VideoPaths {
		err := os.Remove(videoPath)
		if err != nil {
			log.Fatal(err)
		}
	}
}
