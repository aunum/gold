module github.com/pbarker/go-rl

go 1.13

require (
	github.com/awalterschulze/gographviz v0.0.0-20190522210029-fa59802746ab // indirect
	github.com/chewxy/math32 v1.0.4
	github.com/gammazero/deque v0.0.0-20200124200322-7e84b94275b8
	github.com/golang/protobuf v1.3.3 // indirect
	github.com/k0kubun/pp v3.0.1+incompatible
	github.com/leesper/go_rng v0.0.0-20190531154944-a612b043e353 // indirect
	github.com/mattn/go-isatty v0.0.12 // indirect
	github.com/ory/dockertest v3.3.5+incompatible
	github.com/pbarker/log v0.0.0-20200203210754-97f67228913d
	github.com/pbarker/sphere v0.0.0-20200218163301-89282f667248
	github.com/phayes/freeport v0.0.0-20180830031419-95f893ade6f2
	github.com/pkg/errors v0.9.1
	github.com/schwarmco/go-cartesian-product v0.0.0-20180515110546-d5ee747a6dc9
	github.com/skratchdot/open-golang v0.0.0-20200116055534-eef842397966
	github.com/stretchr/testify v1.4.0
	golang.org/x/exp v0.0.0-20200207192155-f17229e696bd
	golang.org/x/sys v0.0.0-20200202164722-d101bd2416d5 // indirect
	golang.org/x/tools v0.0.0-20200207224406-61798d64f025 // indirect
	gonum.org/v1/gonum v0.6.2
	gonum.org/v1/plot v0.0.0-20200111075622-4abb28f724d5
	google.golang.org/grpc v1.27.0
	gopkg.in/yaml.v2 v2.2.8 // indirect
	gorgonia.org/cu v0.9.2 // indirect
	gorgonia.org/gorgonia v0.9.8
	gorgonia.org/tensor v0.9.4
)

replace gorgonia.org/gorgonia => github.com/pbarker/gorgonia v0.0.0-20200209225806-37febf40acbc
