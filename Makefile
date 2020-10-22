
static:
	cd pkg/v1/ui && rice embed-go

demo:
	go run ./pkg/v1/agent/deepq/experiments/cartpole/main.go