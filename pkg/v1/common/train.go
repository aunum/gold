package common

// TrainLoop is a generic training loop
func TrainLoop(episodes, timesteps int, episodeFunc, timestepFunc func()) {
	for ep := 0; ep <= episodes; ep++ {
		if episodeFunc != nil {
			episodeFunc()
		}
		for ts := 0; ts <= timesteps; ts++ {
			if timestepFunc != nil {
				timestepFunc()
			}
		}
	}
}
