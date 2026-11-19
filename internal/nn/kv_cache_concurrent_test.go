package nn

import (
	"fmt"
	"sync"
	"testing"

	"m3tal-llm/internal/tensor"
)

// TestKVCacheConcurrentIsolation models one cache per request: several
// goroutines build, update, inspect and reset their own caches in parallel.
// Run with -race; a shared, mutable cache would show up as a data race or a
// cross-request value.
func TestKVCacheConcurrentIsolation(t *testing.T) {
	const (
		workers  = 8
		steps    = 4
		numLayer = 2
	)
	var wg sync.WaitGroup
	failures := make(chan error, workers)

	for worker := 0; worker < workers; worker++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			cache, err := NewKVCache(numLayer, 1, 2, 4, 16, tensor.Float32)
			if err != nil {
				failures <- err
				return
			}
			k, err := tensor.New([]int{1, 2, 1, 4}, tensor.Float32)
			if err != nil {
				failures <- err
				return
			}
			v, err := tensor.New([]int{1, 2, 1, 4}, tensor.Float32)
			if err != nil {
				failures <- err
				return
			}
			if err := k.SetFloat32At(float32(id), 0, 0, 0, 0); err != nil {
				failures <- err
				return
			}

			for step := 0; step < steps; step++ {
				for layer := 0; layer < numLayer; layer++ {
					if _, _, err := cache.Update(layer, k, v); err != nil {
						failures <- err
						return
					}
				}
			}
			if got := cache.SeqLen(); got != steps {
				failures <- fmt.Errorf("worker %d: seq len %d, want %d", id, got, steps)
				return
			}
			for layer := 0; layer < numLayer; layer++ {
				if got := cache.LayerSeqLen(layer); got != steps {
					failures <- fmt.Errorf("worker %d layer %d: seq len %d, want %d", id, layer, got, steps)
					return
				}
				cached, _, err := cache.Get(layer)
				if err != nil {
					failures <- err
					return
				}
				first, err := cached.Float32At(0, 0, 0, 0)
				if err != nil {
					failures <- err
					return
				}
				if first != float32(id) {
					failures <- fmt.Errorf("worker %d layer %d: cache value %v, want its own %d", id, layer, first, id)
					return
				}
			}

			cache.Reset()
			if got := cache.SeqLen(); got != 0 {
				failures <- fmt.Errorf("worker %d: seq len after reset %d, want 0", id, got)
			}
		}(worker)
	}

	wg.Wait()
	close(failures)
	for err := range failures {
		t.Error(err)
	}
}
