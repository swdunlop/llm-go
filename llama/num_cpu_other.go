//go:build !darwin && !aarch64

package llama

import "runtime"

func numCPU() int {
	return runtime.NumCPU()
}
