package llama

import (
	"runtime"
	"strconv"
	"syscall"
)

func numCPU() int {
	//NOTE(swdunlop): we do not want to use so many threads that part of the model runs on the efficiency cores
	// so we need to use a sysctl to sniff out the number of physical cores.
	//
	// As of this writing, this is just a Mac M1 / M2 thing, but if other big.LITTLE architectures are supported,
	// we'll want to add them, too.
	str, err := syscall.Sysctl("hw.perflevel0.physicalcpu")
	if err == nil {
		n, err := strconv.Atoi(str)
		if err == nil {
			if n > 0 {
				return n
			}
		}
	}

	// fallback to the number of logical CPUs minus 2
	n := runtime.NumCPU() - 2 // assume 2 efficiency cores
	if n < 1 {
		n = 1
	}
	return n
}
