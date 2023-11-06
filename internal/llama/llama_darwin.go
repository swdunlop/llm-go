package llama

/*
#include <stdint.h>
*/
import "C"
import (
	"syscall"
)

func init() {
	n, err := syscall.SysctlUint32(`hw.perflevel0.logicalcpu`)
	if err != nil {
		return
	}
	if n > 0 {
		params.context.n_threads = C.uint32_t(n)
		nThreads = int(n)
	}
}
