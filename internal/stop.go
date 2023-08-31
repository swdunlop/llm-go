package internal

import "bytes"

// NewStopFilter constructs a stop filter with the provided set of case sensitive stop strings.
func NewStopFilter(stops ...string) *StopFilter {
	stopBytes := make([][]byte, 0, len(stops))
	for _, stop := range stops {
		if stop == "" {
			continue
		}
		stopBytes = append(stopBytes, []byte(stop))
	}
	return &StopFilter{stopBytes, make([]byte, 0, 16384)}
}

// A StopFilter buffers content that may contain one of the stops provided to its constructor.
type StopFilter struct {
	stops  [][]byte
	buffer []byte
}

// Filter will append content to its internal buffer and return the portion of the buffer that cannot contain any of
// its stops.  Filter will return true if the buffer contains any of its stops, false otherwise.  Once a stop has been
// fully matched, the stop filter buffer will be emptied.
func (f *StopFilter) Filter(content string) (string, bool) {
	if content == "" {
		return "", false
	}
	f.buffer = append(f.buffer, content...)
	n := len(f.buffer)
	// for each possible start position in the buffer..
	for i := 0; i < n; i++ {
		m := n - i
		// for each stop we are looking for..
		for _, stop := range f.stops {
			// if the stop is longer than the buffer, then the match can only be partial.
			partial := len(stop) > m
			if partial {
				stop = stop[:m]
			}

			if !bytes.HasPrefix(f.buffer[i:][:m], stop) {
				continue
			}
			if !partial {
				content := string(f.buffer[:i])
				f.buffer = f.buffer[0:0:cap(f.buffer)]
				return content, true
			}
			// we have a partial match, but we need to keep going until we find a full match.
			for _, stop := range f.stops {
				ix := bytes.Index(f.buffer[i:], stop)
				if ix < 0 {
					continue
				}
				content := string(f.buffer[:i+ix])
				f.buffer = f.buffer[0:0:cap(f.buffer)]
				return content, true
			}
			content := string(f.buffer[:i])
			f.buffer = f.buffer[i:]
			return content, false
		}
	}
	content = string(f.buffer)
	f.buffer = f.buffer[0:0:cap(f.buffer)]
	return content, false
}

// String returns the content of the internal buffer, which might partially match one of the stops.
func (f *StopFilter) String() string {
	return string(f.buffer)
}
