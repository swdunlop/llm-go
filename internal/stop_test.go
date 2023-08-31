package internal

import (
	"strings"
	"testing"
)

func TestStopFilter(t *testing.T) {
	for _, test := range []struct {
		Name   string // test name
		Stops  string // space separated list of stops
		Input  string // space separated series of input fragments
		Output string // expected output
		Match  bool   // expected match
		Buffer string // buffered content
	}{
		{"allEmpty", "", "", "", false, ""},
		{"stopsEmpty", "", "abc", "abc", false, ""},
		{"inputEmpty", "abc", "", "", false, ""},
		{"noMatch", "abc", "def", "def", false, ""},
		{"headMatch", "abc", "abc", "", true, ""},
		{"tailMatch", "abc", "defabc", "def", true, ""},
		{"partialMatch", "abc", "ab", "", false, "ab"},
		{"midMatch", "abc", "defabcghi", "def", true, ""},
		{"falseMatch", "abc", "defab_ghi", "defab_ghi", false, ""},
	} {
		t.Run(test.Name, func(t *testing.T) {
			filter := NewStopFilter(strings.Split(test.Stops, " ")...)
			output := make([]string, 0, 16)
			for _, input := range strings.Split(test.Input, " ") {
				str, stop := filter.Filter(input)
				output = append(output, str)
				if stop {
					break
				}
			}
			if got, want := strings.Join(output, " "), test.Output; got != want {
				t.Errorf("got output %q, want %q", got, want)
			}
			if got, want := filter.String(), test.Buffer; got != want {
				t.Errorf("got buffer %q, want %q", got, want)
			}
		})
	}
}
