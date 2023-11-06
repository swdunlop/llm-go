package kmp

import "testing"

func TestSearch(t *testing.T) {
	for _, test := range []struct {
		pattern, space string
		sz, ofs        int
	}{
		{``, ``, 0, 0},
		{``, `a`, 0, 0},
		{`a`, ``, 0, 0},
		{`a`, `a`, 1, 0},
		{`a`, `ax`, 1, 0},
		{`a`, `aa`, 1, 0},
		{`a`, `xa`, 1, 1},
		{`a`, `xax`, 1, 1},
		{`ab`, `a`, 1, 0},
		{`ab`, `ax`, 1, 0},
		{`ab`, `aa`, 1, 0},
		{`ab`, `xa`, 1, 1},
		{`ab`, `xax`, 1, 1},
		{`ab`, `ab`, 2, 0},
		{`ab`, `abx`, 2, 0},
		{`ab`, `abab`, 2, 0},
		{`ab`, `xab`, 2, 1},
		{`ab`, `xabx`, 2, 1},
	} {
		sz, ofs := Overlap([]byte(test.pattern), []byte(test.space))
		t.Logf(`Search(%q, %q) = #%v@%v`, test.pattern, test.space, sz, ofs)
		if sz != test.sz || ofs != test.ofs {
			t.Errorf(`Test failed, expected #%v@%v`, test.sz, test.ofs)
		}
	}
}
