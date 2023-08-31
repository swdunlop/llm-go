package llm

import "testing"

func TestParseStrings(t *testing.T) {
	for _, test := range []struct {
		Str    string
		Expect []string
	}{
		{"", []string{}},
		{"a", []string{"a"}},
		{"a b", []string{"a", "b"}},
		{"a b ", []string{"a", "b"}},
		{" a b ", []string{"a", "b"}},
		{" a b", []string{"a", "b"}},
		{" a \tb ", []string{"a", "b"}},
		{`"a"`, []string{"a"}},
		{`"a b"`, []string{"a b"}},
		{`"a b" `, []string{"a b"}},
		{` "a b" `, []string{"a b"}},
		{` "a b"`, []string{"a b"}},
		{` 'a\"b'`, []string{`a"b`}},
		{` 'a\"b'c`, nil},
		{` 'a\"b' c`, []string{`a"b`, "c"}},
	} {
		t.Logf(`testing %q`, test.Str)
		got, err := parseStrings(test.Str)
		if err != nil {
			if test.Expect != nil {
				t.Errorf(`unexpected error: %v`, err)
			}
			continue
		}
		if len(got) != len(test.Expect) {
			t.Errorf(`expected %d items, got %d`, len(test.Expect), len(got))
			continue
		}
		for i := range got {
			if got[i] != test.Expect[i] {
				t.Errorf(`expected %q, got %q`, test.Expect[i], got[i])
			}
		}

	}

}
