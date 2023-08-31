package llm

import (
	"fmt"
	"os"
	"reflect"
	"regexp"
	"strings"
)

// Env constructs a configuration map from OS environment variables with the provided prefix, like "llm_"
func Env(prefix string) map[string]string {
	out := make(map[string]string)
	for _, env := range os.Environ() {
		if !strings.HasPrefix(env, prefix) {
			continue
		}
		env = env[len(prefix):]
		if ix := strings.IndexByte(env, '='); ix != -1 {
			out[env[:ix]] = env[ix+1:]
		}
	}
	return out
}

// Map will marshal the provided structure pointer into a map.
func Map(ref any, out map[string]string) error {
	return forStructFields(ref, func(tag string, v any) error {
		switch v := v.(type) {
		case *string:
			out[tag] = *v
		case *[]string:
			out[tag] = formatStrings(*v)
		default:
			out[tag] = fmt.Sprint(v)
		}
		return nil
	})
}

func formatStrings(in []string) string {
	var buf strings.Builder
	for i, s := range in {
		if i > 0 {
			buf.WriteByte(' ')
		}
		if rxWord.MatchString(s) {
			buf.WriteString(s)
			continue
		}
		buf.WriteByte('"')
		buf.WriteString(stringQuoter.Replace(s))
		buf.WriteByte('"')
	}
	return buf.String()
}

var stringQuoter = strings.NewReplacer(`\`, `\\`, `"`, `\"`, "\n", `\n`, "\r", `\r`, "\t", `\t`)

var rxWord = regexp.MustCompile(`^[\pL\pN!#$%&()*+,.\[\]\\^_{|}~-]+$`)

// Unmap will unmarshal the provided map into the provided structure pointer.
func Unmap(in map[string]string, ref any) error {
	return forStructFields(ref, func(tag string, v any) error {
		value, ok := in[tag]
		if !ok {
			return nil
		}
		switch v := v.(type) {
		case *string:
			*v = value
		case *[]string:
			seq, err := parseStrings(value)
			if err != nil {
				return fmt.Errorf(`%w while parsing %q for %v`, err, value, tag)
			}
			*v = seq
		default:
			_, err := fmt.Sscan(value, v)
			if err != nil {
				return fmt.Errorf(`%w while parsing %q for %v`, err, value, tag)
			}
		}
		return nil
	})
}

type Parser interface {
	Parse(string) error
}

func forStructFields(ref any, fn func(string, any) error) error {
	rv := reflect.ValueOf(ref)
	if rv.Kind() != reflect.Ptr {
		return fmt.Errorf(`expected pointer to Go structures, not %T`, ref)
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf(`expected pointer to Go structures, not %T`, ref)
	}
	rt := rv.Type()
	for i := 0; i < rv.NumField(); i++ {
		ft := rt.Field(i)
		if ft.PkgPath != `` {
			continue
		}
		if ft.Anonymous {
			var err error
			if ft.Type.Kind() == reflect.Struct {
				err = forStructFields(rv.Field(i).Addr().Interface(), fn)
			} else if ft.Type.Kind() == reflect.Ptr && ft.Type.Elem().Kind() == reflect.Struct {
				err = forStructFields(rv.Field(i).Interface(), fn)
			}
			if err != nil {
				return err
			}
			continue
		}

		tag, ok := ``, false
		if tag, ok = ft.Tag.Lookup(`llm`); ok {
		} else if tag, ok = ft.Tag.Lookup(`json`); ok {
		} else if tag, ok = ft.Tag.Lookup(`yaml`); ok {
		} else if tag, ok = ft.Tag.Lookup(`toml`); ok {
		} else if tag, ok = ft.Tag.Lookup(`env`); ok {
		} else {
			continue
		}
		// TODO: sniff for default tags?
		tag = strings.SplitN(tag, `,`, 2)[0]
		if tag == `-` {
			continue
		}
		err := fn(tag, rv.Field(i).Addr().Interface())
		if err != nil {
			return err
		}
	}
	return nil
}

func parseStrings(in string) ([]string, error) {
	if in == "" {
		return nil, nil
	}
	seq := make([]string, 0, 8)
	in = strings.TrimRightFunc(in, isSpace)
	for {
		in = strings.TrimLeftFunc(in, isSpace)
		var s string
		var err error
		s, in, err = parseString(in)
		if err != nil {
			return nil, err
		}
		seq = append(seq, s)
		if in == "" {
			break
		}
		if !isSpace(rune(in[0])) {
			return nil, fmt.Errorf(`expected space after %q`, s)
		}
	}
	return seq, nil
}

func parseString(in string) (string, string, error) {
	switch in[0] {
	case '"':
		return parseQuotedString(in, '"')
	case '\'':
		return parseQuotedString(in, '\'')
	default:
		ix := strings.IndexFunc(in, isSpace)
		if ix == -1 {
			return in, "", nil
		}
		return in[:ix], in[ix:], nil
	}
}

func parseQuotedString(in string, quote byte) (string, string, error) {
	buf := make([]byte, 0, len(in))
	for i := 1; i < len(in); i++ {
		ch := in[i]
		switch ch {
		case '\\':
			i++
			if i >= len(in) {
				return "", "", fmt.Errorf(`unexpected end of string`)
			}
			buf = append(buf, in[i])
		case quote:
			return string(buf), in[i+1:], nil
		default:
			buf = append(buf, ch)
		}
	}
	return "", "", fmt.Errorf(`could not find matching quote %q`, string(quote))
}

func isSpace(r rune) bool {
	switch r {
	case ' ', '\t', '\r', '\n':
		return true
	}
	return false
}
