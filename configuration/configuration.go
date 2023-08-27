package configuration

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Marshal returns the provided structure as a configuration, with similar restrictions to Unmarshal.
func Marshal(v any) (Interface, error) {
	rv := reflect.Indirect(reflect.ValueOf(v))
	if rv.Kind() != reflect.Struct {
		return nil, fmt.Errorf(`marshal can only be used with struct pointers, got %T`, v)
	}
	rt := rv.Type()
	nf := rt.NumField()
	cf := make(Map, nf)
	for i := 0; i < nf; i++ {
		ft := rt.Field(i)
		if ft.PkgPath != `` {
			continue
		}
		name := ft.Name
		if tag := ft.Tag.Get(`cfg`); tag != `` {
			name = tag
		} else if tag := ft.Tag.Get(`yaml`); tag != `` {
			name = tag
		} else if tag := ft.Tag.Get(`json`); tag != `` {
			name = tag
		}
		name = strings.SplitN(name, `,`, 2)[0]
		if name == `` || name == `-` {
			continue
		}
		fv := rv.Field(i)
		if !fv.CanInterface() {
			continue
		}
		switch fv.Kind() {
		case reflect.String:
			cf[name] = []string{fv.String()}
		case reflect.Slice:
			if fv.Type().Elem().Kind() != reflect.String {
				continue
			}
			cf[name] = []string{strings.Join(fv.Interface().([]string), `,`)}
		case reflect.Bool:
			cf[name] = []string{strconv.FormatBool(fv.Bool())}
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			cf[name] = []string{strconv.FormatInt(fv.Int(), 10)}
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			cf[name] = []string{strconv.FormatUint(fv.Uint(), 10)}
		case reflect.Float32, reflect.Float64:
			cf[name] = []string{strconv.FormatFloat(fv.Float(), 'f', -1, 64)}
		default:
			continue
		}
	}
	return cf, nil
}

// Unmarshal uses the provided configuration to unmarshal the exported fields of v, which must be a pointer to a struct.
// The names of the fields are used to look up configuration values, but are overridden by the following struct tags,
// in order of precedence: "cfg", "yaml", "json".
//
// Like with Get, unconfigured fields are left unchanged and the supported field types are string, []string, bool, int
// and floats.
func Unmarshal(v interface{}, cf Interface) error {
	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return fmt.Errorf(`unmarshal can only be used with struct pointers, got %T`, v)
	}
	rv = rv.Elem()
	if rv.Kind() != reflect.Struct {
		return fmt.Errorf(`unmarshal can only be used with struct pointers, got %T`, v)
	}
	rt := rv.Type()
	nf := rt.NumField()
	for i := 0; i < nf; i++ {
		ft := rt.Field(i)
		if ft.PkgPath != `` {
			continue
		}
		name := ft.Name
		if tag := ft.Tag.Get(`cfg`); tag != `` {
			name = strings.SplitN(tag, `,`, 2)[0]
		} else if tag := ft.Tag.Get(`yaml`); tag != `` {
			name = strings.SplitN(tag, `,`, 2)[0]
		} else if tag := ft.Tag.Get(`json`); tag != `` {
			name = strings.SplitN(tag, `,`, 2)[0]
		}
		if name == `` || name == `-` {
			continue
		}
		if err := Get(rv.Field(i).Addr().Interface(), cf, name); err != nil {
			return err
		}
	}
	return nil
}

// Get resolves ref from a named configuration item.  Ref must be a pointer to a string, a slice of strings, a boolean,
// an integer or a float.  If the item is nil, then the ref is left unchanged.
func Get(ref any, cf Interface, name string) error {
	values := cf.GetConfiguration(name)
	if values == nil {
		return nil
	}

	if ref, ok := ref.(*[]string); ok {
		*ref = values
		return nil
	}
	if len(values) == 0 {
		return nil
	}
	if len(values) > 1 {
		return fmt.Errorf(`only one value allowed for %q, got %d`, name, len(values))
	}
	value := values[0]

	switch ref := ref.(type) {
	case *string:
		*ref = value
	case *bool:
		switch value {
		case `true`, `TRUE`, `YES`, `ON`, `1`:
			*ref = true
		case `false`, `FALSE`, `NO`, `OFF`, `0`:
			*ref = false
		default:
			return fmt.Errorf(`invalid boolean value %q for %q`, value, name)
		}
	case *int:
		n, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf(`%w for %q`, err, name)
		}
		*ref = n
	case *float64:
		n, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return fmt.Errorf(`%w for %q`, err, name)
		}
		*ref = n
	case *float32:
		n, err := strconv.ParseFloat(value, 32)
		if err != nil {
			return fmt.Errorf(`%w for %q`, err, name)
		}
		*ref = float32(n)
	default:
		return fmt.Errorf(`unsupported type %T for %q`, ref, name)
	}
	return nil
}

// An Overlay combines multiple configurations, possibly of different types, into
// a single configuration.  The first item offering a named configuration item "wins"
// so it is best to put them in priority order, such as "Overlay(req.Configuration, Environment(`llm_`), svc.Configuration)".
type Overlay []Interface

func (cf Overlay) GetConfiguration(name string) []string {
	for _, it := range cf {
		if it == nil {
			continue
		}
		items := it.GetConfiguration(name)
		if items != nil {
			return items
		}
	}
	return nil
}

// Configured returns the list of items that have been configured in the provided interfaces.
func (cf Overlay) Configured() []string {
	return Configured(cf...)
}

// Configured returns the list of items that have been configured in the provided interfaces, in their presence
// order but with duplicates removed.
func Configured(configurations ...Interface) []string {
	seen := make(map[string]struct{}, 256)
	items := make([]string, 0, 256)
	for _, cf := range configurations {
		for _, item := range cf.Configured() {
			if _, ok := seen[item]; !ok {
				seen[item] = struct{}{}
				items = append(items, item)
			}
		}
	}
	return items
}

// Interface describes a generic interface for providing a configuration for
// named options.
type Interface interface {
	// Get returns the set of configured values for a given name.  This typically
	// returns either nil or a slice with a single string value.  Nil indicates
	// that there was no configuration, and therefore a default can be used.
	GetConfiguration(name string) []string

	// Configured returns the list of items that have been configured.  This list does not need to be unique
	// or sorted.
	Configured() []string
}

// MapOf maps the configured items.
func MapOf(cf Interface) Map {
	items := cf.Configured()
	m := make(Map, len(items))
	for _, item := range items {
		m[item] = cf.GetConfiguration(item)
	}
	return m
}

// Map uses a map of strings to provide a configuration.  This is
// useful for inclusion in a JSON or YAML structure, since it knows how to marshal
// and unmarshal itself.
type Map map[string][]string

// GetConfiguration returns the value for the given name, or nil if there is no value.
func (cf Map) GetConfiguration(name string) []string { return cf[name] }

// Configured returns the list of items that have been configured in the map.
func (cfg Map) Configured() []string {
	items := make([]string, 0, len(cfg))
	for item := range cfg {
		items = append(items, item)
	}
	return items
}

// UnmarshalJSON satisfies json.Unmarshaler using a map of values, where each value
// may be a string, a number, true, false, null, or an array of values that are not arrays.
func (cf *Map) UnmarshalJSON(p []byte) error {
	var m map[string]any
	if err := json.Unmarshal(p, &m); err != nil {
		return err
	}
	*cf = make(Map, len(m))
	for k, v := range m {
		switch v := v.(type) {
		case string:
			(*cf)[k] = []string{v}
		case map[string]any:
			return fmt.Errorf(`nested maps not supported for %q`, k)
		case []any:
			(*cf)[k] = make([]string, len(v))
			for i, v := range v {
				switch v := v.(type) {
				case []any:
				case map[string]any:
				default:
					(*cf)[k][i] = fmt.Sprint(v)
				}
			}
		default:
			(*cf)[k] = []string{fmt.Sprint(v)}
		}
	}
	return nil
}

// MarshalJSON satisfies json.Marshaler by encoding a map of values, where each
// value is either a string, true, false, null, or an array of strings.  This is
// not perfectly symmetric with UnmarshalJSON, which also accepts numbers, which this
// marshaller will not produce.
func (cf Map) MarshalJSON() ([]byte, error) {
	m := make(map[string]any, len(cf))
	for k, v := range cf {
		switch len(v) {
		case 0:
		case 1:
			v := v[0]
			switch v {
			case `true`:
				m[k] = true
			case `false`:
				m[k] = false
			case `null`:
				m[k] = nil
			default:
				m[k] = v
			}
		default:
			seq := make([]any, len(v))
			for i, it := range v {
				switch it {
				case `true`:
					seq[i] = true
				case `false`:
					seq[i] = false
				case `null`:
					seq[i] = nil
				default:
					seq[i] = it
				}
			}
			m[k] = seq
		}
	}
	return json.Marshal(m)
}

// UnmarshalYAML satisfies yaml.Unmarshaler using a map of values, where each value
// may be a string, a number, true, false, null, or an array of values that are not arrays.
func (cf *Map) UnmarshalYAML(value *yaml.Node) error {
	if value.Kind != yaml.MappingNode {
		return fmt.Errorf(`expected a map , got %v`, value.Kind)
	}
	*cf = make(Map, len(value.Content)/2)
	for i := 0; i < len(value.Content); i += 2 {
		name := value.Content[i].Value
		if name == `` {
			return fmt.Errorf(`expected a string key, got %v`, value.Content[i].Kind)
		}
		value := value.Content[i+1]
		switch value.Kind {
		case yaml.ScalarNode:
			(*cf)[name] = []string{value.Value}
		case yaml.SequenceNode:
			(*cf)[name] = make([]string, len(value.Content))
			for i, v := range value.Content {
				if v.Kind != yaml.ScalarNode {
					return fmt.Errorf(`expected a scalar item, got %v`, v.Kind)
				}
				(*cf)[name][i] = v.Value
			}
		default:
			return fmt.Errorf(`expected a scalar or sequence, got %v`, value.Kind)
		}
	}
	return nil
}

// Environment provides a configuration that gets values from the OS environment,
// adding the provided prefix to each lookup.  This is the most common source of
// configuration.  If the prefix is uppercase, then names will be converted to
// uppercase before lookup.
func Environment(prefix string) Interface {
	return environment{
		prefix != `` && strings.ToUpper(prefix) == prefix,
		prefix,
	}
}

type environment struct {
	uppercase bool
	prefix    string
}

func (cf environment) Configured() []string {
	for _, it := range os.Environ() {
		if strings.HasPrefix(it, cf.prefix) {
			return []string{it[len(cf.prefix):]}
		}
	}
	return nil
}

func (cf environment) GetConfiguration(name string) []string {
	if cf.uppercase {
		name = cf.prefix + strings.ToUpper(name)
	} else {
		name = cf.prefix + name
	}
	value, ok := os.LookupEnv(name)
	if !ok {
		return nil
	}
	return []string{value}
}

// With returns a configuration with the named item overloaded with the provided values.
func With(cf Interface, name string, values ...any) Interface {
	items := make([]string, len(values))
	for i, it := range values {
		items[i] = fmt.Sprint(it)
	}
	return with{name, items, cf}
}

type with struct {
	name   string
	values []string
	cf     Interface
}

func (cf with) GetConfiguration(name string) []string {
	if name == cf.name {
		return cf.values
	}
	return cf.cf.GetConfiguration(name)
}

func (cf with) Configured() []string {
	return append([]string{cf.name}, cf.Configured()...)
}
