package internal

import "github.com/nats-io/nats.go"

func Defaults() Options {
	return Options{
		URL: nats.DefaultURL,
	}
}

type Options struct {
	// ClientName specifies the name of the client, which is used in logs and metadata to identify the client for
	// debugging purposes.
	ClientName string `env:"nats_client_name"`

	// URL specifies the NATS client URL used to connect to the NATS server.  This is used if Conn is nil, which is
	// the case if `llm.New("nats")` is used to create the client.  This defaults to `nats://localhost:4222`.
	URL string `env:"nats_url"`

	// NKeyFile provides the path to an nkey seed / secret file that will be used to authenticate with the NATS server.
	// Ignored if a NATS connection is provided.
	NKeyFile string `env:"nats_nk"`

	// CA provides the path to a file containing trusted certificates for verifying the NATS server.  Ignored if a
	// NATS connection is provided.  If not provided, the host certificate authorities will be used.
	CA string `env:"nats_ca"`
}

// Dial will connect to the NATS server using the options provided.
func (opts *Options) Dial(more ...nats.Option) (*nats.Conn, error) {
	options := make([]nats.Option, 0, 8)
	if opts.NKeyFile != `` {
		opt, err := nats.NkeyOptionFromSeed(opts.NKeyFile)
		if err != nil {
			return nil, err
		}
		options = append(options, opt)
	}
	if opts.CA != `` {
		options = append(options, nats.RootCAs(opts.CA))
	}
	if opts.ClientName != `` {
		options = append(options, nats.Name(opts.ClientName))
	}
	options = append(options, more...)
	return nats.Connect(opts.URL, options...)
}
