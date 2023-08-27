{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let 
      pkgs = nixpkgs.legacyPackages.${system}; 
      buildInputs = [ ] ++ (
          if pkgs.stdenv.isDarwin then with pkgs.darwin.apple_sdk_11_0.frameworks; [
            Accelerate
            MetalKit
            MetalPerformanceShaders
          ] else [ ]);
    in {
      devShells.default = pkgs.stdenv.mkDerivation {
        name = "ollama-build";
        inherit buildInputs;
      };
      devShells.worker = pkgs.stdenv.mkDerivation {
        name = "ollama-worker";
        buildInputs = with pkgs; buildInputs ++ [ nats-server natscli ];
      };
    }
  );
}
