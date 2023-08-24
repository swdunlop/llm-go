{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/master";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }: flake-utils.lib.eachDefaultSystem (system:
    let 
      pkgs = nixpkgs.legacyPackages.${system}; 
    in {
      devShells.default = pkgs.stdenv.mkDerivation {
        name = "ollama-dev";
        buildInputs = [ ] ++ (
          if pkgs.stdenv.isDarwin then with pkgs.darwin.apple_sdk_11_0.frameworks; [
            Accelerate
            MetalKit
            MetalPerformanceShaders
          ] else [ ]
        );
      };
    }
  );
}
