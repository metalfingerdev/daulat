# NixOS dev shell: lets pip-installed wheels in .venv find libstdc++ and zlib.
#   nix-shell        # creates/activates .venv
{ pkgs ? import <nixpkgs> { } }:

pkgs.mkShell {
  packages = [ pkgs.python3 ];

  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
    pkgs.stdenv.cc.cc.lib
    pkgs.zlib
  ];

  shellHook = ''
    [ -d .venv ] || python3 -m venv .venv
    source .venv/bin/activate
  '';
}
