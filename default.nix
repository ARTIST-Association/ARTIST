# An example package with dependencies defined via pyproject.toml
{
  config,
  lib,
  dream2nix,
  ...
}: let
  pyproject = lib.importTOML (config.mkDerivation.src + /pyproject.toml);
in {
  imports = [
    dream2nix.modules.dream2nix.pip
  ];

  deps = {nixpkgs, ...}: {
    python = nixpkgs.python3;
    hatch = nixpkgs.hatch;
    hatchling = nixpkgs.python3.pkgs.hatchling;
  };

  inherit (pyproject.project) name version;

  mkDerivation = {
    src = lib.cleanSourceWith {
      src = lib.cleanSource ./.;
      filter = name: type:
        !(builtins.any (x: x) [
          (lib.hasSuffix ".nix" name)
          (lib.hasPrefix "." (builtins.baseNameOf name))
          (lib.hasSuffix "flake.lock" name)
        ]);
    };
  };

  buildPythonPackage = {
    pyproject = true;
    build-system = [ config.deps.python.pkgs.hatchling ];
    pythonImportsCheck = [
      "artist"
    ];
  };

  #paths.lockFile = "lock.${config.deps.stdenv.system}.json";
  pip = {
    # Setting editables.$pkg to a relative or absolute path, as a string, will
    # link this path as an editable install to .dream2nix/editables in
    # devShells. The root package is always installed as editable.
    # editables.charset-normalizer = "/home/my-user/src/charset-normalizer";

    requirementsList =
      pyproject.build-system.requires
      or []
      ++ pyproject.project.dependencies or [];
    flattenDependencies = true;

  };
}
