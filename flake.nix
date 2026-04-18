{
  description = "Python dev env (uv, TF, OpenCV, VSCodium, Vim-like UX)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    flake-utils.url = "github:numtide/flake-utils";
    home-manager.url = "github:nix-community/home-manager";
    home-manager.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, flake-utils, home-manager }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};

        python = pkgs.python313;

        pythonEnv = python.withPackages (ps: with ps; [
          uv
          pip
          setuptools
          wheel
          pyzmq
          ipykernel
          jupyter
          jupyterlab
          ipywidgets
          numpy
          pandas
          matplotlib
          seaborn
          scikit-learn
          black
          isort
          ruff
          pylint
          mypy
          pytest
          tqdm
          loguru

          # OpenCV bindings from Nix, не из pip
          opencv4
        ]);

        libPath = pkgs.lib.makeLibraryPath [
          pkgs.stdenv.cc.cc
          pkgs.zeromq
          pkgs.libsodium
          pkgs.libffi
          pkgs.openssl
          pkgs.opencv4
          pkgs.libGL
          pkgs.libGLU
          pkgs.xorg.libxcb
          pkgs.zlib
          pkgs.glib.out
          pkgs.gtk3
          pkgs.gst_all_1.gstreamer
          pkgs.gst_all_1.gst-plugins-base
          pkgs.gst_all_1.gst-plugins-good
          pkgs.gst_all_1.gst-plugins-bad
          pkgs.gst_all_1.gst-plugins-ugly
        ];

        homeConfig = home-manager.lib.homeManagerConfiguration {
          inherit pkgs;
          modules = [
            {
              home = {
                username = builtins.getEnv "USER";
                homeDirectory = builtins.getEnv "HOME";
                stateVersion = "25.11";
              };

              programs.vscode = {
                enable = true;
                package = pkgs.vscodium-fhs;

                profiles.default = {
                  extensions = with pkgs.vscode-extensions; [
                    ms-python.python
                    ms-python.vscode-pylance
                    ms-toolsai.jupyter
                    ms-toolsai.jupyter-keymap
                    ms-toolsai.jupyter-renderers
                    ms-toolsai.vscode-jupyter-cell-tags
                    ms-toolsai.vscode-jupyter-slideshow
                    ms-toolsai.vscode-tensorboard

                    ms-python.black-formatter
                    ms-python.isort
                    charliermarsh.ruff

                    eamodio.gitlens
                    redhat.vscode-yaml

                    vscodevim.vim
                  ];

                  userSettings = {
                    # Python / editor
                    "python.defaultInterpreterPath" = "./.venv/bin/python";
                    "python.terminal.activateEnvironment" = true;
                    "python.terminal.activateEnvInCurrentTerminal" = true;
                    "python.terminal.launchArgs" = [ "--no-warnings" ];
                    "editor.formatOnSave" = true;
                    "editor.defaultFormatter" = "ms-python.black-formatter";
                    "editor.lineNumbers" = "relative";  # relative line numbers

                    "[python]" = {
                      "editor.formatOnSave" = true;
                      "editor.codeActionsOnSave" = {
                        "source.organizeImports" = true;
                      };
                      "editor.defaultFormatter" = "ms-python.black-formatter";
                    };

                    "black-formatter.args" = [ "--line-length" "88" ];
                    "black-formatter.importStrategy" = "fromEnvironment";

                    "ruff.enable" = true;
                    "jupyter.alwaysTrustNotebooks" = true;

                    # Alt не фокусит меню
                    "window.titleBarStyle" = "custom";
                    "window.menuBarVisibility" = "hidden";
                    "window.customMenuBarAltFocus" = false;
                    "window.enableMenuBarMnemonics" = false;
                    "terminal.integrated.allowMnemonics" = false;

                    # Vim extension: не перехватывать Ctrl
                    "vim.useCtrlKeys" = false;
                    "vim.useSystemClipboard" = true;
                    "vim.statusBarColorControl" = false;
                    "vim.handleKeys" = {
                      "<C-a>" = false;
                      "<C-c>" = false;
                      "<C-f>" = false;
                      "<C-h>" = false;
                      "<C-n>" = false;
                      "<C-p>" = false;
                      "<C-r>" = false;
                      "<C-s>" = false;
                      "<C-v>" = false;
                      "<C-w>" = false;
                      "<C-x>" = false;
                      "<C-z>" = false;
                    };
                  };

                  # Глобальные keybindings профиля
                  keybindings = [
                    # History в редакторе
                    {
                      key = "alt+left";
                      command = "workbench.action.navigateBack";
                      when = "!terminalFocus && canNavigateBack";
                    }
                    {
                      key = "alt+right";
                      command = "workbench.action.navigateForward";
                      when = "!terminalFocus && canNavigateForward";
                    }

                    # Поведение в терминале: сплиты, как по умолчанию
                    {
                      key = "alt+left";
                      command = "workbench.action.terminal.focusPreviousPane";
                      when =
                        "terminalFocus && terminalSplitPaneActive && terminalHasBeenCreated"
                        + " || terminalFocus && terminalSplitPaneActive && terminalProcessSupported";
                    }
                    {
                      key = "alt+right";
                      command = "workbench.action.terminal.focusNextPane";
                      when =
                        "terminalFocus && terminalSplitPaneActive && terminalHasBeenCreated"
                        + " || terminalFocus && terminalSplitPaneActive && terminalProcessSupported";
                    }

                    # Linux default history (пусть живут как запасной вариант)
                    {
                      key = "ctrl+alt+-";
                      command = "workbench.action.navigateBack";
                      when = "canNavigateBack";
                    }
                    {
                      key = "ctrl+shift+-";
                      command = "workbench.action.navigateForward";
                      when = "canNavigateForward";
                    }
                  ];
                };
              };
            }
          ];
        };
      in
      {
        # Чтобы можно было применять home-manager конфиг через flake
        packages.homeConfigurations.default = homeConfig.activationPackage;

        devShells.default = pkgs.mkShell {
          packages = [
            python
            pythonEnv

            pkgs.uv
            pkgs.git
            pkgs.gcc
            pkgs.pkg-config

            pkgs.opencv4
            pkgs.gtk3
            pkgs.zeromq
            pkgs.libsodium
            pkgs.libffi
            pkgs.openssl
            pkgs.libGL
            pkgs.libGLU
            pkgs.xorg.libxcb
            pkgs.zlib
            pkgs.glib
            pkgs.gst_all_1.gstreamer
            pkgs.gst_all_1.gst-plugins-base
            pkgs.gst_all_1.gst-plugins-good
            pkgs.gst_all_1.gst-plugins-bad
            pkgs.gst_all_1.gst-plugins-ugly

            pkgs.jq
          ];

          shellHook = ''
            export LD_LIBRARY_PATH="${libPath}:$LD_LIBRARY_PATH"
            export PKG_CONFIG_PATH="${pkgs.opencv4}/lib/pkgconfig:$PKG_CONFIG_PATH"
            export PYTHONPATH="${pkgs.opencv4}/${python.sitePackages}:$PYTHONPATH"

            if [ ! -d ".venv" ]; then
              echo "Creating uv virtual environment with Python 3.13..."
              uv venv --python ${python}/bin/python .venv
            fi

            source .venv/bin/activate

            echo "Installing packages that should stay in the venv..."
            uv pip install --upgrade pip setuptools wheel

            if ! python -c "import tensorflow" >/dev/null 2>&1; then
              echo "Installing TensorFlow CPU wheel into .venv..."
              uv pip install tensorflow_cpu-2.21.0-cp313-cp313-linux_x86_64.whl
            fi

            uv pip install pyzmq jupyter-client ipykernel

            mkdir -p .vscode

            # Workspace-local settings (могут переопределять глобальные, если нужно)
            cat > .vscode/settings.json << EOF
{
  "python.defaultInterpreterPath": "''${workspaceFolder}/.venv/bin/python",
  "python.terminal.activateEnvironment": true,
  "python.terminal.activateEnvInCurrentTerminal": true,
  "python.terminal.launchArgs": ["--no-warnings"],

  "editor.formatOnSave": true,
  "editor.defaultFormatter": "ms-python.black-formatter",
  "editor.lineNumbers": "relative",

  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.organizeImports": true
    },
    "editor.defaultFormatter": "ms-python.black-formatter"
  },

  "black-formatter.args": ["--line-length", "88"],
  "black-formatter.importStrategy": "fromEnvironment",

  "python.envFile": "''${workspaceFolder}/.env",
  "python.analysis.extraPaths": [
    "${pkgs.opencv4}/${python.sitePackages}",
    "${pkgs.python313Packages.opencv4}/${python.sitePackages}"
  ],

  "python.linting.enabled": true,
  "python.linting.pylintEnabled": false,
  "python.linting.ruffEnabled": true,
  "python.formatting.provider": "black",
  "python.formatting.blackArgs": ["--line-length", "88"],

  "ruff.enable": true,
  "jupyter.alwaysTrustNotebooks": true,

  "window.titleBarStyle": "custom",
  "window.menuBarVisibility": "hidden",
  "window.customMenuBarAltFocus": false,
  "window.enableMenuBarMnemonics": false,
  "terminal.integrated.allowMnemonics": false,

  "vim.useCtrlKeys": false,
  "vim.useSystemClipboard": true,
  "vim.statusBarColorControl": false,
  "vim.handleKeys": {
    "<C-a>": false,
    "<C-c>": false,
    "<C-f>": false,
    "<C-h>": false,
    "<C-n>": false,
    "<C-p>": false,
    "<C-r>": false,
    "<C-s>": false,
    "<C-v>": false,
    "<C-w>": false,
    "<C-x>": false,
    "<C-z>": false
  },

  "editor.codeActionsOnSave": {
    "source.organizeImports": true
  }
}
EOF

            # Workspace keybindings: дублируют профиль, если запуск без home-manager
            cat > .vscode/keybindings.json << EOF
[
  {
    "key": "alt+left",
    "command": "workbench.action.navigateBack",
    "when": "!terminalFocus && canNavigateBack"
  },
  {
    "key": "alt+right",
    "command": "workbench.action.navigateForward",
    "when": "!terminalFocus && canNavigateForward"
  },
  {
    "key": "alt+left",
    "command": "workbench.action.terminal.focusPreviousPane",
    "when": "terminalFocus && terminalSplitPaneActive && terminalHasBeenCreated || terminalFocus && terminalSplitPaneActive && terminalProcessSupported"
  },
  {
    "key": "alt+right",
    "command": "workbench.action.terminal.focusNextPane",
    "when": "terminalFocus && terminalSplitPaneActive && terminalHasBeenCreated || terminalFocus && terminalSplitPaneActive && terminalProcessSupported"
  },
  {
    "key": "ctrl+alt+-",
    "command": "workbench.action.navigateBack",
    "when": "canNavigateBack"
  },
  {
    "key": "ctrl+shift+-",
    "command": "workbench.action.navigateForward",
    "when": "canNavigateForward"
  }
]
EOF

            cat > .env << EOF
LD_LIBRARY_PATH=${libPath}
PYTHONPATH=${pkgs.opencv4}/${python.sitePackages}
EOF

            KERNEL_DIR="$HOME/.local/share/jupyter/kernels/python313-venv"
            mkdir -p "$KERNEL_DIR"

            cat > "$KERNEL_DIR/kernel.json" << EOF
{
  "argv": [
    "$PWD/.venv/bin/python",
    "-m",
    "ipykernel_launcher",
    "-f",
    "{connection_file}"
  ],
  "display_name": "Python 3.13 (.venv + Nix OpenCV)",
  "language": "python",
  "env": {
    "LD_LIBRARY_PATH": "${libPath}",
    "PYTHONPATH": "${pkgs.opencv4}/${python.sitePackages}"
  }
}
EOF

            echo ""
            echo "╔════════════════════════════════════════════════════════════════╗"
            echo "║  Dev environment ready                                        ║"
            echo "║                                                                ║"
            echo "║  • Python: $(python --version)                                 ║"
            echo "║  • Virtual env: .venv (activated)                              ║"
            echo "║  • OpenCV: from Nix package                                    ║"
            echo "║  • TensorFlow: from .venv wheel                                ║"
            echo "║  • VSCodium: Alt menu focus disabled                           ║"
            echo "║  • Alt+Left/Right: history (editor) / panes (terminal)         ║"
            echo "╚════════════════════════════════════════════════════════════════╝"
            echo ""
          '';
        };

        apps.default = {
          type = "app";
          program = "${pkgs.writeShellScript "codium-launch" ''
            exec ${pkgs.vscodium-fhs}/bin/codium "$@"
          ''}";
        };
      });
}
