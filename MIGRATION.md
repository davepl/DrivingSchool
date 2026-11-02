# Setup on ubvmdell

Clone `https://github.com/davepl/DrivingSchool.git` and run `./scripts/setup.sh`.
Python 3.11+ and the system Python venv package are required. Setup creates a
local virtual environment and installs dependencies only. It starts no processes.

Optionally run `./scripts/install-service.sh` to install the user service and
viewer launcher. The installer leaves autostart disabled and does not start the
service. Start it manually with `systemctl --user start driving-school.service`.
The installed service uses the smaller default profile; the GB300 max-load
profile must be explicitly requested. Select a GPU with `DRIVING_GPU`, or preserve
an existing `CUDA_VISIBLE_DEVICES` selection.

Saved classes and checkpoints live in `runs/` and are intentionally excluded
from Git. Transfer that directory separately to retain the trained drivers.
Never transfer the old `.venv` between architectures. The dashboard defaults to
loopback port 8780; use an SSH tunnel when viewing from another computer.

Validation without a long-running workload:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests -v
```
