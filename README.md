# lcelnn

Lua/Torch7 port of the `celnn` cellular neural network library.

Install from a checkout with `luarocks install ./celnn-1.0-1.rockspec`.

```lua
local celNN = require "CelNN";
local torch = require "torch";

local input = torch.ones(32);
local net = celNN.CellularNetwork.new({
    input = input,
    feedback = torch.Tensor{.1, 1, .1},
    control = torch.Tensor{0, 1, 0},
    activation = "piecewiseLinear",
    boundary = "reflect",
});
local result = net:run({t_end = 1, dt = .01,
    return_trajectory = true});
print(result.output);
```

## Ported modules

- `CelNN.lua`: network dynamics, activations, simulation, and results.
- `CelNN/Activations.lua`: built-in scalar activation functions.
- `CelNN/Templates.lua`: built-in templates and template registry.
- `CelNN/Domains.lua`: grid, signal, and grayscale image helpers.
- `CelNN/Serialization.lua`: atomic JSON persistence (requires `lua-cjson`).
- `CelNN/Training.lua`: dataset helpers, fitness, and genetic algorithm.
- `CelNN/Plasticity.lua`: Hebbian/Oja updates and plastic linear layer.
- `CelNN/Differentiable.lua`: repeated scalar or diagonal-channel evolution.
- `CelNN/Visualization.lua`: SVG signal, grid, and trajectory plots.
- `CelNN/Utils.lua`: tensor conversion, stencil centers, and shape helpers.

Torch7 has no automatic tensor gradient tracking. The differentiable-network
module ports the forward model and parameter layout, but does not provide
PyTorch-compatible autograd or BPTT. The adaptive SciPy `solve_ivp` solver is
also unavailable; Euler and semi-implicit Euler are supported. Grid stencils
support one and two dimensions.

Image file helpers require Torch's `image` package. JSON helpers require
`lua-cjson`. The core has no extra dependencies beyond Torch7.

## Run the smoke test

Install Torch7 in a Riema environment, then run:

```sh
riema run <environment> th tests/Smoke.lua
```

The available Riema Lua environments may not include Torch7. Syntax can be
checked without it using:

```sh
riema run <environment> luac -p CelNN.lua CelNN/*.lua tests/Smoke.lua
```

The port is derived from the Apache-2.0 `celnn` project; see `LICENSE`.
