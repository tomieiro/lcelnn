-- Numerical port check based on the Apache-2.0 celnn project.
local celNN = require "CelNN";
local torch = require "torch";

local input = torch.Tensor{1, 1, 1, 1, 1};
local zero = torch.Tensor{0, 0, 0};
local control = torch.Tensor{0, 1, 0};
local net = celNN.CellularNetwork.new({
    input = input,
    feedback = zero,
    control = control,
    activation = "identity",
});
local result = net:run({t_end = .3, dt = .1});
assert(math.abs(result.state[1] - .271) < 1e-6);
assert(math.abs(result.state[5] - .271) < 1e-6);
assert(math.abs(net:derivative(torch.Tensor{0, 0, 0, 0, 0})[3] - 1)
    < 1e-9);

net:reset();
local semi = net:run({t_end = .3, dt = .1,
    solver = "semi_implicit_euler"});
assert(math.abs(semi.state[3] - .2486852) < 1e-6);

local left = celNN.CellularNetwork.new({
    input = input,
    feedback = zero,
    control = torch.Tensor{1, 0, 0},
    activation = "identity",
    boundary = "reflect",
});
assert(math.abs(left:drive(left.state)[1] - 1) < 1e-6);
assert(#celNN.templates.builtinTemplates():names() == 12);

local normalized = celNN.normalizeSignal(torch.Tensor{2, 4, 6});
assert(math.abs(normalized[1] + 1) < 1e-6);
assert(math.abs(normalized[3] - 1) < 1e-6);
print("celnn Lua smoke test passed");
