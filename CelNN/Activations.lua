-- Lua port based on the Apache-2.0 celnn project.
local module = {};

module.piecewiseLinear = function(value)
    return .5 * (math.abs(value + 1) - math.abs(value - 1));
end;
module.saturatedLinear = module.piecewiseLinear;
module.identity = function(value) return value; end;
module.tanhActivation = function(value)
    if math.tanh then return math.tanh(value); end
    local positive, negative = math.exp(value), math.exp(-value);
    return (positive - negative) / (positive + negative);
end;
module.sigmoidActivation = function(value)
    return 1 / (1 + math.exp(-value));
end;
module.signActivation = function(value)
    return value > 0 and 1 or (value < 0 and -1 or 0);
end;
module.reluActivation = function(value) return math.max(0, value); end;
module.piecewise_linear = module.piecewiseLinear;
module.saturated_linear = module.saturatedLinear;
module.tanh_activation = module.tanhActivation;
module.sigmoid_activation = module.sigmoidActivation;
module.sign_activation = module.signActivation;
module.relu_activation = module.reluActivation;

return module;
