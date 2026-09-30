-- Lua port based on the Apache-2.0 celnn project.
local torch = require "torch";

local module = {};
local state = {};
state.__index = state;

function state.new(memory, updates)
    assert(torch.isTensor(memory) and memory:dim() == 3,
        "memory must have shape batch x output x input");
    assert((updates or 0) >= 0, "updates must be non-negative");
    return setmetatable({memory = memory:clone(), updates = updates or 0},
        state);
end

function state.zeros(batchSize, outputSize, inputSize, like)
    assert(batchSize > 0 and outputSize > 0 and inputSize > 0,
        "state dimensions must be positive");
    return state.new(like.new(batchSize, outputSize, inputSize):zero());
end

function state:reset()
    return state.new(self.memory:clone():zero());
end

function state:detach()
    return state.new(self.memory, self.updates);
end

local function activities(pre, post, memory)
    assert(pre:dim() >= 2 and post:dim() >= 2,
        "activities need batch and feature axes");
    assert(pre:dim() == post:dim() and pre:size(1) == post:size(1),
        "pre and post sample axes must match");
    local batchSize, inputSize = pre:size(1), pre:size(pre:dim());
    local outputSize = post:size(post:dim());
    assert(memory:size(1) == batchSize and memory:size(2) == outputSize and
        memory:size(3) == inputSize, "memory shape does not match activities");
    local samples = 1;
    for dimension = 2, pre:dim() - 1 do
        assert(pre:size(dimension) == post:size(dimension),
            "pre and post sample axes must match");
        samples = samples * pre:size(dimension);
    end
    return pre:contiguous():view(batchSize, samples, inputSize),
        post:contiguous():view(batchSize, samples, outputSize);
end

local rule = {};
rule.__index = rule;
function rule.hebbian(learningRate, decay)
    learningRate, decay = learningRate or .01, decay or 1;
    assert(learningRate >= 0 and decay >= 0 and decay <= 1,
        "invalid Hebbian learning parameters");
    return setmetatable({learningRate = learningRate, decay = decay}, rule);
end
function rule.oja(learningRate, decay)
    local result = rule.hebbian(learningRate, decay);
    result.kind = "oja";
    return result;
end
function rule:update(pre, post, memory)
    local inputs, outputs = activities(pre, post, memory);
    local batchSize, samples = inputs:size(1), inputs:size(2);
    local inputSize, outputSize = inputs:size(3), outputs:size(3);
    local result = memory:clone();
    for batch = 1, batchSize do
        for output = 1, outputSize do
            local energy = 0;
            for sample = 1, samples do
                energy = energy + outputs[batch][sample][output] ^ 2;
            end
            energy = energy / samples;
            for input = 1, inputSize do
                local correlation = 0;
                for sample = 1, samples do
                    correlation = correlation +
                        outputs[batch][sample][output] *
                        inputs[batch][sample][input];
                end
                correlation = correlation / samples;
                local old = memory[batch][output][input];
                local update = correlation;
                if self.kind == "oja" then update = update - energy * old; end
                result[batch][output][input] = self.decay * old +
                    self.learningRate * update;
            end
        end
    end
    return result;
end

local plasticity = {};
plasticity.__index = plasticity;
function plasticity.new(updateRule, alpha, options)
    options = options or {};
    assert(updateRule and updateRule.update, "plasticity rule is required");
    assert(not options.memoryLimit or options.memoryLimit > 0,
        "memoryLimit must be positive");
    return setmetatable({
        rule = updateRule,
        alpha = alpha or 1,
        detachUpdates = options.detachUpdates ~= false,
        memoryLimit = options.memoryLimit,
    }, plasticity);
end
function plasticity:newState(batchSize, weight)
    assert(weight:dim() == 2, "weight must be output x input");
    return state.zeros(batchSize, weight:size(1), weight:size(2), weight);
end
function plasticity:effectiveWeight(weight, currentState)
    assert(weight:dim() == 2 and currentState.memory:size(2) ==
        weight:size(1) and currentState.memory:size(3) == weight:size(2),
        "state memory is incompatible with weight");
    local result = currentState.memory:clone();
    for batch = 1, result:size(1) do
        result[batch]:mul(self.alpha):add(weight);
    end
    return result;
end
function plasticity:update(currentState, pre, post)
    local memory = self.rule:update(pre, post, currentState.memory);
    if self.memoryLimit then
        memory:clamp(-self.memoryLimit, self.memoryLimit);
    end
    return state.new(memory, currentState.updates + 1);
end

local linear = {};
linear.__index = linear;
function linear.new(inputSize, outputSize, plastic, withBias)
    assert(inputSize > 0 and outputSize > 0, "feature sizes must be positive");
    local weight = torch.Tensor(outputSize, inputSize)
        :normal(0, 1 / math.sqrt(inputSize));
    local bias = nil;
    if withBias ~= false then bias = torch.Tensor(outputSize):zero(); end
    return setmetatable({
        weight = weight,
        bias = bias,
        plasticity = plastic,
    }, linear);
end
function linear:newState(batchSize)
    return self.plasticity:newState(batchSize, self.weight);
end
function linear:forward(input, currentState, update)
    assert(input:dim() == 2 and input:size(1) ==
        currentState.memory:size(1), "input must be batch x feature");
    assert(input:size(2) == self.weight:size(2),
        "input feature count does not match weight");
    local weights = self.plasticity:effectiveWeight(self.weight, currentState);
    local output = input.new(input:size(1), self.weight:size(1)):zero();
    for batch = 1, input:size(1) do
        output[batch]:mv(weights[batch], input[batch]);
        if self.bias then output[batch]:add(self.bias); end
    end
    if update == false then return output, currentState; end
    return output, self.plasticity:update(currentState, input, output);
end

module.PlasticityState = state;
module.HebbianRule = rule.hebbian;
module.OjaRule = rule.oja;
module.Plasticity = plasticity;
module.PlasticLinear = linear;
return module;
