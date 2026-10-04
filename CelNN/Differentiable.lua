-- Lua port based on the Apache-2.0 celnn project.
local activations = require "CelNN.Activations";
local template = require "CelNN.Template";
local torch = require "torch";

local module = {};
local network = {};
network.__index = network;

local function activate(name, value)
    local fn = type(name) == "function" and name or activations[name];
    assert(fn, "unknown activation: " .. tostring(name));
    return fn(value);
end

local function boundaryIndex(index, length, mode)
    if index >= 1 and index <= length then return index; end
    if mode == "constant" then return nil; end
    if mode == "nearest" then return math.max(1, math.min(length, index)); end
    if mode == "wrap" then return ((index - 1) % length) + 1; end
    if length == 1 then return 1; end
    while index < 1 or index > length do
        if mode == "reflect" then
            if index < 1 then index = 2 - index;
            else index = 2 * length - index; end
        else
            if index < 1 then index = 1 - index;
            else index = 2 * length + 1 - index; end
        end
    end
    return index;
end

function network.new(radius, options)
    options = options or {};
    assert(radius >= 0 and radius == math.floor(radius),
        "radius must be a non-negative integer");
    local channels = options.channels;
    assert(not channels or channels >= 1,
        "channels must be positive");
    assert(not options.sharedChannels or channels,
        "sharedChannels requires channels");
    local causal = options.causal or false;
    local span = causal and radius + 1 or 2 * radius + 1;
    local weightChannels = channels and
        (options.sharedChannels and 1 or channels) or nil;
    local feedback = weightChannels and torch.Tensor(span, weightChannels)
        :zero() or torch.Tensor(span):zero();
    local control = feedback:clone():zero();
    local bias = channels and torch.Tensor(weightChannels):zero() or 0;
    local method = (options.method or "euler"):lower()
        :gsub("%-", "_");
    if method == "semi_implicit" then method = "semi_implicit_euler"; end
    assert(method == "euler" or method == "semi_implicit_euler",
        "method must be euler or semi_implicit_euler");
    local boundary = (options.boundary or "constant"):lower()
        :gsub("^%s*(.-)%s*$", "%1");
    assert(({constant=true, wrap=true, reflect=true, nearest=true,
        mirror=true})[boundary], "unknown boundary mode");
    assert((options.dt or .1) > 0 and (options.steps or 10) >= 0,
        "invalid integration settings");
    return setmetatable({
        radius = radius,
        channels = channels,
        sharedChannels = options.sharedChannels or false,
        causal = causal,
        activation = options.activation or "piecewiseLinear",
        boundary = boundary,
        boundaryValue = options.boundaryValue or 0,
        dt = options.dt or .1,
        steps = options.steps or 10,
        method = method,
        trainable = options.trainable ~= false,
        autograd = false,
        feedback = feedback,
        control = control,
        bias = bias,
    }, network);
end

function network.fromTemplate(value, radius, options)
    options = options or {};
    assert(not options.causal,
        "fromTemplate requires the full centered template");
    local result = network.new(radius, options);
    local feedback = value.feedback;
    local control = value.control;
    local expected = 2 * radius + 1;
    assert(feedback:dim() == 1 and feedback:size(1) == expected and
        control:dim() == 1 and control:size(1) == expected,
        "template extent must match radius");
    for offset = 1, expected do
        if result.channels then
            for channel = 1, result.feedback:size(2) do
                result.feedback[offset][channel] = feedback[offset];
                result.control[offset][channel] = control[offset];
            end
        else
            result.feedback[offset], result.control[offset] =
                feedback[offset], control[offset];
        end
    end
    if result.channels then result.bias:fill(value.bias); end
    return result;
end

function network:toTemplate(name)
    assert(not self.causal,
        "causal templates do not match the centered Template contract");
    assert(not self.channels or self.channels == 1,
        "only scalar or single-channel weights convert to a template");
    local feedback = self.channels and self.feedback:select(2, 1) or
        self.feedback:clone();
    local control = self.channels and self.control:select(2, 1) or
        self.control:clone();
    local bias = self.channels and self.bias[1] or self.bias;
    return template.new({name = name or "learned",
        feedback = feedback, control = control, bias = bias});
end

function network:_shape(value, name)
    assert(torch.isTensor(value), name .. " must be a Torch tensor");
    if self.channels then
        assert(value:dim() >= 2 and value:size(value:dim()) == self.channels,
            name .. " must end with the channel dimension");
    else
        assert(value:dim() >= 1, name .. " must have a spatial dimension");
    end
end

function network:drive(state, input)
    self:_shape(state, "state");
    self:_shape(input, "input");
    assert(state:isSameSizeAs(input), "state and input shapes differ");
    local hasChannels = self.channels ~= nil;
    local channels = self.channels or 1;
    local length = state:size(state:dim() - (hasChannels and 1 or 0));
    local batchSize = 1;
    local batchDimensions = state:dim() - (hasChannels and 2 or 1);
    for dimension = 1, batchDimensions do
        batchSize = batchSize * state:size(dimension);
    end
    local stateView, inputView;
    if hasChannels then
        stateView = state:contiguous():view(batchSize, length, channels);
        inputView = input:contiguous():view(batchSize, length, channels);
    else
        stateView = state:contiguous():view(batchSize, length);
        inputView = input:contiguous():view(batchSize, length);
    end
    local result = state.new():resizeAs(state):zero();
    local resultView;
    if hasChannels then
        resultView = result:view(batchSize, length, channels);
    else
        resultView = result:view(batchSize, length);
    end
    for batch = 1, batchSize do
        for position = 1, length do
            for channel = 1, channels do
                local total = 0;
                for offset = 1, self.feedback:size(1) do
                    local source = position + offset -
                        (self.causal and self.feedback:size(1) or
                        self.radius + 1);
                    source = boundaryIndex(source, length, self.boundary);
                    local sourceState = source and
                        (hasChannels and stateView[batch][source][channel] or
                        stateView[batch][source]);
                    local output = source and
                        activate(self.activation, sourceState) or
                        self.boundaryValue;
                    local weightChannel = self.sharedChannels and 1 or channel;
                    local feedback = hasChannels and
                        self.feedback[offset][weightChannel] or
                        self.feedback[offset];
                    local control = hasChannels and
                        self.control[offset][weightChannel] or
                        self.control[offset];
                    local inputValue = source and
                        (hasChannels and inputView[batch][source][channel] or
                        inputView[batch][source]) or self.boundaryValue;
                    total = total + feedback * output + control * inputValue;
                end
                local bias = hasChannels and self.bias[
                    self.sharedChannels and 1 or channel] or self.bias;
                if hasChannels then
                    resultView[batch][position][channel] = total + bias;
                else
                    resultView[batch][position] = total + bias;
                end
            end
        end
    end
    return result;
end

function network:derivative(state, input)
    return -state + self:drive(state, input);
end

function network:step(state, input, extraDrive)
    local drive = self:drive(state, input);
    if extraDrive then
        assert(extraDrive:isSameSizeAs(state), "extraDrive shape differs");
        drive = drive + extraDrive;
    end
    if self.method == "semi_implicit_euler" then
        assert(not extraDrive, "extraDrive requires Euler integration");
        return (state + self.dt * drive) / (1 + self.dt);
    end
    return state + self.dt * (-state + drive + (extraDrive or 0));
end

function network:forward(input, state)
    state = state or input.new():resizeAs(input):zero();
    assert(input:isSameSizeAs(state), "state and input shapes differ");
    for _ = 1, self.steps do state = self:step(state, input); end
    return state;
end

module.DifferentiableCellularNetwork = network;
return module;
