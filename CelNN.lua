-- Derived from celnn (Apache-2.0); this file is the Lua port.
-- https://github.com/delbem-research/celnn
local activations = require "CelNN.Activations";
local differentiable = require "CelNN.Differentiable";
local domains = require "CelNN.Domains";
local plasticity = require "CelNN.Plasticity";
local template = require "CelNN.Template";
local templates = require "CelNN.Templates";
local utils = require "CelNN.Utils";
local visualization = require "CelNN.Visualization";
local hasCutorch = pcall(require, "cutorch");
local torch = require "torch";

local module = {};
local unpack = table.unpack or unpack;
local network = {};
network.__index = network;

local function asTensor(value, name)
    if torch.isTensor(value) then return value; end
    assert(type(value) == "table", name .. " must be a tensor or table");
    return torch.Tensor(value);
end

local function shapeOf(value)
    local shape = {};
    for dimension = 1, value:dim() do
        shape[dimension] = value:size(dimension);
    end
    return shape;
end

local function sameShape(first, second)
    if first:dim() ~= second:dim() then return false; end
    for dimension = 1, first:dim() do
        if first:size(dimension) ~= second:size(dimension) then
            return false;
        end
    end
    return true;
end

local function eachIndex(shape, visit, index, dimension)
    index, dimension = index or {}, dimension or 1;
    for value = 1, shape[dimension] do
        index[dimension] = value;
        if dimension == #shape then
            visit(index);
        else
            eachIndex(shape, visit, index, dimension + 1);
        end
    end
end

local function getValue(value, index)
    return value:get(unpack(index));
end

local function putValue(value, index, item)
    local args = {};
    for i = 1, #index do args[i] = index[i]; end
    args[#args + 1] = item;
    value:set(unpack(args));
end

local function broadcastValue(value, index, targetShape)
    if not torch.isTensor(value) then return value; end
    assert(value:dim() <= #targetShape, "bias cannot broadcast to state shape");
    local sourceIndex = {};
    local shift = #targetShape - value:dim();
    for dimension = 1, value:dim() do
        local extent = value:size(dimension);
        local target = targetShape[shift + dimension];
        assert(extent == 1 or extent == target,
            "bias cannot broadcast to state shape");
        sourceIndex[dimension] = extent == 1 and 1 or
            index[shift + dimension];
    end
    return getValue(value, sourceIndex);
end

local function mapIndex(index, length, mode)
    if index >= 1 and index <= length then return index; end
    if mode == "constant" then return nil; end
    if mode == "nearest" then return math.max(1, math.min(length, index)); end
    if mode == "wrap" then return ((index - 1) % length) + 1; end
    if length == 1 then return 1; end
    if mode == "reflect" then
        while index < 1 or index > length do
            if index < 1 then index = 2 - index;
            else index = 2 * length - index; end
        end
    elseif mode == "mirror" then
        while index < 1 or index > length do
            if index < 1 then index = 1 - index;
            else index = 2 * length + 1 - index; end
        end
    end
    return index;
end

local function aggregate(values, weights, mode, constant)
    assert(values:dim() == weights:dim(),
        "input and template dimensions must match");
    assert(values:dim() == 1 or values:dim() == 2,
        "only 1D and 2D tensors are supported");
    local valueShape, weightShape = shapeOf(values), shapeOf(weights);
    for dimension = 1, #weightShape do
        assert(weightShape[dimension] % 2 == 1,
            "template dimensions must be odd");
    end
    local result = values.new():resizeAs(values):zero();
    local weightIndex, valueIndex = {}, {};
    eachIndex(valueShape, function(position)
        local sum = 0;
        eachIndex(weightShape, function(offset)
            local inside = true;
            for dimension = 1, #valueShape do
                local mapped = mapIndex(position[dimension] + offset[dimension]
                    - math.floor(weightShape[dimension] / 2) - 1,
                    valueShape[dimension], mode);
                if mapped == nil then inside = false;
                else valueIndex[dimension] = mapped; end
            end
            local sample = inside and getValue(values, valueIndex) or constant;
            for dimension = 1, #weightShape do
                weightIndex[dimension] = offset[dimension];
            end
            sum = sum + getValue(weights, weightIndex) * sample;
        end);
        putValue(result, position, sum);
    end);
    return result;
end

local function toTable(value)
    if not torch.isTensor(value) then return value; end
    if value:dim() == 0 then return value[1]; end
    local result = {};
    for index = 1, value:size(1) do
        result[index] = value:dim() == 1 and value[index] or
            toTable(value[index]);
    end
    return result;
end

function network.new(options)
    options = options or {};
    local input = asTensor(options.input, "input");
    if options.dtype then input = input:type(options.dtype); end
    local device = (options.device or "cpu"):lower();
    assert(device == "cpu" or device == "gpu" or device == "auto",
        "device must be cpu, gpu, or auto");
    if device == "gpu" or device == "auto" then
        if hasCutorch then
            input, device = input:cuda(), "gpu";
        elseif device == "gpu" then
            error("GPU execution requires Torch cutorch");
        else
            device = "cpu";
        end
    end
    assert(input:dim() == 1 or input:dim() == 2,
        "input must be 1D or 2D");
    local feedback = options.feedback;
    if feedback == nil then
        local dimensions = {};
        for i = 1, input:dim() do dimensions[i] = 3; end
        feedback = input.new(unpack(dimensions)):zero();
        local center = {};
        for i = 1, input:dim() do center[i] = 2; end
        putValue(feedback, center, 1);
    else
        feedback = asTensor(feedback, "feedback");
    end
    feedback = feedback:typeAs(input);
    local control = options.control and asTensor(options.control, "control")
        or feedback.new():resizeAs(feedback):zero();
    control = control:typeAs(input);
    assert(sameShape(feedback, control) and
        feedback:dim() == input:dim(), "template shape must match input");
    for dimension = 1, feedback:dim() do
        assert(feedback:size(dimension) % 2 == 1,
            "template dimensions must be odd");
    end
    local initial = options.initial_state and
        asTensor(options.initial_state, "initial_state") or
        input.new():resizeAs(input):zero();
    initial = initial:typeAs(input);
    assert(sameShape(input, initial),
        "initial_state shape must match input");
    local activation = options.activation or "piecewiseLinear";
    local activationFn = type(activation) == "function" and activation or
        activations[activation];
    assert(activationFn, "unknown activation: " .. tostring(activation));
    local boundary = (options.boundary or "constant"):lower()
        :gsub("^%s*(.-)%s*$", "%1");
    assert(({constant=true, wrap=true, reflect=true, nearest=true,
        mirror=true})[boundary], "unknown boundary mode");
    local bias = options.bias or 0;
    if torch.isTensor(bias) then bias = bias:typeAs(input); end
    return setmetatable({
        input = input:clone(),
        state = initial:clone(),
        initialState = initial:clone(),
        feedback = feedback:clone(),
        control = control:clone(),
        bias = bias,
        activation = activation,
        activationFn = activationFn,
        boundary = boundary,
        boundaryValue = options.boundary_value or 0,
        metadata = options.metadata or {},
        device = device,
        stateShape = shapeOf(input),
        dtype = input:type(),
    }, network);
end

function network.fromTemplate(value, input, options)
    options = options or {};
    return network.new({
        input = input,
        initial_state = options.initial_state or value.initialState,
        feedback = value.feedback,
        control = value.control,
        bias = value.bias,
        activation = options.activation,
        boundary = options.boundary,
        boundary_value = options.boundary_value,
        metadata = options.metadata,
    });
end

function network:output(state)
    state = state or self.state;
    local result = state.new():resizeAs(state);
    eachIndex(shapeOf(state), function(index)
        putValue(result, index, self.activationFn(getValue(state, index)));
    end);
    return result;
end

function network:drive(state)
    state = state or self.state;
    local result = aggregate(self:output(state), self.feedback,
        self.boundary, self.boundaryValue);
    result = result + aggregate(self.input, self.control,
        self.boundary, self.boundaryValue);
    eachIndex(shapeOf(result), function(index)
        putValue(result, index, getValue(result, index) +
            broadcastValue(self.bias, index, shapeOf(result)));
    end);
    return result;
end

function network:derivative(state)
    state = state or self.state;
    return -state + self:drive(state);
end

function network:step(dt)
    assert(type(dt) == "number" and dt > 0, "dt must be positive");
    self.state = self.state + dt * self:derivative(self.state);
    return self.state:clone();
end

function network:reset(state)
    state = state and asTensor(state, "initial_state") or self.initialState;
    assert(sameShape(state, self.input),
        "initial_state shape must match input");
    self.initialState = state:clone();
    self.state = state:clone();
end

function network:run(config)
    config = config or {};
    local start = config.t_start or 0;
    local finish = config.t_end or 1;
    local dt = config.dt or .01;
    assert(dt > 0 and finish >= start, "invalid simulation range or dt");
    local solver = (config.solver or "euler"):lower()
        :gsub("%-", "_");
    if solver == "semi_implicit" then solver = "semi_implicit_euler"; end
    assert(solver == "euler" or solver == "semi_implicit_euler",
        "solver must be euler or semi_implicit_euler");
    local times, states = {start}, {self.state:clone()};
    local outputs = {self:output()};
    local time, count, previous = start, 0, self.state:clone();
    while time < finish do
        previous = self.state:clone();
        local remaining = finish - time;
        local stepSize = math.min(dt, remaining);
        if solver == "euler" then
            self:step(stepSize);
        else
            self.state = (self.state + stepSize * self:drive(self.state)) /
                (1 + stepSize);
        end
        if remaining <= dt then time = finish;
        else time = time + stepSize; end
        count = count + 1;
        if config.progress then
            print(string.format("[celnn] step %d, time %.6g", count, time));
        end
        if config.return_trajectory then
            local stride = config.store_every or 1;
            if stride > 0 and (count % stride == 0 or time >= finish) then
                times[#times + 1] = time;
                states[#states + 1] = self.state:clone();
                outputs[#outputs + 1] = self:output();
            end
        end
    end
    local delta = (self.state - previous):abs():max();
    local metadata = {
        solver = solver,
        boundary = self.boundary,
        backend = "torch",
        device = self.device,
        shape = shapeOf(self.state),
    };
    if config.stability_checks ~= false and dt > 1 then
        metadata.warnings = {
            "dt > 1.0 may be numerically unstable for explicit schemes.",
        };
    end
    local result = {
        state = self.state:clone(),
        output = self:output(),
        time = config.return_trajectory and times or {finish},
        metadata = metadata,
        convergence = {
            max_abs_state_delta = delta,
            approx_converged = delta < 1e-6,
        },
        hasTrajectory = config.return_trajectory or false,
        finalTime = finish,
    };
    if config.return_trajectory then
        result.trajectory_state = states;
        result.trajectory_output = outputs;
    end
    return result;
end

function network:toDict()
    assert(type(self.activation) == "string",
        "custom activations cannot be serialized automatically");
    return {
        input = toTable(self.input),
        initial_state = toTable(self.initialState),
        current_state = toTable(self.state),
        feedback = toTable(self.feedback),
        control = toTable(self.control),
        bias = toTable(self.bias),
        activation = self.activation,
        boundary = self.boundary,
        boundary_value = self.boundaryValue,
        dtype = self.dtype,
        metadata = self.metadata,
    };
end

function network.fromDict(data, device)
    local result = network.new({
        input = data.input,
        initial_state = data.initial_state,
        feedback = data.feedback,
        control = data.control,
        bias = data.bias,
        activation = data.activation,
        boundary = data.boundary,
        boundary_value = data.boundary_value,
        dtype = data.dtype,
        device = device or data.device,
        metadata = data.metadata,
    });
    if data.current_state then result.state = asTensor(data.current_state,
        "current_state"):clone(); end
    return result;
end

local simulationConfig = {};
simulationConfig.__index = simulationConfig;
function simulationConfig.new(options)
    options = options or {};
    local result = setmetatable({
        t_start = options.t_start or 0,
        t_end = options.t_end or 1,
        dt = options.dt or .01,
        solver = (options.solver or "euler"):lower()
            :gsub("%-", "_"),
        return_trajectory = options.return_trajectory or false,
        store_every = options.store_every or 1,
        stability_checks = options.stability_checks ~= false,
        progress = options.progress or false,
    }, simulationConfig);
    assert(result.dt > 0 and result.t_end >= result.t_start,
        "invalid simulation range or dt");
    assert(result.store_every > 0, "store_every must be positive");
    return result;
end
function simulationConfig:timePoints()
    local span = self.t_end - self.t_start;
    if span == 0 then return {self.t_start}; end
    local steps = math.floor(span / self.dt);
    local result = {};
    for index = 0, steps do
        result[#result + 1] = self.t_start + self.dt * index;
    end
    if result[#result] < self.t_end then
        result[#result + 1] = self.t_end;
    else
        result[#result] = self.t_end;
    end
    return result;
end
function simulationConfig:toDict()
    local result = {};
    for key, value in pairs(self) do result[key] = value; end
    return result;
end
function simulationConfig.fromDict(data)
    return simulationConfig.new(data);
end

module.CellularNetwork = network;
module.Template = template;
module.SimulationConfig = simulationConfig;
module.activations = activations;
module.piecewiseLinear = activations.piecewiseLinear;
module.saturatedLinear = activations.saturatedLinear;
module.identity = activations.identity;
module.tanhActivation = activations.tanhActivation;
module.sigmoidActivation = activations.sigmoidActivation;
module.signActivation = activations.signActivation;
module.reluActivation = activations.reluActivation;
module.TemplateRegistry = templates.TemplateRegistry;
module.builtinTemplates = templates.builtinTemplates;
module.templates = templates;
module.domains = domains;
module.utils = utils;
module.visualization = visualization;
module.plotSignal = visualization.plotSignal;
module.plotGrid = visualization.plotGrid;
module.plotTrajectory = visualization.plotTrajectory;
module.plotSignalComparison = visualization.plotSignalComparison;
module.randomGrid = domains.randomGrid;
module.impulseGrid = domains.impulseGrid;
module.checkerboardGrid = domains.checkerboardGrid;
module.coordinateGrid = domains.coordinateGrid;
module.normalizeSignal = domains.normalizeSignal;
module.generateSineWave = domains.generateSineWave;
module.generateNoisySine = domains.generateNoisySine;
module.normalizeImage = domains.normalizeImage;
module.denormalizeImage = domains.denormalizeImage;
module.imageToArray = domains.imageToArray;
module.arrayToImage = domains.arrayToImage;
module.loadGrayscale = domains.loadGrayscale;
module.saveGrayscale = domains.saveGrayscale;
module.PlasticityState = plasticity.PlasticityState;
module.HebbianRule = plasticity.HebbianRule;
module.OjaRule = plasticity.OjaRule;
module.Plasticity = plasticity.Plasticity;
module.PlasticLinear = plasticity.PlasticLinear;
module.DifferentiableCellularNetwork =
    differentiable.DifferentiableCellularNetwork;

return module;
