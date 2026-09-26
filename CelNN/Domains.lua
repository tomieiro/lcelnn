-- Lua port based on the Apache-2.0 celnn project.
local visualization = require "CelNN.Visualization";
local hasImage, image = pcall(require, "image");
local torch = require "torch";

local module = {};
local unpack = table.unpack or unpack;

local function make(shape)
    return torch.Tensor(unpack(shape)):zero();
end

local function put(tensor, index, value)
    local args = {};
    for i = 1, #index do args[i] = index[i]; end
    args[#args + 1] = value;
    tensor:set(unpack(args));
end

local function eachIndex(shape, visit, index, dimension)
    index, dimension = index or {}, dimension or 1;
    for i = 1, shape[dimension] do
        index[dimension] = i;
        if dimension == #shape then
            visit(index);
        else
            eachIndex(shape, visit, index, dimension + 1);
        end
    end
end

function module.randomGrid(shape, low, high, seed)
    low, high = low or -1, high or 1;
    assert(low <= high, "invalid random range");
    if seed ~= nil then math.randomseed(seed); end
    local result = make(shape);
    eachIndex(shape, function(index)
        put(result, index, low + (high - low) * math.random());
    end);
    return result;
end

function module.impulseGrid(shape, location, amplitude)
    local result = make(shape);
    local index = {};
    for dimension = 1, #shape do
        local coordinate = location and location[dimension];
        if coordinate == nil then
            coordinate = math.floor((shape[dimension] - 1) / 2);
        end
        assert(coordinate >= 0 and coordinate < shape[dimension],
            "location out of bounds");
        index[dimension] = coordinate + 1;
    end
    assert(not location or #location == #shape,
        "location dimensionality must match shape");
    put(result, index, amplitude or 1);
    return result;
end

function module.checkerboardGrid(shape, values)
    assert(#shape == 2, "checkerboard shape must be 2D");
    values = values or {-1, 1};
    local result = make(shape);
    for row = 1, shape[1] do
        for column = 1, shape[2] do
            result[row][column] = ((row + column) % 2 == 0) and
                values[1] or values[2];
        end
    end
    return result;
end

function module.coordinateGrid(shape, normalize)
    local outputShape = {};
    for i = 1, #shape do outputShape[i] = shape[i]; end
    outputShape[#outputShape + 1] = #shape;
    local result = make(outputShape);
    eachIndex(shape, function(position)
        for axis = 1, #shape do
            local coordinate = position[axis] - 1;
            if normalize ~= false then
                coordinate = shape[axis] > 1 and
                    (2 * coordinate / (shape[axis] - 1) - 1) or 0;
            end
            local index = {};
            for i = 1, #position do index[i] = position[i]; end
            index[#index + 1] = axis;
            put(result, index, coordinate);
        end
    end);
    return result;
end

function module.normalizeSignal(input, lower, upper)
    lower, upper = lower or -1, upper or 1;
    local minimum, maximum = input:min(), input:max();
    if minimum == maximum then
        return input.new():resizeAs(input):fill((lower + upper) / 2);
    end
    return (input - minimum) / (maximum - minimum) * (upper - lower) + lower;
end

function module.generateSineWave(samples, cycles, amplitude)
    samples, cycles, amplitude = samples or 512, cycles or 4, amplitude or 1;
    assert(samples >= 1, "samples must be positive");
    local result = torch.Tensor(samples);
    for i = 1, samples do
        local phase = samples == 1 and 0 or
            (i - 1) * (2 * math.pi * cycles) / (samples - 1);
        result[i] = amplitude * math.sin(phase);
    end
    return result;
end

function module.generateNoisySine(samples, cycles, amplitude, noiseScale, seed)
    samples, noiseScale = samples or 512, noiseScale or .1;
    if seed ~= nil then math.randomseed(seed); end
    local result = module.generateSineWave(samples, cycles, amplitude);
    for i = 1, samples do
        local first = math.max(math.random(), 1e-15);
        local second = math.random();
        local noise = math.sqrt(-2 * math.log(first)) *
            math.cos(2 * math.pi * second);
        result[i] = result[i] + noiseScale * noise;
    end
    return result;
end

function module.normalizeImage(input)
    if not torch.isTensor(input) then input = torch.Tensor(input); end
    if input:nElement() == 0 then return input:double(); end
    local minimum, maximum = input:min(), input:max();
    if minimum >= -1 and maximum <= 1 then return input:double(); end
    return input:double() / 127.5 - 1;
end

function module.denormalizeImage(input)
    return ((input:double():clamp(-1, 1) + 1) * 127.5):round():byte();
end

function module.imageToArray(input)
    return module.normalizeImage(input);
end

function module.arrayToImage(input)
    assert(hasImage, "arrayToImage requires the Torch image package");
    return module.denormalizeImage(input);
end

function module.loadGrayscale(path)
    assert(hasImage, "loadGrayscale requires the Torch image package");
    local loaded = image.load(path, 1, "double");
    if loaded:dim() == 3 then loaded = loaded:select(1, 1); end
    return module.normalizeImage(loaded);
end

function module.saveGrayscale(input, path)
    assert(hasImage, "saveGrayscale requires the Torch image package");
    local pixels = module.denormalizeImage(input):float():div(255);
    if pixels:dim() == 2 then pixels = pixels:view(1, pixels:size(1),
        pixels:size(2)); end
    image.save(path, pixels);
    return path;
end

module.plotSignal = visualization.plotSignal;
module.plotSignalComparison = visualization.plotSignalComparison;

return module;
