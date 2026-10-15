-- Lua port based on the Apache-2.0 celnn project.
local celNN = require "CelNN";
local torch = require "torch";

local module = {};
local gaConfig = {};
gaConfig.__index = gaConfig;
function gaConfig.new(options)
    options = options or {};
    local result = setmetatable({
        popSize = options.popSize or 30,
        generations = options.generations or 20,
        bounds = options.bounds or {-2, 2},
        biasBounds = options.biasBounds,
        crossoverProbability = options.crossoverProbability or .5,
        mutationProbability = options.mutationProbability or .2,
        mutationSigma = options.mutationSigma or .1,
        tournamentSize = options.tournamentSize or 3,
        eliteSize = options.eliteSize or 1,
        seedTemplate = options.seedTemplate,
    }, gaConfig);
    assert(result.popSize >= 2 and result.generations >= 0 and
        result.eliteSize >= 0 and result.eliteSize < result.popSize,
        "invalid population or generation size");
    assert(result.bounds[1] <= result.bounds[2], "invalid bounds");
    assert(result.tournamentSize >= 1, "tournamentSize must be positive");
    return result;
end

local sample = {};
sample.__index = sample;
function sample.new(input, target, initialState, weight)
    input = torch.isTensor(input) and input:clone() or torch.Tensor(input);
    target = torch.isTensor(target) and target:clone() or torch.Tensor(target);
    assert(input:isSameSizeAs(target), "sample input and target shapes differ");
    if initialState and not torch.isTensor(initialState) then
        initialState = torch.Tensor(initialState);
    end
    return setmetatable({input = input, target = target,
        initialState = initialState, weight = weight or 1}, sample);
end

local dataset = {};
dataset.__index = dataset;
function dataset.new(samples)
    assert(type(samples) == "table" and #samples > 0,
        "dataset must contain samples");
    local expected = samples[1].input;
    for _, item in ipairs(samples) do
        assert(item.input:isSameSizeAs(expected),
            "all sample input shapes must match");
    end
    return setmetatable({samples = samples}, dataset);
end
function dataset.fromPairs(inputs, targets, initialStates, weights)
    assert(#inputs == #targets, "input and target counts differ");
    assert(not initialStates or #initialStates == #inputs,
        "initial state count differs");
    assert(not weights or #weights == #inputs, "weight count differs");
    local samples = {};
    for index, input in ipairs(inputs) do
        samples[index] = sample.new(input, targets[index],
            initialStates and initialStates[index], weights and weights[index]);
    end
    return dataset.new(samples);
end

local function mse(target, output)
    assert(target:isSameSizeAs(output), "target and output shapes differ");
    return (output - target):pow(2):mean();
end

local function penalty(feedback, control, bias)
    return feedback:dot(feedback) + control:dot(control) + bias * bias;
end

local function flatten(value)
    local result = {};
    for index = 1, value:nElement() do
        result[index] = value:view(-1)[index];
    end
    return result;
end

local function normal()
    local first = math.max(math.random(), 1e-15);
    return math.sqrt(-2 * math.log(first)) *
        math.cos(2 * math.pi * math.random());
end

local trainer = {};
trainer.__index = trainer;
function trainer.new(seedTemplate, data, config, options)
    options = options or {};
    assert(seedTemplate and data and #data.samples > 0,
        "template and non-empty dataset are required");
    local ga = options.ga or gaConfig.new();
    local bounds = ga.bounds or {-2, 2};
    local biasBounds = ga.biasBounds or bounds;
    local result = setmetatable({
        template = seedTemplate,
        dataset = data,
        simulation = config or celNN.SimulationConfig.new(),
        popSize = ga.popSize or 30,
        generations = ga.generations or 20,
        bounds = bounds,
        biasBounds = biasBounds,
        crossoverProbability = ga.crossoverProbability or .5,
        mutationProbability = ga.mutationProbability or .2,
        mutationSigma = ga.mutationSigma or .1,
        tournamentSize = ga.tournamentSize or 3,
        eliteSize = ga.eliteSize or 1,
        seedTemplate = ga.seedTemplate,
        activation = options.activation or "piecewiseLinear",
        boundary = options.boundary or "reflect",
        boundaryValue = options.boundaryValue or 0,
        regularization = options.regularization or 0,
        lossFunction = options.lossFunction or mse,
        seed = options.seed,
    }, trainer);
    assert(result.popSize >= 2 and result.generations >= 0 and
        result.eliteSize >= 0 and result.eliteSize < result.popSize,
        "invalid genetic algorithm sizes");
    assert(result.tournamentSize >= 1, "tournamentSize must be positive");
    assert(bounds[1] <= bounds[2] and biasBounds[1] <= biasBounds[2],
        "invalid coefficient bounds");
    local seedValue = result.seedTemplate;
    if seedValue then
        assert(seedValue.feedback:isSameSizeAs(seedTemplate.feedback) and
            seedValue.control:isSameSizeAs(seedTemplate.control),
            "seed template shape must match search template");
    end
    return result;
end

function trainer:_genome(value)
    local result = flatten(value.feedback);
    local control = flatten(value.control);
    for _, item in ipairs(control) do result[#result + 1] = item; end
    result[#result + 1] = torch.isTensor(value.bias) and value.bias[1] or
        value.bias;
    return result;
end

function trainer:_decode(genome, suffix)
    local feedbackCount = self.template.feedback:nElement();
    local controlCount = self.template.control:nElement();
    local feedback = self.template.feedback.new():resizeAs(
        self.template.feedback);
    local control = self.template.control.new():resizeAs(
        self.template.control);
    for index = 1, feedbackCount do
        feedback:view(-1)[index] = genome[index];
    end
    for index = 1, controlCount do
        control:view(-1)[index] = genome[feedbackCount + index];
    end
    return celNN.Template.new({
        name = self.template.name .. "_" .. suffix,
        feedback = feedback,
        control = control,
        bias = genome[feedbackCount + controlCount + 1],
        description = self.template.description,
        tags = self.template.tags,
        metadata = self.template.metadata,
    });
end

function trainer:_evaluate(genome)
    local value = self:_decode(genome, "candidate");
    local total = 0;
    for _, item in ipairs(self.dataset.samples) do
        local ok, result = pcall(function()
            local net = celNN.CellularNetwork.new({
                input = item.input,
                initial_state = item.initialState,
                feedback = value.feedback,
                control = value.control,
                bias = value.bias,
                activation = self.activation,
                boundary = self.boundary,
                boundary_value = self.boundaryValue,
            });
            local output = net:run(self.simulation).output;
            return self.lossFunction(item.target, output);
        end);
        if not ok then return 1e12; end
        local loss = result;
        if loss ~= loss or loss == math.huge then return 1e12; end
        total = total + loss;
    end
    local result = total / #self.dataset.samples;
    if self.regularization > 0 then
        local feedback = value.feedback;
        local control = value.control;
        result = result + self.regularization * penalty(
            feedback, control, value.bias);
    end
    return result;
end

local function tournament(population, count, size)
    local result = {};
    for index = 1, count do
        local best = population[math.random(#population)];
        for _ = 2, size do
            local candidate = population[math.random(#population)];
            if candidate.fitness < best.fitness then best = candidate; end
        end
        local genome = {};
        for gene = 1, #best.genome do genome[gene] = best.genome[gene]; end
        result[index] = {genome = genome, fitness = best.fitness};
    end
    return result;
end

local function evaluateTemplate(networkFactory, config, data, options)
    options = options or {};
    local total = 0;
    local lastNetwork = nil;
    for _, item in ipairs(data.samples) do
        local ok, loss = pcall(function()
            local net = networkFactory(item);
            local output = net:run(config).output;
            lastNetwork = net;
            return (options.lossFunction or mse)(item.target, output);
        end);
        if not ok or loss ~= loss or loss == math.huge then return 1e12; end
        total = total + loss;
    end
    local mean = total / #data.samples;
    if options.regularization and options.regularization > 0 and
        lastNetwork then
        mean = mean + options.regularization * penalty(
            lastNetwork.feedback, lastNetwork.control, lastNetwork.bias);
    end
    return mean;
end

function trainer:run()
    if self.seed then math.randomseed(self.seed); end
    local seedGenome = self:_genome(self.seedTemplate or self.template);
    local feedbackCount = self.template.feedback:nElement();
    local genomeSize = #seedGenome;
    local population = {};
    local evaluations = 0;
    for index = 1, self.popSize do
        local genome = {};
        for gene = 1, genomeSize - 1 do
            genome[gene] = self.bounds[1] + math.random() *
                (self.bounds[2] - self.bounds[1]);
        end
        genome[genomeSize] = self.biasBounds[1] + math.random() *
            (self.biasBounds[2] - self.biasBounds[1]);
        if self.seedTemplate and index == 1 then
            for gene = 1, genomeSize do
                local sigmaIndex = gene <= feedbackCount and 1 or 2;
                local sigma = type(self.mutationSigma) == "number" and
                    self.mutationSigma or self.mutationSigma[sigmaIndex];
                local range = gene == genomeSize and self.biasBounds or
                    self.bounds;
                genome[gene] = math.max(range[1], math.min(range[2],
                    seedGenome[gene] + normal() * sigma));
            end
        end
        population[index] = {genome = genome};
        population[index].fitness = self:_evaluate(genome);
        evaluations = evaluations + 1;
    end
    local fitnessHistory, meanHistory, bestTemplates = {}, {}, {};
    for generation = 1, self.generations do
        table.sort(population,
            function(a, b) return a.fitness < b.fitness; end);
        local nextPopulation = {};
        for index = 1, self.eliteSize do
            nextPopulation[index] = population[index];
        end
        local children = tournament(population,
            self.popSize - self.eliteSize, self.tournamentSize);
        for index = 1, #children - 1, 2 do
            if math.random() < self.crossoverProbability and genomeSize > 1 then
                local first = math.random(genomeSize - 1);
                local second = math.random(first + 1, genomeSize);
                for gene = first, second do
                    children[index].genome[gene],
                        children[index + 1].genome[gene] =
                        children[index + 1].genome[gene],
                        children[index].genome[gene];
                end
                children[index].fitness = nil;
                children[index + 1].fitness = nil;
            end
        end
        for _, child in ipairs(children) do
            if math.random() < self.mutationProbability then
                for gene = 1, genomeSize do
                    if math.random() < 1 / genomeSize then
                        local sigma = type(self.mutationSigma) == "number" and
                            self.mutationSigma or self.mutationSigma[gene <=
                            feedbackCount and 1 or 2];
                        child.genome[gene] = child.genome[gene] +
                            normal() * sigma;
                    end
                    local range = gene == genomeSize and self.biasBounds or
                        self.bounds;
                    child.genome[gene] = math.max(range[1],
                        math.min(range[2], child.genome[gene]));
                end
                child.fitness = nil;
            end
            if child.fitness == nil then
                child.fitness = self:_evaluate(child.genome);
                evaluations = evaluations + 1;
            end
            nextPopulation[#nextPopulation + 1] = child;
        end
        population = nextPopulation;
        table.sort(population,
            function(a, b) return a.fitness < b.fitness; end);
        local total = 0;
        for _, individual in ipairs(population) do
            total = total + individual.fitness;
        end
        fitnessHistory[generation] = population[1].fitness;
        meanHistory[generation] = total / #population;
        bestTemplates[generation] = self:_decode(population[1].genome,
            "gen" .. generation);
    end
    table.sort(population, function(a, b) return a.fitness < b.fitness; end);
    local bestTemplate = self:_decode(population[1].genome, "best");
    bestTemplates[#bestTemplates + 1] = bestTemplate;
    return {
        bestTemplate = bestTemplate,
        bestFitness = population[1].fitness,
        fitnessHistory = fitnessHistory,
        meanFitnessHistory = meanHistory,
        bestPerGeneration = bestTemplates,
        nGenerations = self.generations,
        nEvaluations = evaluations,
        populationSize = self.popSize,
        config = {
            populationSize = self.popSize,
            generations = self.generations,
            bounds = self.bounds,
            biasBounds = self.biasBounds,
            crossoverProbability = self.crossoverProbability,
            mutationProbability = self.mutationProbability,
            mutationSigma = self.mutationSigma,
            tournamentSize = self.tournamentSize,
            eliteSize = self.eliteSize,
            regularization = self.regularization,
        },
        seed = self.seed,
    };
end

module.TrainingSample = sample;
module.TrainingDataset = dataset;
module.GAConfig = gaConfig;
module.mseLoss = mse;
module.l2Penalty = penalty;
module.evaluateTemplate = evaluateTemplate;
module.GATrainer = trainer;
return module;
