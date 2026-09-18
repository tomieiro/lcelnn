-- Lua port based on the Apache-2.0 celnn project.
local template = require "CelNN.Template";
local torch = require "torch";

local module = {};
local registry = {};
registry.__index = registry;

local definitions = {
    {
        "edge_detection",
        {{0,0,0},{0,1,0},{0,0,0}},
        {{-1,-1,-1},{-1,8,-1},{-1,-1,-1}},
        -1,
        "Demonstrative 3x3 edge detector.",
        {"image", "edge", "demo"},
    },
    {
        "inversion",
        {{0,0,0},{0,1,0},{0,0,0}},
        {{0,0,0},{1,1,1},{0,0,0}},
        -2,
        "Demonstrative inversion-like operator.",
        {"image", "logic", "demo"},
    },
    {
        "corner_detection",
        {{0,0,0},{0,1,0},{0,0,0}},
        {{-1,-1,-1},{-1,4,-1},{-1,-1,-1}},
        -5,
        "Demonstrative corner detector.",
        {"image", "corner", "demo"},
    },
    {
        "diagonal_line_detection",
        {{0,0,0},{0,1,0},{0,0,0}},
        {{-1,0,1},{0,1,0},{1,0,-1}},
        -4,
        "Demonstrative diagonal-line detector.",
        {"image", "line", "demo"},
    },
    {
        "smoothing_demo",
        {{.05,.1,.05},{.1,.4,.1},{.05,.1,.05}},
        {{0,0,0},{0,.8,0},{0,0,0}},
        0,
        "Simple smoothing-like template.",
        {"image", "smoothing", "demo"},
    },
    {
        "sharpening_demo",
        {{-.05,-.1,-.05},{-.1,1.5,-.1},{-.05,-.1,-.05}},
        {{0,0,0},{0,1,0},{0,0,0}},
        0,
        "Simple sharpening-like template.",
        {"image", "sharpening", "demo"},
    },
    {
        "not_demo", {0,1,0}, {0,-1,0}, 0,
        "Simple 1D inversion-like template.", {"logic", "signal", "demo"},
    },
    {
        "threshold_demo", {.2,1,.2}, {0,1,0}, -.25,
        "Thresholding-like 1D template.",
        {"logic", "signal", "threshold", "demo"},
    },
    {
        "diffusion_like",
        {{0,.2,0},{.2,.2,.2},{0,.2,0}},
        {{0,0,0},{0,0,0},{0,0,0}},
        0,
        "Local smoothing template.",
        {"pattern", "diffusion", "demo"},
    },
    {
        "local_excitation_global_damping_demo",
        {
            {-.02,-.03,-.04,-.03,-.02},
            {-.03,.02,.05,.02,-.03},
            {-.04,.05,1.2,.05,-.04},
            {-.03,.02,.05,.02,-.03},
            {-.02,-.03,-.04,-.03,-.02},
        },
        {
            {0,0,0,0,0}, {0,0,0,0,0}, {0,0,0,0,0},
            {0,0,0,0,0}, {0,0,0,0,0},
        },
        -.1,
        "Excitatory-center pattern template.",
        {"pattern", "diffusion", "demo"},
    },
    {
        "one_d_diffusion", {.15,.70,.15}, {0,0,0}, 0,
        "Simple 1D smoothing template.", {"signal", "diffusion", "demo"},
    },
    {
        "two_d_laplacian_demo",
        {{0,.2,0},{.2,.2,.2},{0,.2,0}},
        {{0,1,0},{1,-4,1},{0,1,0}},
        0,
        "Laplacian-like demonstrative stencil.",
        {"grid", "laplacian", "diffusion", "demo"},
    },
};

function registry.new()
    return setmetatable({_templates = {}}, registry);
end

function registry:register(value, overwrite)
    assert(value and value.name, "template is required");
    assert(overwrite or not self._templates[value.name],
        "template already registered: " .. value.name);
    self._templates[value.name] = value:copy();
end

function registry:get(name)
    assert(self._templates[name],
        "template is not registered: " .. tostring(name));
    return self._templates[name]:copy();
end

function registry:names()
    local result = {};
    for name in pairs(self._templates) do result[#result + 1] = name; end
    table.sort(result);
    return result;
end

function registry:list()
    local result = {};
    for _, name in ipairs(self:names()) do
        result[#result + 1] = self:get(name);
    end
    return result;
end

function registry:remove(name)
    local value = self:get(name);
    self._templates[name] = nil;
    return value;
end

function registry:toDict()
    local result = {templates = {}};
    for _, value in ipairs(self:list()) do
        result.templates[#result.templates + 1] = value:toDict();
    end
    return result;
end

function registry.fromDict(data)
    local result = registry.new();
    for _, value in ipairs(data.templates or {}) do
        result:register(template.fromDict(value));
    end
    return result;
end

function module.builtinTemplates()
    local result = registry.new();
    for _, row in ipairs(definitions) do
        result:register(template.new({
            name = row[1],
            feedback = torch.Tensor(row[2]),
            control = torch.Tensor(row[3]),
            bias = row[4],
            description = row[5],
            tags = row[6],
        }));
    end
    return result;
end

module.TemplateRegistry = registry;
for _, row in ipairs(definitions) do
    module[row[1]:upper()] = template.new({
        name = row[1],
        feedback = torch.Tensor(row[2]),
        control = torch.Tensor(row[3]),
        bias = row[4],
        description = row[5],
        tags = row[6],
    });
end

return module;
