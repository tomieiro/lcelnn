-- Lua port based on the Apache-2.0 celnn project.
local torch = require "torch";

local template = {};
template.__index = template;

local function tableValue(value)
    if not torch.isTensor(value) then return value; end
    if value:dim() == 0 then return value[1]; end
    local result = {};
    for index = 1, value:size(1) do
        if value:dim() == 1 then
            result[index] = value[index];
        else
            result[index] = tableValue(value[index]);
        end
    end
    return result;
end

function template.new(options)
    assert(type(options) == "table" and options.name,
        "template name is required");
    local feedback = torch.isTensor(options.feedback) and
        options.feedback:clone() or torch.Tensor(options.feedback);
    local control = torch.isTensor(options.control) and
        options.control:clone() or torch.Tensor(options.control);
    assert(feedback:dim() == control:dim(),
        "feedback and control dimensions must match");
    for dimension = 1, feedback:dim() do
        assert(feedback:size(dimension) == control:size(dimension),
            "feedback and control shapes must match");
        assert(feedback:size(dimension) % 2 == 1,
            "template dimensions must be odd");
    end
    local initialState = options.initial_state;
    if initialState and not torch.isTensor(initialState) then
        initialState = torch.Tensor(initialState);
    end
    return setmetatable({
        name = options.name,
        feedback = feedback,
        control = control,
        bias = options.bias or 0,
        initialState = initialState and initialState:clone() or nil,
        description = options.description or "",
        tags = options.tags or {},
        metadata = options.metadata or {},
    }, template);
end

function template:copy()
    return template.new({
        name = self.name,
        feedback = self.feedback,
        control = self.control,
        bias = self.bias,
        initial_state = self.initialState,
        description = self.description,
        tags = self.tags,
        metadata = self.metadata,
    });
end

function template:validate()
    assert(self.name and self.feedback:dim() == self.control:dim(),
        "invalid template");
    for dimension = 1, self.feedback:dim() do
        assert(self.feedback:size(dimension) == self.control:size(dimension)
            and self.feedback:size(dimension) % 2 == 1,
            "invalid template shape");
    end
    return self;
end

function template:withBias(bias)
    local result = self:copy();
    result.bias = bias;
    return result;
end

function template:asArrays()
    return self.feedback:clone(), self.control:clone(),
        self.initialState and self.initialState:clone() or nil;
end

function template:toDict()
    return {
        name = self.name,
        feedback = tableValue(self.feedback),
        control = tableValue(self.control),
        bias = tableValue(self.bias),
        initial_state = self.initialState and
            tableValue(self.initialState) or nil,
        description = self.description,
        tags = self.tags,
        metadata = self.metadata,
    };
end

function template.fromDict(data)
    return template.new(data);
end

return template;
