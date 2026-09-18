-- Lua port based on the Apache-2.0 celnn project.
local torch = require "torch";

local module = {};

function module.toFloatArray(value)
    if torch.isTensor(value) then return value:float(); end
    return torch.Tensor(value):float();
end

function module.centerIndex(shape)
    local center = {};
    for dimension, extent in ipairs(shape) do
        center[dimension] = math.floor(extent / 2);
    end
    return center;
end

function module.isBroadcastable(value, targetShape)
    local sourceShape = {};
    if torch.isTensor(value) then
        for dimension = 1, value:dim() do
            sourceShape[dimension] = value:size(dimension);
        end
    end
    if #sourceShape > #targetShape then return false; end
    local shift = #targetShape - #sourceShape;
    for dimension = 1, #sourceShape do
        local source = sourceShape[dimension];
        local target = targetShape[shift + dimension];
        if source ~= 1 and source ~= target then return false; end
    end
    return true;
end

function module.optionalDependencyMessage(packageName, extra, feature)
    return feature .. " requires '" .. packageName .. "'. Install the " ..
        "LuaRocks package '" .. extra .. "'.";
end

return module;
