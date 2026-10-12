-- Lua port based on the Apache-2.0 celnn project.
local celNN = require "CelNN";
local hasCjson, cjson = pcall(require, "cjson");

local module = {};

local function json()
    assert(hasCjson, "JSON helpers require lua-cjson");
    return cjson;
end

function module.saveJson(data, path)
    local target = path .. ".tmp." .. tostring(os.time()) .. "." ..
        tostring(math.random(1, 1000000000));
    local file = assert(io.open(target, "wb"));
    local ok, message = pcall(function()
        file:write(json().encode(data));
        file:flush();
        file:close();
    end);
    if not ok then
        pcall(function() file:close(); end);
        os.remove(target);
        error(message);
    end
    local renamed, reason = os.rename(target, path);
    if not renamed then os.remove(target); error(reason); end
    return path;
end

function module.loadJson(path)
    local file = assert(io.open(path, "rb"));
    local contents = file:read("*a");
    file:close();
    return json().decode(contents);
end

function module.saveTemplateJson(value, path)
    return module.saveJson(value:toDict(), path);
end

function module.loadTemplateJson(path)
    return celNN.Template.fromDict(module.loadJson(path));
end

function module.saveConfigJson(value, path)
    return module.saveJson(value:toDict(), path);
end

function module.loadConfigJson(path)
    return celNN.SimulationConfig.fromDict(module.loadJson(path));
end

function module.saveRegistryJson(value, path)
    return module.saveJson(value:toDict(), path);
end

function module.loadRegistryJson(path)
    return celNN.TemplateRegistry.fromDict(module.loadJson(path));
end

function module.saveNetworkJson(value, path)
    return module.saveJson(value:toDict(), path);
end

function module.loadNetworkJson(path, device)
    return celNN.CellularNetwork.fromDict(module.loadJson(path), device);
end

return module;
