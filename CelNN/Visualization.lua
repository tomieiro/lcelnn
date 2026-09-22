-- Lua port based on the Apache-2.0 celnn project.
local torch = require "torch";

local module = {};
local width, height, margin = 720, 420, 48;

local function escape(value)
    return tostring(value):gsub("&", "&amp;"):gsub("<", "&lt;")
        :gsub(">", "&gt;");
end

local function document(title, body)
    return string.format(
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d">' ..
        '<rect width="100%%" height="100%%" fill="white"/>' ..
        '<text x="%d" y="28" text-anchor="middle" font-size="18">%s' ..
        '</text>%s</svg>', width, height, width / 2, escape(title), body);
end

local function save(svg, path)
    if path then
        local file = assert(io.open(path, "wb"));
        file:write(svg);
        file:close();
    end
    return svg;
end

local function values(input)
    assert(torch.isTensor(input) and input:dim() == 1,
        "signal must be a 1D tensor");
    local result = {};
    for i = 1, input:size(1) do result[i] = input[i]; end
    return result;
end

local function linePlot(series, title, path, labels)
    local minimum, maximum = math.huge, -math.huge;
    for _, line in ipairs(series) do
        for _, value in ipairs(line) do
            minimum = math.min(minimum, value);
            maximum = math.max(maximum, value);
        end
    end
    if minimum == maximum then minimum, maximum = minimum - 1, maximum + 1; end
    local points = {};
    for index, line in ipairs(series) do
        local coords = {};
        for i, value in ipairs(line) do
            local x = margin + (i - 1) * (width - 2 * margin) /
                math.max(#line - 1, 1);
            local y = height - margin - (value - minimum) *
                (height - 2 * margin) / (maximum - minimum);
            coords[#coords + 1] = string.format("%.2f,%.2f", x, y);
        end
        points[#points + 1] = string.format(
            '<polyline fill="none" stroke="%s" stroke-width="2" points="%s"/>',
            index == 1 and "#1565c0" or "#c62828", table.concat(coords, " "));
    end
    local body = '<path d="M48 372 H672 M48 48 V372" stroke="#333"/>' ..
        table.concat(points);
    if labels then body = body .. '<text x="580" y="58">' ..
        escape(labels[1]) .. " / " .. escape(labels[2]) .. "</text>"; end
    return save(document(title, body), path);
end

function module.plotSignal(signal, title, path)
    return linePlot({values(signal)}, title or "Signal", path);
end

function module.plotSignalComparison(reference, candidate, title, path)
    return linePlot({values(reference), values(candidate)},
        title or "Signal comparison", path, {"reference", "candidate"});
end

function module.plotTrajectory(trajectory, title, path)
    local means = {};
    for step, frame in ipairs(trajectory) do
        means[step] = frame:mean();
    end
    return linePlot({means}, title or "Trajectory", path);
end

function module.plotGrid(grid, title, path)
    assert(torch.isTensor(grid) and grid:dim() == 2,
        "grid must be a 2D tensor");
    local rows, columns = grid:size(1), grid:size(2);
    local cellWidth = (width - 2 * margin) / columns;
    local cellHeight = (height - 2 * margin) / rows;
    local body = {};
    for row = 1, rows do
        for column = 1, columns do
            local shade = math.floor((math.max(-1,
                math.min(1, grid[row][column])) + 1) * 127.5);
            body[#body + 1] = string.format(
                '<rect x="%.2f" y="%.2f" width="%.2f" ' ..
                'height="%.2f" fill="rgb(%d,%d,%d)"/>',
                margin + (column - 1) * cellWidth,
                48 + (row - 1) * cellHeight, cellWidth, cellHeight,
                shade, shade, shade);
        end
    end
    return save(document(title or "Grid", table.concat(body)), path);
end

return module;
