package = "CelNN";
version = "1.0-1";
source = {
    url = "git+https://github.com/tomieiro/lcelnn.git",
};
description = {
    summary = "Lua/Torch7 cellular neural network library",
    homepage = "https://github.com/tomieiro/lcelnn",
    license = "Apache-2.0",
};
dependencies = {
    "lua >= 5.1",
    "torch",
};
build = {
    type = "builtin",
    modules = {
        ["CelNN"] = "CelNN.lua",
        ["CelNN.Activations"] = "CelNN/Activations.lua",
        ["CelNN.Differentiable"] = "CelNN/Differentiable.lua",
        ["CelNN.Domains"] = "CelNN/Domains.lua",
        ["CelNN.Plasticity"] = "CelNN/Plasticity.lua",
        ["CelNN.Serialization"] = "CelNN/Serialization.lua",
        ["CelNN.Template"] = "CelNN/Template.lua",
        ["CelNN.Templates"] = "CelNN/Templates.lua",
        ["CelNN.Training"] = "CelNN/Training.lua",
        ["CelNN.Utils"] = "CelNN/Utils.lua",
        ["CelNN.Visualization"] = "CelNN/Visualization.lua",
    },
};
