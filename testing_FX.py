for name, layer in net2.named_modules():
    print(name, layer)

layer_list = list(net.modules())

layers_to_try = []
for i, layer in enumerate(net2.named_modules(),0):
    print(type(layer[1]))
    if isinstance(layer[1],nn.modules.activation.ReLU):
        layers_to_try.append(i)

modules = nn.ModuleList()
for curr_module in net.features:
    modules.append(curr_module)

if 'alexnet' in model_name:
    modules.append(nn.AdaptiveAvgPool2d((6, 6)))
elif 'vgg' in model_name:
    modules.append(nn.AdaptiveAvgPool2d((7, 7)))
modules.append(Flatten())
for curr_module in net.classifier:
    modules.append(curr_module)
new_net = nn.Sequential(*modules)


tmodels.create_feature_extractor(m, return_nodes={f'features.26'})

{f'layer{k}': str(v) for v, k in enumerate([1, 2, 3, 4])}

print(net.get_submodule("features.30"))

# general procedure

# 1. find out which layers are RELU as listed in train_nodes, eval_nodes = get_graph_node_names(net)
# 2. generate new cut network
#           net2 = create_feature_extractor(net, return_nodes={"features.29":"layer1"})
#           the input dict key is the get_graph_nodes_names layer name, value is what it will be called in the output dict
#           net2["layer1"] will get the output
# 3. add a flatten
# 4. get the size of the output with dummy input
# 5. add a linear layer with the correct size        
