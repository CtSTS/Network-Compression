import numpy as np
from sklearn.cluster import KMeans 
from model import *

def model_reassignment_kmeans(model_int_path,Map_path):
    print("Reload model by model_int_path:{} and Map_path:{}".format(model_int_path,Map_path))
    Map = np.load(Map_path,allow_pickle=True)
    if Map[-1] == 'vgg':
        model = VGG(Map[-2],dataset = Map[-3])
    elif Map[-1] == 'mobile':
     
        model = MobileNet(Map[-2],dataset = Map[-3])
    elif Map[-1] == 'res':
        model = ResNet(Map[-2],dataset = Map[-3],cfg_before_slim=Map[-4])
    else:
        print("Wrong type for model in Numpy list")
        return 0
    model.load_state_dict(torch.load(model_int_path))
    layer_count = 0
    for module in model.modules():
        convert_map = { 'int8':int,'float32':float}
        if isinstance(module,nn.Linear) or isinstance(module,nn.BatchNorm2d) or isinstance(module,nn.Conv2d):
            print("----------------")
        else:
            continue
        print(module)
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy().reshape(-1)
        weight_float32 = []
        for i in weight:
            weight_float32.append(Map[layer_count][i])
        weight_float32 = np.asarray(weight_float32,'float32')
        if isinstance(module,nn.Linear):
            module.weight.data = torch.from_numpy(weight_float32.reshape(module.weight.shape[0],module.weight.shape[1])).to(dev)
        elif isinstance(module,nn.BatchNorm2d):
            module.weight.data = torch.from_numpy(weight_float32.reshape(module.weight.shape[0])).to(dev)
        elif isinstance(module,nn.Conv2d):
            module.weight.data = torch.from_numpy(weight_float32.reshape(module.weight.shape[0],module.weight.shape[1],module.weight.shape[2],module.weight.shape[3])).to(dev)
        layer_count = layer_count + 1
    return model

def apply_weight_kmeans(model,bits):
    Map = []
    model_int = model
    for module in model_int.modules():     
        convert_map = { 'int8':int,'float32':float}
        if isinstance(module,nn.Linear) or isinstance(module,nn.BatchNorm2d) or isinstance(module,nn.Conv2d):
            print(module)
        else:
            continue
        #print(module)
        #print(module.weight.shape)
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        mat=weight
        min_ = mat.min()
        max_ = mat.max()
        print("------min------:  " ,min_)
        print("------max------:  " ,max_)
        mat = mat.reshape(-1,1)
        if (mat.shape[0]>2**bits):
            space = np.linspace(min_, max_, num=2**bits)
            kmeans = KMeans(n_clusters=len(space), init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        else:
            space = np.linspace(min_, max_, num=mat.shape[0])
            kmeans = KMeans(n_clusters=mat.shape[0], init=space.reshape(-1,1), n_init=1, precompute_distances=True, algorithm="full")
        kmeans.fit(mat)  
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        label = np.asarray(kmeans.labels_,'int8')
        if isinstance(module,nn.Linear):
            module.weight.data = torch.from_numpy(label.reshape(module.weight.shape[0],module.weight.shape[1])).to(dev)
        elif isinstance(module,nn.BatchNorm2d):
            module.weight.data = torch.from_numpy(label.reshape(module.weight.shape[0])).to(dev)
        elif isinstance(module,nn.Conv2d):
            module.weight.data = torch.from_numpy(label.reshape(module.weight.shape[0],module.weight.shape[1],module.weight.shape[2],module.weight.shape[3])).to(dev)
        for i,j in zip(label,new_weight):
            convert_map[i] = j
        Map.append(convert_map)
    if model.model_info()[-1] == 'res':
        Map.append(model.model_info()[-4])
    Map.append(model.model_info()[-3])    
    Map.append(model.model_info()[-2])
    Map.append(model.model_info()[-1])

    return model_int,Map
    