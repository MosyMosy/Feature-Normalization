# from lab.tsne import generate_features
# from lab.affines import plot
from lab.layers import plot
# from lab.learning_curve import plot
# from lab.tsne import plot

# import os
# dir_list = [o for o in os.listdir('D:/downloaded_DS/miniImagenet') 
#                     if os.path.isdir(os.path.join('D:/downloaded_DS/miniImagenet',o))]

# import csv
# with open('D:/Project/BMS/datasets/split_seed_1/ImageNet_val_labeled.csv', newline='') as f:
#     reader = csv.reader(f)
#     data = list(reader)

# with open("Output.txt", "w") as text_file:    
#     for line in data:
#         is_in = False
#         for dir in dir_list:
#             if dir in line[1]:
#                 is_in = True
#                 break
        
#         if is_in == False:
            # text_file.writelines(line[0]+','+ line[1] + '\n')
    
# import torch
# import torchvision.models as models
# import copy

# model = models.resnet18(pretrained=True)

# sd = {
#     'model': copy.deepcopy(model.state_dict())
# }

# torch.save(sd, 'resnet18-f37072fd.pkl')
