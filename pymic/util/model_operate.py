import torch


def rename_model_variable(input_file, output_file, input_var_list, output_var_list):
    assert(len(input_var_list) == len(output_var_list))
    checkpoint = torch.load(input_file)
    state_dict = checkpoint['model_state_dict']
    for i in range(len(input_var_list)):
        input_var  = input_var_list[i]
        output_var = output_var_list[i]
        state_dict[output_var] = state_dict[input_var]
        state_dict.pop(input_var)
    checkpoint['model_state_dict'] = state_dict
    torch.save(checkpoint, output_file)


def get_average_model(checkpoint_name1, checkpoint_name2, checkpoint_name3, save_name):
    checkpoint1 = torch.load(checkpoint_name1)
    state_dict1 = checkpoint1['model_state_dict']

    checkpoint2 = torch.load(checkpoint_name2)
    state_dict2 = checkpoint2['model_state_dict']

    checkpoint3 = torch.load(checkpoint_name3)
    state_dict3 = checkpoint3['model_state_dict']

    state_dict = {}
    for item in state_dict1:
        print(item)
        state_dict[item] = (state_dict1[item] + state_dict2[item] + state_dict3[item])/3

    save_dict = {'model_state_dict': state_dict}
    torch.save(save_dict, save_name) 

if __name__ == "__main__":
    input_file  = '/home/guotai/disk2t/projects/dlls/training_fetal_brain/exp1/model/unet2dres_bn1_20000.pt'
    output_file = '/home/guotai/disk2t/projects/dlls/training_fetal_brain/exp1/model/unet2dres_bn1_20000_rename.pt'
    input_var_list = ['conv.weight', 'conv.bias']
    output_var_list= ['conv9.weight', 'conv9.bias']
    rename_model_variable(input_file, output_file, input_var_list, output_var_list)