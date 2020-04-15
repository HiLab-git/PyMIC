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

if __name__ == "__main__":
    input_file  = '/home/guotai/disk2t/projects/dlls/training_fetal_brain/exp1/model/unet2dres_bn1_20000.pt'
    output_file = '/home/guotai/disk2t/projects/dlls/training_fetal_brain/exp1/model/unet2dres_bn1_20000_rename.pt'
    input_var_list = ['conv.weight', 'conv.bias']
    output_var_list= ['conv9.weight', 'conv9.bias']
    rename_model_variable(input_file, output_file, input_var_list, output_var_list)