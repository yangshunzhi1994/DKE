import torch.nn.functional as F
import time
import torch

def forward(self,input,logit_s,model_s,model_t,target,index):
    tea_model_copy = model_t
    target_layer_tea = tea_model_copy.layer3[-1].conv2
    stu_model_copy = model_s
    target_layer_stu = stu_model_copy.layer3[-1].conv2
    stu_grad_cam = get_grad_cams(input,target_layer_stu,stu_model_copy,self.Nums)
    tea_grad_cam = get_grad_cams(input,target_layer_tea,tea_model_copy,self.Nums)
    CAM_loss = F.mse_loss(stu_grad_cam, tea_grad_cam)
    logit_loss = gen_logit_loss(stu_grad_cam, tea_grad_cam)
    loss_ce = self.gamma * F.cross_entropy(logit_s, target) 
    loss_cam = self.alpha *CAM_loss
    loss_logit = self.e *logit_loss
    loss = loss_ce + loss_cam + loss_logit 
    return loss

def get_grad_cams(input,target_layer, model,nums):
    features_blobs = []
    gradients_blobs = []

    def hook_feature(module, input, output):
        nonlocal features_blobs
        features_blobs=output

    def hook_gradient(module, grad_in, grad_out):
        nonlocal gradients_blobs
        gradients_blobs = grad_out[0]

    handle_feature = target_layer.register_forward_hook(hook_feature)
    handle_gradient = target_layer.register_full_backward_hook(hook_gradient)


    _, logit = model(input, is_feat=True, preact=True)
    batch_size = logit.shape[0]
    num_classes = logit.shape[1]
    device = logit.device

    # ... 其他代码 ...
    grad_cams_tensor = torch.zeros(batch_size, num_classes, *features_blobs.shape[2:]).to(device)
    _, top_pred_index = torch.topk(logit, nums)
    # 1. Perform a single backward pass for the top `nums` classes
    one_hot_output = torch.zeros(batch_size, num_classes).to(device)
    for i in range(batch_size):
        one_hot_output[i, top_pred_index[i]] = 1

    model.zero_grad()
    logit.backward(gradient=one_hot_output,retain_graph=True)

    # 2. Compute weights for all classes at once
    print(gradients_blobs.shape,33332)
    weights = torch.mean(gradients_blobs, dim=[2, 3], keepdim=True)
    print(weights.shape,3331)

    # 3. Compute Grad-CAM for each class
    for i in range(batch_size):
        for target_class in top_pred_index[i]:
            grad_cams = torch.sum(weights[i, target_class] * features_blobs[i], dim=0, keepdim=True)
            grad_cams = F.normalize(grad_cams, dim=(1,2))
            grad_cams_tensor[i, target_class.item()] = grad_cams.squeeze(0)
    
    handle_feature.remove()
    handle_gradient.remove()
    # print(grad_cams_tensor.shape)

    return grad_cams_tensor

def get_grad_cams1(input,target_layer, model,nums):
    features_blobs = []
    gradients_blobs = []

    def hook_feature(module, input, output):
        nonlocal features_blobs
        features_blobs=output

    def hook_gradient(module, grad_in, grad_out):
        nonlocal gradients_blobs
        gradients_blobs = grad_out[0]

    handle_feature = target_layer.register_forward_hook(hook_feature)
    handle_gradient = target_layer.register_full_backward_hook(hook_gradient)
    # print(features_blobs,2224)

    _, logit = model(input, is_feat=True, preact=True)
    batch_size = logit.shape[0]
    num_classes = logit.shape[1]
    device = logit.device
    one_hot_output = torch.zeros(batch_size, num_classes).to(device)
    grad_cams_tensor = torch.zeros(batch_size, num_classes, *features_blobs.shape[2:]).to(device)


    _, top_pred_index = torch.topk(logit, nums)
    # for target_class in top_pred[0]:
    for i in range(batch_size):
        one_hot_output[i, top_pred_index[i]] = 1
    # one_hot_output[:, top_pred_index[0]] = 1

    model.zero_grad()
    logit.backward(gradient=one_hot_output,retain_graph=True)
    # print(features_blobs,2224)

    if len(features_blobs) == 0 or len(gradients_blobs) == 0:
        print(3334)
        return None

    weights = torch.mean(gradients_blobs, dim=[2, 3], keepdim=True)
    # print(weights, 3334)
    grad_cams = torch.sum(weights * features_blobs, dim=1, keepdim=True)
    grad_cams = F.normalize(grad_cams,dim=(2,3))
    # print(grad_cams.shape,11112)
    handle_feature.remove()
    handle_gradient.remove()
    return grad_cams_tensor

def gen_logit_loss(cams_tea, cams_stu):
    gap_cams_tea =  cams_tea.view(cams_tea.size(0), -1)
    gap_cams_stu =  cams_stu.view(cams_stu.size(0), -1)
    tea_pred = F.softmax(gap_cams_tea,dim=1)
    stu_log = F.log_softmax(gap_cams_stu,dim=1)
    loss_logit = F.kl_div(stu_log, tea_pred,reduction = 'batchmean') 
    return loss_logit