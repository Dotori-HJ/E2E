import torch
from ptflops import get_model_complexity_info

from models.spatial_pooler import TemporalWiseAttentionPooling
from models.video_encoder_archs.slowfast import ResNet3dSlowFast


def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def get_gpu_memory_consumption(model, input):
    before = torch.cuda.max_memory_allocated()
    x = model(input)
    x.mean().backward()
    after = torch.cuda.max_memory_allocated()
    return (after - before) / 1024 / 1024

# def get_flops(model, input):
#     input = input.to(torch.float32)

#         model(input)
#     print(prof.key_averages())


#     # input = torch.randn(1, 2048, 256, 7, 7).cuda()
#     # model = TemporalWiseAttentionPooling(input_dim=2048, base_dim=512).cuda()
#     # print(f'The number of GPU meory: {get_gpu_memory_consumption(model, input)}')
#     # print(f'The number of FLOPs: {get_flops(model, input)}')
#     # print(f'The number of parameters: {count_parameters(model)}')


#     input = torch.randn(1, 256, 32, 7, 7).cuda()
#     model = TemporalWiseAttentionPooling(input_dim=256, base_dim=64).cuda()
    # print(f'The number of GPU meory: {get_gpu_memory_consumption(model, input)}')
#     print(f'The number of FLOPs: {get_flops(model, input)}')
#     print(f'The number of parameters: {count_parameters(model)}')

# def get_flops(model, input):
#     input = input.to(torch.float32)
#     with torch.autograd.profiler.profile() as prof:
#         model(input)
#     flops = sum([v.value for k, v in prof.key_averages().items()]) * input.nelement()
#     return flops

if __name__ == '__main__':


    with torch.autograd.profiler.profile(with_flops=True, with_modules=True) as prof:
        before = torch.cuda.max_memory_allocated()
        backbone = ResNet3dSlowFast(
            None, depth=50,freeze_bn=True, freeze_bn_affine=True, slow_upsample=8,
            pooler='avg'
        ).cuda()
        # with torch.no_grad():
        #     print(f'The number of parameters: {count_parameters(backbone) / 1000 / 1000} M')

        input = torch.randn(1, 3, 256, 224, 224).cuda()
        x = backbone(input)

        after = torch.cuda.max_memory_allocated()
    macs, params = get_model_complexity_info(backbone, (3, 256, 224, 224), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    stats = prof.key_averages().table()
    index = stats.find("FLOPs")
    unit = stats[index - 1]
    print(unit)
    # column_data = [stat for stat in stats]
    # print(column_data)
    flops = 0
    for k in prof.key_averages():
        flops += k.flops
    print(f'The number of GPU meory: {(after - before) / 1024 / 1024 / 1024} GB')
    print(f'Total FLOPs: {flops / 1000 / 1000}M')
    # model_a = TemporalWiseAttentionPooling(input_dim=2048, base_dim=512, num_layers=2).cuda()
    # model_b = TemporalWiseAttentionPooling(input_dim=256, base_dim=64, num_layers=2).cuda()
    # input_a = torch.randn(1, 2048, 256, 7, 7).cuda()
    # input_b = torch.randn(1, 256, 32, 7, 7).cuda()
    # with torch.no_grad():
    #     print(f'The number of parameters: {(count_parameters(model_a) + count_parameters(model_b)) / 1000 / 1000} M')

    # before = torch.cuda.max_memory_allocated()
    # x_a = model_a(input_a)
    # x_b = model_b(input_b)
    # after = torch.cuda.max_memory_allocated()
    # print(f'The number of GPU meory: {(after - before) / 1024 / 1024 / 1024} GB')

