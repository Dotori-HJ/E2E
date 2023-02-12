import torch
from ptflops import get_model_complexity_info

from models.spatial_pooler import TemporalWiseAttentionPooling


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
#     with torch.autograd.profiler.profile() as prof:
#         model(input)
#     flops = prof.total_ops()
#     return flops

# if __name__ == '__main__':
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


with torch.cuda.device(0):
    model = TemporalWiseAttentionPooling(input_dim=256, base_dim=64).cuda()
    # input = torch.randn(1, 2048, 256, 7, 7).cuda()
    # model = TemporalWiseAttentionPooling(input_dim=2048, base_dim=512).cuda()
    print(f'The number of parameters: {count_parameters(model)}')

    input = torch.randn(1, 256, 32, 7, 7).cuda()
    print(f'The number of GPU meory: {get_gpu_memory_consumption(model, input)} MB')
    macs, params = get_model_complexity_info(model, (256, 32, 7, 7), as_strings=True, flops_units=True,
                                            print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))