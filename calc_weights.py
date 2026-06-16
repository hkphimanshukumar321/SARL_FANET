import json

with open('best_weights_and_params (1).json', 'r') as f:
    d = json.load(f)

def norm(arr):
    s = sum(arr)
    return tuple(round(x/s, 2) for x in arr)

print('Latex rows:')
for name, p in d.items():
    rw = norm([p['w_throughput'], p['w_delay'], p['w_failures'], p['w_jitter']])
    mw = norm([p['c_w_dist'], p['c_w_sinr'], p['c_w_mob'], p['c_w_load']])
    hw = norm([p['c_a_energy'], p['c_a_degree'], p['c_a_mobstab'], p['c_a_queue'], p['c_a_risk']])
    
    rw_str = f'({rw[0]}, {rw[1]}, {rw[2]}, {rw[3]})'
    mw_str = f'({mw[0]}, {mw[1]}, {mw[2]}, {mw[3]})'
    hw_str = f'({hw[0]}, {hw[1]}, {hw[2]}, {hw[3]}, {hw[4]})'
    
    lr_val = p['lr']
    lr_str = f'{lr_val:.2e}'.replace('e-0', ' \\times 10^{-') + '}'
    bs = p['batch_size']
    
    print(f'{name.upper()} & {rw_str} & {mw_str} & {hw_str} & lr=${lr_str}$, $B={bs}$ \\\\')
