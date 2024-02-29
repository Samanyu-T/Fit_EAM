import numpy as np

def data_dict(ref_json, my_json, N_Vac, N_H, N_He):

    formation = {}

    for n_h in range(N_H+1):
        for n_vac in range(N_Vac+1):
            for n_he in range(N_He+1):
                    
                    if n_h + n_vac + n_he > 0 :
                        
                        key = 'V%dH%dHe%d' % (n_vac, n_h, n_he)

                        if key in ref_json.keys() and key in my_json.keys():
                            
                            if len(ref_json[key]['dft']['val']) > 0:
                                formation[key] = {}
                                formation[key]['val'] = ref_json[key]['dft']['val'][0]
                                    
                                if len(ref_json[key]['r_vol_dft']['val']) > 0:
                                    formation[key]['rvol'] = ref_json[key]['r_vol_dft']['val'][0]
                                    
                                else:
                                    formation[key]['rvol'] = None

                                formation[key]['pos'] = my_json[key]['xyz_opt']

    return formation

def find_binding(df, defect, add_element, trend_element):

    add_int = df['V%dH%dHe%d' % (add_element[0], add_element[1], add_element[2])]['val']
    
    defect = np.array(defect)

    add_element = np.array(add_element)

    trend_element = np.array(trend_element)

    defect_next = defect + add_element

    key_curr = 'V%dH%dHe%d' % (defect[0],defect[1], defect[2])

    key_next = 'V%dH%dHe%d' % (defect_next[0],defect_next[1], defect_next[2])

    init_config = []
    final_config = []

    while key_next in df.keys():
                
        init_config.append(df[key_curr]['val'])
        
        final_config.append(df[key_next]['val'])

        defect += trend_element

        defect_next += trend_element

        key_curr = 'V%dH%dHe%d' % (defect[0],defect[1], defect[2])
        
        key_next = 'V%dH%dHe%d' % (defect_next[0],defect_next[1], defect_next[2])

    init_config = np.array(init_config)

    final_config = np.array(final_config)
    
    binding = add_int + init_config - final_config

    return binding

def binding_fitting(ref_formations):

    binding = []


    for i in range(3):
        binding.append(find_binding(ref_formations, [i, 0, 1], [0, 0, 1], [0,0,1]))

    if 'V0H1He0' in ref_formations:
        for i in range(1, 6):
            binding.append(find_binding(ref_formations, [0, 0, 1], [0, i, 0], [0,0,1]))

        for i in range(1, 6):
            binding.append(find_binding(ref_formations, [1, 0, 0], [0, i, 0], [0,0,1]))

        for i in range(1, 4):
            binding.append(find_binding(ref_formations, [2, 0, 0], [0, i, 0], [0,0,1]))

    # binding.append(find_binding(ref_formations, [0, 0, 1], [0,0,1], [0,0,1]))
    # binding.append(find_binding(ref_formations, [1, 0, 0], [0,0,1], [0,0,1]))
    # binding.append(find_binding(ref_formations, [2, 0, 0], [0,0,1], [0,0,1]))

    # binding.extend(find_binding(ref_formations, [0, 0, 1], [0,1,0], [0,0,1]))
    # binding.extend(find_binding(ref_formations, [1, 0, 1], [0,1,0], [0,0,1]))

    return binding

def binding_testing(ref_formations):

    binding = []

    for i in range(3):
        binding.extend(find_binding(ref_formations, [i, 0, 1], [0, 0, 1], [0,0,1]))

    for i in range(1, 6):
        binding.extend(find_binding(ref_formations, [0, 0, 1], [0, i, 0], [0,0,1]))

    for i in range(1, 6):
        binding.extend(find_binding(ref_formations, [1, 0, 0], [0, i, 0], [0,0,1]))

    for i in range(1, 4):
        binding.extend(find_binding(ref_formations, [2, 0, 0], [0, i, 0], [0,0,1]))

    return np.array(binding)