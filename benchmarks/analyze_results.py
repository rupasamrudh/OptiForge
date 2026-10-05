import json
import numpy as np

def main():
    with open('benchmarks/stage_8_18_results.json') as f:
        data = json.load(f)

    results = data['results']

    print('=== 1. PUBLIC NETLIB RESULTS ===')
    print(f"{'Model':10s} | {'m':3s} | {'n':3s} | {'DenseLU (ms)':12s} | {'Inc (ms)':10s} | {'PFI (ms)':10s} | {'Inc Spd':8s} | {'PFI Spd':8s}")
    netlib_models = ['AVGAS', 'AFIRO', 'ADLITTLE']
    for mod in netlib_models:
        entries = {e['config']: e for e in results if e['workload_name'].upper() == mod}
        d = entries['A_DenseLU']['timings']['median_ms']
        inc = entries['B_Incremental']['timings']['median_ms']
        pfi = entries['C_PFI']['timings']['median_ms']
        m = entries['A_DenseLU']['m_constraints']
        n = entries['A_DenseLU']['n_variables']
        print(f"{mod:10s} | {m:3d} | {n:3d} | {d:12.2f} | {inc:10.2f} | {pfi:10.2f} | {d/inc:7.2f}x | {d/pfi:7.2f}x")

    print('\n=== 2. SCALE SWEEP RESULTS ===')
    print(f"{'m':3s} | {'n':3s} | {'m_SF':4s} | {'DenseLU (ms)':12s} | {'Inc (ms)':10s} | {'PFI (ms)':10s} | {'Inc Spd':8s} | {'PFI Spd':8s} | {'PFI vs Inc':10s}")
    scale_entries = [e for e in results if e['workload_category'] == 'Scale Sweep LP']
    models = sorted(list(set(e['workload_name'] for e in scale_entries)), key=lambda x: int(x.split('_')[2][1:]))
    for mod in models:
        ents = {e['config']: e for e in scale_entries if e['workload_name'] == mod}
        d = ents['A_DenseLU']['timings']['median_ms']
        inc = ents['B_Incremental']['timings']['median_ms']
        pfi = ents['C_PFI']['timings']['median_ms']
        m = ents['A_DenseLU']['m_constraints']
        n = ents['A_DenseLU']['n_variables']
        m_sf = ents['A_DenseLU']['m_standard_form']
        print(f"{m:3d} | {n:3d} | {m_sf:4d} | {d:12.2f} | {inc:10.2f} | {pfi:10.2f} | {d/inc:7.2f}x | {d/pfi:7.2f}x | {inc/pfi:9.2f}x")

    print('\n=== 3. STRUCTURALLY DIVERSE LP RESULTS ===')
    print(f"{'Workload':25s} | {'m':3s} | {'n':3s} | {'DenseLU':10s} | {'Inc':10s} | {'PFI':10s} | {'Inc Spd':8s} | {'PFI Spd':8s}")
    div_entries = [e for e in results if e['workload_category'] == 'Diverse LP']
    div_models = sorted(list(set(e['workload_name'] for e in div_entries)))
    for mod in div_models:
        ents = {e['config']: e for e in div_entries if e['workload_name'] == mod}
        d = ents['A_DenseLU']['timings']['median_ms']
        inc = ents['B_Incremental']['timings']['median_ms']
        pfi = ents['C_PFI']['timings']['median_ms']
        m = ents['A_DenseLU']['m_constraints']
        n = ents['A_DenseLU']['n_variables']
        print(f"{mod:25s} | {m:3d} | {n:3d} | {d:10.2f} | {inc:10.2f} | {pfi:10.2f} | {d/inc:7.2f}x | {d/pfi:7.2f}x")

    print('\n=== 4. STRUCTURALLY DIVERSE MILP RESULTS ===')
    print(f"{'Workload':22s} | {'Nodes':5s} | {'Cold (ms)':10s} | {'WS-Inc (ms)':11s} | {'WS-PFI (ms)':11s} | {'Adapt-T6':10s} | {'WS Speedup':10s}")
    milp_entries = [e for e in results if e['workload_category'] == 'Diverse MILP']
    milp_models = sorted(list(set(e['workload_name'] for e in milp_entries)))
    for mod in milp_models:
        ents = {e['config']: e for e in milp_entries if e['workload_name'] == mod}
        cold = ents['A_DenseLU_Cold']['timings']['median_ms']
        ws_inc = ents['D_WarmStart_Inc']['timings']['median_ms']
        ws_pfi = ents['C_WarmStart_PFI']['timings']['median_ms']
        adapt = ents['E_Adaptive_PFI_T6']['timings']['median_ms']
        nodes = ents['A_DenseLU_Cold']['nodes']
        print(f"{mod:22s} | {nodes:5d} | {cold:10.2f} | {ws_inc:11.2f} | {ws_pfi:11.2f} | {adapt:10.2f} | {cold/ws_inc:9.2f}x")

    print('\n=== 5. THRESHOLD EXPERIMENT DISTRIBUTION (Across Candidate Thresholds T) ===')
    print(f"{'Model':30s} | {'m_child':7s} | {'T=2':8s} | {'T=4':8s} | {'T=6':8s} | {'T=8':8s} | {'T=10':8s} | {'T=20':8s} | {'T=50':8s} | {'WS-Inc':8s} | {'WS-PFI':8s}")
    thresh_entries = [e for e in results if e['workload_category'] == 'Threshold Crossing']
    thresh_models = sorted(list(set(e['workload_name'] for e in thresh_entries)), key=lambda x: int(x.split('_')[2][1:]))
    for mod in thresh_models:
        ents = {e['config']: e for e in thresh_entries if e['workload_name'] == mod}
        m_con = ents['D_WarmStart_Inc']['m_constraints']
        m_child = m_con + 1
        t2 = ents['E_Adaptive_PFI_T2']['timings']['median_ms']
        t4 = ents['E_Adaptive_PFI_T4']['timings']['median_ms']
        t6 = ents['E_Adaptive_PFI_T6']['timings']['median_ms']
        t8 = ents['E_Adaptive_PFI_T8']['timings']['median_ms']
        t10 = ents['E_Adaptive_PFI_T10']['timings']['median_ms']
        t20 = ents['E_Adaptive_PFI_T20']['timings']['median_ms']
        t50 = ents['E_Adaptive_PFI_T50']['timings']['median_ms']
        ws_inc = ents['D_WarmStart_Inc']['timings']['median_ms']
        ws_pfi = ents['C_WarmStart_PFI']['timings']['median_ms']
        print(f"{mod:30s} | {m_child:7d} | {t2:8.1f} | {t4:8.1f} | {t6:8.1f} | {t8:8.1f} | {t10:8.1f} | {t20:8.1f} | {t50:8.1f} | {ws_inc:8.1f} | {ws_pfi:8.1f}")

    print('\n=== 6. ADAPTIVE DISPATCH DECISIONS (PFI vs Inc Chosen) ===')
    print(f"{'Model':30s} | {'m_child':7s} | {'T=2':14s} | {'T=4':14s} | {'T=6':14s} | {'T=8':14s} | {'T=10':14s} | {'T=20':14s} | {'T=50':14s}")
    for mod in thresh_models:
        ents = {e['config']: e for e in thresh_entries if e['workload_name'] == mod}
        m_child = ents['D_WarmStart_Inc']['m_constraints'] + 1
        decisions = []
        for T in [2, 4, 6, 8, 10, 20, 50]:
            e = ents[f'E_Adaptive_PFI_T{T}']
            pfi_c = e['telemetry']['adaptive_pfi_chosen']
            inc_c = e['telemetry']['adaptive_inc_chosen']
            decisions.append(f"{pfi_c}P/{inc_c}I")
        print(f"{mod:30s} | {m_child:7d} | " + " | ".join(f"{d:14s}" for d in decisions))

if __name__ == '__main__':
    main()
