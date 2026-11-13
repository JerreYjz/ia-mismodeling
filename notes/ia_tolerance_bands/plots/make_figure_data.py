"""Write the data.json of every figure of the IA tolerance-band note.

The only file of the note that knows where the analysis outputs are:
analysis/roman_dchi2/v7/tolerance_bands.json, written by
analysis/roman_dchi2/v7/tolerance_bands.py. Rerun this after that script,
then rerun the plot scripts.

    ~/cocoa/Cocoa/.local/bin/python make_figure_data.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, '..', '..', '..', 'analysis', 'roman_dchi2', 'v7',
                                    'tolerance_bands.json'))


def write(name, data):
    folder = os.path.join(HERE, name)
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, 'data.json'), 'w') as f:
        json.dump(data, f, indent=1)
    print('wrote', os.path.relpath(os.path.join(folder, 'data.json'), HERE))


def bands(res, mask, case):
    out = {}
    for r in res:
        if r['mask'] == mask and r['case'] == case:
            out[r['model']] = {X: {'single': b['single'], 'joint': b['joint']}
                               for X, b in r['bands'].items()}
            out[r['model']]['npts'] = r['npts']
            out[r['model']]['baseline_FoB'] = r['baseline']['FoB']
            out[r['model']]['sigma_S8'] = r['sigma_S8']
    return out


def main():
    src = json.load(open(SRC))
    res = src['results']
    common = ('Tolerance bands of the IA power spectra for Roman cosmic shear: the '
              'fractional perturbation eps of P_X(k, z) -> P_X [1 + sum_j eps_j h_j(k)] '
              '(h_j tent functions in ln k at the nodes, the same at every z; X = GI, II, '
              'or both with the same eps) below which the shift of the best-fit '
              '(Omega_m, S8) stays below FoB = 0.3. Truth: FLAMINGO D3A cosmology, GG + the '
              'stitched FLAMINGO IA (NLA below k_t(z), spline of the measured spectra above) '
              'injected through sim_IA; covariance example1_cov; model refitted with all '
              'parameters free. single: tolerance of node j alone; joint: a box in which '
              'every combination of signs and sizes stays below 0.3 (each node the same '
              'share of the budget, capped at 1). Linear response at the best fit '
              '(analysis/roman_dchi2/v7/tolerance_bands.py -> v7/tolerance_bands.json). '
              'nodes in h/Mpc.')
    for model in ('NLA', 'TATT'):
        write('tolerance_bands_' + model.lower(), {
            'about': common + f' This figure: {model} refitted, the production mask '
                              'k_UV = 0.3/D(z) (IR/UV, alpha = 0.05), case A (Omega_m, A_s '
                              'free); case_B adds n_s, H0, Omega_b with priors. The other '
                              'masks are in tolerance_masks.',
            'nodes': src['nodes_h_per_Mpc'], 'fob_max': src['fob_max'],
            'mask': 'K0/D 0.3', 'model': model,
            'bands': bands(res, 'K0/D 0.3', 'A')[model],
            'case_B': bands(res, 'K0/D 0.3', 'B')[model],
        })
    write('tolerance_masks', {
        'about': common + ' This figure: the bands for the production masks '
                          'k_UV = K0/D(z), K0 = 0.2 to 0.5, case A.',
        'nodes': src['nodes_h_per_Mpc'], 'fob_max': src['fob_max'],
        'K0': [0.2, 0.3, 0.4, 0.5],
        'masks': {mask: bands(res, mask, 'A') for mask in ('K0/D 0.2', 'K0/D 0.3', 'K0/D 0.4', 'K0/D 0.5')},
    })

    ver = {}
    for r in res:
        if r['mask'] == 'K0/D 0.3' and r['case'] == 'A':
            for X in ('GI', 'II'):
                v = r['bands'][X].get('verify')
                if v:
                    ver[f"{r['model']} {X}"] = [{'kind': e['kind'], 'fob_full': e['fob_full'],
                                                 'fob_linear': e['fob_linear'],
                                                 'node': e.get('node')} for e in v]
    write('verification', {
        'about': ('Full non-linear refits at the edges of the linear-response bands, mask '
                  'k_UV = 0.3/D(z), case A: each sign at the single-node edge of the three most '
                  'sensitive nodes ("single") and both worst-case corners of the guaranteed box '
                  '("joint corner"). fob_full: the FoB of the shift of the refit from the '
                  'unperturbed refit, in the unperturbed Sigma; linear response predicts '
                  'fob_linear = 0.3 at every edge. (analysis/roman_dchi2/v7/tolerance_bands.py '
                  '--verify -> tolerance_bands.json, bands.<X>.verify.)'),
        'fob_max': src['fob_max'], 'groups': ver})


if __name__ == '__main__':
    main()
