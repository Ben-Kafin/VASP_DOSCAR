import re
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from os.path import exists, join
import matplotlib.colors as mc, colorsys
import mplcursors

class DosPlotter:
    def __init__(self, directory):
        self.directory = directory
        self.doscar = join(directory, 'DOSCAR')
        self.poscar = join(directory, 'CONTCAR') if exists(join(directory, 'CONTCAR')) else join(directory, 'POSCAR')
        
        if not exists(self.doscar):
            raise FileNotFoundError(f"DOSCAR not found in {directory}")

        self.total_dos = np.array([])
        self.site_dos = np.array([])
        self.energies = np.array([])
        self.ef = 0.0
        
        self.orbitals = []
        self.atomtypes = []
        self.atomnums = []
        self.vesta_label_map = {} # Corrected terminology
        
        # Hard-coded element colors
        self._type_color_map = {
            'Au': 'orange',
            'N': 'blue',
            'C': 'brown',
            'H': 'grey'
        }
        
        self._parse_all()

    def _orbit_base(self, orb):
        if orb.endswith('_up'): return orb[:-3]
        elif orb.endswith('_down'): return orb[:-5]
        return orb

    def _extract_ef_and_nedos(self, raw_header):
        tokens = re.findall(r'[-+]?\d+\.\d{8}|[-+]?\d+', raw_header)
        if len(tokens) < 4: raise ValueError("Header tokens insufficient")
        return int(tokens[2]), float(tokens[3])

    def _parse_doscar(self):
        with open(self.doscar, 'r') as file:
            atomnum = int(file.readline().split()[0])
            for _ in range(4): file.readline()
            nedos, ef = self._extract_ef_and_nedos(file.readline().strip())
            
            energies_list = []
            total_dos_list = []
            site_dos_list = []
            
            for i in range(atomnum + 1):
                if i != 0: file.readline()
                block_data = []
                for j in range(nedos):
                    line = [float(x) for x in file.readline().split()]
                    if i == 0: 
                        energies_list.append(line[0])
                        total_dos_list.append(line[1:])
                    else:
                        block_data.append(line[1:])
                if i > 0:
                    site_dos_list.append(block_data)
        
        self.energies = np.array(energies_list) - ef
        self.ef = ef
        self.total_dos = np.array(total_dos_list)
        self.site_dos = np.array(site_dos_list)

        num_cols = self.site_dos.shape[2]
        mapping = {
            3: ['s', 'p', 'd'],
            6: ['s_up','s_down','p_up','p_down','d_up','d_down'],
            9: ['s','py','pz','px','dxy','dyz','dz2','dxz','dx2-y2'],
            16: ['s','py','pz','px','dxy','dyz','dz2','dxz','dx2-y2','fy3x2','fxyz','fyz2','fz3','fxz2','fzx2','fx3'],
            18: ['s_up','s_down','py_up','py_down','pz_up','pz_down','px_up','px_down','dxy_up','dxy_down','dyz_up','dyz_down','dz2_up','dz2_down','dxz_up','dxz_down','dx2-y2_up','dx2-y2_down'],
            32: ['s_up','s_down','py_up','py_down','pz_up','pz_down','px_up','px_down','dxy_up','dxy_down','dyz_up','dyz_down','dz2_up','dz2_down','dxz_up','dxz_down','dx2-y2_up','dx2-y2_down','fy3x2_up','fy3x2_down','fxyz_up','fxyz_down','fyz2_up','fyz2_down','fz3_up','fz3_down','fxz2_up','fxz2_down','fzx2_up','fzx2_down','fx3_up','fx3_down']
        }
        self.orbitals = mapping.get(num_cols, [])

    def _parse_poscar(self):
        with open(self.poscar, 'r') as file:
            lines = file.readlines()
            self.atomtypes = lines[5].split()
            self.atomnums = [int(i) for i in lines[6].split()]

    def _parse_all(self):
        self._parse_doscar()
        self._parse_poscar()
        current_global = 1
        for idx, t in enumerate(self.atomtypes):
            for n_rel in range(1, self.atomnums[idx] + 1):
                self.vesta_label_map[current_global] = f"{t}{n_rel}"
                current_global += 1

    def _lighten_color(self, color, amount=0.3):
        c = mc.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)

    def plot_dos_cursors(self, nums=None, types=None, orbitals=None, full=False):
        fig, ax = plt.subplots()
        data_lines = []
        
        if full:
            total_sum = np.sum(self.total_dos, axis=1)
            line, = ax.plot(self.energies, total_sum, color='k', label='total DOS')
            data_lines.append(line)
        else:
            selected_atoms = []
            counter = 1
            for idx, t in enumerate(self.atomtypes):
                for j in range(1, self.atomnums[idx] + 1):
                    if ((not types) or (t in types)) and ((not nums) or (counter in nums)):
                        selected_atoms.append(counter)
                    counter += 1

            plot_orbitals = orbitals if orbitals else self.orbitals
            unique_bases = sorted(set(self._orbit_base(o) for o in plot_orbitals))
            styles = ['-', '--', ':', '-.'] + [(0, (3+i, 2)) for i in range(max(0, len(unique_bases)-4))]
            linestyle_map = dict(zip(unique_bases, styles))

            for a in selected_atoms:
                global_idx = a
                curr = 0
                for idx, count in enumerate(self.atomnums):
                    if global_idx <= curr + count:
                        element = self.atomtypes[idx]
                        break
                    curr += count
                
                base_color = self._type_color_map.get(element, 'grey')
                vesta_label = self.vesta_label_map[a]
                
                for orb in plot_orbitals:
                    if orb not in self.orbitals: continue
                    col_idx = self.orbitals.index(orb)
                    y_vals = self.site_dos[a-1, :, col_idx]
                    ls = linestyle_map[self._orbit_base(orb)]
                    plot_color = self._lighten_color(base_color, 0.3) if orb.endswith('_up') else base_color
                    line, = ax.plot(self.energies, y_vals, color=plot_color, linestyle=ls, label=f"{vesta_label} – {orb}")
                    data_lines.append(line)

        atom_proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes]
        leg1 = ax.legend(atom_proxies, self.atomtypes, title="Atoms (VESTA-style)", loc='upper right', frameon=False)
        ax.add_artist(leg1)

        orb_proxies = [Line2D([0], [0], color='black', linestyle=linestyle_map[b], lw=1.5) for b in unique_bases]
        ax.legend(orb_proxies, unique_bases, title="Orbitals", loc='upper left', frameon=False)

        # Optimized cursor prevents warnings and eliminates lag
        cursor = mplcursors.cursor(data_lines, hover=True)
        cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
        
        ax.set_xlabel('energy – $E_f$ / eV')
        ax.set_ylabel('DOS / states eV⁻¹')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    v_dir = r'C:/Users/Benjamin Kafin/Documents/VASP/SAM/zigzag/kpoints551/dpl_corr/kp551'
    plotter = DosPlotter(v_dir)
    plotter.plot_dos_cursors()