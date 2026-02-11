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
        self.vesta_label_map = {} 
        
        # Hard-coded VESTA-style element colors
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

    def _get_element_by_index(self, a):
        curr = 0
        for idx, count in enumerate(self.atomnums):
            if a <= curr + count:
                return self.atomtypes[idx]
            curr += count
        return 'grey'

    def _parse_doscar(self):
        with open(self.doscar, 'r') as file:
            atomnum = int(file.readline().split()[0])
            for _ in range(4): file.readline()
            nedos, ef = self._extract_ef_and_nedos(file.readline().strip())
            
            energies_list, total_dos_list, site_dos_list = [], [], []
            
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
            9: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'],
            18: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down', 'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down', 'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down'],
            32: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down', 'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down', 'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down', 'fy3x2_up', 'fy3x2_down', 'fxyz_up', 'fxyz_down', 'fyz2_up', 'fyz2_down', 'fz3_up', 'fz3_down', 'fxz2_up', 'fxz2_down', 'fzx2_up', 'fzx2_down', 'fx3_up', 'fx3_down']
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

    def plot_dos_cursors(self, nums=None, types=None):
        fig, ax = plt.subplots()
        atom_sum_lines, atom_orb_lines = {}, {} 
        self.active_atom, self.orb_cursor = None, None 
        
        selected_atoms = []
        counter = 1
        for idx, t in enumerate(self.atomtypes):
            for j in range(1, self.atomnums[idx] + 1):
                if ((not types) or (t in types)) and ((not nums) or (counter in nums)):
                    selected_atoms.append(counter)
                counter += 1

        unique_bases = sorted(set(self._orbit_base(o) for o in self.orbitals))
        styles = ['-', '--', ':', '-.'] + [(0, (3+i, 2)) for i in range(max(0, len(unique_bases)-4))]
        linestyle_map = dict(zip(unique_bases, styles))

        BASE_Z, FRONT_Z_ATOM, FRONT_Z_ORB = 2, 10, 11

        for a in selected_atoms:
            element = self._get_element_by_index(a)
            base_color = self._type_color_map.get(element, 'grey')
            
            y_sum = np.sum(self.site_dos[a-1], axis=1)
            sum_line, = ax.plot(self.energies, y_sum, color=base_color, lw=2, 
                                label=self.vesta_label_map[a], picker=True, pickradius=3, zorder=BASE_Z)
            atom_sum_lines[a] = sum_line
            
            orb_list = []
            for orb in self.orbitals:
                col_idx = self.orbitals.index(orb)
                y_orb = self.site_dos[a-1, :, col_idx]
                ls = linestyle_map[self._orbit_base(orb)]
                p_color = self._lighten_color(base_color, 0.3) if orb.endswith('_up') else base_color
                o_line, = ax.plot(self.energies, y_orb, color=p_color, linestyle=ls, 
                                  lw=1.2, visible=False, label=f"{self.vesta_label_map[a]} – {orb}", 
                                  zorder=BASE_Z-1)
                orb_list.append(o_line)
            atom_orb_lines[a] = orb_list

        atom_proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes]
        ax.legend(atom_proxies, self.atomtypes, title="Atoms (VESTA-style)", loc='upper right', frameon=False)
        orb_proxies = [Line2D([0], [0], color='black', linestyle=linestyle_map[b], lw=1.5) for b in unique_bases]
        ax.legend(orb_proxies, unique_bases, title="Orbitals", loc='upper left', frameon=False).set_zorder(100)

        def update_plot_visuals():
            if self.orb_cursor:
                self.orb_cursor.remove()
                self.orb_cursor = None

            # User-adjustable desaturation factor (0 = grey, 1 = full color)
            S = 0.25 

            if self.active_atom is None:
                for a, line in atom_sum_lines.items():
                    element = self._get_element_by_index(a)
                    line.set_color(self._type_color_map.get(element, 'grey'))
                    line.set_alpha(1.0)
                    line.set_zorder(BASE_Z)
                    for o_line in atom_orb_lines[a]: o_line.set_visible(False)
            else:
                for a, line in atom_sum_lines.items():
                    element = self._get_element_by_index(a)
                    orig_color = mc.to_rgb(self._type_color_map.get(element, 'grey'))
                    
                    if a == self.active_atom:
                        line.set_color(orig_color)
                        line.set_alpha(1.0)
                        line.set_zorder(FRONT_Z_ATOM)
                        active_orbs = atom_orb_lines[a]
                        for o_line in active_orbs:
                            o_line.set_visible(True)
                            o_line.set_zorder(FRONT_Z_ORB)
                        self.orb_cursor = mplcursors.cursor(active_orbs, hover=True)
                        self.orb_cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
                    else:
                        # Partial Desaturation Logic [cite: 2026-02-11]
                        lumi = 0.299*orig_color[0] + 0.587*orig_color[1] + 0.114*orig_color[2]
                        new_rgb = (S * np.array(orig_color)) + ((1 - S) * lumi)
                        
                        line.set_color(new_rgb)
                        line.set_alpha(0.05) 
                        line.set_zorder(BASE_Z)
                        for o_line in atom_orb_lines[a]: o_line.set_visible(False)
            fig.canvas.draw_idle()

        def on_pick(event):
            clicked_idx = next((a for a, line in atom_sum_lines.items() if line == event.artist), None)
            if clicked_idx is None: return
            self.active_atom = None if self.active_atom == clicked_idx else clicked_idx
            update_plot_visuals()

        fig.canvas.mpl_connect('pick_event', on_pick)
        ax.set_xlabel('energy – $E_f$ / eV')
        ax.set_ylabel('DOS / states eV⁻¹')
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    v_dir = r'C:/dir'
    plotter = DosPlotter(v_dir)
    plotter.plot_dos_cursors()
