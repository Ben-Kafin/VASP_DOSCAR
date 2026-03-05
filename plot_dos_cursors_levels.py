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
        self.element_dos = {}
        self.molecule_dos = None
        
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
        mol_sum = np.zeros_like(self.energies)
        for idx, t in enumerate(self.atomtypes):
            atom_indices = range(current_global - 1, current_global - 1 + self.atomnums[idx])
            e_dos = np.sum(self.site_dos[atom_indices, :, :], axis=(0, 2))
            self.element_dos[t] = e_dos
            if t != 'Au':
                mol_sum += e_dos
            for n_rel in range(1, self.atomnums[idx] + 1):
                self.vesta_label_map[current_global] = f"{t}{n_rel}"
                current_global += 1
        self.molecule_dos = mol_sum

    def _lighten_color(self, color, amount=0.3):
        c = mc.to_rgb(color)
        h, l, s = colorsys.rgb_to_hls(*c)
        return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)

    def plot_dos_cursors(self, nums=None, types=None):
        fig, ax = plt.subplots()
        self.plot_level = 0 
        self.active_element, self.active_atom = None, None
        
        unique_bases = sorted(set(self._orbit_base(o) for o in self.orbitals))
        styles = ['-', '--', ':', '-.'] + [(0, (3+i, 2)) for i in range(max(0, len(unique_bases)-4))]
        linestyle_map = dict(zip(unique_bases, styles))

        def _rescale_y_axis():
            lines = ax.get_lines()
            active_maxes = [np.max(l.get_ydata()) for l in lines if l.get_visible() and l.get_alpha() == 1.0]
            if active_maxes:
                ax.set_ylim(0, max(active_maxes) * 1.1)

        def update_plot_visuals():
            ax.cla() 
            S = 0.25 

            if self.plot_level == 0:
                total_y = np.sum(self.total_dos[:, :int(self.total_dos.shape[1]/2)], axis=1) if self.total_dos.shape[1] > 1 else self.total_dos[:, 0]
                ax.plot(self.energies, total_y, color='black', lw=2.5, label='Total DOS', picker=True, pickradius=5)
                ax.legend(loc='upper right', frameon=False)

            elif self.plot_level == 1:
                total_y = np.sum(self.total_dos[:, :int(self.total_dos.shape[1]/2)], axis=1) if self.total_dos.shape[1] > 1 else self.total_dos[:, 0]
                ax.plot(self.energies, total_y, color='black', lw=1.5, alpha=0.1, zorder=1)
                # Partition picking logic [cite: 2026-03-05]
                ax.plot(self.energies, self.element_dos['Au'], color='orange', lw=2, label='Au', picker=True, pickradius=5, zorder=2)
                ax.plot(self.energies, self.molecule_dos, color='black', lw=2, label='Molecule', picker=True, pickradius=5, zorder=2)
                proxies = [Line2D([0], [0], color='orange', lw=2), Line2D([0], [0], color='black', lw=2)]
                ax.legend(proxies, ['Au', 'Molecule'], title="Partition", loc='upper right', frameon=False)

            elif self.plot_level == 2:
                for t in self.atomtypes:
                    ax.plot(self.energies, self.element_dos[t], color=self._type_color_map.get(t, 'grey'), 
                            lw=2, label=t, picker=True, pickradius=3)
                proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes]
                ax.legend(proxies, self.atomtypes, title="Atom Types", loc='upper right', frameon=False)

            elif self.plot_level == 3:
                ax.plot(self.energies, self.element_dos[self.active_element], color=self._type_color_map.get(self.active_element, 'grey'), 
                        lw=2, alpha=0.15, zorder=1)
                for a_idx, label in self.vesta_label_map.items():
                    if label.startswith(self.active_element):
                        y_sum = np.sum(self.site_dos[a_idx-1], axis=1)
                        ax.plot(self.energies, y_sum, color=self._type_color_map.get(self.active_element, 'grey'), 
                                lw=2, label=label, picker=True, pickradius=3, zorder=2)
                proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes]
                ax.legend(proxies, self.atomtypes, title="Atom Types", loc='upper right', frameon=False)

            elif self.plot_level == 4:
                for a_idx, label in self.vesta_label_map.items():
                    if label.startswith(self.active_element):
                        y_sum = np.sum(self.site_dos[a_idx-1], axis=1)
                        element_color = self._type_color_map.get(self.active_element, 'grey')
                        if a_idx == self.active_atom:
                            ax.plot(self.energies, y_sum, color=element_color, lw=2.5, alpha=1.0, zorder=5)
                            orb_artists = []
                            for orb in self.orbitals:
                                col_idx = self.orbitals.index(orb)
                                y_orb = self.site_dos[a_idx-1, :, col_idx]
                                ls = linestyle_map[self._orbit_base(orb)]
                                p_color = self._lighten_color(element_color, 0.3) if orb.endswith('_up') else element_color
                                o_line, = ax.plot(self.energies, y_orb, color=p_color, linestyle=ls, lw=1.2, label=f"{label} – {orb}", zorder=10)
                                orb_artists.append(o_line)
                            cursor = mplcursors.cursor(orb_artists, hover=True)
                            cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
                        else:
                            orig = mc.to_rgb(element_color)
                            lumi = 0.299*orig[0] + 0.587*orig[1] + 0.114*orig[2]
                            faded_color = (S * np.array(orig)) + ((1 - S) * lumi)
                            ax.plot(self.energies, y_sum, color=faded_color, lw=1.5, alpha=0.05, zorder=2)
                atom_proxies = [Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes]
                leg1 = ax.legend(atom_proxies, self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
                ax.add_artist(leg1)
                orb_proxies = [Line2D([0], [0], color='black', linestyle=linestyle_map[b], lw=1.5) for b in unique_bases]
                ax.legend(orb_proxies, unique_bases, title="Orbitals", loc='upper left', frameon=False)

            ax.set_xlabel('energy – $E_f$ / eV')
            ax.set_ylabel('DOS / states eV⁻¹')
            _rescale_y_axis()
            fig.canvas.draw_idle()

        def on_pick(event):
            label = event.artist.get_label()
            if self.plot_level == 0:
                self.plot_level = 1
            elif self.plot_level == 1:
                # Triggers for both 'Au' and 'Molecule' artists [cite: 2026-03-05]
                self.plot_level = 2
            elif self.plot_level == 2:
                self.active_element = label
                self.plot_level = 3
            elif self.plot_level == 3:
                self.active_atom = next(k for k, v in self.vesta_label_map.items() if v == label)
                self.active_element = self._get_element_by_index(self.active_atom)
                self.plot_level = 4
            update_plot_visuals()

        def on_click(event):
            if event.inaxes != ax: return
            if fig.canvas.manager.toolbar.mode == "" and not event.dblclick:
                hit = any(l.contains(event)[0] for l in ax.get_lines() if l.get_picker())
                if not hit:
                    self.plot_level = max(0, self.plot_level - 1)
                    update_plot_visuals()

        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('button_press_event', on_click)
        update_plot_visuals()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    v_dir = r'C:/dir'
    plotter = DosPlotter(v_dir)
    plotter.plot_dos_cursors()