from typing import Union, Iterable, Optional, List
import pathlib as pl

class Layer():
    """ A Layer of a sample
    Attributes:
        Element: Element(s) of the Layer according to the SESSA notation but without enclosing slashes /
        Thickness: Thickness of the Layer in Angstrom
    """
    def __init__(self, element: Union[str, tuple], thickness:float)->None:
        if isinstance(element, str):
            self.Element = element
        elif isinstance(element, list):
            self.Element = ''.join([f'(/{i[0]}/){i[1]}' for i in element])
        self.Thickness = thickness
    
    def __repr__(self) -> str:
        if len(self.Element) > 8: 
            return str(str(self.Element).replace('/', '').replace('(', '_').replace(')', '_') + '_' + str(self.Thickness))[1:]
        return (self.Element) + '_' + str(self.Thickness)


class Experiment():
    """ An Experiment which consists of Layer objects
        The Layers are ordered from top to bottom
    Functions:
        create_folders: creates the session folder and one for results if necessary
        write_session_file: creates a file in the session folder which holds the SESSA instructions
        simulate: calls the SESSA executable and passes the session file from write_session_file to simulate spectra
    """
    def __init__(self, layers: Iterable[Layer],
                 root_dir: Union[str, pl.Path],
                 sessa_dir: Union[str, pl.Path],
                 exp_dir: Union[str, pl.Path],
                 name: Optional[str]=None,
                 etching: Optional[int]=None,
                 contamination: Optional[bool]=None,
                 shifts_probability: Optional[float]=0)-> None:
        self.Layers = [l for l in layers]
        self.root_dir = root_dir
        self.sessa_dir = sessa_dir
        self.name = name
        self.exp_dir = exp_dir
        self.etching = etching
        self.contamination = contamination
        self.lines_file = None
        self.tmp_folder = f'{self.root_dir}\\data\\simfiles\\{self.exp_dir}\\tmp\\'
        self.shifts_probability = shifts_probability

    def __repr__(self) -> str:
        layersstr = ''
        for i, layer in enumerate(self.Layers):
            if i==0:
                layersstr = str(layer)[:-3]
            else:
                layersstr = layersstr + '_' + str(layer)
        # layersstr = '_'.join([str(i) for i in self.Layers])
        # if len(layersstr)>20 : layersstr = str(self.Layers[0])
        if self.name is None:
            if 'multi' in self.exp_dir:
                return layersstr
            else:
                return layersstr.replace('/', '').replace('(', '').replace(')', '')
        else:
            return self.name

    def write_session_string(self, simulate: bool=True) -> str:
        import random
        etching_str = '100_etching' if self.etching!=None else 'separate'

        commands = ['\\PROJECT LOAD SESSION "C:\Program Files (x86)\Sessa v2.2.0\\bin/Sessa_ini.ses"',
                    '\\SPECTROMETER SET RANGE 486.6:1486.6 REGION 1',]

        for i, layer in enumerate(self.Layers):
            if '/' not in layer.Element: layer.Element = f'/{layer.Element}/'
            if i==0:
                commands.append(f'\\SAMPLE SET MATERIAL {layer.Element}')
            elif layer.Element:
                commands.append(f'\\SAMPLE ADD LAYER {layer.Element} THICKNESS {layer.Thickness} ABOVE 0')

        if self.contamination:
            commands.append(f'\\SAMPLE ADD LAYER /C/O/ THICKNESS {int(6)} ABOVE 0')

        if simulate:
            import math
            probability = self.shifts_probability
            peak_shifts = self.set_chemical_shifts()
            sampled_shifts = random.sample(peak_shifts, k=math.floor(probability * len(peak_shifts)))
            commands.extend(sampled_shifts)
            
            commands.extend([
                '\\MODEL SET CONVERGENCE 1.000e-02',
                '\\MODEL SET SE true',
                '\\MODEL SIMULATE',
                # f'\\MODEL SAVE INTENSITIES "{self.root_dir}\\data\\simulation_data\\{self.exp_dir}\\{str(self)}_output.txt"',
                f'\\MODEL SAVE SPECTRA "{self.root_dir}\\data\\simulation_data\\{self.exp_dir}\\{str(self)}_{etching_str}_spectra.spc"'
            ])
        
            
            if self.etching is not None:
                for i in range(self.etching)[::2]:
                    commands.extend([
                        '\\SAMPLE DELETE LAYER 2',
                        '\\SAMPLE DELETE LAYER 2',
                        '\\MODEL SIMULATE',
                        f'\\MODEL SAVE SPECTRA "{self.root_dir}\\data\\simulation_data\\{self.exp_dir}\\{str(self)}_{100-((i+2)*5)}_etching_spectra.spc"'
                    ])
                
        return commands

    def write_session_file(self) -> str:
        '''
        Writes a session file based on an experiment, it can be used to simulate.
        '''
        commands = self.write_session_string(simulate=True)

        with open(f'{self.root_dir}\\data\\simfiles\\{self.exp_dir}\\{str(self)}.txt', 'w') as f:
            f.writelines('\n'.join(commands))
        filename_abs = f'{self.root_dir}\\data\\simfiles\\{self.exp_dir}\\{str(self)}.txt'

        return filename_abs

    def get_chemical_shifts(self) -> list:
        import subprocess
        import os
        commands = self.write_session_string(simulate=False)
        if not os.path.exists(self.tmp_folder):
            os.makedirs(self.tmp_folder)
        commands.append(f'\\PROJECT SAVE OUTPUT "{self.tmp_folder}{str(self)}"')
        filename = f'{self.root_dir}\\data\\simfiles\\{self.exp_dir}\\tmp\\{str(self)}.txt'
        with open(filename, 'w') as f:
            f.writelines('\n'.join(commands))
        os.chdir(self.sessa_dir)
        p = subprocess.Popen('sessa.exe -s "%s"' % filename)
        p.wait(4)
        lines_file = f'{self.tmp_folder}{str(self)}sam_peak.txt'
        with open(lines_file, 'r') as f:
            lines = f.readlines()
        self.lines_file = lines

    def cleanup(self) -> None:
        import os
        for filename in os.listdir(self.tmp_folder):
            if filename.endswith(".txt"):
                os.remove(os.path.join(self.tmp_folder, filename))

    def set_chemical_shifts(self) -> None:
        import pandas as pd
        self.get_chemical_shifts()  # sets self.lines_file
        self.cleanup()
        commands = []

        spacer = 80*'-'
        first, second = [i for i, line in enumerate(self.lines_file) 
                         if line.startswith(spacer)][:2]
        xpslines = self.lines_file[first+1:second]
        xpslines = [l.split()[:-1] for l in xpslines]
        # build DataFrame from lines
        xps_df = pd.DataFrame(xpslines, columns=['num',
                                                 'Element',
                                                 'Orbital',
                                                 'Kinetic Energy [eV]',
                                                 'type'])
        # filter DataFrame for relevant lines
        for n, entry in enumerate(xps_df.iterrows()):
            # print(f'Searching for {[entry[1].Element, entry[1].Orbital]}')
            shift = self.find_similar_chemical_shifts([entry[1].Element, entry[1].Orbital])
            if shift is None:
                # print('None found')
                continue
            commands.append(f"\\SAMPLE PEAK SET POSITION {shift} PEAK {n+1} SUBPEAK 1")
        
        return commands

    def find_similar_chemical_shifts(self, search: List) -> None:
        '''
        Finds the most similar chemical shifts from the database
        '''
        import pandas as pd
        import config
        import re
        from fuzzywuzzy import process
        import numpy as np
        import base
        
        chemshift_db = pd.read_parquet(config.CHEMSHIFTS_DB_PATH)
        chemshift_db = chemshift_db.astype({"Energy": 'float',
                                            "LineDesign": 'category',
                                            "LineID": 'category'})
        
        pl_df = chemshift_db[chemshift_db['LineID'] == 'Photoelectron Line']
        pl_df.set_index(['ElmStudy', 'LineDesign'], inplace=True)
        # filter database
        filtered = pl_df.filter(like=search[0], axis=0)\
            .filter(like=search[1], axis=0)[['StrFormula', 'Energy', 'TitleEng']]

        # find all binding energies and kinetic energies
        BEs = filtered[filtered['TitleEng']=='Binding Energy (eV)']
        KEs = filtered[filtered['TitleEng']=='Kinetic Energy (eV)']
        # print('BEs: ', BEs)
        # print('KEs: ', KEs)
        
        if len(BEs)==0 and len(KEs)==0:
            return None
        
        BEs_Valid = BEs[abs(BEs['Energy']-np.median(BEs['Energy'])) < BEs['Energy'].mean()/10]# 10% deviation from median
        KEs_Valid = KEs[abs(KEs['Energy']-np.median(KEs['Energy'])) < KEs['Energy'].mean()/10]# 10% deviation from median

        BEs_Valid['Energy'] = base.binding_energy_list_to_kinetic_energy_al_kalpha(BEs_Valid['Energy']) # convert
        BEs_Valid['TitleEng'] = 'Kinetic Energy (eV)'  # rename

        new_db = pd.concat([BEs_Valid, KEs_Valid])
        # print(f'new_db: {new_db}')
        #  find best matches
        search_db = [''.join(re.findall('[A-Za-z]+', x)) for x in new_db['StrFormula'].to_list()]
        # print(search_db)
        letters = re.findall('[A-Za-z]+', str(self))
        search_string = ''.join(letters)
        # print(f'matching {search_string} with the db {search_db}')
        results = process.extractOne(search_string, search_db)
        # print(f'Results: {results}')
        if results:
            res = new_db[new_db['StrFormula'] == results[0]]
            if len(results)>0 and len(res)>0:
                choice = res.sample(1)["StrFormula"].values[0]
                energy = res.sample(1)['Energy'].values[0]
                return energy
        
    def simulate(self) -> None:
        '''
        Simulates the spectra with SESSA
        '''
        import subprocess
        import os
        filename = self.write_session_file()
        os.chdir(self.sessa_dir)
        SWHIDE = 0
        info = subprocess.STARTUPINFO()
        info.dwFlags = subprocess.STARTF_USESHOWWINDOW
        info.wShowWindow = SWHIDE
        p = subprocess.Popen('sessa.exe -s "%s"' % filename, startupinfo=info)
        p.wait()
        return None