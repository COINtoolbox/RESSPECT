import sys
import os
# sys.path.append('/home/mi/mymedia/SALT3')
sys.path.append(os.environ['SALT3_DIR'])
from salt3.pipeline.pipeline import SALT3pipe
import configparser

class SNANAHook():
    def __init__(self,data_folder=None,phot_version=None,snid_file=None,salt2mu_prefix=None,
                 fitres_prefix=None, results_dir=None,
                 stages=['lcfit'],glue=False,**kwargs):
        self.gen_input(data_folder=data_folder,phot_version=phot_version,snid_file=snid_file,
                       salt2mu_prefix=salt2mu_prefix,fitres_prefix=fitres_prefix,
                       results_dir=results_dir, **kwargs)
        self.stages = stages
        self.glue = glue

    def run(self):
        self.pipe = self.MyPipe()
        self.pipe.run()

    def MyPipe(self,**kwargs):
        pipe = SALT3pipe(finput=self.pipeinput)
        pipe.build(data=False,mode='customize',onlyrun=self.stages)
        pipe.configure()
        if 'cosmofit' in self.stages:
            pipe.CosmoFit.finput = pipe.CosmoFit.finput.replace('.temp.'+pipe.timestamp,'')
        if self.glue:
            if 'lcfit' in self.stages and 'getmu' in self.stages:
                pipe.glue(['lcfit','getmu'])
#             if 'getmu' in self.stages and 'cosmofit' in self.stages:
#                 pipe.glue(['getmu','cosmofit'])
        return pipe
    
    def gen_input(self,data_folder=None,phot_version=None,snid_file=None,salt2mu_prefix=None,
                  fitres_prefix=None,combined_fitres=None,result_dir='results/salt3',
                  outfile='salt3pipeinput.txt',tempfile='salt3pipeinput_template.txt',**kwargs):
        salt3pipe = SALT3pipe(finput=os.path.expandvars(tempfile))
        config = configparser.ConfigParser()
        config.read(salt3pipe.finput)
        m2df = salt3pipe._multivalues_to_df
        setkeys = m2df(salt3pipe._get_config_option(config,'lcfitting','set_key')).set_index('key')
        if data_folder is not None:
            setkeys.loc['PRIVATE_DATA_PATH','value'] = data_folder
        if phot_version is not None:
            setkeys.loc['VERSION_PHOTOMETRY','value'] = phot_version
        if snid_file is not None:
            setkeys.loc['SNCID_LIST_FILE','value'] = snid_file
        if fitres_prefix is not None:
            setkeys.loc['TEXTFILE_PREFIX','value'] = fitres_prefix
        setkey_string = self._df_to_string(setkeys.reset_index(),has_section=True)
        config['lcfitting']['set_key'] = setkey_string
        
        setkeys = m2df(salt3pipe._get_config_option(config,'getmu','set_key')).set_index('key')
        if salt2mu_prefix is not None:
            setkeys.loc['prefix','value'] = salt2mu_prefix
        if combined_fitres is not None:
            setkeys.loc['file','value'] = combined_fitres
        setkey_string = self._df_to_string(setkeys.reset_index(),has_section=False)
        config['getmu']['set_key'] = setkey_string
        
        #add result_dir and check if output folder exists
        for sec in ['lcfitting','getmu']:
            outname = config[sec]['outinput']
            outname = outname.replace('REPLACE_RESULT_DIR',result_dir)
            config[sec]['outinput'] = outname
            folder = os.path.split(outname)[0]
            if not os.path.isdir(folder):
                os.mkdir(folder)
        #replace cosmofit outinput
        for sec in ['cosmofit']:
            outname = config[sec]['outinput']
            outname = outname.replace('REPLACE_SALT2MU_M0DIF',salt2mu_prefix+'.M0DIF')
            config[sec]['outinput'] = outname
        
        with open(outfile, 'w') as configfile:
            config.write(configfile)
        self.pipeinput = outfile
        
    def _df_to_string(self,df,has_section=True,has_label=True):
        outstring = '{}\n'.format(len(df.columns)) 
        for i,row in df.iterrows():
            if has_section:
                if not has_label:
                    outstring += '{} {} {}\n'.format(row['section'],row['key'],row['value'])
                else:
                    outstring += '{} {} {} {}\n'.format(row['label'],row['section'],row['key'],row['value'])                    
            else:
                outstring += '{} {}\n'.format(row['key'],row['value'])
        return outstring

