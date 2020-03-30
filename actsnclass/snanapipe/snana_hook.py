import sys
sys.path.append('/home/mi/mymedia/SALT3/salt3')
from pipeline.pipeline import SALT3pipe
import configparser

class SNANAHook():
    def __init__(self,data_folder=None,phot_version=None,snid_file=None,salt2mu_prefix=None,**kwargs):
        self.gen_input(data_folder=data_folder,phot_version=phot_version,snid_file=snid_file,
                       salt2mu_prefix=salt2mu_prefix,**kwargs)

    def run(self):
        self.pipe = self.MyPipe()
        self.pipe.run()

    def MyPipe(self,**kwargs):
        pipe = SALT3pipe(finput=self.pipeinput)
        pipe.build(data=False,mode='customize',onlyrun=['lcfit','getmu'])
        pipe.configure()
        pipe.glue(['lcfit','getmu'])
        return pipe
    
    def gen_input(self,data_folder=None,phot_version=None,snid_file=None,salt2mu_prefix=None,
                  outfile='salt3pipeinput.txt',tempfile='salt3pipeinput_template.txt'):
        salt3pipe = SALT3pipe(finput='salt3pipeinput_template.txt')
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
        setkey_string = self._df_to_string(setkeys.reset_index(),has_section=True)
        config['lcfitting']['set_key'] = setkey_string
        
        setkeys = m2df(salt3pipe._get_config_option(config,'getmu','set_key')).set_index('key')
        if salt2mu_prefix is not None:
            setkeys.loc['prefix','value'] = salt2mu_prefix
        setkey_string = self._df_to_string(setkeys.reset_index(),has_section=False)
        config['getmu']['set_key'] = setkey_string
        
        with open(outfile, 'w') as configfile:
            config.write(configfile)
        self.pipeinput = outfile
        
    def _df_to_string(self,df,has_section=True):
        outstring = '{}\n'.format(len(df.columns)) 
        for i,row in df.iterrows():
            if has_section:
                outstring += '{} {} {}\n'.format(row['section'],row['key'],row['value'])
            else:
                outstring += '{} {}\n'.format(row['key'],row['value'])
        return outstring

